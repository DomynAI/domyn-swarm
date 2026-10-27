# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import io
import math
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    StringConstraints,
    field_validator,
    model_validator,
)
import yaml

from domyn_swarm import utils
from domyn_swarm.config.backend import BackendConfig
from domyn_swarm.config.defaults import default_for
from domyn_swarm.config.plan import DeploymentPlan, PlanBuilder
from domyn_swarm.config.settings import get_settings
from domyn_swarm.config.watchdog import WatchdogConfig
from domyn_swarm.helpers.io import to_path
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger(__name__)

_DYNAMO_REJECTED_VLLM_FLAGS = ("--enable-auto-tool-choice", "--tool-call-parser")


class DomynLLMSwarmConfig(BaseModel):
    """Validated configuration for one serving swarm.

    Field descriptions are the canonical user-facing reference for individual
    settings. Use `read` and `persist` for YAML-backed configuration and
    `build_plan` to resolve the configured serving and compute backends.
    """

    # model / revision --------------------------------------------------------
    model: str = Field(
        description=(
            "Hugging Face model ID or local path. Passed verbatim to `vllm serve`; "
            "must resolve to a local directory or an offline Hugging Face model in "
            "`HF_HOME`."
        ),
    )
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, to_lower=True, max_length=38),
    ] = Field(
        description=(
            "Name of the swarm. Stripped of surrounding whitespace, lower-cased and "
            "limited to 38 characters so it fits backend resource-name limits. The "
            "swarm's unique id is this name plus a short suffix."
        ),
    )
    revision: str | None = Field(
        default=None,
        description="Git tag or commit for the model, when loading it from Hugging Face.",
    )

    # resources ---------------------------------------------------------------
    replicas: int = Field(
        default=1,
        description=("How many independent vLLM clusters to launch. Useful for A/B tests."),
    )
    gpus_per_replica: int = Field(
        default=4,
        description=("How many GPUs each replica uses. Also sets vLLM's `--tensor-parallel-size`."),
    )

    gpus_per_node: int = Field(
        description="Number of GPUs per node (vLLM)",
        default=4,
        ge=1,
        le=4,
    )
    replicas_per_node: int | None = Field(
        description="Number of model replicas per node (vLLM)", default=None
    )
    nodes: int | None = Field(
        description="Number of nodes to use for the swarm (vLLM)", default=None
    )
    cpus_per_task: int | None = Field(
        description="Number of CPUs per task (vLLM)",
        ge=1,
        default=None,
    )
    mem_per_cpu: str | None = Field(
        default=None,
        description=(
            "Memory to request per CPU, e.g. `4GB`. Reserved: no backend currently "
            "reads this field."
        ),
    )
    wait_endpoint_s: int = Field(
        default=1200,
        description=(
            "Seconds the load-balancer script waits for the endpoint to come up before giving up."
        ),
    )

    image: str | utils.EnvPath = Field(
        default_factory=default_for("image"),
        description=(
            "Container image for the vLLM replicas: a path to a Singularity image "
            "on the Slurm backend, or a Docker image on Lepton."
        ),
    )

    args: str = Field(
        default="",
        description=(
            "Extra CLI flags passed verbatim to `python -m vllm.entrypoints.openai.api_server`, "
            "or to `dynamo.vllm` with the Dynamo router."
        ),
    )
    port: int = Field(
        default=8000,
        description="Port on which each replica's OpenAI-compatible API listens.",
    )

    home_directory: utils.EnvPath = Field(
        default_factory=lambda: utils.EnvPath(get_settings().home),
        description="Home directory where logs and state are stored",
    )

    backend: BackendConfig | None = Field(
        description="Backend configuration for the swarm",
    )
    _plan: DeploymentPlan | None = PrivateAttr(default=None)

    env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Environment variables to set on the replica processes, as a mapping of name to value."
        ),
    )
    watchdog: WatchdogConfig = Field(
        default_factory=WatchdogConfig,
        description=(
            "Watchdog settings governing how the spawned vLLM replicas are monitored and restarted."
        ),
    )

    # Convenience accessor
    def get_deployment_plan(self) -> DeploymentPlan | None:
        return self._plan

    def build_plan(self) -> DeploymentPlan:
        builder = PlanBuilder(self)
        self._plan = builder.build()
        return self._plan

    @field_validator("backend")
    @classmethod
    def not_empty(cls, v: BackendConfig | None) -> BackendConfig:
        if not v:
            raise ValueError("At least one backend must be configured")
        return v

    @staticmethod
    def _resolve_ray_metrics(backend: dict, requires_ray: bool) -> None:
        """Auto-resolve ``monitoring.ray_metrics.enabled`` on a backend dict.

        Mirrors how ``watchdog.ray.enabled`` is derived: when monitoring is on
        and ``ray_metrics.enabled`` hasn't been set explicitly, it becomes
        ``True`` iff the deployment requires Ray; explicit values are left
        untouched.

        Args:
            backend: The (possibly mutated in place) backend config dict.
            requires_ray: Whether this deployment requires Ray.
        """
        endpoint = backend.get("endpoint")
        if not isinstance(endpoint, dict):
            return
        mon = endpoint.get("monitoring")
        if not (isinstance(mon, dict) and mon.get("enabled")):
            return
        rm = mon.setdefault("ray_metrics", {})
        if isinstance(rm, dict) and rm.get("enabled") is None:
            rm["enabled"] = bool(requires_ray)

    @classmethod
    def read(cls, path: str) -> "DomynLLMSwarmConfig":
        config_path = to_path(path)
        return _load_swarm_config(config_path.open())

    def persist(self, path: str | Path) -> None:
        config_path = to_path(path)
        with config_path.open("w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f)

    @model_validator(mode="before")
    @classmethod
    def validate_resource_allocations(cls, data: Any) -> "DomynLLMSwarmConfig":
        """Validate and auto-compute all derived resource allocation fields."""
        replicas = data.get("replicas", 1)
        gpus_per_replica = data.get("gpus_per_replica", 4)
        gpus_per_node = data.get("gpus_per_node", 4)
        replicas_per_node = data.get("replicas_per_node")

        # Replicas per node
        if replicas_per_node is None:
            if gpus_per_replica <= gpus_per_node:
                capacity = gpus_per_node // gpus_per_replica
                replicas_per_node = min(capacity, replicas)
            else:
                replicas_per_node = None

        # Nodes
        if replicas_per_node:
            nodes = math.ceil(replicas / replicas_per_node)
        else:
            nodes = math.ceil((replicas * gpus_per_replica) / gpus_per_node)

        if nodes < 1:
            raise ValueError("Number of nodes must be >= 1")

        # CPUs per task
        cpus_per_task = data.get("cpus_per_task")
        if cpus_per_task is None:
            cpus_per_task = max(1, 32 // replicas_per_node) if replicas_per_node else 32

        # Requires Ray?
        requires_ray = gpus_per_replica > gpus_per_node and nodes > 1
        # Ensure watchdog config exists and update ray settings
        if "watchdog" not in data:
            data["watchdog"] = {}
        if "ray" not in data["watchdog"]:
            data["watchdog"]["ray"] = {}
        data["watchdog"]["ray"]["enabled"] = requires_ray

        if requires_ray and gpus_per_replica % gpus_per_node != 0:
            raise ValueError(
                "When gpus_per_replica > gpus_per_node, gpus_per_replica "
                "must be a multiple of gpus_per_node"
            )

        # Fill computed fields
        data["replicas_per_node"] = replicas_per_node
        data["nodes"] = nodes
        data["cpus_per_task"] = cpus_per_task

        # Update backend configurations with computed values
        backend = data.get("backend", [])
        if backend:
            if not isinstance(backend, dict):
                backend = backend.model_dump()
            if backend.get("type") == "slurm" and "requires_ray" not in backend:
                backend["requires_ray"] = requires_ray
            # Ray multi-node deployments render from a dedicated template.
            # Select it automatically unless the user pinned a custom path.
            if backend.get("type") == "slurm" and requires_ray and "template_path" not in backend:
                from domyn_swarm.config import slurm as _slurm_mod

                backend["template_path"] = (
                    utils.EnvPath(_slurm_mod.__file__).parent.parent
                    / "templates"
                    / "llm_swarm_ray.sh.j2"
                )

            cls._resolve_ray_metrics(backend, requires_ray)

        data["backend"] = backend

        return data

    @model_validator(mode="after")
    def _finalize_ray_metrics(self) -> "DomynLLMSwarmConfig":
        """Resolve ``ray_metrics.enabled`` to a concrete bool in all cases.

        The before-validator (``validate_resource_allocations``) only auto-resolves
        ``ray_metrics.enabled`` when a monitoring block is present and enabled. This
        after-validator runs once all nested objects exist (with their defaults) and
        is the final authority: it guarantees ``ray_metrics.enabled`` is never left
        as ``None`` after a full config is validated, regardless of whether
        monitoring is disabled or the monitoring block was absent entirely. Explicit
        ``True``/``False`` values set by the user are always respected.
        """
        from domyn_swarm.config.slurm import SlurmConfig

        be = self.backend
        if isinstance(be, SlurmConfig):
            rm = be.endpoint.monitoring.ray_metrics
            if rm.enabled is None:
                rm.enabled = bool(be.endpoint.monitoring.enabled and be.requires_ray)
        return self

    @model_validator(mode="after")
    def _reject_dynamo_with_ray(self) -> "DomynLLMSwarmConfig":
        from domyn_swarm.config.slurm import SlurmConfig

        be = self.backend
        if isinstance(be, SlurmConfig) and be.requires_ray and be.endpoint.router.type == "dynamo":
            raise ValueError(
                "Dynamo routing does not support Ray mode (one replica across several nodes)"
            )
        return self

    @model_validator(mode="after")
    def _reject_vllm_tool_call_args_with_dynamo(self) -> "DomynLLMSwarmConfig":
        from domyn_swarm.config.slurm import SlurmConfig

        be = self.backend
        if not (isinstance(be, SlurmConfig) and be.endpoint.router.type == "dynamo"):
            return self
        # vLLM accepts underscores in flag names and normalizes them to hyphens.
        flags = {token.split("=", 1)[0].replace("_", "-") for token in self.args.split()}
        rejected = [flag for flag in _DYNAMO_REJECTED_VLLM_FLAGS if flag in flags]
        if rejected:
            raise ValueError(
                f"dynamo.vllm exits on {' and '.join(rejected)} in `args`. Use the Dynamo "
                "flags --dyn-tool-call-parser and --dyn-reasoning-parser instead"
            )
        return self


def _load_swarm_config(
    config_file: io.TextIOWrapper,
    *,
    replicas: int | None = None,
) -> DomynLLMSwarmConfig:
    """Load YAML, inject driver_script if given, apply replicas override."""
    cfg_dict = yaml.safe_load(config_file)
    cfg = DomynLLMSwarmConfig.model_validate(cfg_dict)
    # override default only if user passed something truthy
    if replicas:
        cfg.replicas = replicas
    return cfg
