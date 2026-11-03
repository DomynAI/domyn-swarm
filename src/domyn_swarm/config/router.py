# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Load-balancer router selection for the Slurm endpoint."""

import json
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from domyn_swarm import utils

THUNDERAGENT_POLICY = (
    Path(__file__).parent.parent / "templates" / "routers" / ("dynamo_thunderagent.yaml")
)

_PROGRAM_SCHEDULING_CONFIG = {"program_scheduling_enable_key": "auto", "global_queue": True}

# The templates render each key into a shell `export KEY=...`.
_EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class _RouterBase(BaseModel):
    """Fields and hooks shared by every router."""

    serve_engine: ClassVar[str] = "vllm"

    def metrics_target(self) -> tuple[str, str] | None:
        """Return the `(address, path)` Prometheus scrapes, or `None` without metrics."""
        return None


class NginxRouter(_RouterBase):
    """Plain nginx balancing with `least_conn`, the default.

    nginx sends each request to the replica with the fewest active connections.
    """

    type: Literal["nginx"] = Field(
        default="nginx",
        description="Router that balances requests across replicas.",
    )


class _ExternalRouter(_RouterBase):
    """Router that runs as a separate process behind Nginx."""

    image: str | utils.EnvPath | None = Field(
        default=None,
        description=(
            "Path to a Singularity image that runs the router. Set exactly one of "
            "`image` or `binary`."
        ),
    )
    binary: str | utils.EnvPath | None = Field(
        default=None,
        description=(
            "Router executable on the load-balancer node, either a bare name on "
            "PATH or an absolute path. Set exactly one of `image` or `binary`."
        ),
    )
    port: int = Field(
        default=9200,
        description=(
            "Local port on the load-balancer node where the router listens. nginx "
            "forwards the `llm` location to it."
        ),
    )
    metrics_port: int = Field(
        default=9201,
        description="Local port where the router serves Prometheus metrics.",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Extra command-line arguments appended to the router command.",
    )
    env: dict[_EnvName, str] = Field(
        default_factory=dict,
        description=(
            "Extra environment variables set for the router process. Each name is a "
            "shell identifier: letters, digits and underscores, not starting with a digit."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_launcher(self) -> "_ExternalRouter":
        if (self.image is None) == (self.binary is None):
            raise ValueError("set exactly one of `image` or `binary`")
        return self

    def metrics_target(self) -> tuple[str, str] | None:
        return (f"127.0.0.1:{self.metrics_port}", "/metrics")


class VllmRouter(_ExternalRouter):
    """The vLLM production-stack router."""

    type: Literal["vllm-router"] = Field(
        default="vllm-router",
        description="Router that balances requests across replicas.",
    )
    policy: Literal["cache_aware", "consistent_hash", "program", "round_robin"] = Field(
        default="cache_aware",
        description=(
            "Routing policy. `program` is cache-aware routing with program scheduling enabled."
        ),
    )

    def policy_args(self) -> list[str]:
        """Return the router command-line arguments that select `policy`."""
        if self.policy == "program":
            return [
                "--policy",
                "cache_aware",
                "--enable-program-scheduling",
                "--program-scheduling-config-json",
                json.dumps(_PROGRAM_SCHEDULING_CONFIG, separators=(",", ":")),
            ]
        return ["--policy", self.policy]


class RamjetRouter(_ExternalRouter):
    """The ramjet router."""

    type: Literal["ramjet"] = Field(
        default="ramjet",
        description="Router that balances requests across replicas.",
    )
    policy: Literal["prefix", "load"] = Field(
        default="prefix",
        description=(
            "Replica affinity. `prefix` routes requests that share a prompt prefix "
            "to the same replica; `load` routes on replica load only."
        ),
    )

    def policy_env(self) -> dict[str, str]:
        """Return the environment variables that select `policy`."""
        return {"RJ_AFFINITY": self.policy}


class DynamoRouter(_RouterBase):
    """The NVIDIA Dynamo frontend, which also switches replicas to Dynamo workers."""

    serve_engine: ClassVar[str] = "dynamo"

    type: Literal["dynamo"] = Field(
        default="dynamo",
        description="Router that balances requests across replicas.",
    )
    port: int = Field(
        default=9200,
        description=(
            "Local port on the load-balancer node where the Dynamo frontend "
            "listens. nginx forwards the `llm` location to it. The frontend also "
            "serves Prometheus metrics on this port."
        ),
    )
    args: list[str] = Field(
        default_factory=list,
        description="Extra command-line arguments appended to the frontend command.",
    )
    env: dict[_EnvName, str] = Field(
        default_factory=dict,
        description=(
            "Extra environment variables set for the frontend process. Each name is a "
            "shell identifier: letters, digits and underscores, not starting with a digit."
        ),
    )
    policy: Literal["kv_session_affinity", "thunderagent", "kv", "round_robin"] = Field(
        default="kv_session_affinity",
        description=(
            "Routing policy. `kv_session_affinity` routes on KV-cache overlap and "
            "keeps a session on one replica. `thunderagent` adds the ThunderAgent "
            "request classifier. `kv` routes on KV-cache overlap alone."
        ),
    )
    session_affinity_ttl_s: int = Field(
        default=1800,
        description=(
            "Seconds an idle session stays bound to its replica. Applies to "
            "`kv_session_affinity` and `thunderagent`."
        ),
    )
    policy_config: utils.EnvPath | None = Field(
        default=None,
        description=(
            "Path to a router policy YAML. With `thunderagent` it replaces the shipped "
            "ThunderAgent file, and with `kv_session_affinity` the frontend loads it. "
            "`kv` and `round_robin` ignore it."
        ),
    )
    venv: utils.EnvPath | None = Field(
        default=None,
        description=(
            "Python virtual environment with Dynamo installed. Its `bin/python` "
            "runs the frontend and the workers."
        ),
    )
    kv_events_port: int = Field(
        default=20080,
        description="Base port on which workers publish KV-cache events to the frontend.",
    )

    @property
    def uses_kv_events(self) -> bool:
        """Whether the policy routes on the KV-cache events that workers publish."""
        return self.policy != "round_robin"

    def resolved_policy_config(self) -> str | None:
        """Return the policy YAML the frontend loads, or `None` when it needs none."""
        if self.policy_config is not None:
            return str(self.policy_config)
        if self.policy == "thunderagent":
            return str(THUNDERAGENT_POLICY)
        return None

    def frontend_args(self) -> list[str]:
        """Return the frontend command-line arguments that select `policy`."""
        if not self.uses_kv_events:
            return ["--router-mode", "round-robin"]
        args = ["--router-mode", "kv", "--router-kv-events"]
        if self.policy == "kv":
            return args
        args += [
            "--router-session-affinity-mode",
            "soft",
            "--router-session-affinity-ttl-secs",
            str(self.session_affinity_ttl_s),
        ]
        policy_config = self.resolved_policy_config()
        if policy_config is not None:
            args += ["--router-policy-config", policy_config]
        return args

    def metrics_target(self) -> tuple[str, str] | None:
        return (f"127.0.0.1:{self.port}", "/metrics")


RouterConfig = Annotated[
    NginxRouter | VllmRouter | RamjetRouter | DynamoRouter,
    Field(discriminator="type"),
]
