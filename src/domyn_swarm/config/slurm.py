# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from domyn_swarm import utils
from domyn_swarm.config.defaults import default_for
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.config.router import NginxRouter, RouterConfig
from domyn_swarm.platform.protocols import ServingHandle

_DCGM_DEFAULT_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04"


class GpuExporterConfig(BaseModel):
    """Optional per-node GPU metrics exporter for replica nodes.

    Disabled by default. `kind` selects both the exporter implementation and the
    metric vocabulary the bundled dashboard expects, so changing it changes which
    dashboard `domyn-swarm monitor --gpu` loads.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Run a GPU exporter on every replica node. Requires "
            "`monitoring.enabled`; on its own it does nothing."
        ),
    )
    kind: Literal["nvidia_smi", "dcgm"] = Field(
        default="nvidia_smi",
        description=(
            "Which exporter to run. `nvidia_smi` is a small static binary that "
            "works unprivileged wherever `nvidia-smi` is present. `dcgm` emits "
            "NVIDIA's standard `DCGM_FI_*` series but is pinned to the 3.x line, "
            "because 4.x aborts when run unprivileged."
        ),
    )
    image: str | None = Field(
        default=None,
        description=(
            "Singularity image for the exporter. Required for `nvidia_smi` with "
            "`mode: container`; `dcgm` falls back to a public NVIDIA image."
        ),
    )
    binary: str | None = Field(
        default=None,
        description=(
            "Exporter binary for `mode: binary`. Defaults to "
            "`nvidia_gpu_exporter` on PATH, and is required for `dcgm`."
        ),
    )
    port: int = Field(
        default=9835,
        description="Port each node's exporter listens on for Prometheus to scrape.",
    )

    def resolved_binary(self, *, mode: str) -> str:
        if self.binary:
            return self.binary
        if self.kind == "nvidia_smi":
            return "nvidia_gpu_exporter"
        raise ValueError("gpu_exporter.binary is required for kind='dcgm' with mode='binary'")

    def resolved_image(self, *, mode: str) -> str | None:
        if self.image:
            return self.image
        if self.kind == "dcgm":
            return _DCGM_DEFAULT_IMAGE
        return None  # nvidia_smi container mode requires an explicit image


class RayMetricsConfig(BaseModel):
    """Optional scraping of Ray's per-node Prometheus metrics (`ray_*`).

    Only meaningful for Ray multi-node replicas. Left unset it resolves itself:
    on when monitoring is on and the deployment requires Ray, off otherwise.
    """

    enabled: bool | None = Field(
        default=None,
        description=(
            "Scrape Ray's own `ray_*` metrics from every node. Unset means auto: "
            "`true` when monitoring is enabled and the deployment requires Ray, "
            "`false` otherwise. An explicit value is always respected, and the "
            "field is never left unset on a validated config."
        ),
    )
    port: int = Field(
        default=8090,
        description=(
            "Ray's `--metrics-export-port`. Fixed rather than ephemeral so the "
            "per-node announce files Prometheus reads have stable contents."
        ),
    )


class MonitoringConfig(BaseModel):
    """Optional Prometheus-based monitoring sidecars for the load-balancer node.

    Disabled by default; when disabled the load balancer behaves exactly as it
    did before monitoring existed. Slurm only.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Run Prometheus and an Nginx exporter alongside the load balancer. "
            "Master switch: with it off, every other field here is ignored."
        ),
    )
    mode: Literal["container", "binary"] = Field(
        default="container",
        description=(
            "Whether the sidecars run from Singularity images or from binaries "
            "already on the node's PATH."
        ),
    )
    prometheus_image: str | None = Field(
        default_factory=default_for("slurm.endpoint.prometheus_image", None),
        description="Singularity image running Prometheus. Required for `mode: container`.",
    )
    nginx_exporter_image: str | None = Field(
        default_factory=default_for("slurm.endpoint.nginx_exporter_image", None),
        description=(
            "Singularity image running nginx-prometheus-exporter. Required for `mode: container`."
        ),
    )
    prometheus_binary: str = Field(
        default="prometheus",
        description="Prometheus binary name or path, for `mode: binary`.",
    )
    nginx_exporter_binary: str = Field(
        default="nginx-prometheus-exporter",
        description="nginx-prometheus-exporter binary name or path, for `mode: binary`.",
    )
    port: int = Field(
        default=9090,
        description=(
            "Port Prometheus listens on, on the load-balancer node. Reached "
            "through the load balancer rather than directly, so this is not the "
            "port you connect to."
        ),
    )
    exporter_port: int = Field(
        default=9113,
        description="Port the Nginx exporter serves its metrics on for Prometheus to scrape.",
    )
    route_prefix: str = Field(
        default="/prometheus",
        description=(
            "Path under the swarm's endpoint where Prometheus is served. A "
            "leading slash is added if missing."
        ),
    )
    scrape_interval: str = Field(
        default="15s",
        description="How often Prometheus scrapes every target.",
    )
    retention: str = Field(
        default="12h",
        description=(
            "How long Prometheus keeps samples. The database is node-local and "
            "dies with the load-balancer job, so this only caps a single run."
        ),
    )
    gpu_exporter: GpuExporterConfig = Field(
        default_factory=GpuExporterConfig,
        description="Per-node GPU metrics exporter. Off by default.",
    )
    ray_metrics: RayMetricsConfig = Field(
        default_factory=RayMetricsConfig,
        description="Scraping of Ray's own metrics. Auto-enabled for Ray deployments.",
    )

    @field_validator("route_prefix")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("route_prefix must be a non-empty string")
        return v if v.startswith("/") else f"/{v}"

    @model_validator(mode="after")
    def _validate_gpu_exporter_combo(self) -> "MonitoringConfig":
        if not self.gpu_exporter.enabled:
            return self
        if (
            self.mode == "container"
            and self.gpu_exporter.kind == "nvidia_smi"
            and self.gpu_exporter.image is None
        ):
            raise ValueError(
                "nvidia_smi container mode needs an explicit gpu_exporter.image "
                "(build from images/gpu_exporter_nvidia_smi.def) or use mode=binary."
            )
        if self.mode == "binary" and self.gpu_exporter.kind == "dcgm":
            raise ValueError(
                "dcgm exporter is only supported in container mode (the launch "
                "template runs it via `singularity exec`); set mode=container."
            )
        return self


class SlurmEndpointConfig(BaseModel):
    """Configuration for the load-balancer job fronting the replicas."""

    cpus_per_task: int = Field(
        default=32,
        description=(
            "vCPUs for the driver process that launches and monitors the swarm. "
            "Job steps use the same value unless `require_allocated_node` is set."
        ),
    )
    mem: str = Field(
        default="16GB",
        description=(
            "Physical memory for the driver job. Job steps use the same value "
            "unless `require_allocated_node` is set."
        ),
    )
    threads_per_core: int = Field(
        default=1,
        description="SMT threads to request per physical core.",
    )
    wall_time: str = Field(
        default="24:00:00",
        description="Slurm time limit for the driver job.",
    )
    enable_proxy_buffering: bool = Field(
        default=True,
        description=(
            "Enable Nginx response and request buffering in the generated load "
            "balancer config. Turn it off for streaming responses that should reach "
            "the client as they are produced."
        ),
    )
    nginx_timeout: str | int = Field(
        default="60s",
        description=(
            "HTTP timeout for the load balancer's proxied requests to the model "
            "replicas. Applied to Nginx's connect, send and read timeouts."
        ),
    )
    port: int = Field(
        default=9000,
        description="External port exposed by the Nginx load balancer.",
    )
    nginx_worker_processes: Annotated[int, Field(gt=0)] | Literal["auto"] | None = Field(
        default=None,
        description=(
            "Number of Nginx worker processes on the load balancer. Nginx defaults "
            "to a single worker, which becomes the throughput ceiling once a swarm "
            "has many replicas. Defaults to `cpus_per_task`, so the workers stay "
            "within the job's CPU allocation and leave the monitoring sidecars "
            "schedulable; set `auto` to let Nginx count the node's cores itself."
        ),
    )
    nginx_upstream_keepalive: int = Field(
        default=32,
        description=(
            "Idle connections to the replicas that each Nginx worker keeps alive "
            "for reuse. Without it Nginx speaks HTTP/1.0 upstream and opens a new "
            "TCP connection per request. The load balancer holds up to "
            "`nginx_upstream_keepalive * nginx_worker_processes` idle connections "
            "spread across all replicas; set 0 to restore the old per-request "
            "connection behaviour."
        ),
        ge=0,
    )
    collector_port: int = Field(
        default=9100,
        description=(
            "TCP port on the load-balancer node where the watchdog collector "
            "listens for replica status updates. Replicas read the resolved "
            "address from the swarm's `serving/collector.env`, so only the load "
            "balancer needs this. A `COLLECTOR_PORT` variable exported in the "
            "submission environment overrides it."
        ),
    )
    nginx_image: str | utils.EnvPath = Field(
        default_factory=default_for("slurm.endpoint.nginx_image"),
        description=(
            "Path to a Singularity image running Nginx as the swarm's load "
            "balancer. Required on the Slurm backend."
        ),
    )
    qos: str | None = Field(
        default=None,
        description=(
            "QoS override for the load-balancer job. When unset it stays `None`, and "
            "the effective QoS is resolved at submission time as `endpoint.qos` or "
            "`SlurmConfig.qos`; it is not a resolved copy of `SlurmConfig.qos`."
        ),
    )
    poll_interval: int = Field(
        default=10,
        description=(
            "Seconds between `sacct` status checks while waiting for the load "
            "balancer to become ready."
        ),
    )
    require_allocated_node: bool = Field(
        default=False,
        description=(
            "Refuse to build an `srun` command unless already inside a Slurm "
            "allocation other than the load-balancer job. Guards against large data "
            "jobs accidentally running on the load-balancer node. Job steps then get "
            "all the memory of that allocation (`--mem=0`) and its `--cpus-per-task` "
            "value. The `--mem` and `--cpus-per-task` options of `job submit` "
            "override both."
        ),
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description=(
            "Prometheus and GPU-exporter sidecars running alongside the load "
            "balancer. Off by default, and Slurm-only."
        ),
    )
    router: RouterConfig = Field(
        default_factory=NginxRouter,
        description=(
            "Router that balances requests across replicas. nginx stays the front "
            "door on `port` for every router type and forwards to the selected "
            "router. Defaults to plain nginx balancing."
        ),
    )


class SlurmConfig(BaseModel):
    """Configuration for SLURM-based deployments."""

    type: Literal["slurm"] = Field(
        default="slurm",
        description="Backend discriminator; always `slurm` for this model.",
    )
    partition: str = Field(
        default_factory=default_for("slurm.partition"),
        description="Slurm partition to submit to.",
    )
    account: str = Field(
        default_factory=default_for("slurm.account"),
        description="Slurm account or charge code.",
    )
    qos: str = Field(
        default_factory=default_for("slurm.qos"),
        description=(
            "Slurm QoS for the cluster and load-balancer jobs. The load balancer "
            "can override it with `endpoint.qos`."
        ),
    )

    # Ray-related settings
    requires_ray: bool | None = Field(
        description=(
            "Whether replicas form a Ray cluster spanning several nodes. "
            "Derived rather than set by hand: true when one replica needs more "
            "GPUs than a single node has, which also requires `gpus_per_replica` "
            "to be a multiple of `gpus_per_node`. It selects the sbatch template "
            "and turns on the Ray watchdog and Ray metrics."
        ),
        default=None,
    )
    ray_port: int = Field(
        default=6379,
        description="Port for Ray's GCS / head node inside each replica.",
    )
    ray_dashboard_port: int = Field(
        default=8265,
        description="Port for the optional Ray dashboard.",
    )

    modules: list[str] = Field(
        default_factory=list,
        description=(
            "Environment modules to `module load` at the top of the generated "
            "cluster sbatch script."
        ),
    )
    preamble: list[str] = Field(
        default_factory=list,
        description=(
            "Additional lines inserted near the top of the generated cluster sbatch "
            "script, before the module loads. Use for extra sbatch directives or "
            "shell setup."
        ),
    )

    template_path: utils.EnvPath = Field(
        default_factory=lambda: (
            utils.EnvPath(__file__).with_suffix("").parent.parent / "templates" / "llm_swarm.sh.j2"
        ),
        description=(
            "Path to the Jinja2 template for the cluster sbatch script. "
            "Auto-filled and normally left alone: a Ray deployment renders from "
            "`llm_swarm_ray.sh.j2` and a single-node one from `llm_swarm.sh.j2`. "
            "Setting it pins the template and disables that choice."
        ),
    )
    nginx_template_path: utils.EnvPath = Field(
        default_factory=lambda: (
            utils.EnvPath(__file__).with_suffix("").parent.parent / "templates" / "nginx.conf.j2"
        ),
        description=(
            "Path to the Jinja2 template for the Nginx config. Auto-filled; there is "
            "normally no need to set it."
        ),
    )

    time_limit: str = Field(
        default="36:00:00",
        description="Overall Slurm wall-clock limit for the allocation.",
    )
    requeue: bool = Field(
        default_factory=default_for("slurm.requeue", True),
        description=(
            "Emit `#SBATCH --requeue` so replica jobs may be requeued after a "
            "node failure or preemption, and let the Ray head ask for a requeue "
            "when the cluster dies. Turn it off on sites that forbid requeueing; "
            "a failed replica then stays down instead of coming back."
        ),
    )
    exclude_nodes: str | None = Field(
        default=None,
        description=("Nodes to exclude, passed through to Slurm, e.g. `node[001-004]`."),
    )
    node_list: str | None = Field(
        default=None,
        description="Explicit node list to run on, e.g. `node[005-008]`.",
    )
    mail_user: str | None = Field(
        default=None,
        description=(
            "Email address for Slurm END and FAIL notifications about the resources "
            "domyn-swarm deploys. Notifications are disabled when unset."
        ),
    )
    endpoint: SlurmEndpointConfig = Field(
        default_factory=SlurmEndpointConfig,
        description="Configuration for the Nginx load-balancer job.",
    )

    venv_path: utils.EnvPath | None = Field(
        default=None,
        description=("Virtual environment used by the driver process, not by the containers."),
    )
    env: dict[str, str] | None = Field(
        default=None,
        description=("Additional environment variables set on every job this backend submits."),
    )

    mounts: list[str] = Field(
        default_factory=list,
        description=(
            "Extra Singularity bind mounts for the vLLM containers. Each entry is "
            "either '/path' (bound at the same path inside the container) or "
            "'/host/path:/container/path' (with an optional ':ro'/':rw' suffix). "
            "Appended verbatim to the container's bind list."
        ),
    )

    system_mounts: list[str] = Field(
        default_factory=default_for("slurm.system_mounts", []),
        description=(
            "Site-specific host paths the containers need, such as the cluster's "
            "software stack or its interconnect devices (e.g. '/leonardo/prod/opt' "
            "on CINECA Leonardo). Same syntax as `mounts`, but each entry is bound "
            "only when its host path exists on the node, so a path belonging to "
            "another site is skipped with a warning instead of failing the job. "
            "Belongs in the site `defaults.yaml` under `slurm.system_mounts` rather "
            "than in a per-deployment config."
        ),
    )

    @field_validator("mounts", "system_mounts")
    @classmethod
    def _validate_mounts(cls, value: list[str]) -> list[str]:
        """Validate the format of each bind mount specification.

        Checks that every entry is non-empty, has at most three colon-separated
        segments (source[:dest[:opts]]), and uses an absolute source path. Host
        path existence is intentionally not checked here: for `mounts` the
        container runtime reports it, and `system_mounts` are checked on the node
        at launch time.

        Args:
            value: The list of mount specifications.

        Returns:
            The validated list of mount specifications.

        Raises:
            ValueError: If any entry is empty, has too many ':' segments, or has
                a non-absolute source path.
        """
        for mount in value:
            if not mount or not mount.strip():
                raise ValueError("mount entry must be a non-empty string")
            segments = mount.split(":")
            if len(segments) > 3:
                raise ValueError(
                    f"invalid mount spec '{mount}': expected at most 'source:dest:opts'"
                )
            if not segments[0].startswith("/"):
                raise ValueError(f"invalid mount spec '{mount}': source must be an absolute path")
        return value

    def build(self, cfg_ctx) -> DeploymentPlan:
        """Builds the deployment plan for SLURM-based deployments."""
        from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
        from domyn_swarm.backends.serving.slurm import SlurmServingBackend
        from domyn_swarm.backends.serving.slurm_driver import SlurmDriver

        driver = SlurmDriver(cfg=cfg_ctx)
        serving = SlurmServingBackend(cfg=self, driver=driver)

        def _make_compute(handle: ServingHandle) -> SlurmComputeBackend:
            """Build the Slurm compute backend from a ready serving handle."""
            lb_jobid = handle.meta.get("lb_jobid")
            lb_node = handle.meta.get("lb_node")
            if not lb_jobid or not lb_node:
                raise RuntimeError(
                    "Slurm serving handle is missing load-balancer metadata "
                    f"(lb_jobid={lb_jobid!r}, lb_node={lb_node!r}); the endpoint "
                    "is not ready."
                )
            return SlurmComputeBackend(cfg=self, lb_jobid=lb_jobid, lb_node=lb_node)

        def _validate_handle(handle: ServingHandle) -> None:
            """Reject a handle the Slurm serving backend could not query.

            ``SlurmServingBackend.status()`` subscripts ``meta["jobid"]``
            directly, so a handle without one -- a state record written before
            the replica job existed, say -- would blow up with a bare KeyError
            deep inside ``status()`` instead of here.
            """
            if handle.meta.get("jobid") is None:
                raise ValueError("State file does not contain valid job IDs")

        serving_spec = self.model_dump(exclude_none=True) | cfg_ctx.model_dump(
            include={
                "replicas",
                "nodes",
                "gpus_per_replica",
                "gpus_per_node",
                "replicas_per_node",
            },
            exclude_none=True,
        )

        return DeploymentPlan(
            name_hint="slurm",
            serving=serving,
            compute=None,
            compute_factory=_make_compute,
            handle_validator=_validate_handle,
            serving_spec=serving_spec,
            job_resources={},
            extras={},
            platform="slurm",
        )
