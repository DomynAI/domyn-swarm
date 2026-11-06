# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
import subprocess

import requests
from requests import RequestException

from domyn_swarm.backends.compute.slurm_helpers import _cancel_slurm
from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.backends.serving.slurm_readiness import (
    SLURM_BAD_STATES,
    SLURM_WAIT_STATES,
    SlurmReadiness,
)
from domyn_swarm.config.settings import get_settings
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.platform.protocols import (
    ServingBackend,
    ServingHandle,
    ServingPhase,
    ServingStatus,
)


@dataclass
class SlurmServingBackend(ServingBackend):  # type: ignore[misc]
    """Adapt your existing Slurm flow into the ServingBackend protocol.

    * create_or_update -> submit replicas + LB job
    * wait_ready       -> poll with LBHealthChecker, populate .url
    * delete           -> scancel both jobs

    spec dict (suggested keys):
        {
          "name": str,
          "cfg": DomynLLMSwarmConfig,   # your config object
        }
    """

    cfg: SlurmConfig
    driver: SlurmDriver
    readiness: SlurmReadiness | None = None

    def create_or_update(self, name: str, spec: dict, extras: dict) -> ServingHandle:
        settings = get_settings()
        replicas = spec.get("replicas", 1)
        nodes = spec.get("nodes", 1)
        gpus_per_node = spec.get("gpus_per_node", 1)
        gpus_per_replica = spec.get("gpus_per_replica", 1)
        replicas_per_node = spec.get("replicas_per_node", 1)

        swarm_directory = spec.get("swarm_directory", settings.home / "swarms" / name)
        colocate = self.cfg.endpoint.colocate
        if colocate:
            self.driver.stage_colocated_endpoint(name, replicas, swarm_directory)

        try:
            jobid = self.driver.submit_replicas(
                name,
                replicas,
                nodes,
                gpus_per_node,
                gpus_per_replica,
                replicas_per_node,
                swarm_directory,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to submit replicas Slurm job: {e}") from e
        # A colocated endpoint has no job of its own: until `wait_ready` reads the
        # first array task's job ID, the replica job stands in for it.
        lb_jobid = (
            jobid
            if colocate
            else self.driver.submit_endpoint(name, jobid, replicas, swarm_directory)
        )
        return ServingHandle(
            id=str(lb_jobid),
            url="",  # filled in wait_ready()
            meta={
                "jobid": jobid,
                "lb_jobid": lb_jobid,
                "port": self.cfg.endpoint.port,
                "name": name,
                "swarm_directory": str(swarm_directory),
            },
        )

    def wait_ready(self, handle: ServingHandle, timeout_s: int, extras: dict) -> ServingHandle:
        swarm_dir = extras.get("swarm_directory")
        swarm_name = handle.meta.get("name", "")

        if swarm_dir:
            watchdog_db = Path(f"{swarm_dir}/watchdog.db")
        else:
            watchdog_db = get_settings().home / "swarms" / swarm_name / "watchdog.db"

        probe = self.readiness or SlurmReadiness(
            driver=self.driver,
            endpoint_port=self.cfg.endpoint.port,
            poll_interval_s=self.cfg.endpoint.poll_interval,
            watchdog_db=watchdog_db,
            swarm_name=swarm_name,
            endpoint_job_file=(
                Path(handle.meta["swarm_directory"]) / "serving" / "endpoint.job"
                if self.cfg.endpoint.colocate
                else None
            ),
        )
        return probe.wait_ready(handle, timeout_s)

    def delete(self, handle: ServingHandle) -> None:
        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        if jobid:
            _cancel_slurm(str(jobid))
        if lb_jobid and lb_jobid != jobid:
            _cancel_slurm(str(lb_jobid))

    def ensure_ready(self, handle: ServingHandle | None):
        """Ensure the current serving handle is ready, or raise if not."""
        if handle is None:
            raise RuntimeError("Serving handle is null.")
        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        lb_node = handle.meta.get("lb_node")
        endpoint = handle.url
        if not all([jobid, lb_jobid, lb_node, endpoint]):
            raise RuntimeError(
                "Swarm not ready (jobid/lb_jobid/lb_node/endpoint): "
                f"{jobid}/{lb_jobid}/{lb_node}/{endpoint}"
            )

    def status(self, handle: ServingHandle) -> ServingStatus:
        rep = self.driver.get_job_state(handle.meta["jobid"])
        lb = self.driver.get_job_state(handle.meta["lb_jobid"])

        if rep in SLURM_BAD_STATES or lb in SLURM_BAD_STATES:
            return ServingStatus(ServingPhase.FAILED, handle.url, {"rep": rep, "lb": lb})
        if (
            rep in SLURM_WAIT_STATES
            or lb in SLURM_WAIT_STATES
            or rep == "UNKNOWN"
            or lb == "UNKNOWN"
        ):
            return ServingStatus(ServingPhase.PENDING, handle.url, {"rep": rep, "lb": lb})

        lb_node = handle.meta.get("lb_node") or self.driver.get_node_from_jobid(
            handle.meta["lb_jobid"]
        )
        base = f"http://{lb_node}:{self.cfg.endpoint.port}"
        api_token = get_settings().resolved_api_token
        try:
            if api_token:
                headers = {"Authorization": f"Bearer {api_token.get_secret_value()}"}
                r = requests.get(f"{base}/health", headers=headers, timeout=1.5)
            else:
                r = requests.get(f"{base}/health", timeout=1.5)
            http_ok = r.status_code == 200
            model_ok = False
            try:
                data = r.json()
                names = {m.get("id") for m in data.get("data", []) if isinstance(m, dict)}
                expected = handle.meta.get("model")
                model_ok = (expected in names) if expected else http_ok
            except Exception:
                model_ok = http_ok
        except RequestException:
            http_ok = model_ok = False

        if http_ok and model_ok:
            if not handle.url:
                handle.url = base
            return ServingStatus(
                ServingPhase.RUNNING,
                handle.url or base,
                {"rep": rep, "lb": lb, "http": 200},
            )

        return ServingStatus(
            ServingPhase.INITIALIZING,
            handle.url,
            {"rep": rep, "lb": lb, "http": "unready"},
        )
