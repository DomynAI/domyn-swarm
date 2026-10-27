# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from importlib import resources
from pathlib import Path
import subprocess
import tempfile

import jinja2

import domyn_swarm
from domyn_swarm.config.router import VllmRouter
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.helpers.data import get_device_slices
from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.helpers.logger import setup_logger
import domyn_swarm.runtime.collector as collector_mod
import domyn_swarm.runtime.lb_supervisor as supervisor_mod
import domyn_swarm.runtime.watchdog as watchdog_mod
import domyn_swarm.runtime.watchdog_args as watchdog_args_mod

logger = setup_logger(__name__)


class SlurmDriver:
    """Submit and inspect Slurm serving allocations for one swarm."""

    def __init__(self, cfg: DomynLLMSwarmConfig):
        self.cfg: DomynLLMSwarmConfig = cfg

    def submit_replicas(
        self,
        job_name: str,
        replicas: int,
        nodes: int,
        gpus_per_node: int,
        gpus_per_replica: int,
        replicas_per_node: int,
        swarm_directory: str,
    ) -> int:
        """Submit the replica array job and return its Slurm job ID."""

        assert isinstance(self.cfg.backend, SlurmConfig)
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.backend.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        script_txt = env.get_template(self.cfg.backend.template_path.name).render(
            cfg=self.cfg,
            job_name=job_name,
            path_exists=path_exists,
            is_folder=is_folder,
            cuda_visible_devices=get_device_slices(gpus_per_node, gpus_per_replica),
            swarm_directory=swarm_directory,
            watchdog_script_path=Path(watchdog_mod.__file__).resolve().as_posix(),
            build_watchdog_args=watchdog_args_mod.build_watchdog_args,
            args_to_str=watchdog_args_mod.args_to_str,
            dswarm_agent_version=domyn_swarm.__version__,
            dcgm_core_counters_path=Path(
                str(resources.files("domyn_swarm.data.dcgm").joinpath("core-counters.csv"))
            ).as_posix(),
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(script_txt)
            script_path = fh.name

        sbatch_cmd = ["sbatch", "--parsable", "--export=ALL"]
        array_spec = None
        if self.cfg.backend.requires_ray:
            logger.info(
                "Detected gpus_per_replica > gpus_per_node with "
                "multiple nodes: enabling Ray support."
            )
            array_spec = f"0-{replicas - 1}%{replicas}"
            # In this case, the nodes are the total number of nodes to be allocated
            # So we divide by replicas to get the number of nodes per array task
            sbatch_cmd.append(f"--nodes={nodes // replicas}")
        elif nodes and nodes >= 1 and replicas >= 1:
            array_spec = f"0-{nodes - 1}%{nodes}"
            sbatch_cmd.append("--nodes=1")
            sbatch_cmd.append(f"--ntasks-per-node={replicas_per_node}")

        if array_spec is not None:
            sbatch_cmd.extend(["--array", array_spec])
        sbatch_cmd.append(script_path)

        out = subprocess.check_output(sbatch_cmd, text=True).strip()
        job_id = out.split(";")[0]

        logger.info(f"Submitted replicas job {job_id} with command: {' '.join(sbatch_cmd)}")

        return int(job_id)

    def submit_endpoint(
        self, job_name: str, dep_jobid: int, replicas: int, swarm_directory: str
    ) -> int:
        """Submit the load balancer job to Slurm with a dependency on the replica array job."""
        assert isinstance(self.cfg.backend, SlurmConfig)

        router = self.cfg.backend.endpoint.router
        if isinstance(router, VllmRouter) and router.policy == "consistent_hash":
            logger.warning(
                "vllm-router consistent_hash sends every request without a session ID "
                "to one replica"
            )

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.backend.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        serving_dir = Path(swarm_directory) / "serving"
        serving_dir.mkdir(parents=True, exist_ok=True)

        # Static nginx server block (dynamic upstreams come from the supervisor).
        server_conf = env.get_template("nginx_server.conf.j2").render(cfg=self.cfg)
        (serving_dir / "10-server.conf").write_text(server_conf)

        # Main nginx config (mounted at $HOST_DIR/nginx.conf by lb.sh.j2).
        (Path(swarm_directory) / "nginx.conf").write_text(
            env.get_template("nginx.conf.j2").render(cfg=self.cfg)
        )

        # Prometheus config (only when monitoring is enabled).
        mon = self.cfg.backend.endpoint.monitoring
        if getattr(mon, "enabled", False):
            if mon.mode == "container":
                targets_path = "/etc/prometheus/serving/targets.json"
                gpu_targets_path = "/etc/prometheus/serving/gpu_targets.json"
                ray_targets_path = "/etc/prometheus/serving/ray_targets.json"
            else:
                targets_path = (serving_dir / "targets.json").as_posix()
                gpu_targets_path = (serving_dir / "gpu_targets.json").as_posix()
                ray_targets_path = (serving_dir / "ray_targets.json").as_posix()
            prom_yml = env.get_template("prometheus.yml.j2").render(
                monitoring=mon,
                targets_path=targets_path,
                gpu_targets_path=gpu_targets_path,
                ray_targets_path=ray_targets_path,
                router_metrics=self.cfg.backend.endpoint.router.metrics_target(),
                cfg=self.cfg,
            )
            (serving_dir / "prometheus.yml").write_text(prom_yml)

        lb_script_txt = env.get_template("lb.sh.j2").render(
            cfg=self.cfg,
            job_name=job_name,
            dep_jobid=dep_jobid,
            replicas=replicas,
            swarm_directory=swarm_directory,
            collector_script_path=Path(collector_mod.__file__).resolve().as_posix(),
            supervisor_script_path=Path(supervisor_mod.__file__).resolve().as_posix(),
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(lb_script_txt)
            script_path = fh.name

        cmd = [
            "sbatch",
            "--parsable",
            "--dependency",
            f"after:{dep_jobid}",
            "--export",
            f"ALL,DEP_JOBID={dep_jobid}",
            script_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        logger.info(f"Submitted load balancer job {out} with command: {' '.join(cmd)}")
        return int(out)

    def get_node_from_jobid(self, jobid: int) -> str:
        """Get the head node of the job given its job ID."""
        nodespec = subprocess.check_output(
            ["squeue", "-j", str(jobid), "-h", "-O", "NodeList:2048"],
            text=True,
        ).strip()
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodespec],
            text=True,
        ).splitlines()

        if not nodes:
            raise RuntimeError(f"Could not find nodes for job ID {jobid}")
        head_node = nodes[0]
        logger.info(f"Job ID {jobid} is running on head node {head_node}")

        return head_node

    def get_job_state(self, jobid: int) -> str:
        """Return the current state of a Slurm job.

        Queries `squeue`, `sacct`, and `scontrol` in order so active and
        recently completed jobs are both discoverable.

        Args:
            jobid: Slurm job ID to query.

        Returns:
            The normalized Slurm state, or `UNKNOWN` if no command reports one.
        """

        def _run(cmd: list[str]) -> str:
            try:
                res = subprocess.run(cmd, check=False, text=True, capture_output=True)
                if res.returncode == 0 and res.stdout:
                    return res.stdout.strip()
            except Exception:
                pass
            return ""

        out = _run(["squeue", "-j", str(jobid), "-h", "-o", "%T"])
        if out:
            # %T returns a single token (e.g., RUNNING). Be cautious if someone changes the format.
            state = out.split()[0].upper()
            # Some sites prepend headers if -h is ignored; guard that:
            if state != "STATE":
                return state

        out = _run(["sacct", "-j", str(jobid), "-o", "State", "-n", "-X", "-P"])
        if out:
            # sacct may still return multiple lines in some schedulers (array parent/children).
            # Take the first non-empty token; with -P, fields are ';'-separated,
            # but we asked only State.
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                # Drop trailing details like "CANCELLED by 12345" or "FAILED+"
                s = s.split()[0].split("+")[0].upper()
                if s and s != "STATE":
                    return s

        out = _run(["scontrol", "show", "job", "-o", str(jobid)])
        if out:
            # one line of key=val pairs, e.g., "... JobState=FAILED Reason=..."
            for field in out.split():
                if field.startswith("JobState="):
                    state = field.split("=", 1)[1].split()[0].upper()
                    if state:
                        return state

        return "UNKNOWN"
