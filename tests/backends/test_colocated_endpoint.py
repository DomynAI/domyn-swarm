# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
import textwrap
from unittest.mock import MagicMock, patch

import jinja2
import pytest

from domyn_swarm.backends.serving import slurm_readiness
from domyn_swarm.backends.serving.slurm_driver import SlurmDriver, endpoint_job_file
from domyn_swarm.backends.serving.slurm_readiness import SlurmReadiness
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.platform.protocols import ServingHandle

TPL_DIR = Path("src/domyn_swarm/templates")


def _cfg(tmp_path, *, colocate: bool, ray: bool = False) -> DomynLLMSwarmConfig:
    return DomynLLMSwarmConfig.model_validate(
        {
            "name": "s",
            "model": "/models/m",
            "image": "/images/vllm.sif",
            "replicas": 1 if ray else 4,
            "gpus_per_replica": 8 if ray else 1,
            "gpus_per_node": 4,
            "home_directory": str(tmp_path),
            "backend": {
                "type": "slurm",
                "account": "a",
                "partition": "p",
                "qos": "q",
                "endpoint": {"nginx_image": "/images/nginx.sif", "colocate": colocate},
            },
        }
    )


def _replica_script(cfg: DomynLLMSwarmConfig, swarm_dir: Path) -> str:
    scripts: list[str] = []

    def fake_sbatch(cmd, **_):
        script = Path(cmd[-1])
        scripts.append(script.read_text())
        script.unlink()
        return "123"

    with patch(
        "domyn_swarm.backends.serving.slurm_driver.subprocess.check_output",
        side_effect=fake_sbatch,
    ):
        SlurmDriver(cfg).submit_replicas(
            "s",
            cfg.replicas,
            cfg.nodes,
            cfg.gpus_per_node,
            cfg.gpus_per_replica,
            cfg.replicas_per_node,
            str(swarm_dir),
        )
    return scripts[0]


def _probe(tmp_path, driver) -> SlurmReadiness:
    return SlurmReadiness(
        driver=driver,
        endpoint_port=9000,
        poll_interval_s=0,
        console=MagicMock(),
        endpoint_job_file=tmp_path / "endpoint.job",
    )


def test_staging_writes_lb_script_without_submitting_a_job(tmp_path):
    swarm_dir = tmp_path / "swarm"

    with patch("domyn_swarm.backends.serving.slurm_driver.subprocess.check_output") as sbatch:
        SlurmDriver(_cfg(tmp_path, colocate=True)).stage_colocated_endpoint("s", 4, str(swarm_dir))

    sbatch.assert_not_called()
    lb_script = (swarm_dir / "serving" / "lb.sh").read_text()
    assert "nginx" in lb_script
    assert (swarm_dir / "nginx.conf").exists()


def test_staging_removes_the_endpoint_record_of_a_previous_deployment(tmp_path):
    swarm_dir = tmp_path / "swarm"
    stale = endpoint_job_file(swarm_dir)
    stale.parent.mkdir(parents=True)
    stale.write_text("1000 1001 old-node\n")

    SlurmDriver(_cfg(tmp_path, colocate=True)).stage_colocated_endpoint("s", 4, str(swarm_dir))

    assert not stale.exists()


def test_first_array_task_starts_the_colocated_load_balancer(tmp_path):
    swarm_dir = tmp_path / "swarm"
    script = _replica_script(_cfg(tmp_path, colocate=True), swarm_dir)

    assert 'if [[ "$REPL_ID" == 0 ]]; then' in script
    assert f'bash "{swarm_dir}/serving/lb.sh"' in script
    assert 'echo "$REPL_JOB_ID $SLURM_JOB_ID $HEAD_NODE"' in script
    assert script.index("serving/lb.sh") < script.index("waiting for collector.env")
    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("ray", [False, True])
def test_replica_script_checks_the_load_balancer_until_it_ends(tmp_path, ray):
    script = _replica_script(_cfg(tmp_path, colocate=True, ray=ray), tmp_path / "swarm")

    keep_alive = script.index("entering Ray + vLLM health loop" if ray else "# Keep job alive")
    assert "check_colocated_lb" in script[keep_alive:]
    assert "until collector_load_env; do" in script
    assert "COLLECTOR_RESTART_COUNT" in script
    assert script.count("check_colocated_lb") >= 3
    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_ray_exit_trap_keeps_stopping_the_load_balancer(tmp_path):
    script = _replica_script(_cfg(tmp_path, colocate=True, ray=True), tmp_path / "swarm")

    exit_traps = [
        line for line in script.splitlines() if line.startswith("trap ") and line.endswith(" EXIT")
    ]
    assert exit_traps[-1] == "trap 'final_sync_ray_logs; stop_colocated_lb' EXIT"


def test_replica_script_does_not_start_a_load_balancer_by_default(tmp_path):
    script = _replica_script(_cfg(tmp_path, colocate=False), tmp_path / "swarm")

    assert "lb.sh" not in script
    assert "endpoint.job" not in script


def _run_colocated_lb(tmp_path: Path, lb_body: str, driver: str) -> subprocess.CompletedProcess:
    """Run the colocated load-balancer partial with a fake `lb.sh` and `scancel`."""
    swarm_dir = tmp_path / "swarm"
    (swarm_dir / "serving").mkdir(parents=True)
    (swarm_dir / "serving" / "lb.sh").write_text(textwrap.dedent(lb_body))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scancel = bin_dir / "scancel"
    scancel.write_text(f'#!/bin/bash\necho "$@" > "{tmp_path}/scancel.args"\n')
    scancel.chmod(0o755)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TPL_DIR))
    partial = env.get_template("_colocated_lb.sh.j2").render(swarm_directory=swarm_dir)
    script = "\n".join(
        [
            "set -euo pipefail",
            f'PATH="{bin_dir}:$PATH"',
            "REPL_ID=0 REPL_JOB_ID=4242 SLURM_JOB_ID=4243 HEAD_NODE=node-a",
            partial,
            textwrap.dedent(driver),
        ]
    )
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=30)


def test_replica_task_exits_and_cancels_the_array_when_the_load_balancer_dies(tmp_path):
    result = _run_colocated_lb(
        tmp_path,
        lb_body="sleep 0.5; exit 3\n",
        driver="""
        check_colocated_lb
        touch "$SWARM_MARK"
        sleep 2
        check_colocated_lb
        echo still-running
        """.replace("$SWARM_MARK", str(tmp_path / "collector.env")),
    )

    assert result.returncode == 1
    assert (tmp_path / "collector.env").exists()
    assert "still-running" not in result.stdout
    assert "load balancer exited" in result.stderr
    assert (tmp_path / "scancel.args").read_text().strip() == "4242"
    assert (tmp_path / "swarm" / "serving" / "endpoint.job").read_text() == "4242 4243 node-a\n"


def test_replica_task_stops_the_load_balancer_when_it_exits(tmp_path):
    stopped = tmp_path / "lb.stopped"
    result = _run_colocated_lb(
        tmp_path,
        lb_body=f"""
        trap 'touch "{stopped}"; exit 0' TERM
        while true; do sleep 0.1; done
        """,
        driver="""
        sleep 0.5
        check_colocated_lb
        exit 0
        """,
    )

    assert result.returncode == 0, result.stderr
    assert stopped.exists()
    assert not (tmp_path / "scancel.args").exists()


def test_readiness_takes_lb_job_and_node_from_the_endpoint_job_file(tmp_path):
    (tmp_path / "endpoint.job").write_text("4242 4243 node-a\n")
    driver = MagicMock()
    driver.get_job_state.return_value = "RUNNING"
    handle = ServingHandle(id="4242", url="", meta={"jobid": 4242, "lb_jobid": 4242})

    with patch.object(slurm_readiness, "wait_http_200"):
        out = _probe(tmp_path, driver).wait_ready(handle, timeout_s=5)

    assert out.meta["lb_jobid"] == 4243
    assert out.meta["lb_node"] == "node-a"
    assert out.url == "http://node-a:9000"
    driver.get_node_from_jobid.assert_not_called()


def test_readiness_fails_when_replicas_end_before_reporting_the_endpoint(tmp_path):
    driver = MagicMock()
    driver.get_job_state.side_effect = ["RUNNING", "RUNNING", "FAILED"]
    handle = ServingHandle(id="4242", url="", meta={"jobid": 4242, "lb_jobid": 4242})

    with pytest.raises(RuntimeError, match="Replica array ended in FAILED"):
        _probe(tmp_path, driver).wait_ready(handle, timeout_s=5)


def test_readiness_ignores_an_endpoint_record_from_another_array_job(tmp_path):
    endpoint_job = tmp_path / "endpoint.job"
    endpoint_job.write_text("1000 1001 old-node\n")
    driver = MagicMock()

    def publish_current_endpoint(_delay):
        endpoint_job.write_text("4242 4243 node-a\n")

    driver.get_job_state.return_value = "RUNNING"
    handle = ServingHandle(id="4242", url="", meta={"jobid": 4242, "lb_jobid": 4242})

    with (
        patch.object(slurm_readiness.time, "sleep", side_effect=publish_current_endpoint),
        patch.object(slurm_readiness, "wait_http_200"),
    ):
        out = _probe(tmp_path, driver).wait_ready(handle, timeout_s=5)

    assert out.meta["lb_jobid"] == 4243
    assert out.meta["lb_node"] == "node-a"


def test_readiness_fails_when_replicas_complete_without_reporting_the_endpoint(tmp_path):
    driver = MagicMock()
    driver.get_job_state.side_effect = ["RUNNING", "RUNNING", "COMPLETED"]
    handle = ServingHandle(id="4242", url="", meta={"jobid": 4242, "lb_jobid": 4242})

    with pytest.raises(RuntimeError, match="ended in COMPLETED before starting the load balancer"):
        _probe(tmp_path, driver).wait_ready(handle, timeout_s=5)


def test_readiness_times_out_when_the_endpoint_record_never_appears(tmp_path):
    driver = MagicMock()
    driver.get_job_state.return_value = "RUNNING"
    handle = ServingHandle(id="4242", url="", meta={"jobid": 4242, "lb_jobid": 4242})
    clock = iter([0.0, 2.0, 4.0, 6.0])

    with (
        patch.object(slurm_readiness.time, "monotonic", side_effect=lambda: next(clock)),
        patch.object(slurm_readiness, "wait_http_200") as probe,
        pytest.raises(TimeoutError, match="did not start the load balancer"),
    ):
        _probe(tmp_path, driver).wait_ready(handle, timeout_s=5)

    probe.assert_not_called()


def test_http_probe_gets_the_time_left_after_the_endpoint_record(tmp_path):
    (tmp_path / "endpoint.job").write_text("4242 4243 node-a\n")
    driver = MagicMock()
    driver.get_job_state.return_value = "RUNNING"
    handle = ServingHandle(id="4242", url="", meta={"jobid": 4242, "lb_jobid": 4242})
    clock = iter([100.0, 103.0])

    with (
        patch.object(slurm_readiness.time, "monotonic", side_effect=lambda: next(clock)),
        patch.object(slurm_readiness, "wait_http_200") as probe,
    ):
        _probe(tmp_path, driver).wait_ready(handle, timeout_s=10)

    assert probe.call_args.kwargs["timeout_s"] == 7
