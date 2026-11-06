# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from domyn_swarm.backends.serving import slurm_readiness
from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.backends.serving.slurm_readiness import SlurmReadiness
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.platform.protocols import ServingHandle


def _cfg(tmp_path, *, colocate: bool) -> DomynLLMSwarmConfig:
    return DomynLLMSwarmConfig.model_validate(
        {
            "name": "s",
            "model": "/models/m",
            "image": "/images/vllm.sif",
            "replicas": 4,
            "gpus_per_replica": 1,
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
        SlurmDriver(cfg).submit_replicas("s", 4, 1, 4, 1, 4, str(swarm_dir))
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


def test_first_array_task_starts_the_colocated_load_balancer(tmp_path):
    swarm_dir = tmp_path / "swarm"
    script = _replica_script(_cfg(tmp_path, colocate=True), swarm_dir)

    assert 'if [[ "$REPL_ID" == 0 ]]; then' in script
    assert f'bash "{swarm_dir}/serving/lb.sh"' in script
    assert 'echo "$SLURM_JOB_ID $HEAD_NODE"' in script
    assert script.index("serving/lb.sh") < script.index("waiting for collector.env")
    result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_replica_script_does_not_start_a_load_balancer_by_default(tmp_path):
    script = _replica_script(_cfg(tmp_path, colocate=False), tmp_path / "swarm")

    assert "lb.sh" not in script
    assert "endpoint.job" not in script


def test_readiness_takes_lb_job_and_node_from_the_endpoint_job_file(tmp_path):
    (tmp_path / "endpoint.job").write_text("4243 node-a\n")
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
