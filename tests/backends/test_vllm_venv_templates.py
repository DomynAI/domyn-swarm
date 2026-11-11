# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.config.swarm import DomynLLMSwarmConfig

VENV = "/venvs/vllm"


def _cfg(tmp_path, **backend) -> DomynLLMSwarmConfig:
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
                "endpoint": {"nginx_image": "/images/nginx.sif"},
                **backend,
            },
        }
    )


def _submitted_scripts(cfg: DomynLLMSwarmConfig, tmp_path: Path) -> tuple[str, str]:
    """Return the replica and load-balancer sbatch scripts the driver would submit."""
    scripts: list[str] = []

    def fake_sbatch(cmd, **_):
        script = Path(cmd[-1])
        scripts.append(script.read_text())
        script.unlink()
        return "123"

    driver = SlurmDriver(cfg)
    with patch(
        "domyn_swarm.backends.serving.slurm_driver.subprocess.check_output",
        side_effect=fake_sbatch,
    ):
        driver.submit_replicas("s", 4, 1, 4, 1, 4, str(tmp_path / "swarm"))
        driver.submit_endpoint("s", 123, 4, str(tmp_path / "swarm"))
    replicas_script, lb_script = scripts
    return replicas_script, lb_script


@pytest.fixture
def venv_scripts(tmp_path) -> tuple[str, str]:
    return _submitted_scripts(_cfg(tmp_path, vllm_venv=VENV), tmp_path)


@pytest.fixture
def container_scripts(tmp_path) -> tuple[str, str]:
    return _submitted_scripts(_cfg(tmp_path), tmp_path)


def test_replicas_run_vllm_from_the_venv_without_a_container(venv_scripts):
    replicas_script, _ = venv_scripts

    assert f'export PATH="{VENV}/bin:$PATH"' in replicas_script
    assert "vllm serve /models/m" in replicas_script
    assert "DSWARM_CONTAINER_RUNTIME" not in replicas_script


def test_watchdog_runs_from_the_installed_package(venv_scripts):
    replicas_script, _ = venv_scripts

    assert "/opt/watchdog.py \\" not in replicas_script
    assert "domyn_swarm/runtime/watchdog.py \\" in replicas_script


def test_lb_helpers_use_venv_python_and_nginx_keeps_its_container(venv_scripts):
    _, lb_script = venv_scripts

    assert f'HELPER_PYTHON=("{VENV}/bin/python")' in lb_script
    assert '"${HELPER_PYTHON[@]}" /' in lb_script
    assert "/images/vllm.sif" not in lb_script
    assert '"$DSWARM_CONTAINER_RUNTIME" instance start' in lb_script
    assert "/images/nginx.sif" in lb_script


def test_container_mode_runs_vllm_and_helpers_in_the_image(container_scripts):
    replicas_script, lb_script = container_scripts

    assert '"$DSWARM_CONTAINER_RUNTIME" exec --bind $MOUNTS' in replicas_script
    assert "python3 /opt/watchdog.py \\" in replicas_script
    assert 'HELPER_PYTHON=("$DSWARM_CONTAINER_RUNTIME" exec' in lb_script
    assert '"/images/vllm.sif" python3)' in lb_script
    assert lb_script.count('"${HELPER_PYTHON[@]}" /') == 3


@pytest.mark.parametrize("scripts", ["venv_scripts", "container_scripts"])
def test_rendered_scripts_are_valid_bash(scripts, request):
    for script in request.getfixturevalue(scripts):
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
