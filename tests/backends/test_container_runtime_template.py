# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re
import subprocess

import jinja2
import pytest

from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig

TPL_DIR = Path("src/domyn_swarm/templates")


def _environment() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render_runtime_detector() -> str:
    try:
        return _environment().get_template("_container_runtime.sh.j2").render()
    except jinja2.TemplateNotFound:
        pytest.fail("container runtime detector template is missing")


def _install_fake_runtime(directory: Path, name: str) -> None:
    runtime = directory / name
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)


def _run_runtime_detector(tmp_path: Path, *runtimes: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for runtime in runtimes:
        _install_fake_runtime(bin_dir, runtime)

    script = "\n".join(
        [
            _render_runtime_detector(),
            'printf "%s\\n%s\\n" "$CONTAINER_RUNTIME" "$CONTAINER_RUNTIME_STATE_DIR"',
        ]
    )
    return subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": str(bin_dir)},
        check=False,
    )


def test_apptainer_is_preferred_when_both_runtimes_exist(tmp_path):
    result = _run_runtime_detector(tmp_path, "apptainer", "singularity")

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["apptainer", f"{tmp_path}/.apptainer"]


def test_singularity_is_used_when_apptainer_is_missing(tmp_path):
    result = _run_runtime_detector(tmp_path, "singularity")

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["singularity", f"{tmp_path}/.singularity"]


def test_missing_runtime_exits_with_actionable_error(tmp_path):
    result = _run_runtime_detector(tmp_path)

    assert result.returncode != 0
    assert "apptainer" in result.stderr
    assert "singularity" in result.stderr


def _cfg(*, requires_ray: bool) -> DomynLLMSwarmConfig:
    backend = SlurmConfig(
        partition="p",
        account="a",
        qos="q",
        endpoint=SlurmEndpointConfig(nginx_image="nginx.sif"),
    )
    return DomynLLMSwarmConfig(
        name="runtime-test",
        image="vllm.sif",
        model="model",
        replicas=1,
        gpus_per_node=4,
        gpus_per_replica=8 if requires_ray else 1,
        env={"HF_HOME": "/hf"},
        backend=backend.model_dump(),
    )


def _render(template_name: str, *, requires_ray: bool) -> str:
    cfg = _cfg(requires_ray=requires_ray)
    context = {
        "cfg": cfg,
        "job_name": "runtime-test",
        "swarm_directory": "/swarm",
        "path_exists": lambda *_: False,
        "is_folder": lambda *_: False,
        "cuda_visible_devices": ["0", "1", "2", "3"],
        "watchdog_script_path": "/watchdog.py",
        "dswarm_agent_version": "0.0.0",
        "build_watchdog_args": lambda *args, **kwargs: [],
        "args_to_str": lambda *_: "",
        "dcgm_core_counters_path": "/core-counters.csv",
        "dep_jobid": 1,
        "replicas": 1,
        "collector_script_path": "/collector.py",
        "supervisor_script_path": "/supervisor.py",
    }
    return _environment().get_template(template_name).render(**context)


@pytest.mark.parametrize(
    ("template_name", "requires_ray"),
    [
        ("llm_swarm.sh.j2", False),
        ("llm_swarm_ray.sh.j2", True),
        ("lb.sh.j2", False),
    ],
)
def test_generated_slurm_scripts_use_detected_runtime(template_name, requires_ray):
    rendered = _render(template_name, requires_ray=requires_ray)

    assert "command -v apptainer" in rendered
    assert re.search(r'"\$CONTAINER_RUNTIME" (?:exec|instance)', rendered)
    assert not re.search(r"(?m)^\s*(?:apptainer|singularity) (?:exec|instance)", rendered)


def test_replica_environment_is_exported_for_both_runtimes():
    rendered = _render("llm_swarm.sh.j2", requires_ray=False)

    assert 'export APPTAINERENV_HF_HOME="/hf"' in rendered
    assert 'export SINGULARITYENV_HF_HOME="/hf"' in rendered


def test_ray_workers_receive_runtime_and_environment_aliases():
    rendered = _render("llm_swarm_ray.sh.j2", requires_ray=True)

    assert "export CONTAINER_RUNTIME" in rendered
    assert "export APPTAINERENV_VLLM_HOST_IP=" in rendered
    assert "export SINGULARITYENV_VLLM_HOST_IP=" in rendered
    assert 'src="$CONTAINER_RUNTIME_STATE_DIR/instances/logs/' in rendered
