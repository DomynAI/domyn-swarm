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


def _run_runtime_detector(
    tmp_path: Path,
    *runtimes: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for runtime in runtimes:
        _install_fake_runtime(bin_dir, runtime)

    script = "\n".join(
        [
            _render_runtime_detector(),
            'printf "%s\\n%s\\n" "$DSWARM_CONTAINER_RUNTIME" "$DSWARM_CONTAINER_RUNTIME_STATE_DIR"',
        ]
    )
    process_environment = {"HOME": str(tmp_path), "PATH": str(bin_dir)}
    process_environment.update(environment or {})
    return subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True,
        text=True,
        env=process_environment,
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


@pytest.mark.parametrize(
    ("runtime", "variable", "directory"),
    [
        ("apptainer", "APPTAINER_CONFIGDIR", "/apptainer-config"),
        ("singularity", "SINGULARITY_CONFIGDIR", "/singularity-config"),
    ],
)
def test_runtime_state_directory_honors_runtime_configuration(
    tmp_path, runtime, variable, directory
):
    result = _run_runtime_detector(
        tmp_path,
        runtime,
        environment={variable: directory},
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [runtime, directory]


def _cfg(
    *,
    requires_ray: bool,
    modules: tuple[str, ...] = (),
    monitoring: bool = False,
) -> DomynLLMSwarmConfig:
    endpoint = SlurmEndpointConfig(
        nginx_image="nginx.sif",
        monitoring={
            "enabled": monitoring,
            "mode": "binary",
            "gpu_exporter": {"enabled": monitoring},
        },
    )
    backend = SlurmConfig(
        partition="p",
        account="a",
        qos="q",
        modules=list(modules),
        endpoint=endpoint,
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


def _render(
    template_name: str,
    *,
    requires_ray: bool,
    modules: tuple[str, ...] = (),
    monitoring: bool = False,
    environment: dict[str, str] | None = None,
) -> str:
    cfg = _cfg(
        requires_ray=requires_ray,
        modules=modules,
        monitoring=monitoring,
    )
    cfg.env.update(environment or {})
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
    assert re.search(r'"\$DSWARM_CONTAINER_RUNTIME" (?:exec|instance)', rendered)
    assert not re.search(r"(?m)^\s*(?:apptainer|singularity) (?:exec|instance)", rendered)


def test_replica_environment_is_exported_for_both_runtimes():
    rendered = _render("llm_swarm.sh.j2", requires_ray=False)

    assert 'export APPTAINERENV_HF_HOME="/hf"' in rendered
    assert 'export SINGULARITYENV_HF_HOME="/hf"' in rendered


def test_ray_workers_receive_runtime_and_environment_aliases():
    rendered = _render("llm_swarm_ray.sh.j2", requires_ray=True)

    assert "export DSWARM_CONTAINER_RUNTIME" in rendered
    assert "export APPTAINERENV_VLLM_HOST_IP=" in rendered
    assert "export SINGULARITYENV_VLLM_HOST_IP=" in rendered
    assert 'src="$DSWARM_CONTAINER_RUNTIME_STATE_DIR/instances/logs/' in rendered


def test_load_balancer_loads_configured_modules_before_runtime_detection():
    rendered = _render(
        "lb.sh.j2",
        requires_ray=False,
        modules=("apptainer/1.4",),
    )

    assert rendered.index("module load apptainer/1.4") < rendered.index("command -v apptainer")


def test_ray_checks_each_node_before_starting_the_head():
    rendered = _render("llm_swarm_ray.sh.j2", requires_ray=True)

    execution = rendered.index("# Execute multi-node deployment steps")
    check = rendered.index("\nverify_container_runtimes\n", execution)
    head_start = rendered.index("\nstart_ray_head\n", execution)
    assert check < head_start
    assert (
        "--ntasks-per-node=1" in rendered[rendered.index("verify_container_runtimes()") : execution]
    )


def test_ray_remote_steps_detect_their_local_runtime():
    rendered = _render(
        "llm_swarm_ray.sh.j2",
        requires_ray=True,
        monitoring=True,
    )

    worker_start = rendered.index("bash -c '")
    worker_end = rendered.index("' 2>&1 &", worker_start)
    assert "command -v apptainer" in rendered[worker_start:worker_end]

    exporter_start = rendered.index("launch_gpu_exporter()")
    exporter_end = rendered.index("\n}", exporter_start)
    assert "command -v apptainer" in rendered[exporter_start:exporter_end]


def test_embedded_runtime_detection_ends_before_the_following_command():
    rendered = _render("llm_swarm_ray.sh.j2", requires_ray=True)

    assert "export DSWARM_CONTAINER_RUNTIME_STATE_DIR\n    ' >/dev/null" in rendered
    assert (
        "export DSWARM_CONTAINER_RUNTIME_STATE_DIR\n       export APPTAINERENV_VLLM_HOST_IP="
    ) in rendered


def test_missing_ray_log_directory_does_not_abort_the_job():
    rendered = _render("llm_swarm_ray.sh.j2", requires_ray=True)

    assert 'src="$DSWARM_CONTAINER_RUNTIME_STATE_DIR/instances/logs/$(hostname)/$USER"' in (
        rendered
    )
    assert 'if [ -d "$src" ]; then' in rendered


def test_configured_container_runtime_variable_does_not_replace_launcher():
    rendered = _render(
        "llm_swarm.sh.j2",
        requires_ray=False,
        environment={"CONTAINER_RUNTIME": "inside-container"},
    )

    assert 'export CONTAINER_RUNTIME="inside-container"' in rendered
    assert '"$DSWARM_CONTAINER_RUNTIME" exec' in rendered
