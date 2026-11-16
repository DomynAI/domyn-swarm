# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil
import subprocess

import jinja2
import pytest

import domyn_swarm
from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.runtime import watchdog_args as watchdog_args_mod

NUMACTL = 'numactl --membind="$DSWARM_NUMA_NODES" "$DSWARM_CONTAINER_RUNTIME"'


def _slurm_config(**kwargs) -> SlurmConfig:
    return SlurmConfig(
        partition="p",
        account="a",
        qos="q",
        endpoint=SlurmEndpointConfig(nginx_image="/path/to/nginx.sif"),
        **kwargs,
    )


def _render(numa_membind: str | None, gpus_per_replica: int = 1) -> str:
    """Render the cluster script the way SlurmDriver does; more GPUs than a node selects Ray."""
    import domyn_swarm.runtime.watchdog as watchdog_mod

    cfg = DomynLLMSwarmConfig(
        name="numa",
        image="/path/to/vllm.sif",
        model="m",
        gpus_per_node=4,
        gpus_per_replica=gpus_per_replica,
        backend={"type": "slurm"}
        | _slurm_config(numa_membind=numa_membind).model_dump(exclude_unset=True),
    )
    template_path = cfg.backend.template_path
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path.parent),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_path.name).render(
        cfg=cfg,
        job_name="numa",
        path_exists=path_exists,
        is_folder=is_folder,
        cuda_visible_devices="0,1,2,3",
        swarm_directory="/tmp/swarm",
        watchdog_script_path=Path(watchdog_mod.__file__).resolve().as_posix(),
        build_watchdog_args=watchdog_args_mod.build_watchdog_args,
        args_to_str=watchdog_args_mod.args_to_str,
        dswarm_agent_version=domyn_swarm.__version__,
    )


def _numa_block(rendered: str) -> str:
    start = rendered.index("if ! command -v numactl")
    end = rendered.index("\n", rendered.index('echo "NUMA memory binding'))
    return rendered[start:end]


def test_numa_membind_defaults_to_unset():
    assert _slurm_config().numa_membind is None


@pytest.mark.parametrize("value", ["cpu", "0", "0-3", "0,2", "0-1,4-5"])
def test_numa_membind_accepts_cpu_and_node_lists(value):
    assert _slurm_config(numa_membind=value).numa_membind == value


@pytest.mark.parametrize("value", ["", "CPU", "all", "0-", "0;rm -rf /", "$(id)", " 0-3"])
def test_numa_membind_rejects_other_values(value):
    with pytest.raises(ValueError):
        _slurm_config(numa_membind=value)


@pytest.mark.parametrize("gpus_per_replica", [1, 8])
def test_unset_numa_membind_renders_no_numactl(gpus_per_replica):
    rendered = _render(None, gpus_per_replica)
    assert "numactl" not in rendered
    assert "DSWARM_NUMA_NODES" not in rendered


def test_single_node_replica_runs_under_numactl():
    rendered = _render("0-3")
    assert rendered.count(NUMACTL) == 1
    assert 'DSWARM_NUMA_NODES="0-3"' in rendered


def test_ray_head_workers_and_server_run_under_numactl():
    rendered = _render("0-3", gpus_per_replica=8)
    assert rendered.count(NUMACTL) == 3
    assert f"{NUMACTL} instance start --bind $MOUNTS" in rendered
    assert f"{NUMACTL} instance start --bind '\"$MOUNTS\"'" in rendered
    assert f"{NUMACTL} exec --bind $MOUNTS" in rendered


def _run_cpu_detection(tmp_path: Path, cpulists: dict[int, str]) -> subprocess.CompletedProcess:
    """Run the rendered `cpu` detection against a fake sysfs, with a stub numactl on PATH."""
    sysfs = tmp_path / "node"
    for node, cpulist in cpulists.items():
        (sysfs / f"node{node}").mkdir(parents=True)
        (sysfs / f"node{node}" / "cpulist").write_text(cpulist + "\n")
    (sysfs / "possible").write_text("0-15\n")  # sibling files that the glob must not match
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "numactl").write_text("#!/bin/sh\n")
    (bin_dir / "numactl").chmod(0o755)

    block = _numa_block(_render("cpu")).replace("/sys/devices/system/node", sysfs.as_posix())
    return subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail\nexport PATH="{bin_dir}:$PATH"\n{block}\necho "$DSWARM_NUMA_NODES"',
        ],
        capture_output=True,
        text=True,
    )


def test_cpu_mode_binds_to_numa_nodes_with_cpus(tmp_path):
    # Grace Hopper with 4 superchips: nodes 0-3 hold the CPUs, the rest are GPU memory.
    proc = _run_cpu_detection(
        tmp_path, {0: "0-71", 1: "72-143", 2: "144-215", 3: "216-287", 4: "", 12: ""}
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "0,1,2,3"


def test_cpu_mode_fails_without_numa_nodes_with_cpus(tmp_path):
    proc = _run_cpu_detection(tmp_path, {4: "", 5: ""})
    assert proc.returncode == 1
    assert "no NUMA node with CPUs" in proc.stderr


def test_numa_membind_fails_without_numactl(tmp_path):
    block = _numa_block(_render("0-3"))
    bash = shutil.which("bash")
    assert bash is not None
    proc = subprocess.run(
        [bash, "-c", block], capture_output=True, text=True, env={"PATH": tmp_path.as_posix()}
    )
    assert proc.returncode == 1
    assert "numactl is not available" in proc.stderr
