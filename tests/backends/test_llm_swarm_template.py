# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import jinja2

from domyn_swarm.config.router import NginxRouter
from domyn_swarm.config.slurm import GpuExporterConfig

TPL_DIR = Path("src/domyn_swarm/templates")


def _cfg(gpu_enabled, kind="nvidia_smi", requeue=True):
    """Build a minimal config-like object exposing the real GpuExporterConfig methods."""
    gx = GpuExporterConfig(enabled=gpu_enabled, kind=kind)
    mon = SimpleNamespace(enabled=True, mode="binary", gpu_exporter=gx)
    ep = SimpleNamespace(port=9000, monitoring=mon, ray_port=6379, router=NginxRouter())
    backend = SimpleNamespace(
        endpoint=ep,
        requires_ray=False,
        account="a",
        qos="q",
        partition="p",
        time_limit="1:00:00",
        requeue=requeue,
        preamble=[],
        modules=[],
        exclude_nodes=None,
        node_list=None,
        mail_user=None,
    )
    return SimpleNamespace(
        backend=backend,
        gpus_per_node=4,
        gpus_per_replica=1,
        replicas=4,
        replicas_per_node=4,
        cpus_per_task=8,
        image="vllm.sif",
        model="m",
        args="",
        port=9000,
        env={"HF_HOME": "/hf"},
        mail_user=None,
    )


def _render(cfg):
    """Render llm_swarm.sh.j2 with the same callables the driver passes."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("llm_swarm.sh.j2").render(
        cfg=cfg,
        job_name="jx",
        swarm_directory="/swarm",
        path_exists=lambda *_: True,
        is_folder=lambda *_: True,
        cuda_visible_devices=["0", "1", "2", "3"],
        watchdog_script_path="/w.py",
        dswarm_agent_version="0.0.0",
        build_watchdog_args=lambda *a, **k: [],
        args_to_str=lambda *_: "",
    )


def test_gpu_exporter_launch_present_when_enabled():
    out = _render(_cfg(True))
    assert "nvidia_gpu_exporter" in out
    assert "gpu-$(hostname).target" in out
    assert "gpu-owner-" in out


def test_gpu_exporter_absent_when_disabled():
    out = _render(_cfg(False))
    assert "nvidia_gpu_exporter" not in out
    assert "gpu-owner-" not in out


def test_watchdog_wraps_vllm_when_enabled():
    """The watchdog supervises `vllm serve` as its child process."""
    cfg = _cfg(False)
    cfg.watchdog = SimpleNamespace(enabled=True)

    out = _render(cfg)

    assert "python3 /opt/watchdog.py" in out
    assert "vllm serve" in out


def test_watchdog_is_not_launched_when_disabled():
    """`watchdog.enabled: false` is documented as the master switch."""
    cfg = _cfg(False)
    cfg.watchdog = SimpleNamespace(enabled=False)

    out = _render(cfg)

    # The script stays bind-mounted (harmless); what matters is that it is not invoked.
    assert "python3 /opt/watchdog.py" not in out
    assert "vllm serve" in out, "the server itself must still start"


def test_requeue_directive_present_when_enabled():
    assert "#SBATCH --requeue" in _render(_cfg(False, requeue=True))


def test_requeue_directive_absent_when_disabled():
    """Sites that forbid requeueing must not get the directive at all."""
    assert "#SBATCH --requeue" not in _render(_cfg(False, requeue=False))
