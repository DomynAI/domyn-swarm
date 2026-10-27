# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Byte-for-byte parity gate for the llm_swarm template split.

The monolithic ``llm_swarm.sh.j2`` was split into ``_swarm_common.sh.j2`` +
``llm_swarm.sh.j2`` (vLLM-only) + ``llm_swarm_ray.sh.j2`` (Ray). These snapshots
capture the pre-split rendering with monitoring OFF and assert the split
templates render identically. They also permanently guard the
"monitoring off => byte-for-byte unchanged" invariant: an intentional change to
the base (monitoring-off) rendering must update the fixtures deliberately.
"""

from pathlib import Path
from types import SimpleNamespace

import jinja2

from domyn_swarm.config.router import NginxRouter
from domyn_swarm.config.slurm import GpuExporterConfig

TPL_DIR = Path("src/domyn_swarm/templates")
FIX_DIR = Path("tests/backends/fixtures")


def _cfg(requires_ray: bool):
    gx = GpuExporterConfig(enabled=False)
    mon = SimpleNamespace(enabled=False, mode="binary", gpu_exporter=gx)
    ep = SimpleNamespace(port=9000, monitoring=mon, ray_port=6379, router=NginxRouter())
    backend = SimpleNamespace(
        endpoint=ep,
        requires_ray=requires_ray,
        ray_port=6379,
        ray_dashboard_port=8265,
        account="a",
        qos="q",
        partition="p",
        time_limit="1:00:00",
        requeue=True,
        preamble=[],
        modules=[],
        exclude_nodes=None,
        node_list=None,
        mail_user=None,
    )
    return SimpleNamespace(
        backend=backend,
        gpus_per_node=4,
        gpus_per_replica=8 if requires_ray else 1,
        replicas=1 if requires_ray else 4,
        replicas_per_node=1 if requires_ray else 4,
        cpus_per_task=8,
        image="vllm.sif",
        model="m",
        args="",
        port=9000,
        env={"HF_HOME": "/hf"},
        mail_user=None,
    )


def _render(template_name: str, cfg) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(
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


def _norm(s: str) -> str:
    """Ignore trailing-newline differences (cosmetic in a shell script, and the
    repo's end-of-file-fixer hook normalizes them anyway). All meaningful script
    content and internal whitespace is still compared exactly."""
    return s.rstrip("\n") + "\n"


def test_vllm_only_split_matches_baseline():
    expected = (FIX_DIR / "llm_swarm_baseline_novray.txt").read_text()
    assert _norm(_render("llm_swarm.sh.j2", _cfg(requires_ray=False))) == _norm(expected)


def test_ray_split_matches_baseline():
    expected = (FIX_DIR / "llm_swarm_baseline_ray.txt").read_text()
    assert _norm(_render("llm_swarm_ray.sh.j2", _cfg(requires_ray=True))) == _norm(expected)
