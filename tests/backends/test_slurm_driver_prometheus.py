# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jinja2

from domyn_swarm.config.router import NginxRouter, VllmRouter

TPL_DIR = Path("src/domyn_swarm/templates")


def _render(**ctx) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("prometheus.yml.j2").render(**ctx)


class _Mon:
    def __init__(self, gpu_enabled, ray_enabled=False):
        self.scrape_interval = "15s"
        self.exporter_port = 9113

        class _GX:
            enabled = gpu_enabled
            port = 9835

        self.gpu_exporter = _GX()

        class _RX:
            enabled = ray_enabled

        self.ray_metrics = _RX()


class _Cfg:
    class backend:
        class endpoint:
            port = 9000
            router = NginxRouter()

    name = "swarm-x"


def test_prometheus_gpu_jobs_present_when_enabled():
    out = _render(
        monitoring=_Mon(True), targets_path="/t.json", gpu_targets_path="/g.json", cfg=_Cfg()
    )
    assert "job_name: gpu\n" in out or "job_name: gpu" in out
    assert "gpu_ownership" in out


def test_prometheus_gpu_jobs_absent_when_disabled():
    out = _render(
        monitoring=_Mon(False), targets_path="/t.json", gpu_targets_path="/g.json", cfg=_Cfg()
    )
    assert "job_name: gpu" not in out
    assert "gpu_ownership" not in out


def test_prometheus_ray_job_present_when_enabled():
    out = _render(
        monitoring=_Mon(False, ray_enabled=True),
        targets_path="/t.json",
        gpu_targets_path="/g.json",
        ray_targets_path="/serving/ray_targets.json",
        cfg=_Cfg(),
    )
    assert "job_name: ray" in out
    assert "ray_targets.json" in out


def test_prometheus_ray_job_absent_when_disabled():
    out = _render(
        monitoring=_Mon(False, ray_enabled=False),
        targets_path="/t.json",
        gpu_targets_path="/g.json",
        ray_targets_path="/serving/ray_targets.json",
        cfg=_Cfg(),
    )
    assert "job_name: ray" not in out


def _router_job(router):
    cfg = _Cfg()
    cfg.backend.endpoint.router = router
    return _render(
        monitoring=_Mon(False),
        targets_path="/t.json",
        gpu_targets_path="/g.json",
        cfg=cfg,
        router_metrics=router.metrics_target(),
    )


def test_prometheus_router_job_present_for_vllm_router():
    out = _router_job(VllmRouter(binary="vllm-router"))
    assert "job_name: router" in out
    assert "127.0.0.1:9201" in out
    assert "metrics_path: /metrics" in out.split("job_name: router")[1].split("job_name")[0]


def test_prometheus_router_job_absent_for_nginx():
    assert "job_name: router" not in _router_job(NginxRouter())
