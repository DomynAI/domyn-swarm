# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

# tests/backends/test_lb_template_render.py
from pathlib import Path
from types import SimpleNamespace

import jinja2

from domyn_swarm.config.router import NginxRouter

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "domyn_swarm" / "templates"


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _main_cfg(cpus_per_task: int = 32, nginx_worker_processes=None) -> SimpleNamespace:
    return SimpleNamespace(
        backend=SimpleNamespace(
            endpoint=SimpleNamespace(
                cpus_per_task=cpus_per_task,
                nginx_worker_processes=nginx_worker_processes,
                router=NginxRouter(),
            )
        )
    )


def test_nginx_main_sets_run_pid_and_includes_confd():
    out = _env().get_template("nginx.conf.j2").render(cfg=_main_cfg())
    assert "pid /run/nginx.pid;" in out
    assert "include /etc/nginx/conf.d/*.conf;" in out
    assert "/tmp/nginx.pid" not in out
    assert "upstream vllm" not in out


def test_nginx_main_defaults_workers_to_cpus_per_task():
    out = _env().get_template("nginx.conf.j2").render(cfg=_main_cfg(cpus_per_task=32))
    assert "worker_processes 32;" in out


def test_nginx_main_honours_explicit_worker_processes():
    out = (
        _env()
        .get_template("nginx.conf.j2")
        .render(cfg=_main_cfg(cpus_per_task=32, nginx_worker_processes="auto"))
    )
    assert "worker_processes auto;" in out

    out = (
        _env()
        .get_template("nginx.conf.j2")
        .render(cfg=_main_cfg(cpus_per_task=32, nginx_worker_processes=4))
    )
    assert "worker_processes 4;" in out


def test_server_conf_has_llm_proxy_and_health():
    class EP:
        port = 9000
        nginx_timeout = "60s"
        router = NginxRouter()
        enable_proxy_buffering = True
        monitoring = SimpleNamespace(
            enabled=False,
            port=9090,
            route_prefix="/prometheus",
            exporter_port=9113,
            gpu_exporter=SimpleNamespace(enabled=False, kind="nvidia_smi", port=9835),
        )

    class Backend:
        requires_ray = False
        ray_dashboard_port = 8265
        ray_port = 6379
        endpoint = EP()

    class Cfg:
        backend = Backend()

    out = _env().get_template("nginx_server.conf.j2").render(cfg=Cfg())
    assert "listen 9000;" in out
    assert "proxy_pass http://llm;" in out
    assert "location = /health" in out
    assert "proxy_buffer_size 64k;" in out
    assert "/ray/" not in out  # ray disabled


def test_server_conf_includes_ray_when_enabled():
    class EP:
        port = 9000
        nginx_timeout = "60s"
        router = NginxRouter()
        enable_proxy_buffering = False
        monitoring = SimpleNamespace(
            enabled=False,
            port=9090,
            route_prefix="/prometheus",
            exporter_port=9113,
            gpu_exporter=SimpleNamespace(enabled=False, kind="nvidia_smi", port=9835),
        )

    class Backend:
        requires_ray = True
        ray_dashboard_port = 8265
        ray_port = 6379
        endpoint = EP()

    class Cfg:
        backend = Backend()

    out = _env().get_template("nginx_server.conf.j2").render(cfg=Cfg())
    assert "location ^~ /ray/" in out
    assert "proxy_buffering off;" in out
    assert "return 301 /ray/dashboard/;" in out


class _EP:
    cpus_per_task = 32
    mem = "16GB"
    threads_per_core = 1
    wall_time = "24:00:00"
    port = 9000
    nginx_image = "/img/nginx.sif"
    nginx_timeout = "60s"
    enable_proxy_buffering = True
    collector_port = 9100
    nginx_upstream_keepalive = 32
    router = NginxRouter()
    monitoring = SimpleNamespace(
        enabled=False,
        mode="container",
        prometheus_image="/img/prom.sif",
        nginx_exporter_image="/img/nginxexp.sif",
        prometheus_binary="prometheus",
        nginx_exporter_binary="nginx-prometheus-exporter",
        port=9090,
        exporter_port=9113,
        route_prefix="/prometheus",
        scrape_interval="15s",
        retention="12h",
        gpu_exporter=SimpleNamespace(enabled=False, kind="nvidia_smi", port=9835),
    )


class _Backend:
    account = "acct"
    qos = "qos"
    partition = "part"
    mail_user = None
    requires_ray = False
    ray_dashboard_port = 8265
    ray_port = 6379
    endpoint = _EP()


class _Cfg:
    image = "/img/python.sif"
    wait_endpoint_s = 1800
    backend = _Backend()


def _render_lb():
    return (
        _env()
        .get_template("lb.sh.j2")
        .render(
            cfg=_Cfg(),
            job_name="job",
            dep_jobid=111,
            replicas=2,
            swarm_directory="/swarm/s1",
            collector_script_path="/opt/watchdog_collector.py",
            supervisor_script_path="/opt/lb_supervisor.py",
        )
    )


def test_lb_starts_supervisor_and_mounts_main_conf():
    out = _render_lb()
    assert "/opt/lb_supervisor.py" in out
    assert "nginx.conf:/etc/nginx/nginx.conf:ro" in out
    # The old inline generator is gone:
    assert "generate_nginx_conf" not in out
    # supervisor is invoked twice: one-shot init + steady-state loop
    assert out.count("lb_supervisor.py") >= 2
    assert "--once" in out
    assert "SUPERVISOR_PID=$!" in out
    # host-side reload mechanism (nginx is in its own PID namespace)
    assert "nginx -s reload" in out
    assert "RELOAD_PID=$!" in out
    assert "--pid-file" not in out
    # trap tears down both collector and supervisor
    assert "kill $COLLECTOR_PID $SUPERVISOR_PID" in out


def test_lb_still_starts_collector():
    out = _render_lb()
    assert "watchdog_collector.py" in out


def test_lb_gates_startup_on_a_listening_collector():
    out = _render_lb()
    # The collector is launched through the lifecycle helpers, which prove it
    # bound its port before the job carries on.
    assert "collector_wait_ready()" in out
    assert "collector_start_or_die" in out
    assert '--ready-file "$COLLECTOR_READY_FILE"' in out


def _cfg_with_monitoring(enabled: bool):
    class Mon:
        pass

    mon = Mon()
    mon.enabled = enabled
    mon.port = 9090
    mon.route_prefix = "/prometheus"
    mon.exporter_port = 9113
    mon.gpu_exporter = SimpleNamespace(enabled=False, kind="nvidia_smi", port=9835)

    class EP:
        port = 9000
        nginx_timeout = "60s"
        router = NginxRouter()
        enable_proxy_buffering = True
        monitoring = mon

    class Backend:
        requires_ray = False
        ray_dashboard_port = 8265
        ray_port = 6379
        endpoint = EP()

    class Cfg:
        backend = Backend()

    return Cfg()


def test_server_conf_adds_prometheus_locations_when_enabled():
    out = _env().get_template("nginx_server.conf.j2").render(cfg=_cfg_with_monitoring(True))
    assert "location /prometheus/" in out
    assert "proxy_pass http://127.0.0.1:9090" in out
    assert "location = /nginx_status" in out
    assert "stub_status;" in out
    assert "allow 127.0.0.1;" in out
    assert "deny all;" in out


def test_server_conf_no_prometheus_locations_when_disabled():
    out = _env().get_template("nginx_server.conf.j2").render(cfg=_cfg_with_monitoring(False))
    assert "/prometheus/" not in out
    assert "/nginx_status" not in out


def test_server_conf_prometheus_location_follows_route_prefix():
    cfg = _cfg_with_monitoring(True)
    cfg.backend.endpoint.monitoring.route_prefix = "/metrics"
    out = _env().get_template("nginx_server.conf.j2").render(cfg=cfg)
    assert "location /metrics/ {" in out
    assert "location /prometheus/" not in out


def test_prometheus_yml_render():
    from types import SimpleNamespace

    class Mon:
        scrape_interval = "15s"
        exporter_port = 9113
        route_prefix = "/prometheus"
        gpu_exporter = SimpleNamespace(enabled=False, kind="nvidia_smi", port=9835)
        ray_metrics = SimpleNamespace(enabled=False, port=8090)

    out = (
        _env()
        .get_template("prometheus.yml.j2")
        .render(
            monitoring=Mon(),
            targets_path="/etc/prometheus/serving/targets.json",
            cfg=SimpleNamespace(name="my-swarm", model="Qwen/Qwen3-32B"),
        )
    )
    assert "scrape_interval: 15s" in out
    assert "job_name: vllm" in out
    assert "/etc/prometheus/serving/targets.json" in out
    assert "127.0.0.1:9113" in out
    assert 'swarm: "my-swarm"' in out


def _render_lb_with_monitoring(enabled: bool):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        image="/img/python.sif",
        wait_endpoint_s=1800,
        backend=SimpleNamespace(
            account="acct",
            qos="qos",
            partition="part",
            mail_user=None,
            requires_ray=False,
            ray_dashboard_port=8265,
            ray_port=6379,
            endpoint=SimpleNamespace(
                cpus_per_task=32,
                mem="16GB",
                threads_per_core=1,
                wall_time="24:00:00",
                port=9000,
                nginx_image="/img/nginx.sif",
                nginx_timeout="60s",
                enable_proxy_buffering=True,
                router=NginxRouter(),
                monitoring=SimpleNamespace(
                    enabled=enabled,
                    mode="container",
                    prometheus_image="/img/prom.sif",
                    nginx_exporter_image="/img/nginxexp.sif",
                    prometheus_binary="prometheus",
                    nginx_exporter_binary="nginx-prometheus-exporter",
                    port=9090,
                    exporter_port=9113,
                    route_prefix="/prometheus",
                    scrape_interval="15s",
                    retention="12h",
                    gpu_exporter=SimpleNamespace(enabled=False, kind="nvidia_smi", port=9835),
                    ray_metrics=SimpleNamespace(enabled=False, port=8090),
                ),
            ),
        ),
    )
    return (
        _env()
        .get_template("lb.sh.j2")
        .render(
            cfg=cfg,
            job_name="job",
            dep_jobid=111,
            replicas=2,
            swarm_directory="/swarm/s1",
            collector_script_path="/opt/watchdog_collector.py",
            supervisor_script_path="/opt/lb_supervisor.py",
        )
    )


def test_lb_launches_sidecars_only_when_enabled():
    out = _render_lb_with_monitoring(enabled=True)
    assert "prometheus" in out
    assert "nginx-prometheus-exporter" in out
    assert "--web.route-prefix" in out
    assert "--emit-targets" in out
    assert "/etc/prometheus/serving" in out
    out_off = _render_lb_with_monitoring(enabled=False)
    assert "--web.route-prefix" not in out_off
    assert "--emit-targets" not in out_off


def test_lb_passes_upstream_keepalive_to_both_supervisor_calls():
    out = _render_lb()
    assert out.count("--upstream-keepalive 32") == 2
