# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import jinja2

from domyn_swarm.config.router import NginxRouter

TPL_DIR = Path("src/domyn_swarm/templates")


def _cfg(mon_enabled, gpu_enabled):
    gx = SimpleNamespace(enabled=gpu_enabled, port=9835)
    mon = SimpleNamespace(
        enabled=mon_enabled, route_prefix="/prometheus", port=9090, gpu_exporter=gx
    )
    ep = SimpleNamespace(
        port=9000,
        nginx_timeout="60s",
        enable_proxy_buffering=True,
        monitoring=mon,
        router=NginxRouter(),
    )
    backend = SimpleNamespace(endpoint=ep, requires_ray=False)
    return SimpleNamespace(backend=backend)


def _render(cfg):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("nginx_server.conf.j2").render(cfg=cfg)


def test_gpu_ownership_location_present_when_enabled():
    out = _render(_cfg(True, True))
    assert "location = /gpu_ownership" in out
    assert "alias /etc/nginx/conf.d/gpu_ownership.prom;" in out
    assert "default_type text/plain" in out
    assert "allow 127.0.0.1" in out
    assert "deny all" in out


def test_gpu_ownership_location_absent_when_gpu_disabled():
    out = _render(_cfg(True, False))
    assert "gpu_ownership" not in out


def test_server_conf_proxies_upstream_over_http11():
    out = _render(_cfg(mon_enabled=False, gpu_enabled=False))
    assert "proxy_http_version 1.1;" in out
    assert 'proxy_set_header Connection "";' in out
