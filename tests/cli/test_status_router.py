# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The router type shown by `domyn-swarm status`."""

import io
from pathlib import Path

import pytest
from rich.console import Console

from domyn_swarm.cli.main import _build_status_json
from domyn_swarm.cli.tui.status import render_swarm_status
from domyn_swarm.config.router import VllmRouter
from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.state.db import make_session_factory
from domyn_swarm.core.state.models import JobRecord, SwarmRecord
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.platform.protocols import ServingHandle, ServingPhase, ServingStatus


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / SwarmStateManager.DB_NAME
    monkeypatch.setattr(SwarmStateManager, "_get_db_path", classmethod(lambda cls: path))
    session_factory = make_session_factory(path)
    with session_factory() as s:
        engine = s.get_bind()
        SwarmRecord.__table__.create(bind=engine, checkfirst=True)
        JobRecord.__table__.create(bind=engine, checkfirst=True)
    return path


def _swarm(tmp_path: Path, **endpoint_overrides) -> DomynLLMSwarm:
    cfg = DomynLLMSwarmConfig(
        name="fake",
        hf_home=".",
        model="m",
        image="/path/to/vllm.sif",
        replicas=1,
        gpus_per_node=1,
        gpus_per_replica=1,
        replicas_per_node=1,
        nodes=1,
        port=1000,
        cpus_per_task=2,
        mem_per_cpu="1GB",
        wait_endpoint_s=1200,
        home_directory=tmp_path,
        backend=SlurmConfig(
            type="slurm",
            partition="p",
            account="a",
            qos="q",
            requires_ray=False,
            endpoint=SlurmEndpointConfig(
                cpus_per_task=1,
                nginx_image="/path/to/vllm.sif",
                mem="1GB",
                threads_per_core=1,
                wall_time="24:00:00",
                enable_proxy_buffering=True,
                nginx_timeout="60s",
                **endpoint_overrides,
            ),
        ),
    )
    return DomynLLMSwarm(
        name="swarm",
        cfg=cfg,
        serving_handle=ServingHandle(
            id="1234",
            url="http://lb:9003",
            meta={"lb_jobid": 1234, "jobid": 5678, "lb_node": "lrdn4759", "port": 9003},
        ),
        endpoint="http://lb:9003",
        delete_on_exit=True,
    )


def _status_json(swarm: DomynLLMSwarm) -> dict:
    return _build_status_json(
        name="swarm",
        swarm=swarm,
        serving_status=ServingStatus(phase=ServingPhase.RUNNING, url="http://lb:9003"),
        replica_summary=None,
        replica_rows=[],
        errors=[],
    )


def _load_balancer(payload: dict) -> dict:
    return next(j for j in payload["slurm_jobs"] if j["role"] == "load_balancer")


def test_status_json_reports_the_configured_router(db_path: Path, tmp_path: Path) -> None:
    swarm = _swarm(tmp_path, router=VllmRouter(binary="vllm-router"))

    payload = _status_json(swarm)

    assert payload["schema_version"] == 1
    assert _load_balancer(payload) == {
        "role": "load_balancer",
        "jobid": 1234,
        "node": "lrdn4759",
        "router": "vllm-router",
    }


def test_status_json_reports_nginx_for_a_stored_swarm_without_router(
    db_path: Path, tmp_path: Path
) -> None:
    _swarm(tmp_path)._state_mgr.save(deployment_name="swarm")
    session_factory = make_session_factory(db_path)
    with session_factory() as s:
        rec = s.get(SwarmRecord, "swarm")
        cfg = dict(rec.cfg)
        backend = dict(cfg["backend"])
        endpoint = {k: v for k, v in backend["endpoint"].items() if k != "router"}
        backend["endpoint"] = endpoint
        cfg["backend"] = backend
        rec.cfg = cfg
        s.commit()

    loaded = SwarmStateManager.load(deployment_name="swarm")

    assert _load_balancer(_status_json(loaded))["router"] == "nginx"


def test_status_table_shows_the_router(tmp_path: Path) -> None:
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    st = ServingStatus(phase=ServingPhase.RUNNING, url="http://lb:9003")

    render_swarm_status("swarm", "slurm", st, router="vllm-router", console=console)

    assert "Router" in buf.getvalue()
    assert "vllm-router" in buf.getvalue()
