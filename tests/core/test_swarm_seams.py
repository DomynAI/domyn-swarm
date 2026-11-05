# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The public seams of DomynLLMSwarm: platform, status, rehydration, purity."""

from pathlib import Path

import pytest

from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.state.db import make_session_factory
from domyn_swarm.core.state.models import JobRecord, SwarmRecord
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.platform.protocols import ServingHandle


def _init_schema(db_path: Path) -> None:
    """Create the swarm and job tables in a temporary SQLite file."""
    session_factory = make_session_factory(db_path)
    with session_factory() as s:
        engine = s.get_bind()
        SwarmRecord.__table__.create(bind=engine, checkfirst=True)
        JobRecord.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point SwarmStateManager at a temporary SQLite DB with the schema created."""
    path = tmp_path / SwarmStateManager.DB_NAME
    monkeypatch.setattr(SwarmStateManager, "_get_db_path", classmethod(lambda cls: path))
    _init_schema(path)
    return path


@pytest.fixture
def slurm_cfg(tmp_path: Path) -> DomynLLMSwarmConfig:
    """A minimal valid Slurm config rooted in a temporary home directory."""
    return DomynLLMSwarmConfig(
        name="fake",
        hf_home=".",
        image="image",
        model="Qwen/Qwen3-32B",
        revision=None,
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
            partition="partition",
            account="account",
            qos="qos",
            requires_ray=False,
            endpoint=SlurmEndpointConfig(
                cpus_per_task=1,
                nginx_image="/path/to/vllm.sif",
                mem="1GB",
                threads_per_core=1,
                wall_time="24:00:00",
                enable_proxy_buffering=True,
                nginx_timeout="60s",
            ),
        ),
    )


@pytest.fixture
def swarm(db_path: Path, slurm_cfg: DomynLLMSwarmConfig) -> DomynLLMSwarm:
    """A deployed-looking swarm wired to the temporary state DB."""
    return DomynLLMSwarm(
        name="swarm",
        cfg=slurm_cfg,
        serving_handle=ServingHandle(
            id="1234",
            url="http://lb:9003",
            meta={
                "lb_jobid": 1234,
                "jobid": 5678,
                "lb_node": "lrdn4759",
                "port": 9003,
                "name": "swarm",
            },
        ),
        endpoint="http://lb:9003",
        delete_on_exit=True,
    )


def test_platform_is_public(swarm: DomynLLMSwarm) -> None:
    """The platform is part of the public surface, not a private attribute."""
    assert swarm.platform == "slurm"
    assert not hasattr(swarm, "_platform")


def test_saved_swarm_reports_its_platform_in_list_all(swarm: DomynLLMSwarm) -> None:
    """Regression: `domyn-swarm swarm list` showed an empty backend column.

    `_platform` was a PrivateAttr, so it never reached the persisted payload
    that list_all() read it back out of.
    """
    swarm._state_mgr.save(deployment_name="swarm")

    rows = SwarmStateManager.list_all()
    row = next(r for r in rows if r["deployment_name"] == "swarm")

    assert row["platform"] == "slurm"


def test_serving_status_queries_the_backend(
    swarm: DomynLLMSwarm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller can ask for live serving status without touching privates."""
    from domyn_swarm.platform.protocols import ServingPhase, ServingStatus

    seen: dict = {}

    def fake_status(handle):
        seen["handle"] = handle
        return ServingStatus(phase=ServingPhase.RUNNING, url="http://lb:9003")

    monkeypatch.setattr(swarm._deployment.serving, "status", fake_status)

    result = swarm.serving_status()

    assert result.phase is ServingPhase.RUNNING
    assert seen["handle"] is swarm.serving_handle


def test_serving_status_on_an_undeployed_swarm_is_unknown(
    db_path: Path, slurm_cfg: DomynLLMSwarmConfig
) -> None:
    """Asking a never-deployed swarm for status is a question, not an assertion."""
    from domyn_swarm.platform.protocols import ServingPhase

    undeployed = DomynLLMSwarm(name="never-up", cfg=slurm_cfg)

    assert undeployed.serving_status().phase is ServingPhase.UNKNOWN


def test_from_record_builds_a_complete_swarm(swarm: DomynLLMSwarm) -> None:
    """Rehydration produces a finished object, not one to be back-filled."""
    swarm._state_mgr.save(deployment_name="swarm")

    loaded = SwarmStateManager.load(deployment_name="swarm")

    assert loaded.serving_handle is not None
    assert loaded.serving_handle.id == "1234"
    assert loaded.platform == "slurm"
    compute = loaded._require_compute_backend()
    assert compute.lb_jobid == 1234
    assert compute.lb_node == "lrdn4759"


def test_down_stops_a_swarm_whose_endpoint_never_became_ready(
    swarm: DomynLLMSwarm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: loading a record without ``lb_node`` raised, so ``down`` cancelled nothing.

    The load balancer reports its node only once it is healthy, so a swarm whose
    replicas crash at startup is persisted without one.
    """
    from domyn_swarm.backends.serving import slurm as slurm_serving

    assert swarm.serving_handle is not None
    del swarm.serving_handle.meta["lb_node"]
    swarm._state_mgr.save(deployment_name="swarm")
    cancelled: list[str] = []
    monkeypatch.setattr(slurm_serving, "_cancel_slurm", cancelled.append)

    SwarmStateManager.load(deployment_name="swarm").down()

    assert cancelled == ["5678", "1234"]
    assert all(r["deployment_name"] != "swarm" for r in SwarmStateManager.list_all())


def test_state_manager_does_not_touch_swarm_internals() -> None:
    """The persistence layer must not write private attributes of the swarm."""
    import inspect

    from domyn_swarm.core.state import state_manager

    source = inspect.getsource(state_manager)

    assert "swarm._deployment" not in source
    assert "swarm._platform" not in source


def test_adopt_attaches_a_handle_to_a_deployment(
    db_path: Path, slurm_cfg: DomynLLMSwarmConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`adopt` is the public seam replacing a write to Deployment._handle.

    Asserted through observable behaviour: a deployment with no handle reports
    UNKNOWN, and one that has adopted a handle forwards it to the backend.
    """
    from domyn_swarm.platform.protocols import ServingPhase, ServingStatus

    fresh = DomynLLMSwarm(name="adopter", cfg=slurm_cfg)
    assert fresh._deployment.status().phase is ServingPhase.UNKNOWN

    handle = ServingHandle(id="9999", url="http://other:9003", meta={"lb_jobid": 1})
    seen: dict = {}

    def fake_status(h):
        seen["handle"] = h
        return ServingStatus(phase=ServingPhase.RUNNING, url="http://other:9003")

    monkeypatch.setattr(fresh._deployment.serving, "status", fake_status)

    fresh._deployment.adopt(handle)

    assert fresh._deployment.status().phase is ServingPhase.RUNNING
    assert seen["handle"] is handle


def test_from_record_rejects_an_incoherent_slurm_record(swarm: DomynLLMSwarm) -> None:
    """A persisted Slurm handle without a jobid is rejected at rehydration.

    The Slurm serving backend indexes ``handle.meta["jobid"]`` directly, so a
    record missing it must fail loudly here rather than with a bare KeyError
    from deep inside ``status()``. The check belongs to the plan, which owns
    platform-specific knowledge; the swarm only asks it to run.
    """
    assert swarm.serving_handle is not None
    swarm.serving_handle.meta["jobid"] = None
    swarm._state_mgr.save(deployment_name="swarm")

    with pytest.raises(ValueError, match="job IDs"):
        SwarmStateManager.load(deployment_name="swarm")


def test_construction_creates_no_directories(
    db_path: Path, slurm_cfg: DomynLLMSwarmConfig, tmp_path: Path
) -> None:
    """Instantiating a swarm must not touch the filesystem."""
    before = set(tmp_path.rglob("*"))

    DomynLLMSwarm(name="inert", cfg=slurm_cfg)

    assert set(tmp_path.rglob("*")) == before


def test_ensure_directories_is_idempotent(db_path: Path, slurm_cfg: DomynLLMSwarmConfig) -> None:
    """The directory layout is created on demand, and calling twice is safe."""
    swarm = DomynLLMSwarm(name="dirs", cfg=slurm_cfg)

    swarm.ensure_directories()
    swarm.ensure_directories()

    for sub in (
        "serving",
        "jobs",
        "checkpoints",
        "logs/endpoint",
        "logs/replicas",
        "logs/slurm",
    ):
        assert (swarm.swarm_dir / sub).is_dir(), f"missing {sub}"


def test_construction_imports_no_serving_backend(
    db_path: Path, slurm_cfg: DomynLLMSwarmConfig
) -> None:
    """A cheap constructor is what lets read-only consumers use the real swarm.

    Guards the deletion of load_monitor_view, whose docstring justified itself
    by the ~400 modules a constructor used to pull in.
    """
    import subprocess
    import sys

    script = (
        "import sys;"
        "import domyn_swarm.core.swarm as m;"
        "assert 'domyn_swarm.backends.serving.slurm' not in sys.modules, "
        "'serving backend imported at module load'"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_monitor_reads_through_the_normal_load_path() -> None:
    """There is one read path for persisted swarms, not two."""
    from domyn_swarm.core.state import state_manager

    assert not hasattr(state_manager.SwarmStateManager, "load_monitor_view")
