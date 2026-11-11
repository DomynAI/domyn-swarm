# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Test the state persistence."""

import importlib.util
from pathlib import Path
import sqlite3

import pytest

from domyn_swarm.config.lepton import LeptonConfig
from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.state.db import make_session_factory
from domyn_swarm.core.state.migrate import upgrade_head
from domyn_swarm.core.state.models import JobRecord, SwarmRecord
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.exceptions import JobNotFoundError
from domyn_swarm.platform.protocols import ServingHandle


def _init_schema(db_path: Path) -> None:
    """Create the swarm table in the given SQLite file (for tests)."""
    session_factory = make_session_factory(db_path)
    with session_factory() as s:
        engine = s.get_bind()
        # SwarmRecord is a declarative model → its __table__ knows how to create itself
        SwarmRecord.__table__.create(bind=engine, checkfirst=True)
        JobRecord.__table__.create(bind=engine, checkfirst=True)


class TestSwarmStateManager:
    """Tests for swarm state persistence via SwarmStateManager."""

    @pytest.fixture
    def db_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """
        Point SwarmStateManager at a temporary SQLite DB and create the schema.
        """
        path = tmp_path / SwarmStateManager.DB_NAME

        def mock_db_path(cls) -> Path:
            # cls is SwarmStateManager; we ignore it and always return the temp path
            return path

        # Patch the classmethod target with a plain function; Python will bind cls.
        monkeypatch.setattr(SwarmStateManager, "_get_db_path", classmethod(mock_db_path))

        # Create the schema (swarm table) in that DB
        _init_schema(path)

        return path

    @pytest.fixture
    def swarm(
        self,
        db_path: Path,
    ) -> DomynLLMSwarm:
        """Construct a DomynLLMSwarm wired to the temporary DB via SwarmStateManager."""
        return DomynLLMSwarm(
            name="swarm",
            cfg=DomynLLMSwarmConfig(
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
            ),
            serving_handle=ServingHandle(
                id="1234",
                url="http://lb:9003",
                meta=dict(
                    lb_jobid=1234,
                    jobid=5678,
                    lb_node="lrdn4759",
                    port=9003,
                    name="swarm",
                ),
            ),
            endpoint="http://lb:9003",
            delete_on_exit=True,
        )

    @pytest.fixture
    def state_manager(self, swarm: DomynLLMSwarm) -> SwarmStateManager:
        """Return the SwarmStateManager attached to the swarm."""
        # DomynLLMSwarm should attach its own SwarmStateManager as _state_mgr
        return swarm._state_mgr  # type: ignore[attr-defined]

    @pytest.fixture
    def lepton_swarm(self, db_path: Path) -> DomynLLMSwarm:
        """Construct a Lepton-backed DomynLLMSwarm wired to the temporary DB.

        Lepton serving handles carry no ``jobid``/``lb_jobid``/``lb_node`` keys
        (those are Slurm-only concepts), so this fixture's ``serving_handle.meta``
        deliberately omits them.
        """
        return DomynLLMSwarm(
            name="lepton-swarm",
            cfg=DomynLLMSwarmConfig(
                name="fake-lepton",
                image="image",
                model="Qwen/Qwen3-32B",
                revision=None,
                replicas=1,
                gpus_per_replica=1,
                backend=LeptonConfig(
                    type="lepton",
                    workspace_id="ws-123",
                ),
            ),
            serving_handle=ServingHandle(
                id="lepton-endpoint",
                url="https://lepton.example/v1",
                meta=dict(
                    name="lepton-swarm",
                    workspace_id="ws-123",
                ),
            ),
            endpoint="https://lepton.example/v1",
            delete_on_exit=True,
        )

    # --------------------------------------------------------------------- #
    # Basic path / save / load behaviour
    # --------------------------------------------------------------------- #

    def test_get_db_path(self) -> None:
        """The DB name must be correct."""
        db_path = SwarmStateManager._get_db_path()
        assert db_path.name == SwarmStateManager.DB_NAME

    def test_save_persists_record(self, state_manager: SwarmStateManager) -> None:
        """Saving a swarm creates/updates a SwarmRecord row."""
        state_manager.save(deployment_name="name")

        rows = SwarmStateManager.list_all()
        assert any(r["deployment_name"] == "name" for r in rows)

    def test_load_slurm(self, state_manager: SwarmStateManager) -> None:
        """Loading a previously saved swarm reconstructs DomynLLMSwarm."""
        state_manager.save(deployment_name="fake")

        swarm = SwarmStateManager.load(deployment_name="fake")
        assert isinstance(swarm, DomynLLMSwarm)
        # Serving handle should be wired back
        assert swarm.serving_handle is not None
        assert swarm._require_compute_backend() is not None  # type: ignore[attr-defined]

    @pytest.mark.skipif(
        importlib.util.find_spec("leptonai") is None, reason="leptonai does not support Python 3.14"
    )
    def test_load_lepton_without_jobid_does_not_raise(self, lepton_swarm: DomynLLMSwarm) -> None:
        """Lepton handles have no jobid; load() must not apply the Slurm-only guard.

        Regression test for the platform-specific guard in
        ``SwarmStateManager.load()``: it must only reject missing job IDs for
        Slurm swarms, and must pass Lepton swarms through unchanged even though
        their serving handle has no ``jobid`` key at all.
        """
        state_mgr = lepton_swarm._state_mgr  # type: ignore[attr-defined]
        assert "jobid" not in lepton_swarm.serving_handle.meta  # type: ignore[union-attr]
        state_mgr.save(deployment_name="fake-lepton")

        swarm = SwarmStateManager.load(deployment_name="fake-lepton")

        assert isinstance(swarm, DomynLLMSwarm)
        assert swarm._plan is not None  # type: ignore[attr-defined]
        assert swarm._plan.platform == "lepton"  # type: ignore[attr-defined]
        assert swarm.serving_handle is not None
        assert swarm.serving_handle.url == "https://lepton.example/v1"
        assert swarm.serving_handle.meta.get("workspace_id") == "ws-123"
        assert swarm._require_compute_backend() is not None  # type: ignore[attr-defined]

    # --------------------------------------------------------------------- #
    # Error cases around missing job metadata
    # --------------------------------------------------------------------- #

    def test_load_null_jobid(self, state_manager: SwarmStateManager) -> None:
        """Null Job ID in serving_handle.meta should raise."""
        state_manager.swarm.serving_handle.meta["jobid"] = None  # type: ignore[union-attr]
        state_manager.save(deployment_name="fake")

        with pytest.raises(ValueError, match="job IDs"):
            SwarmStateManager.load(deployment_name="fake")

    def test_load_null_lb_jobid(self, state_manager: SwarmStateManager) -> None:
        """Null LB Job ID in serving_handle.meta should raise."""
        state_manager.swarm.serving_handle.meta["lb_jobid"] = None  # type: ignore[union-attr]
        state_manager.save(deployment_name="fake")

        with pytest.raises(ValueError, match="job IDs"):
            SwarmStateManager.load(deployment_name="fake")

    def test_load_null_lb_node(self, state_manager: SwarmStateManager) -> None:
        """A swarm whose endpoint never became ready loads, but cannot run jobs."""
        state_manager.swarm.serving_handle.meta["lb_node"] = None  # type: ignore[union-attr]
        state_manager.save(deployment_name="fake")

        swarm = SwarmStateManager.load(deployment_name="fake")

        with pytest.raises(RuntimeError, match="lb_node"):
            swarm._require_compute_backend()  # type: ignore[attr-defined]

    def test_load_unknown_job_raises(self, state_manager: SwarmStateManager) -> None:
        """Loading a non-existing deployment_name raises JobNotFoundError."""
        with pytest.raises(JobNotFoundError, match="fake"):
            SwarmStateManager.load(deployment_name="fake")

    # --------------------------------------------------------------------- #
    # Delete behaviour
    # --------------------------------------------------------------------- #

    def test_delete_record_removes_row(self, state_manager: SwarmStateManager) -> None:
        """delete_record should remove the row for the given deployment."""
        state_manager.save(deployment_name="swarm")

        # sanity check: it exists
        rows = SwarmStateManager.list_all()
        assert any(r["deployment_name"] == "swarm" for r in rows)

        state_manager.delete_record("swarm")

        rows_after = SwarmStateManager.list_all()
        assert all(r["deployment_name"] != "swarm" for r in rows_after)

    # --------------------------------------------------------------------- #
    # Convenience helpers: get_last_swarm_name / list_by_base_name
    # --------------------------------------------------------------------- #

    def test_get_last_swarm_name_none_when_empty(self, state_manager: SwarmStateManager) -> None:
        """get_last_swarm_name returns None when there are no rows."""
        assert SwarmStateManager.get_last_swarm_name() is None

    def test_get_last_swarm_name_after_save(self, state_manager: SwarmStateManager) -> None:
        """get_last_swarm_name returns the last saved deployment_name."""
        state_manager.save(deployment_name="only-swarm")
        assert SwarmStateManager.get_last_swarm_name() == "only-swarm"

    def test_list_by_base_name_filters_by_prefix(self, state_manager: SwarmStateManager) -> None:
        """list_by_base_name returns deployment_names matching '<base>-*'."""
        # Use the same swarm instance but save under different deployment names
        state_manager.save(deployment_name="foo-abc")
        state_manager.save(deployment_name="foo-def")
        state_manager.save(deployment_name="bar-123")

        matches = SwarmStateManager.list_by_base_name("foo")
        assert set(matches) == {"foo-abc", "foo-def"}

        matches_bar = SwarmStateManager.list_by_base_name("bar")
        assert matches_bar == ["bar-123"]

    def test_jobs_crud(self, state_manager: SwarmStateManager) -> None:
        state_manager.save(deployment_name="swarm-1")

        job_id = SwarmStateManager.create_job(
            deployment_name="swarm-1",
            provider="slurm",
            kind="step",
            status="RUNNING",
            external_id="12345.0",
            name="my-job",
            command=["python", "-m", "mod"],
            resources={"cpus_per_task": 2},
        )

        rows = SwarmStateManager.list_jobs("swarm-1")
        assert any(r["job_id"] == job_id for r in rows)

        job = SwarmStateManager.get_job(job_id)
        assert job["deployment_name"] == "swarm-1"
        assert job["external_id"] == "12345.0"
        assert job["status"] == "RUNNING"

        SwarmStateManager.update_job(job_id, status="SUCCEEDED", raw_status="COMPLETED")
        job2 = SwarmStateManager.get_job(job_id)
        assert job2["status"] == "SUCCEEDED"
        assert job2["raw_status"] == "COMPLETED"

    def test_get_job_by_external_id(self, state_manager: SwarmStateManager) -> None:
        state_manager.save(deployment_name="swarm-1")

        job_id = SwarmStateManager.create_job(
            deployment_name="swarm-1",
            provider="slurm",
            kind="step",
            status="RUNNING",
            external_id="12345.0",
            name="my-job",
            command=["python", "-m", "mod"],
            resources={"cpus_per_task": 2},
        )

        rec = SwarmStateManager.get_job_by_external_id("12345.0")
        assert rec["job_id"] == job_id
        assert rec["deployment_name"] == "swarm-1"
        assert rec["external_id"] == "12345.0"

    def test_upgrade_head_creates_jobs_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        db_path = tmp_path / "swarm.db"

        def mock_db_path(cls) -> Path:
            return db_path

        monkeypatch.setattr(SwarmStateManager, "_get_db_path", classmethod(mock_db_path))

        upgrade_head(db_path.as_posix())

        conn = sqlite3.connect(db_path.as_posix())
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()

        assert "swarm" in tables
        assert "jobs" in tables
