# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ulid import ULID

from domyn_swarm.config.settings import get_settings
from domyn_swarm.exceptions import JobNotFoundError
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.platform.protocols import JobStatus

if TYPE_CHECKING:
    from ..swarm import DomynLLMSwarm

from .db import make_session_factory
from .models import JobRecord, SwarmRecord

logger = setup_logger(__name__)


def _escape_like(s: str) -> str:
    # Escape %, _ and \ for SQLite LIKE ... ESCAPE '\'
    return s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def _job_status_value(status: JobStatus | str) -> str:
    """Normalize persisted job status to its string value.

    Args:
        status: A ``JobStatus`` enum instance or raw status string.

    Returns:
        String status value suitable for DB persistence.
    """
    return status.value if isinstance(status, JobStatus) else str(status)


class SwarmStateManager:
    DB_NAME = "swarm.db"

    def __init__(self, swarm: "DomynLLMSwarm"):
        self.swarm = swarm

    # --- path helpers ---
    @classmethod
    def _resolve_db_path(cls) -> Path:
        """Compute the DB path without triggering schema upgrade."""
        return get_settings().home / cls.DB_NAME

    @classmethod
    def _get_db_path(cls) -> Path:
        from .autoupgrade import ensure_db_up_to_date

        ensure_db_up_to_date(noisy=False)
        return cls._resolve_db_path()

    # --- CRUD ---
    def save(self, deployment_name: str) -> None:
        session_factory = make_session_factory(self._get_db_path())
        # Build payload
        cfg_obj = self.swarm.cfg
        swarm_obj = self.swarm.model_dump(
            mode="json", exclude={"cfg": True}, by_alias=True, exclude_none=True
        )
        handle = self.swarm.serving_handle
        if handle is None:
            raise ValueError("Null Serving handle")

        handle_dict = dataclasses.asdict(handle)

        rec = SwarmRecord(
            deployment_name=deployment_name,
            swarm=swarm_obj,
            cfg=cfg_obj.model_dump(mode="json", by_alias=True, exclude_none=True),
            serving_handle=handle_dict,
        )
        with session_factory() as s:
            # UPSERT semantics: merge by PK
            s.merge(rec)
            s.commit()
        logger.debug(f"State saved for swarm {deployment_name}.")

    @classmethod
    def load(cls, deployment_name: str):
        """Rehydrate a swarm from its persisted record.

        Args:
            deployment_name: The deployment name to look up.

        Returns:
            A fully-built :class:`DomynLLMSwarm`.

        Raises:
            JobNotFoundError: If no record exists for that name.
        """
        from ... import DomynLLMSwarm  # avoid import cycles

        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rec = s.get(SwarmRecord, deployment_name)
        if rec is None:
            raise JobNotFoundError(deployment_name)

        return DomynLLMSwarm.from_record(rec.swarm, rec.cfg, rec.serving_handle)

    def delete_record(self, deployment_name: str) -> None:
        session_factory = make_session_factory(self._get_db_path())
        with session_factory() as s:
            rec = s.get(SwarmRecord, deployment_name)
            if rec:
                s.delete(rec)
                s.commit()

    @classmethod
    def delete_records(cls, deployment_names: Iterable[str]) -> int:
        """Delete multiple swarm records by deployment name.

        Args:
            deployment_names (Iterable[str]): Names of deployments to delete.

        Returns:
            int: Number of records deleted.
        """
        names = [name for name in deployment_names if name]
        if not names:
            return 0
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            deleted = (
                s.query(SwarmRecord)
                .filter(SwarmRecord.deployment_name.in_(names))
                .delete(synchronize_session=False)
            )
            s.commit()
        return int(deleted or 0)

    @classmethod
    def list_all(cls) -> list[dict[str, Any]]:
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rows = s.query(SwarmRecord).order_by(SwarmRecord.creation_dt.desc()).all()
        return [
            {
                "deployment_name": r.deployment_name,
                "swarm": r.swarm,
                "cfg": r.cfg,
                "serving_handle": r.serving_handle,
                "creation_dt": r.creation_dt.isoformat(),
                # Optional: convenience fan-out of commonly used fields:
                "endpoint": (r.swarm or {}).get("endpoint", ""),
                "platform": (r.cfg or {}).get("backend", {}).get("type", ""),
                "name": (r.swarm or {}).get("name", r.deployment_name),
            }
            for r in rows
        ]

    @classmethod
    def get_creation_dt(cls, deployment_name: str) -> datetime | None:
        """Return the persisted creation timestamp for a deployment, if known."""
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rec = s.get(SwarmRecord, deployment_name)
        return rec.creation_dt if rec is not None else None

    @classmethod
    def get_last_swarm_name(cls) -> str | None:
        """Get the last deployment name.

        Returns:
            str | None: Deployment name or None if no deployments exist.
        """
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rec = s.query(SwarmRecord).order_by(SwarmRecord.creation_dt.desc()).first()
        if rec is None:
            return None
        return rec.deployment_name

    @classmethod
    def list_by_base_name(cls, base_name: str) -> list[str]:
        pattern = f"{_escape_like(base_name)}-%"
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rows = (
                s.query(SwarmRecord)
                .filter(SwarmRecord.deployment_name.like(pattern, escape="\\"))
                .order_by(SwarmRecord.creation_dt.desc())
                .all()
            )
        return [r.deployment_name for r in rows]

    # --- Jobs table (submitted compute jobs) ---
    @classmethod
    def create_job(
        cls,
        *,
        deployment_name: str,
        provider: str,
        kind: str,
        status: JobStatus | str,
        raw_status: str | None = None,
        external_id: str | None = None,
        name: str | None = None,
        command: list[str] | None = None,
        resources: dict | None = None,
        log_paths: dict | None = None,
        error: str | None = None,
    ) -> str:
        """Create a new job record in the local state DB.

        Args:
            deployment_name: Swarm deployment name (FK to `swarm.deployment_name`).
            provider: Backend provider name (e.g., "slurm").
            kind: Job kind (e.g., "step", "sbatch").
            status: Normalized status value.
            raw_status: Optional backend-specific status.
            external_id: Optional backend-specific identifier (e.g. Slurm job/step id).
            name: Optional user-friendly name.
            command: Optional argv list to persist (must not include secrets).
            resources: Optional resources dict to persist.
            log_paths: Optional log paths dict to persist.
            error: Optional error message.

        Returns:
            The generated job ID.
        """
        job_id = str(ULID()).lower()
        session_factory = make_session_factory(cls._get_db_path())
        rec = JobRecord(
            job_id=job_id,
            deployment_name=deployment_name,
            provider=provider,
            kind=kind,
            status=_job_status_value(status),
            raw_status=raw_status,
            external_id=external_id,
            name=name,
            command=command,
            resources=resources,
            log_paths=log_paths,
            error=error,
        )
        with session_factory() as s:
            s.add(rec)
            s.commit()
        return job_id

    @classmethod
    def update_job(
        cls,
        job_id: str,
        *,
        status: JobStatus | str | None = None,
        raw_status: str | None = None,
        external_id: str | None = None,
        name: str | None = None,
        log_paths: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update fields for an existing job record.

        Args:
            job_id: Internal job ID.
            status: Optional normalized status value.
            raw_status: Optional backend-specific status string.
            external_id: Optional backend-specific identifier.
            name: Optional job name.
            log_paths: Optional log paths dict.
            error: Optional error string.

        Raises:
            ValueError: If the job ID does not exist.
        """
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rec = s.get(JobRecord, job_id)
            if rec is None:
                raise ValueError(f"Job not found: {job_id}")
            if status is not None:
                rec.status = _job_status_value(status)
            if raw_status is not None:
                rec.raw_status = raw_status
            if external_id is not None:
                rec.external_id = external_id
            if name is not None:
                rec.name = name
            if log_paths is not None:
                rec.log_paths = log_paths
            if error is not None:
                rec.error = error

            rec.update_dt = datetime.now(UTC).replace(tzinfo=None)
            s.commit()

    @classmethod
    def get_job(cls, job_id: str) -> dict[str, Any]:
        """Fetch a single job record by its internal job ID.

        Args:
            job_id: Internal job ID.

        Returns:
            A JSON-serializable dict.

        Raises:
            ValueError: If the job ID does not exist.
        """
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            rec = s.get(JobRecord, job_id)
            if rec is None:
                raise ValueError(f"Job not found: {job_id}")
            return {
                "job_id": rec.job_id,
                "deployment_name": rec.deployment_name,
                "provider": rec.provider,
                "kind": rec.kind,
                "status": rec.status,
                "raw_status": rec.raw_status,
                "external_id": rec.external_id,
                "name": rec.name,
                "command": rec.command,
                "resources": rec.resources,
                "log_paths": rec.log_paths,
                "error": rec.error,
                "creation_dt": rec.creation_dt.isoformat() if rec.creation_dt else None,
                "update_dt": rec.update_dt.isoformat() if rec.update_dt else None,
            }

    @classmethod
    def get_job_by_external_id(
        cls, external_id: str, *, deployment_name: str | None = None
    ) -> dict[str, Any]:
        """Fetch a job record by provider external ID.

        Args:
            external_id: Provider external ID (e.g. Slurm step id).
            deployment_name: Optional swarm deployment name to disambiguate.

        Returns:
            A JSON-serializable dict for the matched job record.

        Raises:
            ValueError: If no match is found or match is ambiguous.
        """
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            q = s.query(JobRecord).filter(JobRecord.external_id == external_id)
            if deployment_name:
                q = q.filter(JobRecord.deployment_name == deployment_name)

            rows = q.order_by(JobRecord.update_dt.desc(), JobRecord.creation_dt.desc()).all()
            if not rows:
                scope = f" in deployment '{deployment_name}'" if deployment_name else ""
                raise ValueError(f"Job not found for external_id '{external_id}'{scope}.")
            if len(rows) > 1 and deployment_name is None:
                raise ValueError(
                    "Multiple jobs matched external_id "
                    f"'{external_id}'. Provide --name/--deployment-name to disambiguate."
                )

            rec = rows[0]
            return {
                "job_id": rec.job_id,
                "deployment_name": rec.deployment_name,
                "provider": rec.provider,
                "kind": rec.kind,
                "status": rec.status,
                "raw_status": rec.raw_status,
                "external_id": rec.external_id,
                "name": rec.name,
                "command": rec.command,
                "resources": rec.resources,
                "log_paths": rec.log_paths,
                "error": rec.error,
                "creation_dt": rec.creation_dt.isoformat() if rec.creation_dt else None,
                "update_dt": rec.update_dt.isoformat() if rec.update_dt else None,
            }

    @classmethod
    def list_jobs(
        cls,
        deployment_name: str,
        *,
        limit: int = 50,
        statuses: list[JobStatus | str] | None = None,
    ) -> list[dict[str, Any]]:
        """List jobs for a deployment, newest first.

        Args:
            deployment_name: Swarm deployment name (FK).
            limit: Maximum number of rows to return.
            statuses: Optional list of statuses to filter on.

        Returns:
            List of JSON-serializable dicts.
        """
        session_factory = make_session_factory(cls._get_db_path())
        with session_factory() as s:
            q = (
                s.query(JobRecord)
                .filter(JobRecord.deployment_name == deployment_name)
                .order_by(JobRecord.creation_dt.desc())
            )
            if statuses:
                q = q.filter(JobRecord.status.in_([_job_status_value(st) for st in statuses]))
            rows = q.limit(limit).all()
        return [
            {
                "job_id": r.job_id,
                "deployment_name": r.deployment_name,
                "provider": r.provider,
                "kind": r.kind,
                "status": r.status,
                "raw_status": r.raw_status,
                "external_id": r.external_id,
                "name": r.name,
                "command": r.command,
                "resources": r.resources,
                "log_paths": r.log_paths,
                "error": r.error,
                "creation_dt": r.creation_dt.isoformat() if r.creation_dt else None,
                "update_dt": r.update_dt.isoformat() if r.update_dt else None,
            }
            for r in rows
        ]
