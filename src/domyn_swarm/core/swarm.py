# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
import uuid
import warnings

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
)
from ulid import ULID

from domyn_swarm import utils
from domyn_swarm.config.plan import DeploymentContext, DeploymentPlan
from domyn_swarm.config.settings import get_settings
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.job_run import JobRunSpec
from domyn_swarm.deploy.deployment import Deployment
from domyn_swarm.helpers.io import to_path
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.helpers.swarm import generate_swarm_name
from domyn_swarm.platform.protocols import (
    ComputeBackend,
    JobHandle,
    JobProbe,
    JobStatus,
    ServingHandle,
    ServingPhase,
    ServingStatus,
    coerce_job_status,
)

from ..core.state.state_manager import SwarmStateManager

if TYPE_CHECKING:
    from domyn_swarm.jobs import SwarmJob

logger = setup_logger(__name__, level=logging.INFO)


class DomynLLMSwarm(BaseModel):
    """Context manager orchestrating a distributed LLM serving swarm.

    Provides a unified interface for deploying, managing and interacting with a
    large language model serving cluster on a compute backend such as Slurm or
    Lepton. It handles the whole lifecycle from resource allocation to cleanup,
    with state persistence and job submission built in.

    A swarm consists of a load balancer (nginx) distributing requests, several
    vLLM server instances serving the model, a head node coordinating jobs and
    running user scripts, and persisted state enabling recovery and reconnection.

    On Slurm the swarm is deployed as a job array with roles assigned by
    `SLURM_NODEID`: node 0 runs the load balancer and the user driver, nodes 1
    to N run vLLM servers. Cloud platforms are reached through the deployment
    abstractions instead.

    State is persisted automatically - deployment metadata and resource handles,
    configuration, platform identifiers such as job IDs and node assignments, and
    endpoint URLs - which is what allows a swarm to be recovered after a failure
    or reattached from another process via `from_state`.

    Attributes:
        cfg: Deployment parameters, resource requirements and platform-specific
            settings.
        endpoint: Public URL of the deployed load balancer. Set once deployment
            succeeds through the context manager.
        delete_on_exit: Whether to clean up all allocated resources when leaving
            the context manager. Useful for temporary deployments.
        serving_handle: Platform-specific handle for the serving deployment,
            carrying metadata such as job IDs, node assignments and status.
        model: Name or path of the model being served. May be set after
            initialization to switch models.
        preflight: Whether to check the config's filesystem paths before
            deploying. Off for sites where they are visible only from the
            compute nodes.

    Raises:
        RuntimeError: If resource allocation fails; the message carries
            diagnostic information.
        subprocess.CalledProcessError: Propagated from a failed job submission.

    Note:
        The swarm must be used as a context manager for resources to be managed
        correctly. Startup waits up to `cfg.wait_endpoint_s` for the endpoint.
        Paths passed to job submission resolve relative to the execution
        environment, `ENDPOINT` and `MODEL` are set automatically for submitted
        jobs, and checkpoint directories are created as needed. Cleanup failures
        are logged but do not prevent the context from exiting.

    Example:
        Basic deployment, cleaned up on exit:

        ::

            cfg = DomynLLMSwarmConfig.read("config.yaml")
            with DomynLLMSwarm(cfg=cfg) as swarm:
                # Reachable at swarm.endpoint
                swarm.submit_job(
                    my_job,
                    run=JobRunSpec(
                        input_path=Path("data.parquet"),
                        output_path=Path("results.parquet"),
                    ),
                )

        Persistent deployment, reattached later from another process:

        ::

            swarm = DomynLLMSwarm(cfg=cfg, delete_on_exit=False)
            with swarm:
                pass
            # Resources remain allocated
            swarm = DomynLLMSwarm.from_state("my-deployment-abc123")

        Detached job submission:

        ::

            with DomynLLMSwarm(cfg=cfg) as swarm:
                handle = swarm.submit_job(
                    job=LongRunningJob(),
                    run=JobRunSpec(
                        input_path=Path("large_dataset.parquet"),
                        output_path=Path("results.parquet"),
                        detach=True,
                    ),
                )
                print(handle.pid, handle.external_id)

        Running a script on the head node:

        ::

            with DomynLLMSwarm(cfg=cfg) as swarm:
                swarm.submit_script(Path("analysis.py"), extra_args=["--mode", "evaluation"])

    See Also:
        SwarmJob: Base class for jobs executable within the swarm.
        DomynLLMSwarmConfig: Configuration schema and validation.
    """

    cfg: DomynLLMSwarmConfig
    name: str = Field(
        default_factory=lambda data: data.get(
            "name", generate_swarm_name(data["cfg"].name, data["cfg"].backend.type)
        ),
        description="Unique name for this swarm deployment",
    )
    endpoint: str | None = None  # LB endpoint, set after job submission
    delete_on_exit: bool | None = (
        False  # Delete the resources for this cluster at the end of the job
    )
    serving_handle: ServingHandle | None = None  # ServingHandle, set after deployment
    preflight: bool = Field(
        default=True,
        description=(
            "Check that the config's image, model and mount paths exist before "
            "deploying. Turn off where they are visible only from the nodes."
        ),
    )
    swarm_dir: utils.EnvPath = Field(
        description="Directory where swarm-related files are stored",
        default_factory=lambda data: data["cfg"].home_directory / "swarms" / data["name"],
    )
    watchdog_db_path: utils.EnvPath = Field(
        description="Path to the watchdog SQLite database file",
        default_factory=lambda data: data["swarm_dir"] / "watchdog.db",
    )

    _plan_cache: DeploymentPlan | None = PrivateAttr(default=None)
    _deployment_cache: Deployment | None = PrivateAttr(default=None)

    @computed_field
    @property
    def model(self) -> str:
        """
        The model name, either from the config or the job submission.
        If not set, defaults to the config's model.
        """
        return self.cfg.model

    @model.setter
    def model(self, value: str):
        """
        Setter for the model name. This allows setting the model after
        the swarm has been created, e.g., when loading from a state file.
        """
        self.cfg.model = value

    @property
    def platform(self) -> str:
        """The platform this swarm deploys to, e.g. ``"slurm"`` or ``"lepton"``.

        Derived from ``cfg.backend.type``, which is the authoritative value and
        is present in every persisted record.
        """
        assert self.cfg.backend is not None, "Swarm config has no backend"
        return self.cfg.backend.type

    def model_post_init(self, __context: Any) -> None:
        """Wire up in-memory collaborators. Performs no I/O."""
        self._cleaned = False
        self._state_mgr = SwarmStateManager(self)

    def ensure_directories(self) -> None:
        """Create this swarm's on-disk layout. Idempotent.

        Called before deploying (the sbatch templates reference these paths and
        Slurm rejects a submission whose output directory is missing) and before
        writing checkpoints on a swarm reattached via :meth:`from_state`, which
        does not pass through the context manager.
        """
        for directory in (
            self.swarm_dir,
            self.swarm_dir / "serving",
            self.swarm_dir / "jobs",
            self.swarm_dir / "checkpoints",
            self.swarm_dir / "logs" / "endpoint",
            self.swarm_dir / "logs" / "replicas",
            self.swarm_dir / "logs" / "slurm",
        ):
            os.makedirs(directory, exist_ok=True)

    @property
    def _plan(self) -> DeploymentPlan:
        """The deployment plan, built on first access.

        Returns:
            The plan for this swarm's configured backend.

        Raises:
            RuntimeError: If the config carries no backend, so no plan exists.
        """
        if self._plan_cache is None:
            plan = self.cfg.get_deployment_plan()
            if plan is None and self.cfg.backend is not None:
                plan = self.cfg.build_plan()
            if plan is None:
                raise RuntimeError("Swarm config has no backend, so no plan can be built")
            self._plan_cache = plan
        return self._plan_cache

    @property
    def _deployment(self) -> Deployment:
        """The deployment, built on first access from the plan.

        Returns:
            The (serving, compute) pair this swarm drives.
        """
        if self._deployment_cache is None:
            plan = self._plan
            self._deployment_cache = Deployment(
                serving=plan.serving,
                compute=plan.compute,  # type: ignore[arg-type]
                extras=plan.extras | {"swarm_directory": str(self.swarm_dir)},
            )
        return self._deployment_cache

    def __enter__(self):
        # Before anything is created on disk. Skipped when re-attaching: a
        # deployed swarm's resources exist whatever its images look like now.
        if self.serving_handle is None and self.preflight:
            from domyn_swarm.config.preflight import raise_on_path_problems

            raise_on_path_problems(self.cfg)

        self.ensure_directories()

        # If instantiating from state, the swarm will be already deployed,
        # so just return self
        if self.serving_handle is not None:
            return self

        assert self._deployment is not None

        serving_spec = dict(self._plan.serving_spec, swarm_directory=self.swarm_dir)
        ctx = DeploymentContext(
            serving_spec=serving_spec,
            job_resources=self._plan.job_resources,
            extras=self._plan.extras | {"swarm_directory": str(self.swarm_dir)},
            timeout_s=self._plan.timeout_s or self.cfg.wait_endpoint_s,
            shared_env=self._plan.shared_env,
            image=self._plan.image,
        )

        logger.info(f"Creating deployment [cyan]{self.name}[/cyan] on {self.platform}...")

        handle = self._deployment.up(self.name, ctx)

        # Record the handle and persist it *before* waiting for readiness. From
        # here on the platform holds real resources, so both this object and the
        # state DB must know about them -- otherwise a failure below would strand
        # the allocation with no way for `domyn-swarm down` to find it.
        self.serving_handle = handle
        self._persist(self.name)

        try:
            handle = self._deployment.wait_ready(
                timeout_s=ctx.timeout_s or self.cfg.wait_endpoint_s
            )

            # Update the handle and deployment with the ready state
            self.serving_handle = handle
            self.endpoint = handle.url
            self._deployment.compute = self._make_compute_backend(handle)

            # Persist again now that the endpoint is known
            self._persist(self.name)
        except Exception:
            # KeyboardInterrupt is deliberately not caught: interrupting `up` is a
            # user decision, and the CLI asks whether to cancel or keep waiting.
            self._cleanup_failed_startup()
            raise

        return self

    def _cleanup_failed_startup(self) -> None:
        """Release resources allocated by a startup that did not complete.

        Never raises: a teardown failure must not mask the error that caused the
        startup to fail. When teardown does fail the state record is deliberately
        left in place so the deployment stays reachable via ``domyn-swarm down``.
        """
        logger.warning(f"Deployment [cyan]{self.name}[/cyan] failed to start; releasing resources")
        try:
            self.cleanup()
        except Exception as exc:
            logger.error(
                f"Could not release resources for [cyan]{self.name}[/cyan]: {exc}. "
                f"The deployment is still recorded -- run `domyn-swarm down {self.name}` to retry."
            )

    def __exit__(self, exc_type, exc, tb):
        if self.delete_on_exit:
            self.cleanup()

    def _persist(self, deployment_name: str):
        """Save the state.

        Args:
            deployment_name (str): Deployment name.
        """
        self.cfg.persist(self.swarm_dir / "config.yaml")
        self._state_mgr.save(deployment_name)

    @classmethod
    def from_state(cls, deployment_name: str) -> DomynLLMSwarm:
        """Initialize a swarm from a saved state.

        Args:
            deployment_name (str): Deployment name.

        Returns:
            DomynLLMSwarm: Loaded swarm.
        """
        return SwarmStateManager.load(deployment_name)

    @classmethod
    def from_record(
        cls,
        swarm_payload: dict,
        cfg_payload: dict,
        handle_payload: dict,
    ) -> DomynLLMSwarm:
        """Build a complete swarm from persisted state.

        The compute backend is not built here: a Slurm swarm whose endpoint
        never became ready has no load-balancer node, and teardown and status
        must still work for it. :meth:`_require_compute_backend` builds the
        backend on first use.

        Args:
            swarm_payload: The persisted ``swarm`` column, without ``cfg``.
            cfg_payload: The persisted ``cfg`` column.
            handle_payload: The persisted ``serving_handle`` column.

        Returns:
            A swarm with its handle adopted.

        Raises:
            ValueError: If the record's serving handle is missing metadata the
                platform needs; see
                :meth:`~domyn_swarm.config.plan.DeploymentPlan.validate_serving_handle`.
        """
        swarm = cls.model_validate({**swarm_payload, "cfg": cfg_payload})
        handle = ServingHandle(**handle_payload)

        assert swarm._plan is not None, "Swarm was built without a deployment plan"
        swarm._plan.validate_serving_handle(handle)

        swarm.serving_handle = handle
        swarm._deployment.adopt(handle)
        return swarm

    def _delete_record(self) -> None:
        """Delete swarm from the DB

        Args:
            deployment_name (str): Deployment name.
        """
        self._state_mgr.delete_record(self.name)

    def _record_job_submission(
        self,
        *,
        name: str,
        command: list[str],
        resources: dict | None,
        kind: str,
        status: JobStatus,
        external_id: str | None = None,
        error: str | None = None,
    ) -> str | None:
        """Persist job metadata in the local swarm DB.

        Args:
            name: Job name.
            command: Command argv list (no secrets).
            resources: Resource dict (if any).
            kind: Job kind (e.g., "step", "script").
            status: Job status.
            external_id: Optional backend external identifier.
            error: Optional error message.

        Returns:
            Job record ID if created, else None.
        """
        try:
            job_id = SwarmStateManager.create_job(
                deployment_name=self.name,
                provider=self.platform,
                kind=kind,
                status=status,
                external_id=external_id,
                name=name,
                command=command,
                resources=resources,
                error=error,
            )
            return job_id
        except Exception as exc:
            logger.warning("Failed to persist job record: %s", exc)
            return None

    def _update_job_submission(
        self,
        *,
        job_id: str | None,
        status: JobStatus,
        external_id: str | None = None,
        log_paths: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update job submission state in the local swarm DB.

        Args:
            job_id: Internal persisted job ID.
            status: Job status.
            external_id: Optional backend external identifier.
            log_paths: Optional backend log path dictionary.
            error: Optional error message.
        """
        if not job_id:
            return
        try:
            SwarmStateManager.update_job(
                job_id,
                status=status,
                external_id=external_id,
                log_paths=log_paths,
                error=error,
            )
        except Exception as exc:
            logger.warning("Failed to update job record %s: %s", job_id, exc)

    def _submit_with_tracking(
        self,
        *,
        name: str,
        command: list[str],
        resources: dict | None,
        kind: str,
        submit: Callable[[], object | None],
    ) -> JobHandle:
        """Submit a compute job and persist lifecycle transitions.

        Args:
            name: Job name.
            command: Command argv list to persist.
            resources: Optional resource requirements.
            kind: Job kind (for persistence).
            submit: Backend submission callable.

        Returns:
            Submitted job handle.

        Raises:
            RuntimeError: If backend submission returns ``None``.
            Exception: Re-raises backend submission errors.
        """
        job_record_id = self._record_job_submission(
            name=name,
            command=command,
            resources=resources,
            kind=kind,
            status=JobStatus.PENDING,
        )
        try:
            raw_handle = submit()
        except Exception as exc:
            self._update_job_submission(
                job_id=job_record_id,
                status=JobStatus.FAILED,
                error=str(exc),
            )
            raise
        if raw_handle is None:
            error_msg = "Failed to submit job to compute backend."
            self._update_job_submission(
                job_id=job_record_id,
                status=JobStatus.FAILED,
                error=error_msg,
            )
            raise RuntimeError(error_msg)
        if isinstance(raw_handle, JobHandle):
            job_handle = raw_handle
        else:
            raw_meta = getattr(raw_handle, "meta", None)
            job_handle = JobHandle(
                id=str(getattr(raw_handle, "id", name)),
                status=coerce_job_status(getattr(raw_handle, "status", JobStatus.PENDING)),
                meta=dict(raw_meta) if isinstance(raw_meta, dict) else {},
            )

        normalized_status = coerce_job_status(job_handle.status)
        job_handle.status = normalized_status
        self._update_job_submission(
            job_id=job_record_id,
            status=normalized_status,
            external_id=job_handle.external_id,
            log_paths=job_handle.meta.get("log_paths"),
        )
        if job_record_id:
            job_handle.meta["job_id"] = job_record_id
        return job_handle

    def submit_job(
        self,
        job: SwarmJob,
        *,
        run: JobRunSpec,
    ) -> JobHandle:
        """Launch a serialized :class:`~domyn_swarm.SwarmJob` inside the current
        SLURM swarm allocation.

        The *job* object is converted to keyword arguments via
        :py:meth:`SwarmJob.to_kwargs`, transmitted to the head node
        (where ``SLURM_NODEID == 0``), reconstructed by
        ``domyn_swarm.jobs.cli.run``, and executed under ``srun``.

        Args:
            job: The job instance to execute.
            run: How the job should run: input/output paths, sharding,
                checkpointing, engine selection, and resources. See
                :class:`~domyn_swarm.core.job_run.JobRunSpec` for the full
                field list and their semantics.

        Returns:
            JobHandle: Compute job handle with normalized status and metadata.

        Raises:
            RuntimeError: The swarm is not ready (`self.serving_handle` or
                `self.endpoint` is ``None``).
            FileNotFoundError: `run.input_path` does not exist.
            subprocess.CalledProcessError: Propagated when the synchronous
                ``srun`` command exits with a non-zero status code.

        Note:
            The constructed command is logged with *rich* for transparency, e.g.::

                srun --jobid=<...> --nodelist=<...> --ntasks=1 --overlap ...
                    python -m domyn_swarm.jobs.cli.run --job-class=<module:Class> ...

        Example:
            ::

                swarm.submit_job(
                    my_job,
                    run=JobRunSpec(
                        input_path=Path("batch.parquet"),
                        output_path=Path("predictions.parquet"),
                        num_shards=4,
                    ),
                )
        """
        num_shards = run.num_shards
        if run.num_threads is not None:
            warnings.warn(
                "`num_threads` is deprecated and will be removed in a future release; "
                "use `num_shards`. The value has always been a shard count, not a "
                "thread count, and it is part of the checkpoint layout.",
                DeprecationWarning,
                stacklevel=2,
            )
            num_shards = run.num_threads

        # A swarm reattached via `from_state` never passed through the context
        # manager, so its layout may not exist yet.
        self.ensure_directories()

        checkpoint_dir = (
            self.swarm_dir / "checkpoints"
            if run.checkpoint_dir is None
            else to_path(run.checkpoint_dir)
        )

        input_parquet = to_path(run.input_path)
        output_parquet = to_path(run.output_path)

        from domyn_swarm.jobs import JobBuilder

        job_class = JobBuilder.to_class_path(job)
        job_kwargs = JobBuilder.to_kwargs_json(job)

        python_interpreter, image, resources, env = self._compose_runtime()
        resources = self._merge_resources(resources, None, run.job_resources)
        env = self._augment_job_env(env, job, job_class, ray_address=run.ray_address)

        exe = self._build_job_command(
            job=job,
            job_class=job_class,
            job_kwargs=job_kwargs,
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            num_shards=num_shards,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=run.checkpoint_interval,
            runner=run.runner,
            engine=run.engine,
            no_resume=run.no_resume,
            no_checkpointing=run.no_checkpointing,
            checkpoint_tag=run.checkpoint_tag,
            shard_output=run.shard_output,
            shard_mode=run.shard_mode,
            global_resume=run.global_resume,
            limit=run.limit,
            ray_address=env.get("DOMYN_SWARM_RAY_ADDRESS"),
            python_interpreter=str(python_interpreter),
        )

        job_name = job.name.lower() if job.name else f"{self.name}-job"
        ulid = str(ULID())
        job_name = f"{self.name}-{job_name}-{ulid.lower()}"

        logger.info(
            f"Submitting {job.__class__.__name__} [cyan]{job_name}[/cyan] job "
            f"to swarm {self.name} on {self.platform}"
        )

        return self._submit_with_tracking(
            name=job_name[:36],  # type: ignore[arg-type]
            command=[*map(str, exe)],
            resources=resources,
            kind="step",
            submit=lambda: self._deployment.run(
                name=job_name[:36],  # type: ignore[arg-type]
                image=image,
                command=exe,
                env=env,
                resources=resources,
                detach=run.detach,
            ),
        )

    def _augment_job_env(
        self,
        env: dict,
        job: SwarmJob,
        job_class: str,
        *,
        ray_address: str | None,
    ) -> dict:
        """Augment the base environment for job submission.

        Args:
            env: Base environment variables.
            job: Job instance being submitted.
            job_class: Fully qualified job class name.
            ray_address: Optional Ray address override.

        Returns:
            Environment dictionary for the job.
        """
        env = dict(env)
        env.update(
            {
                "ENDPOINT": self.endpoint,
                "MODEL": self.model,
                "JOB_CLASS": job_class,
            }
        )
        token = get_settings().resolved_api_token
        if token:
            env["DOMYN_SWARM_API_TOKEN"] = token.get_secret_value()
            env["VLLM_API_KEY"] = token.get_secret_value()

        if getattr(job, "data_backend", None) == "ray":
            resolved_ray_address = (
                ray_address
                or env.get("DOMYN_SWARM_RAY_ADDRESS")
                or env.get("RAY_ADDRESS")
                or os.environ.get("DOMYN_SWARM_RAY_ADDRESS")
                or os.environ.get("RAY_ADDRESS")
            )
            if not resolved_ray_address:
                raise ValueError(
                    "Ray backend requires an explicit ray address. Provide --ray-address or set "
                    "DOMYN_SWARM_RAY_ADDRESS/RAY_ADDRESS in the swarm environment."
                )
            env["DOMYN_SWARM_RAY_ADDRESS"] = resolved_ray_address
        return env

    def _build_job_command(
        self,
        *,
        job: SwarmJob,
        job_class: str,
        job_kwargs: str,
        input_parquet: Path,
        output_parquet: Path,
        num_shards: int,
        checkpoint_dir: str | Path,
        checkpoint_interval: int | None,
        runner: str,
        engine: str | None,
        no_resume: bool,
        no_checkpointing: bool,
        checkpoint_tag: str | None,
        shard_output: bool,
        shard_mode: str,
        global_resume: bool,
        limit: int | None,
        ray_address: str | None,
        python_interpreter: str,
    ) -> list[str]:
        """Build the job runner command to execute inside the swarm.

        Args:
            job: Job instance being submitted.
            job_class: Fully qualified job class name.
            job_kwargs: Serialized job kwargs JSON.
            input_parquet: Input dataset path.
            output_parquet: Output dataset path.
            num_shards: Number of shards to split the input into.
            checkpoint_dir: Checkpoint directory.
            checkpoint_interval: Checkpoint interval override.
            runner: Deprecated runner implementation name; consulted only when
                `engine` is not given.
            engine: Execution engine to use, emitted as `--engine=...` when given.
            no_resume: Whether to ignore existing checkpoints.
            no_checkpointing: Whether to disable checkpointing.
            checkpoint_tag: Optional checkpoint tag override.
            shard_output: Whether to write shard outputs to a directory.
            shard_mode: Sharding strategy.
            global_resume: Whether to enable global resume.
            limit: Optional input row limit.
            ray_address: Optional Ray address override.
            python_interpreter: Python interpreter path to use.

        Returns:
            Command list for job execution.
        """
        exe = [
            python_interpreter,
            "-m",
            "domyn_swarm.jobs.cli.run",
            f"--job-class={job_class}",
            f"--model={self.model}",
            f"--input-parquet={input_parquet}",
            f"--output-parquet={output_parquet}",
            f"--endpoint={self.endpoint}",
            f"--num-shards={num_shards}",
            f"--checkpoint-dir={checkpoint_dir}",
            f"--checkpoint-interval={checkpoint_interval or job.checkpoint_interval}",
            f"--runner={runner}",
            "--job-kwargs",
            job_kwargs,
        ]

        if no_resume:
            exe.append("--no-resume")
        if no_checkpointing:
            exe.append("--no-checkpointing")
        if checkpoint_tag:
            exe.append(f"--checkpoint-tag={checkpoint_tag}")
        if engine is not None:
            exe.append(f"--engine={engine}")
        if shard_output:
            exe.append("--shard-output")
        if shard_mode:
            exe.append(f"--shard-mode={shard_mode}")
        if global_resume:
            exe.append("--global-resume")
        if limit:
            exe.append(f"--limit={limit}")
        if ray_address:
            exe.append(f"--ray-address={ray_address}")
        return exe

    def _compose_runtime(self, extra_env: dict | None = None):
        """Build interpreter/image/resources/env in one place."""
        self._deployment.ensure_ready()
        compute = self._require_compute_backend()
        python_interpreter = compute.default_python(self.cfg)
        image = (
            self._plan.image
            if self._plan and self._plan.image
            else compute.default_image(self.cfg.backend)
        )
        resources = self._merge_resources(
            compute.default_resources(self.cfg.backend),
            self._plan.job_resources if self._plan else None,
            None,
        )

        env = {
            "ENDPOINT": self.endpoint,
            "MODEL": self.model,
        }
        if self._plan and self._plan.shared_env:
            env.update(self._plan.shared_env)
        # Global backend env from config
        if getattr(self.cfg, "backend", None) and getattr(self.cfg.backend, "env", None):
            env.update(self.cfg.backend.env)  # type: ignore[attr-defined]
        # Backend-specific overrides
        overrides = compute.default_env(self.cfg)
        if overrides:
            env.update(overrides)
        # Call-site extras
        if extra_env:
            env.update(extra_env)

        return str(python_interpreter), image, resources, env

    @staticmethod
    def _merge_resources(
        base: dict | None,
        plan_resources: dict | None,
        overrides: dict | None,
    ) -> dict | None:
        if not base and not plan_resources and not overrides:
            return None
        merged: dict = {}
        if base:
            merged.update(base)
        if plan_resources:
            merged.update(plan_resources)
        if overrides:
            merged.update(overrides)
        return merged

    def submit_script(
        self,
        script_path: Path,
        detach: bool = False,
        extra_args: list[str] | None = None,
        job_resources: dict | None = None,
    ) -> JobHandle:
        """Submit a Python script to the compute backend for execution.

        This method validates the script path, composes the runtime environment,
        and submits the script for execution via the configured deployment backend.

        Args:
            script_path (Path): Path to the Python script to be executed.
            detach (bool, optional): If True, run the script in detached mode.
                If False, run synchronously. Defaults to False.
            extra_args (list[str] | None, optional): Additional command-line arguments
                to pass to the script. Defaults to None.
            job_resources: Scheduler resource overrides for the job step. They
                take precedence over the compute backend and plan defaults.

        Returns:
            JobHandle: Submitted job handle with normalized status and metadata.

        Raises:
            FileNotFoundError: If the script file does not exist (only checked for SLURM platform).
            RuntimeError: If the script submission to the compute backend fails.

        Example:
            >>> swarm = Swarm(...)
            >>> # Submit script synchronously
            >>> swarm.submit_script(Path("my_script.py"))

            >>> # Submit script in detached mode with arguments
            >>> handle = swarm.submit_script(
            ...     Path("my_script.py"), detach=True, extra_args=["--config", "config.yaml"]
            ... )
            >>> handle.pid
        """

        # Basic validation
        if self.platform == "slurm" and not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Compose runtime (interpreter/image/resources/env) once
        python_interpreter, image, resources, env = self._compose_runtime()
        resources = self._merge_resources(resources, None, job_resources)

        # Build the command
        args = extra_args or []
        command = [python_interpreter, str(script_path), *args]

        script_name = f"{self.name.lower()}-script"
        return self._submit_with_tracking(
            name=script_name,
            command=[*map(str, command)],
            resources=resources,
            kind="script",
            submit=lambda: self._deployment.run(
                name=script_name,  # type: ignore[arg-type]
                image=image,
                command=command,
                env=env,
                resources=resources,
                detach=detach,
            ),
        )

    def _require_compute_backend(self) -> ComputeBackend:
        """Return the compute backend, building it from the serving handle on first use.

        Raises:
            RuntimeError: If the swarm has no serving handle, or the platform
                cannot build a compute backend from it because the endpoint is
                not ready.
        """
        if self._deployment.compute is None and self.serving_handle is not None:
            self._deployment.compute = self._make_compute_backend(self.serving_handle)
        compute = self._deployment.compute
        if compute is None:
            raise RuntimeError("Compute backend is not initialized for this swarm.")
        return compute

    def wait_job(self, handle: JobHandle, *, stream_logs: bool = True) -> JobStatus:
        """Wait for a submitted compute job to reach a terminal state.

        Args:
            handle: Job handle to wait on.
            stream_logs: Whether to stream backend logs while waiting.

        Returns:
            Normalized terminal job status.
        """
        compute = self._require_compute_backend()
        status = compute.wait(handle, stream_logs=stream_logs)
        normalized_status = coerce_job_status(status)
        handle.status = normalized_status
        return normalized_status

    def cancel_job(self, handle: JobHandle) -> JobStatus:
        """Cancel a submitted compute job.

        Args:
            handle: Job handle to cancel.

        Returns:
            Final normalized status after cancellation.
        """
        compute = self._require_compute_backend()
        compute.cancel(handle)
        normalized_status = coerce_job_status(
            getattr(handle, "status", JobStatus.CANCELLED),
            default=JobStatus.CANCELLED,
        )
        if normalized_status != JobStatus.CANCELLED:
            normalized_status = JobStatus.CANCELLED
        handle.status = normalized_status
        return normalized_status

    def refresh_job_status(self, job_id: str) -> dict[str, Any]:
        """Refresh a persisted job status via backend probe (best effort).

        Args:
            job_id: Internal persisted job identifier.

        Returns:
            Updated job record payload, including transient refresh metadata:
            ``refresh_source`` and ``refresh_error``.
        """
        record = SwarmStateManager.get_job(job_id)
        external_id = record.get("external_id")
        status = coerce_job_status(record.get("status", JobStatus.PENDING))
        handle = JobHandle(
            id=str(external_id or job_id),
            status=status,
            meta={
                "job_id": job_id,
                "external_id": external_id,
            },
        )

        if not external_id:
            record["refresh_source"] = "db"
            record["refresh_error"] = "Missing external_id; backend probe skipped."
            return record

        compute = self._require_compute_backend()
        probe: JobProbe
        try:
            probe = compute.probe(handle)
        except Exception as exc:
            refresh_error = str(exc)
            logger.warning("Job status probe failed for %s: %s", job_id, refresh_error)
            with contextlib.suppress(Exception):
                SwarmStateManager.update_job(job_id, error=refresh_error)
            refreshed = SwarmStateManager.get_job(job_id)
            refreshed["refresh_source"] = "backend"
            refreshed["refresh_error"] = refresh_error
            return refreshed

        try:
            SwarmStateManager.update_job(
                job_id,
                status=probe.status,
                raw_status=probe.raw_status,
                external_id=handle.external_id or external_id,
                error=probe.error,
            )
        except Exception as exc:
            logger.warning("Failed to persist probed status for %s: %s", job_id, exc)

        refreshed = SwarmStateManager.get_job(job_id)
        refreshed["refresh_source"] = probe.source
        refreshed["refresh_error"] = probe.error
        return refreshed

    def cleanup(self):
        """Release the swarm's resources and drop its state record.

        Idempotent: calling it again after a successful teardown is a no-op, so
        overlapping paths (a failed startup, ``__exit__``, an explicit ``down()``)
        cannot issue a second cancel against the platform.

        The record is dropped only once teardown succeeds -- if ``down`` raises,
        the record survives so the deployment stays reachable for a retry.
        """
        if self._cleaned:
            return
        # Handle first: a swarm that never deployed needs no deployment built
        # just to decide there is nothing to tear down.
        if self.serving_handle and self._deployment:
            self._deployment.down(self.serving_handle)
        self._delete_record()
        self._cleaned = True

    def down(self):
        """Manually clean up the swarm resources."""
        self.cleanup()

    def _deployment_name(self) -> str:
        unique_id = uuid.uuid4()
        short_id = str(unique_id)[:8]
        return f"{self.cfg.name}-{short_id}"

    def _make_compute_backend(self, handle: ServingHandle) -> ComputeBackend:
        """Build the compute backend for a ready serving handle."""
        assert self._plan is not None
        return self._plan.make_compute_backend(handle)

    def status(self) -> ServingStatus:
        """Get the current status of the swarm.

        Safe to call at any point in the lifecycle: a swarm that has not been
        deployed reports :attr:`ServingPhase.UNKNOWN` rather than raising.

        Returns:
            The backend-reported serving status, falling back to the endpoint
            this swarm already knows about when the backend reports none.
        """
        if self._deployment is None:
            return ServingStatus(phase=ServingPhase.UNKNOWN, url=self.endpoint)
        status = self._deployment.status()
        if status.url is None:
            status.url = self.endpoint
        return status

    def serving_status(self) -> ServingStatus:
        """Ask the serving backend for this swarm's live status.

        Unlike :meth:`status`, which reports what this object already knows,
        this queries the backend directly. Safe at any point in the lifecycle:
        a swarm that was never deployed reports
        :attr:`~domyn_swarm.platform.protocols.ServingPhase.UNKNOWN`.

        Returns:
            The backend-reported serving status.
        """
        # Handle first, so a never-deployed swarm answers without building a
        # deployment (and importing its serving backend) to do it.
        if self.serving_handle is None or self._deployment is None:
            return ServingStatus(phase=ServingPhase.UNKNOWN, url=self.endpoint)
        return self._deployment.serving.status(self.serving_handle)
