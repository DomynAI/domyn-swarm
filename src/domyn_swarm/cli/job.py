# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from typing import Literal

from rich.console import Console
import typer

from domyn_swarm.cli import job_helpers as helpers
from domyn_swarm.cli.errors import exit_on_config_path_error
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.utils.env_path import EnvPath


class _LazyDomynLLMSwarm:
    """Proxy that imports the swarm implementation only for job execution."""

    def __call__(self, *args, **kwargs):
        from domyn_swarm.core.swarm import DomynLLMSwarm

        return DomynLLMSwarm(*args, **kwargs)

    def from_state(self, *args, **kwargs):
        """Load a swarm from persisted state."""
        from domyn_swarm.core.swarm import DomynLLMSwarm

        return DomynLLMSwarm.from_state(*args, **kwargs)


class _LazyLogger:
    """Logger proxy that avoids importing Rich logging during CLI discovery."""

    def __init__(self) -> None:
        self._logger: logging.Logger | None = None

    def _get(self) -> logging.Logger:
        if self._logger is None:
            from domyn_swarm.helpers.logger import setup_logger

            self._logger = setup_logger("domyn_swarm.cli", level=logging.INFO)
        return self._logger

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


DomynLLMSwarm = _LazyDomynLLMSwarm()
logger = _LazyLogger()

job_app = typer.Typer(help="Submit a workload to a Domyn-Swarm allocation.")


SKIP_PREFLIGHT_OPTION = typer.Option(
    False,
    "--skip-preflight",
    help=(
        "Deploy without checking that the config's image, model and mount paths "
        "exist on this host. Only relevant with --config."
    ),
)


JOB_MEM_OPTION = typer.Option(
    None,
    "--mem",
    help=(
        "Slurm only. Memory for the job step, for example `64G`. `0` gives the step "
        "all the memory of its allocation. Default: `0` with "
        "`endpoint.require_allocated_node`, otherwise `endpoint.mem`."
    ),
)
JOB_CPUS_PER_TASK_OPTION = typer.Option(
    None,
    "--cpus-per-task",
    min=1,
    help=(
        "Slurm only. CPUs for the job step. Default: the `--cpus-per-task` value of "
        "the enclosing allocation with `endpoint.require_allocated_node`, otherwise "
        "`endpoint.cpus_per_task`."
    ),
)


def _load_swarm_config(*args, **kwargs):
    """Load a swarm config lazily."""
    from domyn_swarm.config.swarm import _load_swarm_config as load_swarm_config

    return load_swarm_config(*args, **kwargs)


@job_app.command("submit-script")
def submit_script(
    script_file: Path = typer.Argument(..., exists=True, readable=True),
    config: typer.FileText | None = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        help="YAML that defines/creates a new swarm",
    ),
    name: str | None = typer.Option(None, "-n", "--name", exists=True, help="Swarm name."),
    args: list[str] = typer.Argument(None, help="extra CLI args passed to script"),
    skip_preflight: bool = SKIP_PREFLIGHT_OPTION,
    mem: str | None = JOB_MEM_OPTION,
    cpus_per_task: int | None = JOB_CPUS_PER_TASK_OPTION,
):
    """
    Run an *arbitrary* Python file inside the swarm head node.
    """
    if config is not None and name is not None:
        logger.error("Either --config or --jobid must be provided, not both.")
        raise typer.Exit(1)

    if config:
        cfg = _load_swarm_config(config)
        job_resources = helpers.job_step_resources(mem=mem, cpus_per_task=cpus_per_task, cfg=cfg)
        with (
            exit_on_config_path_error(source=config.name),
            DomynLLMSwarm(cfg=cfg, preflight=not skip_preflight) as swarm,
        ):
            handle = swarm.submit_script(script_file, extra_args=args, job_resources=job_resources)
            helpers.emit_submission_json(
                handle=handle,
                command="submit-script",
                swarm_name=swarm.name,
            )

    elif name is None:
        raise RuntimeError("State is null")

    else:
        swarm = DomynLLMSwarm.from_state(deployment_name=name)
        job_resources = helpers.job_step_resources(
            mem=mem, cpus_per_task=cpus_per_task, cfg=swarm.cfg
        )
        handle = swarm.submit_script(script_file, extra_args=args, job_resources=job_resources)
        helpers.emit_submission_json(
            handle=handle,
            command="submit-script",
            swarm_name=swarm.name,
        )


def _parse_job_path(value: str) -> Path:
    return Path(str(EnvPath(value)))


@job_app.command("submit")
def submit_job(
    job_class: str = typer.Argument(
        default="domyn_swarm.jobs:ChatCompletionJob",
        help="Job class to run, in the form `module:ClassName`",
    ),
    input: Path = typer.Option(..., "--input", parser=_parse_job_path, metavar="ENVPATH"),
    output: Path = typer.Option(..., "--output", parser=_parse_job_path, metavar="ENVPATH"),
    input_column: str = typer.Option("messages", "--input-column"),
    output_column: str = typer.Option("results", "--output-column"),
    id_column: str | None = typer.Option(
        None,
        "--id-column",
        "--id-col",
        help="Optional column name used for stable row ids.",
    ),
    job_kwargs: str = typer.Option(
        "{}", "--job-kwargs", help="JSON dict forwarded to job constructor"
    ),
    data_backend: str | None = typer.Option(
        None,
        "--data-backend",
        help="Data backend for IO (pandas, polars, ray).",
    ),
    native_backend: bool | None = typer.Option(
        None,
        "--native-backend/--no-native-backend",
        help="Use native backend batches when supported.",
    ),
    native_batch_size: int | None = typer.Option(
        None,
        "--native-batch-size",
        help="Batch size for native backend mode (optional).",
    ),
    backend_read_kwargs: str | None = typer.Option(
        None,
        "--backend-read-kwargs",
        help="JSON dict forwarded to backend read() call.",
    ),
    backend_write_kwargs: str | None = typer.Option(
        None,
        "--backend-write-kwargs",
        help="JSON dict forwarded to backend write() call.",
    ),
    job_name: str | None = typer.Option(None, "--job-name", help="Optional job name for logging"),
    config: typer.FileText | None = typer.Option(
        None, "-c", "--config", exists=True, help="YAML that starts a fresh swarm"
    ),
    name: str | None = typer.Option(None, "-n", "--name", help="Swarm name."),
    checkpoint_dir: Path | None = typer.Option(
        None,
        "--checkpoint-dir",
        "-cd",
        help="Directory to store checkpoints (default: .checkpoint/, no checkpoints)",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        "--ignore-checkpoints",
        help="Ignore existing checkpoints for this run (forces recompute).",
    ),
    no_checkpointing: bool = typer.Option(
        False,
        "--no-checkpointing",
        help="Disable checkpointing entirely (no read/write checkpoint state).",
    ),
    runner: str = typer.Option(
        "pandas",
        "--runner",
        help="Deprecated; use --engine instead. Runner implementation for non-ray "
        "backends (pandas, arrow).",
    ),
    engine: str | None = typer.Option(
        None,
        "--engine",
        help="Execution engine to use (pandas, arrow, polars, ray). Takes precedence "
        "over the deprecated --runner/--data-backend pair.",
    ),
    checkpoint_interval: int = typer.Option(
        32,
        "--checkpoint-interval",
        "-ci",
        help="Batch size for processing input DataFrame (default: 32)",
    ),
    max_concurrency: int = typer.Option(
        32,
        "--max-concurrency",
        "-mc",
        help="Number of concurrent requests to process (default: 32)",
    ),
    retries: int = typer.Option(
        5,
        "--retries",
        "-r",
        help="Number of retries for failed requests (default: 5)",
    ),
    timeout: float = typer.Option(
        600,
        "--timeout",
        "-t",
        help="Timeout for each request in seconds (default: 600)",
    ),
    num_shards: int = typer.Option(
        1,
        "--num-shards",
        "-ns",
        help=(
            "How many shards to split the input into. This is part of the checkpoint "
            "layout: keep it fixed across resumes of the same job, or completed rows "
            "will be reprocessed."
        ),
    ),
    num_threads: int | None = typer.Option(
        None,
        "--num-threads",
        "-nt",
        hidden=True,
        help="Deprecated alias for --num-shards.",
    ),
    shard_output: bool = typer.Option(
        False,
        "--shard-output",
        help="When output is a directory, write shards directly from checkpoint outputs instead "
        "of assembling the merged result in memory first. Supported by the pandas and polars "
        "runners; ignored by the arrow runner. Either way the directory is written as one "
        "parquet file per shard (based on --num-shards). On the pandas and polars runners "
        "this also makes --global-resume a no-op: per-shard checkpoint resume alone already "
        "covers it, as long as --num-shards stays fixed across resumes.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit the size to be read from the input dataset. "
        "Useful when debugging and testing to reduce the size of the dataset",
    ),
    shard_mode: Literal["id", "index"] = typer.Option(
        "id",
        "--shard-mode",
        help="How to split input when --num-shards > 1. "
        "'id' uses stable id hashing (resume-friendly), "
        "'index' uses legacy row order sharding. "
        "Keep --num-shards fixed across resumes to avoid reshuffling shards.",
    ),
    global_resume: bool = typer.Option(
        False,
        "--global-resume",
        help="When resuming with sharded execution, filter inputs using global "
        "done ids across all shards (useful if --limit changed).",
    ),
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Detach the job from the current terminal"
    ),
    mail_user: str | None = typer.Option(
        None,
        "--mail-user",
        "-m",
        help="Email address to receive job notifications. "
        "If set, email notifications will be enabled.",
    ),
    checkpoint_tag: str | None = typer.Option(
        None,
        "--checkpoint-tag",
        help="An optional tag to be used when checkpointing is enabled. "
        "It will be used in place of the default uuid-based tag.",
    ),
    ray_address: str | None = typer.Option(
        None,
        "--ray-address",
        help="Ray cluster address to connect to when --data-backend=ray (optional).",
    ),
    skip_preflight: bool = SKIP_PREFLIGHT_OPTION,
    mem: str | None = JOB_MEM_OPTION,
    cpus_per_task: int | None = JOB_CPUS_PER_TASK_OPTION,
):
    """
    Submit a strongly-typed job to the swarm for DataFrame processing.

    This command processes input data through a SwarmJob (DataFrame-in → DataFrame-out)
    within the swarm environment. Jobs are defined by a class that handles the actual
    processing logic, and the input/output are managed as DataFrames with configurable
    column mappings.

    The job can either create a new swarm from a configuration file or connect to an
    existing swarm by name. Progress is tracked through checkpoints, and the operation
    supports concurrent processing with configurable retry logic.
    """
    if config is not None and name is not None:
        logger.error("Either --config or --name must be provided, not both.")
        raise typer.Exit(1)

    job_kwargs = helpers.build_job_kwargs_json(
        job_kwargs=job_kwargs,
        data_backend=data_backend,
        native_backend=native_backend,
        native_batch_size=native_batch_size,
        id_column_name=id_column,
        backend_read_kwargs=backend_read_kwargs,
        backend_write_kwargs=backend_write_kwargs,
    )
    if num_threads is not None:
        typer.echo(
            "warning: --num-threads is deprecated, use --num-shards. The value has "
            "always been a shard count, not a thread count.",
            err=True,
        )
        num_shards = num_threads

    run_spec = helpers.JobRunSpec(
        input_path=input,
        output_path=output,
        shard_output=shard_output,
        checkpoint_dir=checkpoint_dir,
        no_resume=no_resume,
        no_checkpointing=no_checkpointing,
        runner=runner,
        engine=engine,
        num_shards=num_shards,
        limit=limit,
        shard_mode=shard_mode,
        global_resume=global_resume,
        detach=detach,
        mail_user=mail_user,
        ray_address=ray_address,
        checkpoint_tag=checkpoint_tag,
    )

    if config:
        cfg = _load_swarm_config(config)
        run_spec = replace(
            run_spec,
            job_resources=helpers.job_step_resources(mem=mem, cpus_per_task=cpus_per_task, cfg=cfg),
        )
        swarm_ctx = DomynLLMSwarm(cfg=cfg, preflight=not skip_preflight)
        try:
            with exit_on_config_path_error(source=config.name), swarm_ctx as swarm:
                job = helpers.build_job_for_swarm(
                    swarm=swarm,
                    job_class=job_class,
                    job_kwargs=job_kwargs,
                    job_name=job_name,
                    input_column=input_column,
                    output_column=output_column,
                    checkpoint_interval=checkpoint_interval,
                    max_concurrency=max_concurrency,
                    retries=retries,
                    timeout=timeout,
                )
                handle = helpers.submit_loaded_job(
                    swarm=swarm,
                    request=helpers.JobSubmitRequest(job=job, run=run_spec),
                )
                helpers.emit_submission_json(
                    handle=handle,
                    command="submit",
                    swarm_name=swarm.name,
                )
        except KeyboardInterrupt:
            helpers.maybe_cancel_swarm_on_keyboard_interrupt(swarm_ctx)
    elif name is None:
        raise RuntimeError("Swarm name is null.")

    else:
        swarm = DomynLLMSwarm.from_state(deployment_name=name)
        run_spec = replace(
            run_spec,
            job_resources=helpers.job_step_resources(
                mem=mem, cpus_per_task=cpus_per_task, cfg=swarm.cfg
            ),
        )
        job = helpers.build_job_for_swarm(
            swarm=swarm,
            job_class=job_class,
            job_kwargs=job_kwargs,
            job_name=job_name,
            input_column=input_column,
            output_column=output_column,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
        )
        handle = helpers.submit_loaded_job(
            swarm=swarm,
            request=helpers.JobSubmitRequest(job=job, run=run_spec),
        )
        helpers.emit_submission_json(
            handle=handle,
            command="submit",
            swarm_name=swarm.name,
        )


@job_app.command("list")
def list_jobs(
    name: str = typer.Option(..., "-n", "--name", help="Swarm deployment name."),
    status: list[str] | None = typer.Option(
        None,
        "--status",
        help="Filter by status (repeatable). Values: PENDING,RUNNING,SUCCEEDED,FAILED,CANCELLED.",
    ),
    limit: int = typer.Option(50, "--limit", min=1, help="Maximum number of jobs to return."),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-parseable JSON instead of the default TUI view.",
    ),
) -> None:
    """List jobs for a swarm.

    Args:
        name: Swarm deployment name.
        status: Optional status filter list.
        limit: Maximum number of jobs to show.
        json_output: Whether to emit JSON output.
    """
    statuses = helpers.parse_status_filters(status)
    status_values = [status_value.value for status_value in statuses] if statuses else None
    rows = SwarmStateManager.list_jobs(name, limit=limit, statuses=status_values)
    if json_output:
        helpers.emit_job_list_json(
            swarm_name=name,
            jobs=rows,
            limit=limit,
            statuses=statuses,
        )
        return

    from domyn_swarm.cli.tui.job_view import render_job_list

    console = Console()
    render_job_list(rows, swarm_name=name, console=console)


@job_app.command("status")
def status_job(
    job_id: str = typer.Argument(..., help="Internal Domyn job ID from the local state DB."),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Probe the backend for live status and persist it to the local state DB. "
        "By default the recorded status is shown without contacting the backend.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-parseable JSON instead of the default TUI view.",
    ),
) -> None:
    """Show details for a recorded job.

    Args:
        job_id: Internal job ID.
        refresh: Whether to probe the backend for live status before display.
        json_output: Whether to emit JSON output.
    """
    try:
        row = SwarmStateManager.get_job(job_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not refresh:
        row["refresh_source"] = "db"
        row["refresh_error"] = None
    else:
        deployment_name = str(row.get("deployment_name") or "")
        if deployment_name:
            try:
                swarm = DomynLLMSwarm.from_state(deployment_name=deployment_name)
                row = swarm.refresh_job_status(job_id)
            except Exception as exc:
                row["refresh_source"] = "backend"
                row["refresh_error"] = str(exc)
        else:
            row["refresh_source"] = "db"
            row["refresh_error"] = "Missing deployment_name; backend probe skipped."

    if json_output:
        helpers.emit_job_status_json(job=row)
        return

    from domyn_swarm.cli.tui.job_view import render_job_status

    console = Console()
    render_job_status(row, console=console)


@job_app.command("wait")
def wait_job(
    job_id: str | None = typer.Option(
        None, "--job-id", help="Internal Domyn job ID from the local state DB."
    ),
    external_id: str | None = typer.Option(
        None, "--external-id", help="Provider external ID (for example Slurm step id)."
    ),
    handle_json: str | None = typer.Option(
        None,
        "--handle-json",
        help="JSON handle payload (use '-' to read from stdin). If omitted, stdin is auto-read.",
    ),
    name: str | None = typer.Option(
        None,
        "-n",
        "--name",
        help="Optional deployment name used to disambiguate --external-id or handle payload.",
    ),
    stream_logs: bool = typer.Option(
        True,
        "--stream-logs/--no-stream-logs",
        help="Stream backend logs while waiting (when supported by backend).",
    ),
) -> None:
    """Wait for a submitted job to reach a terminal state.

    Args:
        job_id: Internal job ID selector.
        external_id: Provider external-id selector.
        handle_json: Handle JSON selector, literal payload or stdin marker.
        name: Optional deployment name hint.
        stream_logs: Whether to stream backend logs while waiting.
    """
    target = helpers.resolve_job_target(
        job_id=job_id,
        external_id=external_id,
        handle_json=handle_json,
        deployment_name=name,
    )
    swarm = DomynLLMSwarm.from_state(deployment_name=target.swarm_name)
    target.handle.status = swarm.wait_job(target.handle, stream_logs=stream_logs)
    if target.job_id:
        SwarmStateManager.update_job(
            target.job_id,
            status=target.handle.status,
            external_id=target.handle.external_id,
        )
    helpers.emit_job_control_json(command="wait", target=target, status=target.handle.status)


@job_app.command("cancel")
def cancel_job(
    job_id: str | None = typer.Argument(
        None, help="Internal Domyn job ID from the local state DB."
    ),
    external_id: str | None = typer.Option(
        None, "--external-id", help="Provider external ID (for example Slurm step id)."
    ),
    handle_json: str | None = typer.Option(
        None,
        "--handle-json",
        help="JSON handle payload (use '-' to read from stdin). If omitted, stdin is auto-read.",
    ),
    name: str | None = typer.Option(
        None,
        "-n",
        "--name",
        help="Optional deployment name used to disambiguate --external-id or handle payload.",
    ),
) -> None:
    """Cancel a submitted job.

    Args:
        job_id: Internal job ID selector (positional).
        external_id: Provider external-id selector.
        handle_json: Handle JSON selector, literal payload or stdin marker.
        name: Optional deployment name hint.
    """
    target = helpers.resolve_job_target(
        job_id=job_id,
        external_id=external_id,
        handle_json=handle_json,
        deployment_name=name,
    )
    swarm = DomynLLMSwarm.from_state(deployment_name=target.swarm_name)
    target.handle.status = swarm.cancel_job(target.handle)

    if target.job_id:
        SwarmStateManager.update_job(
            target.job_id,
            status=target.handle.status,
            external_id=target.handle.external_id,
        )
    helpers.emit_job_control_json(command="cancel", target=target, status=target.handle.status)
