# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import logging
import sys
from typing import Annotated

import typer
from typer.core import TyperGroup

from .monitor import monitor as _monitor_cmd

# Sub-apps that pull in heavy dependencies (pandas, openai, leptonai, alembic).
# Imported on first use so light root commands stay fast. Maps command name ->
# (module, app attribute, help text shown in --help).
_LAZY_SUBAPPS: dict[str, tuple[str, str, str]] = {
    "job": (
        "domyn_swarm.cli.job",
        "job_app",
        "Submit a workload to a Domyn-Swarm allocation.",
    ),
    "pool": (
        "domyn_swarm.cli.pool",
        "pool_app",
        "Submit a pool of swarm allocations from a YAML config.",
    ),
    "init": (
        "domyn_swarm.cli.init",
        "init_app",
        "Initialize a new Domyn-Swarm configuration.",
    ),
    "swarm": (
        "domyn_swarm.cli.swarm",
        "swarm_app",
        "Manage swarm allocations.",
    ),
    "db": (
        "domyn_swarm.cli.db",
        "db_app",
        "Manage the Domyn-Swarm state database.",
    ),
}


class LazyGroup(TyperGroup):
    """Root group that imports heavy sub-command modules only when invoked.

    Keeps light root commands (version, up, down, status) from importing the
    job/pool/swarm/db subsystems at startup. ``--help`` and the no-args listing
    still resolve sub-commands (importing them) so the help output is complete.
    """

    def list_commands(self, ctx):
        return sorted(set(super().list_commands(ctx)) | set(_LAZY_SUBAPPS))

    def get_command(self, ctx, cmd_name):
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        spec = _LAZY_SUBAPPS.get(cmd_name)
        if spec is None:
            return None
        module_path, attr, help_text = spec
        from typer.main import get_group as _typer_to_click_group

        sub_app = getattr(importlib.import_module(module_path), attr)
        wrapper = typer.Typer()
        wrapper.add_typer(sub_app, name=cmd_name, help=help_text)
        click_cmd = _typer_to_click_group(wrapper).get_command(ctx, cmd_name)
        assert click_cmd is not None
        self.add_command(click_cmd, cmd_name)
        return click_cmd


class _LazyDomynLLMSwarm:
    """Proxy that imports the swarm implementation only when a command needs it."""

    def __call__(self, *args, **kwargs):
        from ..core.swarm import DomynLLMSwarm

        return DomynLLMSwarm(*args, **kwargs)

    def from_state(self, *args, **kwargs):
        """Load a swarm from persisted state."""
        from ..core.swarm import DomynLLMSwarm

        return DomynLLMSwarm.from_state(*args, **kwargs)


class _LazySwarmStateManager:
    """Proxy for state-manager class methods used by the root CLI commands."""

    def load(self, *args, **kwargs):
        """Load a saved swarm record."""
        from ..core.state.state_manager import SwarmStateManager

        return SwarmStateManager.load(*args, **kwargs)

    def list_by_base_name(self, *args, **kwargs):
        """List swarm names matching a configured base name."""
        from ..core.state.state_manager import SwarmStateManager

        return SwarmStateManager.list_by_base_name(*args, **kwargs)

    def get_last_swarm_name(self, *args, **kwargs):
        """Return the most recently created swarm name."""
        from ..core.state.state_manager import SwarmStateManager

        return SwarmStateManager.get_last_swarm_name(*args, **kwargs)

    def get_creation_dt(self, *args, **kwargs):
        """Return the persisted creation timestamp for a deployment."""
        from ..core.state.state_manager import SwarmStateManager

        return SwarmStateManager.get_creation_dt(*args, **kwargs)


class _LazyLogger:
    """Logger proxy that avoids importing Rich logging during CLI discovery."""

    def __init__(self) -> None:
        self._logger: logging.Logger | None = None

    def _get(self) -> logging.Logger:
        if self._logger is None:
            from ..helpers.logger import setup_logger

            self._logger = setup_logger("domyn_swarm.cli", level=logging.INFO)
        return self._logger

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


DomynLLMSwarm = _LazyDomynLLMSwarm()
SwarmStateManager = _LazySwarmStateManager()

app = typer.Typer(name="domyn-swarm CLI", no_args_is_help=True, cls=LazyGroup)

# Sub-apps (job/pool/init/swarm/db) are registered lazily by LazyGroup; see
# _LAZY_SUBAPPS above. Only light root commands are wired eagerly below.

app.command("monitor", short_help="Open grafatui against a swarm's Prometheus")(_monitor_cmd)

logger = _LazyLogger()


def _get_console():
    """Create the Rich console only for commands that render Rich output."""
    from rich.console import Console

    return Console()


def _load_swarm_config(*args, **kwargs):
    """Load a swarm config while keeping config imports off the CLI startup path."""
    from ..config.swarm import _load_swarm_config as load_swarm_config

    return load_swarm_config(*args, **kwargs)


def _pick_one(*args, **kwargs):
    """Prompt for one swarm name from a list."""
    from domyn_swarm.utils.cli import _pick_one as pick_one

    return pick_one(*args, **kwargs)


def ensure_db_up_to_date(*args, **kwargs):
    """Run the database auto-upgrade helper lazily."""
    from ..core.state.autoupgrade import ensure_db_up_to_date as ensure

    return ensure(*args, **kwargs)


def get_version() -> str:
    """Return the installed package version."""
    from ..utils.version import get_version as resolve_version

    return resolve_version()


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """
    Main entrypoint
    """
    # Commands that do not touch swarm state should not pay for DB migrations.
    if ctx.invoked_subcommand in {"db", "init", "version"}:
        return

    # Safe to call for everything else; it's idempotent and guarded.
    ensure_db_up_to_date(noisy=True)


@app.command("version", short_help="Show the version of the domyn-swarm CLI")
def version(short: bool = False):
    v = get_version()
    typer.echo(v if short else f"domyn-swarm CLI Version: {v}")
    raise typer.Exit()


@app.command("up", short_help="Launch a swarm allocation with a configuration")
def launch_up(
    config: Annotated[
        typer.FileText,
        typer.Option(..., "-c", "--config", help="Path to YAML config for LLMSwarmConfig"),
    ],
    replicas: Annotated[
        int | None,
        typer.Option(
            "--replicas",
            "-r",
            help="Number of replicas for the swarm allocation. Defaults to 1.",
        ),
    ] = None,
    skip_preflight: Annotated[
        bool,
        typer.Option(
            "--skip-preflight",
            help=(
                "Submit without checking that the config's image, model and mount "
                "paths exist on this host."
            ),
        ),
    ] = False,
):
    """
    Launch a swarm allocation with the given configuration.
    The configuration must be provided as a YAML file.
    """
    from domyn_swarm.backends.serving.slurm_readiness import SwarmReplicaFailure
    from domyn_swarm.config.preflight import check_config_paths, format_path_problems

    cfg = _load_swarm_config(config, replicas=replicas)

    if not skip_preflight:
        problems = check_config_paths(cfg)
        if problems:
            source = getattr(config, "name", None)
            typer.echo(format_path_problems(problems, source=source), err=True)
            raise typer.Exit(code=2)

    swarm_ctx = DomynLLMSwarm(cfg=cfg, preflight=not skip_preflight)

    def _safe_down(ctx) -> None:
        try:
            ctx.down()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    try:
        with swarm_ctx as swarm:
            # Print ONLY the name to stdout so command substitution works if not tty:
            #   SWARM_NAME=$(domyn-swarm up -c config.yaml)
            if not sys.stdout.isatty():
                typer.echo(swarm.name)
            raise typer.Exit(code=0)
    except KeyboardInterrupt as ki:
        if typer.confirm("KeyboardInterrupt detected. Cancel the swarm allocation?"):
            _safe_down(swarm_ctx)
            typer.echo("Swarm allocation cancelled by user", err=True)
            raise typer.Abort() from ki
        typer.echo(f"Waiting for swarm {swarm_ctx.name}…", err=True)
    except SwarmReplicaFailure as srf:
        # Resources were already released by DomynLLMSwarm on startup failure.
        logger.error(f"Swarm replica failure detected: {srf}")
        raise typer.Exit(code=1) from srf


@app.command(
    "status",
    short_help="Check the status of the swarm allocation given its state file",
)
def check_status(
    name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the swarm allocation to check status for. ",
        ),
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output format: 'table' (default, Rich TUI) or 'json' (stable, machine-readable).",
            case_sensitive=False,
        ),
    ] = "table",
) -> None:
    """
    Check the status of the swarm allocation.

    This command will read the DB and print the status of the swarm allocation.
    If a name is provided, it will check the status of that specific allocation.
    """
    from domyn_swarm.core.state.watchdog import read_swarm_summary
    from domyn_swarm.runtime.status import read_replica_statuses

    fmt = output.lower()
    if fmt not in {"table", "json"}:
        raise typer.BadParameter("--output must be 'table' or 'json'")

    swarm = SwarmStateManager.load(deployment_name=name)
    if swarm.serving_handle is None:
        raise ValueError("Swarm does not have a serving handle.")

    errors: list[str] = []
    serving_status = None
    try:
        serving_status = swarm.status()
    except Exception as e:
        logger.debug(f"Could not query serving status: {e}")
        errors.append(f"serving_status: {e}")

    replica_summary = None
    replica_rows = []

    try:
        replica_summary = read_swarm_summary(swarm.watchdog_db_path, swarm_id=name)
    except Exception as e:
        logger.debug(f"Could not read swarm replica summary: {e}")
        errors.append(f"replica_summary: {e}")

    try:
        replica_rows = read_replica_statuses(swarm.watchdog_db_path, swarm_id=name)
    except Exception as e:
        logger.debug(f"Could not read swarm replica rows: {e}")
        errors.append(f"replica_rows: {e}")

    if fmt == "json":
        payload = _build_status_json(
            name=name,
            swarm=swarm,
            serving_status=serving_status,
            replica_summary=replica_summary,
            replica_rows=replica_rows,
            errors=errors,
        )
        import json as _json

        typer.echo(_json.dumps(payload, indent=2, default=str))
        return

    from ..cli.tui import render_status
    from ..platform.protocols import ServingPhase, ServingStatus

    render_status(
        (
            name,
            swarm.platform,
            serving_status or ServingStatus(phase=ServingPhase.UNKNOWN, url=swarm.endpoint),
        ),
        replica_summary=replica_summary,
        replica_rows=replica_rows,
        console=_get_console(),
    )


def _build_status_json(
    *,
    name: str,
    swarm,
    serving_status,
    replica_summary,
    replica_rows,
    errors: list[str],
) -> dict:
    """Build the stable JSON payload for `ds status -o json`.

    Schema is a public contract — change with care.
    """
    from datetime import UTC

    from ..platform.protocols import ServingPhase

    phase = serving_status.phase.value if serving_status is not None else ServingPhase.UNKNOWN.value
    endpoint = (
        serving_status.url if serving_status is not None and serving_status.url else swarm.endpoint
    )

    summary_dict = None
    if replica_summary is not None:
        summary_dict = {
            "total": replica_summary.total,
            "running": replica_summary.running,
            "http_ready": replica_summary.http_ready,
            "failed": replica_summary.failed,
            "fail_reasons": replica_summary.fail_reasons,
            "example_fail_reason": replica_summary.example_fail_reason,
        }

    replicas_list = [
        {
            "replica_id": r.replica_id,
            "node": r.node,
            "port": r.port,
            "state": r.state,
            "http_ready": bool(r.http_ready) if r.http_ready is not None else None,
            "exit_code": r.exit_code,
            "exit_signal": r.exit_signal,
            "fail_reason": r.fail_reason,
            "last_seen": r.last_seen,
            "url": (f"http://{r.node}:{r.port}" if r.node and r.port is not None else None),
        }
        for r in replica_rows
    ]

    if replica_summary is not None and replica_summary.total > 0:
        ready = (
            phase == ServingPhase.RUNNING.value
            and replica_summary.http_ready == replica_summary.total
            and replica_summary.failed == 0
        )
    else:
        ready = phase == ServingPhase.RUNNING.value

    started_at_iso: str | None = None
    expires_at_iso: str | None = None
    try:
        creation_dt = SwarmStateManager.get_creation_dt(name)
    except Exception as e:
        logger.debug(f"Could not read creation_dt: {e}")
        creation_dt = None
        errors.append(f"creation_dt: {e}")

    if creation_dt is not None:
        started_at = (
            creation_dt if creation_dt.tzinfo is not None else creation_dt.replace(tzinfo=UTC)
        )
        started_at_iso = started_at.isoformat()

        if swarm.platform == "slurm":
            from ..helpers.slurm import parse_slurm_time_limit

            time_limit = getattr(swarm.cfg.backend, "time_limit", None)
            delta = parse_slurm_time_limit(time_limit) if time_limit else None
            if delta is not None:
                expires_at_iso = (started_at + delta).isoformat()

    slurm_jobs: list[dict] | None = None
    if swarm.platform == "slurm" and swarm.serving_handle is not None:
        meta = swarm.serving_handle.meta or {}
        jobs: list[dict] = []
        if meta.get("jobid") is not None:
            jobs.append({"role": "swarm", "jobid": meta.get("jobid")})
        if meta.get("lb_jobid") is not None:
            jobs.append(
                {
                    "role": "load_balancer",
                    "jobid": meta.get("lb_jobid"),
                    "node": meta.get("lb_node"),
                }
            )
        slurm_jobs = jobs

    payload: dict = {
        "schema_version": 1,
        "name": name,
        "backend": swarm.platform,
        "phase": phase,
        "state": phase,
        "ready": ready,
        "endpoint": endpoint,
        "started_at": started_at_iso,
        "expires_at": expires_at_iso,
        "replicas": {
            "summary": summary_dict,
            "items": replicas_list,
        },
        "slurm_jobs": slurm_jobs,
        "detail": (serving_status.detail if serving_status is not None else None),
        "errors": errors or None,
    }
    return payload


def down_by_name(name: str, yes: bool) -> None:
    """
    Shut down a swarm by its name, with optional confirmation.
    This function loads a swarm by name and shuts it down. If the swarm is currently
    running and the yes parameter is False, it will prompt the user for confirmation
    before proceeding with the shutdown.
    Args:
        name (str): The name of the swarm to shut down.
        yes (bool): If True, skip confirmation prompt for running swarms.
    Returns:
        None
    Raises:
        typer.Exit: If user cancels the shutdown when prompted for confirmation.
    Example:
        >>> down_by_name("my-swarm", yes=False)
        # Will prompt for confirmation if swarm is running
        >>> down_by_name("my-swarm", yes=True)
        # Will shut down without confirmation
    """
    swarm = SwarmStateManager.load(deployment_name=name)
    serving_status = swarm.status()
    if serving_status.phase == "RUNNING" and not yes:
        confirm = typer.confirm(f"Are you sure you want to shut down the running swarm {name}?")
        if not confirm:
            typer.secho("Aborting shutdown.", fg="red")
            raise typer.Exit()
    swarm.down()
    typer.echo(f"✅ Swarm {name} shutdown request sent.")


def down_by_config(config: typer.FileText, yes: bool, all_: bool, select: bool) -> None:
    """
    Shut down swarms matching the base name in the provided config file.
    If multiple swarms match, the user can choose to shut down all, select one,
    or be prompted for confirmation.
    Args:
        config (typer.FileText): File handle to the YAML config containing the swarm name.
        yes (bool): If True, skip confirmation prompts.
        all_ (bool): If True, shut down all matching swarms.
        select (bool): If True, prompt the user to select one swarm to shut down.
    Raises:
        typer.Exit: Exits the program after shutting down swarms or if no action is taken.
    """
    logger.info(f"Loading swarm config from {config.name} to find matching swarms")
    base_name = _load_swarm_config(config).name
    matches = SwarmStateManager.list_by_base_name(base_name)

    logger.info(f"Found {len(matches)} matching swarms for base name '{base_name}'")

    if not matches:
        _get_console().print(f"[yellow]No swarms found for base name '{base_name}'.[/]")
        raise typer.Exit(0)

    if len(matches) == 1 and not all_:
        name = matches[0]
        _get_console().print(f"[cyan]Found 1 match:[/] {name}")
        if not yes and not typer.confirm(f"Destroy swarm '{name}'?"):
            raise typer.Abort()
        down_by_name(name=name, yes=yes)  # reuse your existing down logic
        raise typer.Exit(0)

    if all_:
        if not yes:
            from rich.table import Table

            table = Table(title="Swarms to destroy")
            table.add_column("deployment_name")
            for n in matches:
                table.add_row(n)
            _get_console().print(table)
            if not typer.confirm(f"Destroy ALL {len(matches)} swarms above?"):
                raise typer.Abort()
        for n in matches:
            down_by_name(name=n, yes=True)
        raise typer.Exit(0)

    if select:
        name = _pick_one(matches, _get_console())
        if not yes and not typer.confirm(f"Destroy swarm '{name}'?"):
            raise typer.Abort()
        down_by_name(name=name, yes=True)
        raise typer.Exit(0)

    if matches:
        console = _get_console()
        console.print(
            f"[red]Multiple swarms found for base name '{base_name}'. "
            "Please specify one using --name, use --select to pick "
            "one or use --all to delete all of them.[/]"
        )
        for n in matches:
            console.print(f" - {n}")
        raise typer.Exit(1)


@app.command("down", short_help="Shut down a swarm allocation")
def down(
    name: str | None = typer.Argument(default=None, help="Swarm name."),
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Force shutdown without confirmation.",
        ),
    ] = False,
    select: bool = typer.Option(
        False, "--select", help="Pick a single swarm when multiple matches."
    ),
    all_: bool = typer.Option(False, "--all", help="Tear down all matching swarms."),
    config: typer.FileText | None = typer.Option(
        None, "-c", "--config", help="Path to YAML config (must contain 'name')."
    ),
):
    """
    Shut down a swarm allocation.

    If NAME is not provided, attempts to find the last created swarm.
    If a CONFIG is provided, uses its 'name' field to find matching swarms.
    Use --all to shut down all matching swarms, or --select to pick one interactively.
    """
    if name is None:
        logger.warning("Swarm name not provided for shutdown")
        if config is not None:
            down_by_config(config=config, yes=yes, all_=all_, select=select)

        logger.info(
            "No config provided or no matches found; attempting to shut down the last swarm."
        )
        name = SwarmStateManager.get_last_swarm_name()
        if name:
            logger.info(f"Shutting down the last swarm: [bold cyan]{name}[/]")
            down_by_name(name=name, yes=yes)
        else:
            logger.error("No swarms found to shut down.")
            raise typer.Exit(code=1)
    else:
        down_by_name(name=name, yes=yes)
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
