# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from rich.box import HEAVY
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from domyn_swarm.cli.tui.components import (
    _add_if,
    _color_state,
    _fmt_http,
    _phase_badge,
)
from domyn_swarm.cli.tui.tables import _kv_table
from domyn_swarm.core.state.watchdog import SwarmReplicaSummary
from domyn_swarm.platform.protocols import ServingStatus


# ---- Styling helpers --------------------------------------------------------
def render_swarm_status(
    name: str,
    backend: str,
    st: ServingStatus,
    *,
    replica_summary: SwarmReplicaSummary | None = None,
    replica_rows: list | None = None,
    router: str | None = None,
    console: Console | None = None,
) -> None:
    """Renders a comprehensive status view for a single swarm deployment.

    Displays a full-width status panel with swarm information including name, backend,
    phase, endpoint URL, and diagnostic details. The output is optimized for terminal
    display with styled text, tables, and panels using the Rich library.

    Args:
        name: The name/identifier of the swarm to display.
        backend: The deployment backend (e.g., 'slurm', 'lepton'). Will be displayed
            in uppercase.
        st: The serving status object containing phase, URL, and detailed diagnostic
            information.
        router: Optional load-balancer router type, shown for Slurm swarms.
        console: Optional Rich Console instance for output. If None, creates a new
            Console instance.

    Note:
        The diagnostic section adapts to different backends:
        - For Slurm: Shows replica ('rep') and load balancer ('lb') states
        - For Lepton: Shows raw platform state ('raw_state')
        - HTTP status is displayed for all backends when available
        - Additional diagnostic keys are shown in an 'Extra' section

    Example:
        >>> status = ServingStatus(phase="running", url="http://localhost:8000")
        >>> render_swarm_status("my-swarm", "slurm", status)
    """
    console = console or Console()

    banner = Text.assemble(
        ("SWARM ", "bold dim"),
        (name, "bold bright_cyan"),
        ("  •  ", "dim"),
        (backend.upper(), "bold magenta"),
    )

    info = _kv_table()
    info.add_row("Phase", _phase_badge(st.phase))
    url_txt = Text(st.url or "—")
    if st.url:
        url_txt.stylize(f"link {st.url}")
    info.add_row("Endpoint", url_txt)
    _add_if(info, "Router", router)

    # Diagnostics section (existing behaviour)
    diag = _kv_table()
    _add_if(diag, "HTTP", _fmt_http(st.detail))

    if st.detail:
        rep = st.detail.get("rep")
        lb = st.detail.get("lb")
        if rep is not None or lb is not None:
            diag.add_row("Replica", _color_state(rep))
            diag.add_row("LB", _color_state(lb))

        raw = st.detail.get("raw_state")
        if raw is not None:
            diag.add_row("Platform", _color_state(raw))

        extras = {k: v for k, v in st.detail.items() if k not in {"http", "rep", "lb", "raw_state"}}
        if extras:
            extras_kv = ", ".join(f"{k}={v}" for k, v in extras.items())
            diag.add_row("Extra", Text(extras_kv, overflow="fold"))

    diag_panel = (
        Panel(
            diag,
            title="Diagnostics",
            border_style="dim",
            padding=(1, 2),
            box=HEAVY,
        )
        if len(diag.rows) > 0
        else Text("")
    )

    # Replica health block from watchdog
    replica_panel: Panel | Text
    if replica_summary is not None or replica_rows:
        rep_table = Table(box=None, show_header=False, pad_edge=False)
        rep_table.add_column("Key", style="bold dim")
        rep_table.add_column("Value")

        if replica_summary is not None:
            rep_table.add_row("Total replicas", str(replica_summary.total))
            rep_table.add_row("Running", str(replica_summary.running))
            rep_table.add_row("HTTP ready", str(replica_summary.http_ready))
            rep_table.add_row("Failed", str(replica_summary.failed))

        # Failure reasons table (if any)
        reasons_panel: Panel | Text = Text("")
        if replica_summary and replica_summary.fail_reasons:
            reasons_table = Table(
                title=None,
                box=None,
                show_header=True,
                header_style="bold",
                pad_edge=False,
            )
            reasons_table.add_column("Reason")
            reasons_table.add_column("Count", justify="right")

            for reason, count in sorted(
                replica_summary.fail_reasons.items(),
                key=lambda kv: (-kv[1], kv[0]),
            ):
                label = reason or "unknown"
                reasons_table.add_row(label, str(count))

            reasons_panel = Panel(
                reasons_table,
                title="Failure reasons",
                border_style="red",
                padding=(0, 1),
            )

        rows_panel: Panel | Text = Text("")
        if replica_rows:
            rows_table = Table(
                title=None,
                box=None,
                show_header=True,
                header_style="bold",
                pad_edge=False,
            )
            rows_table.add_column("Replica", justify="right")
            rows_table.add_column("State")
            rows_table.add_column("HTTP", justify="right")
            rows_table.add_column("Node")
            rows_table.add_column("Port", justify="right")
            rows_table.add_column("Fail reason")

            for row in replica_rows:
                rows_table.add_row(
                    str(getattr(row, "replica_id", "")),
                    str(getattr(row, "state", "")),
                    str(getattr(row, "http_ready", "")),
                    str(getattr(row, "node", "") or ""),
                    str(getattr(row, "port", "") or ""),
                    str(getattr(row, "fail_reason", "") or ""),
                )

            rows_panel = Panel(
                rows_table,
                title="Replica rows",
                border_style="cyan",
                padding=(0, 1),
            )

        replica_body = Group(
            rep_table if replica_summary is not None else Text(""),
            Text(),  # spacer
            reasons_panel if isinstance(reasons_panel, Panel) else Text(""),
            Text(),  # spacer
            rows_panel if isinstance(rows_panel, Panel) else Text(""),
        )

        replica_panel = Panel(
            replica_body,
            title="Replicas (watchdog)",
            border_style="magenta",
            padding=(1, 2),
            box=HEAVY,
        )
    else:
        replica_panel = Text("")

    body = Group(
        Rule(banner, style="cyan"),
        Text(),  # spacer
        info,
        Text(),  # spacer
        diag_panel,
        Text(),  # spacer
        replica_panel,
    )

    panel = Panel(
        body,
        title=f"[b cyan]{name}[/] — [magenta]{backend.upper()}[/]",
        subtitle=(st.url or ""),
        border_style="cyan",
        padding=(1, 2),
        box=HEAVY,
        expand=True,
    )
    console.print(panel)
