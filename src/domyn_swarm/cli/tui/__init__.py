# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import cast

from rich.console import Console

from domyn_swarm.core.state.watchdog import SwarmReplicaSummary

from ...platform.protocols import ServingStatus
from ..tui.status import render_swarm_status
from ..tui.status_list import render_multi_status


def render_status(
    items: Iterable[tuple[str, str, ServingStatus]] | tuple[str, str, ServingStatus],
    *,
    replica_summary: SwarmReplicaSummary | None = None,
    replica_rows: list | None = None,
    router: str | None = None,
    console: Console | None = None,
) -> None:
    """
    Convenience wrapper:
      - If you pass a single (name, backend, status) tuple, render full-width single view.
      - If you pass an iterable of tuples, render compact multi view.
    """
    if isinstance(items, tuple) and len(items) == 3:
        name, backend, st = cast(tuple[str, str, ServingStatus], items)
        render_swarm_status(
            name,
            backend,
            st,
            replica_summary=replica_summary,
            replica_rows=replica_rows,
            router=router,
            console=console,
        )
        return

    items_list = list(items)  # type: ignore[arg-type]
    if len(items_list) == 1:
        name, backend, st = items_list[0]
        render_swarm_status(
            name,
            backend,
            st,
            replica_summary=replica_summary,
            replica_rows=replica_rows,
            router=router,
            console=console,
        )
    else:
        render_multi_status(items_list, console=console)


__all__ = ["render_multi_status", "render_status", "render_swarm_status"]
