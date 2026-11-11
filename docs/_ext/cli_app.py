# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Generate the CLI reference with Typer's own Markdown formatter."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def render_cli_reference() -> str:
    """Render every CLI command, including the lazily loaded subcommands."""
    return subprocess.check_output(
        [
            sys.executable,
            "-m",
            "typer",
            "domyn_swarm.cli.main",
            "utils",
            "docs",
            "--name",
            "domyn-swarm",
        ],
        text=True,
    )


def generate(app: Sphinx) -> None:
    """Write the CLI reference before Sphinx reads the Markdown include."""
    out_dir = Path(app.srcdir) / "_generated"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "cli.md").write_text(render_cli_reference(), encoding="utf-8")


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the CLI reference generator."""
    app.connect("builder-inited", generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
