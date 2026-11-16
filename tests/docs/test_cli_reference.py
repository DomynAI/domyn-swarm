# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from cli_app import render_cli_reference

from domyn_swarm.cli.main import _LAZY_SUBAPPS


def test_cli_reference_documents_commands_and_options() -> None:
    reference = render_cli_reference()
    root_commands = {"version", "up", "status", "down", "monitor"} | _LAZY_SUBAPPS.keys()
    for command in root_commands:
        assert f"## `domyn-swarm {command}`" in reference
    for command in (
        "db prune",
        "db stamp",
        "db upgrade",
        "init defaults",
        "job cancel",
        "job list",
        "job status",
        "job submit",
        "job submit-script",
        "job wait",
        "pool pool",
        "swarm describe",
        "swarm list",
    ):
        assert f"### `domyn-swarm {command}`" in reference
    for option in ("--install-completion", "--show-completion", "--checkpoint-tag"):
        assert option in reference
