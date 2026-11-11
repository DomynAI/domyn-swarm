# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
from types import SimpleNamespace

import jinja2
import pytest


def _helpers(colocate: bool) -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader("src/domyn_swarm/templates"))
    cfg = SimpleNamespace(backend=SimpleNamespace(endpoint=SimpleNamespace(colocate=colocate)))
    return env.get_template("_collector_env.sh.j2").render(cfg=cfg)


def _run(tmp_path: Path, body: str, *, colocate: bool = True):
    script = "\n".join(
        [
            "set -euo pipefail",
            f'COLLECTOR_ENV_FILE="{tmp_path}/collector.env"',
            _helpers(colocate),
            body,
        ]
    )
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=5)


@pytest.mark.parametrize(
    ("metadata", "state"),
    [
        ("", "JobState=RUNNING Restarts=1"),
        ("COLLECTOR_ARRAY_JOB_ID=1000\nCOLLECTOR_RESTART_COUNT=1", "JobState=RUNNING Restarts=1"),
        ("COLLECTOR_ARRAY_JOB_ID=4242\nCOLLECTOR_RESTART_COUNT=0", "JobState=RUNNING Restarts=1"),
        ("COLLECTOR_ARRAY_JOB_ID=4242\nCOLLECTOR_RESTART_COUNT=1", "JobState=PENDING Restarts=1"),
        ("COLLECTOR_ARRAY_JOB_ID=4242\nCOLLECTOR_RESTART_COUNT=1", "JobState=RUNNING"),
    ],
)
def test_nonzero_task_rejects_stale_collector_before_task_zero_starts(tmp_path, metadata, state):
    (tmp_path / "collector.env").write_text(
        f"COLLECTOR_HOST=old-node\nCOLLECTOR_PORT=9001\n{metadata}\n"
    )
    result = _run(
        tmp_path,
        f"""
        REPL_ID=1 REPL_JOB_ID=4242 SLURM_ARRAY_JOB_ID=4242 SLURM_JOB_ID=4244
        scontrol() {{ printf '%s\\n' '{state}'; }}
        if collector_load_env; then echo accepted-stale-record; exit 1; fi
        """,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("array", [False, True])
def test_reader_uses_the_collectors_restart_count_not_its_own(tmp_path, array):
    (tmp_path / "collector.env").write_text(
        "COLLECTOR_HOST=current-node\nCOLLECTOR_PORT=9001\n"
        "COLLECTOR_ARRAY_JOB_ID=4242\nCOLLECTOR_RESTART_COUNT=1\n"
    )
    owner = "4242_0" if array else "4242"
    result = _run(
        tmp_path,
        f"""
        REPL_JOB_ID=4242 SLURM_JOB_ID=4242 SLURM_RESTART_COUNT=3
        {"SLURM_ARRAY_JOB_ID=4242" if array else "unset SLURM_ARRAY_JOB_ID"}
        scontrol() {{
          [[ "$*" == 'show job {owner} -o' ]] || return 1
          echo 'JobState=RUNNING Restarts=1'
        }}
        collector_load_env
        printf '%s:%s\\n' "$COLLECTOR_HOST" "$COLLECTOR_PORT"
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "current-node:9001"


def test_reader_retries_after_scheduler_failure(tmp_path):
    (tmp_path / "collector.env").write_text(
        "COLLECTOR_HOST=current-node\nCOLLECTOR_PORT=9001\n"
        "COLLECTOR_ARRAY_JOB_ID=4242\nCOLLECTOR_RESTART_COUNT=1\n"
    )
    result = _run(
        tmp_path,
        """
        REPL_JOB_ID=4242 SLURM_JOB_ID=4242
        scontrol() { return 1; }
        if collector_load_env; then exit 1; fi
        scontrol() { echo 'JobState=RUNNING Restarts=1'; }
        collector_load_env
        """,
    )
    assert result.returncode == 0, result.stderr


def test_nonzero_task_waits_for_task_zero_to_publish_its_new_attempt(tmp_path):
    (tmp_path / "collector.env").write_text(
        "COLLECTOR_HOST=old-node\nCOLLECTOR_PORT=9001\n"
        "COLLECTOR_ARRAY_JOB_ID=4242\nCOLLECTOR_RESTART_COUNT=0\n"
    )
    result = _run(
        tmp_path,
        """
        REPL_ID=1 REPL_JOB_ID=4242 SLURM_ARRAY_JOB_ID=4242 SLURM_JOB_ID=4244
        scontrol() { echo 'JobState=RUNNING Restarts=1'; }
        if collector_load_env; then exit 1; fi
        (
          COLLECTOR_HOST=new-node COLLECTOR_PORT=9002 SLURM_RESTART_COUNT=1
          collector_publish_env
        )
        collector_load_env
        printf '%s:%s\\n' "$COLLECTOR_HOST" "$COLLECTOR_PORT"
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "new-node:9002"


@pytest.mark.parametrize("colocate", [False, True])
def test_writer_replaces_the_complete_record_atomically(tmp_path, colocate):
    (tmp_path / "collector.env").write_text("previous-record\n")
    result = _run(
        tmp_path,
        """
        COLLECTOR_HOST=current-node COLLECTOR_PORT=9001
        SLURM_ARRAY_JOB_ID=4242 DEP_JOBID=4243 SLURM_RESTART_COUNT=1
        mv() {
          [[ "$(cat "$COLLECTOR_ENV_FILE")" == previous-record ]] || exit 1
          command mv "$@"
        }
        collector_publish_env
        source "$COLLECTOR_ENV_FILE"
        printf '%s:%s\\n' "$COLLECTOR_HOST" "$COLLECTOR_PORT"
        """,
        colocate=colocate,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "current-node:9001"
    record = (tmp_path / "collector.env").read_text()
    assert ("COLLECTOR_ARRAY_JOB_ID=4242" in record) == colocate
    assert ("COLLECTOR_RESTART_COUNT=1" in record) == colocate
    assert not (tmp_path / "collector.env.tmp").exists()


def test_separate_endpoint_accepts_the_existing_record_format(tmp_path):
    (tmp_path / "collector.env").write_text("COLLECTOR_HOST=current-node\nCOLLECTOR_PORT=9001\n")
    result = _run(tmp_path, "collector_load_env", colocate=False)
    assert result.returncode == 0, result.stderr
