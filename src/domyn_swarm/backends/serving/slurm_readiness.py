# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import time

from rich.console import Console
from rich.status import Status

from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.core.state.watchdog import SwarmReplicaSummary, read_swarm_summary
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.platform.http_probe import wait_http_200
from domyn_swarm.platform.protocols import ServingHandle
from domyn_swarm.platform.readiness import ServingReadiness
from domyn_swarm.runtime.status import read_replica_statuses

SLURM_BAD_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "BOOT_FAIL",
    "NODE_FAIL",
}

SLURM_WAIT_STATES = {"PENDING", "CONFIGURING"}


class SwarmReplicaFailure(RuntimeError):
    """Raised when watchdog DB indicates that all replicas are in terminal states."""

    pass


logger = setup_logger(__name__)


class SlurmReadiness(ServingReadiness):
    """
    Slurm readiness: wait for replica array & LB jobs to be RUNNING,
    resolve LB node, then poll LB HTTP /health for 200.
    """

    def __init__(
        self,
        *,
        driver: SlurmDriver,
        endpoint_port: int,
        poll_interval_s: float = 10.0,
        console: Console | None = None,
        watchdog_db: Path | None = None,
        swarm_name: str | None = None,
        endpoint_job_file: Path | None = None,
    ):
        self.driver = driver
        self.endpoint_port = endpoint_port
        self.poll_interval_s = poll_interval_s
        self.console = console or Console()
        self.watchdog_db = watchdog_db
        self.swarm_name = swarm_name
        self.endpoint_job_file = endpoint_job_file
        self.last_status_tuple: tuple[int, int, int, int] | None = (
            None  # total, running, http_ready, failed
        )

    def wait_ready(self, handle: ServingHandle, timeout_s: int) -> ServingHandle:
        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        if jobid is None or lb_jobid is None:
            raise RuntimeError("SlurmReadiness requires 'jobid' and 'lb_jobid' in handle.meta")

        with self.console.status("[bold green]Waiting for LB and replicas to start...") as status:
            self._wait_jobs_running(jobid, lb_jobid, status)
            # Queue time is not bounded; the endpoint gets `timeout_s` once its jobs run.
            deadline = time.monotonic() + timeout_s
            if self.endpoint_job_file is not None:
                handle.meta["lb_jobid"], handle.meta["lb_node"] = self._wait_endpoint_job(
                    jobid, status, deadline
                )
            lb_node = handle.meta.get("lb_node") or self.driver.get_node_from_jobid(lb_jobid)
            handle.meta["lb_node"] = lb_node
            status.update(f"[yellow]LB job running on {lb_node}, probing HTTP…")

            base = f"http://{lb_node}:{self.endpoint_port}"

            def _watchdog_tick() -> None:
                self._watchdog_probe(status)

            wait_http_200(
                f"{base}/health",
                timeout_s=max(0, math.ceil(deadline - time.monotonic())),
                poll_interval_s=self.poll_interval_s,
                on_tick=_watchdog_tick,
            )
            handle.url = base
            logger.info(f"[bold green]LB healthy → {handle.url}")
            return handle

    def _wait_jobs_running(self, jobid: int, lb_jobid: int, status: Status):
        while True:
            rep = self.driver.get_job_state(jobid)
            lb = self.driver.get_job_state(lb_jobid)

            if rep == "UNKNOWN" or lb == "UNKNOWN":
                status.update(f"[yellow]squeue UNKNOWN for job {jobid} or {lb_jobid}, retrying …")
            elif rep in SLURM_BAD_STATES:
                raise RuntimeError(f"Replica array ended in {rep}")
            elif lb in SLURM_BAD_STATES:
                raise RuntimeError(f"LB job ended in {lb}")
            elif rep in SLURM_WAIT_STATES:
                status.update("[yellow]Waiting for replicas to start …")
            elif lb in SLURM_WAIT_STATES:
                status.update("[yellow]Waiting for LB job to start …")
            elif rep == "RUNNING" and lb == "RUNNING":
                return
            time.sleep(self.poll_interval_s)

    def _wait_endpoint_job(self, jobid: int, status: Status, deadline: float) -> tuple[int, str]:
        """Wait for the replica job to report where its colocated load balancer runs.

        The first array task writes ``<array job ID> <its job ID> <its node>`` to the
        endpoint job file before it starts the load balancer. Job steps target that
        task's job ID: the array job ID does not identify a single allocation.
        Records from another array job are ignored.

        Raises:
            RuntimeError: The replica job ended before writing the record.
            TimeoutError: No record appeared before ``deadline`` (``time.monotonic``).
        """
        assert self.endpoint_job_file is not None
        status.update("[yellow]Waiting for the replica job to start the load balancer …")
        while (record := self._read_endpoint_job(jobid)) is None:
            state = self.driver.get_job_state(jobid)
            if state in SLURM_BAD_STATES or state == "COMPLETED":
                raise RuntimeError(
                    f"Replica array ended in {state} before starting the load balancer"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Replica job {jobid} did not start the load balancer: "
                    f"no record in {self.endpoint_job_file}"
                )
            time.sleep(self.poll_interval_s)
        return record

    def _read_endpoint_job(self, jobid: int) -> tuple[int, str] | None:
        assert self.endpoint_job_file is not None
        try:
            fields = self.endpoint_job_file.read_text().split()
        except FileNotFoundError:
            return None
        if len(fields) != 3 or fields[0] != str(jobid):
            return None
        return int(fields[1]), fields[2]

    def _watchdog_probe(self, status: Status) -> None:
        """
        Called periodically by wait_http_200 via `_watchdog_tick`.

        Uses the watchdog DB (if configured) to:

        - Update the status line with replica counts.
        - Abort early if all replicas are in terminal states.
        """
        if not (self.watchdog_db and self.swarm_name):
            return  # nothing configured → no-op

        summary: SwarmReplicaSummary | None = read_swarm_summary(self.watchdog_db, self.swarm_name)
        if summary is None:
            # DB not present yet, or no rows written yet
            return

        if self.last_status_tuple == (
            summary.total,
            summary.running,
            summary.http_ready,
            summary.failed,
        ):
            return  # no change since last time

        self.last_status_tuple = (
            summary.total,
            summary.running,
            summary.http_ready,
            summary.failed,
        )

        # Update the CLI status with live replica information
        msg = (
            f"[yellow]Waiting for replicas for swarm '{self.swarm_name}': "
            f"total={summary.total}, running={summary.running}, "
            f"http_ready={summary.http_ready}, failed_or_exited={summary.failed}…[/]"
        )
        status.update(msg)

        if (
            summary.total > 0
            and summary.running == 0
            and summary.http_ready == 0
            and summary.failed > 0
        ):
            rows = read_replica_statuses(self.watchdog_db, self.swarm_name)
            row_hint = ""
            if rows:
                first = rows[0]
                row_hint = (
                    f" replica={first.replica_id} state={first.state} "
                    f"reason={first.fail_reason or 'unknown'}"
                )
            reason_frag = (
                f" Example: {summary.example_fail_reason}" if summary.example_fail_reason else ""
            )
            status.update(
                f"[red]All replicas failed for swarm '{self.swarm_name}' "
                f"(failed_or_exited={summary.failed}/{summary.total}).{reason_frag}{row_hint}[/]"
            )
            raise SwarmReplicaFailure(
                f"All replicas for swarm '{self.swarm_name}' are in terminal "
                f"states (failed_or_exited={summary.failed}, total={summary.total}).{row_hint}"
            )
