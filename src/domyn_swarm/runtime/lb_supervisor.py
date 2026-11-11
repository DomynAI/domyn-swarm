#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Domyn-Swarm load-balancer supervisor.

Keeps nginx's dynamic upstreams (and, when monitoring is enabled, the Prometheus
``targets.json``) in sync with the ``replica-*.head`` files that each vLLM replica
writes into the serving directory. Runs inside the same Python image as the
collector and is launched by the LB sbatch job (see ``templates/lb.sh.j2``).

Generation-only: the supervisor writes ``00-upstreams.conf``, the router-neutral
``replicas.txt`` (one ``host:port`` per line) and, when ``--emit-targets`` is
set, ``targets.json``. It does **not** reload nginx.
nginx reload is performed host-side by ``lb.sh.j2`` via ``inotifywait`` (or
equivalent polling). Cross-container ``SIGHUP`` is not possible on this platform
because ``singularity instance start`` uses a private PID namespace, so a process
inside the vLLM container cannot signal the nginx master process running in a
separate Singularity instance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import sys
import time


def read_head_entries(serving_dir: Path) -> list[tuple[int, str]]:
    """Return ``(replica_id, host:port)`` pairs, ordered by replica id.

    Args:
        serving_dir: Directory containing ``replica-<id>.head`` files, each
            holding a single ``host:port`` line.

    Returns:
        ``(replica_id, addr)`` pairs sorted ascending by the integer replica id
        encoded in the filename. Empty if no head files exist yet.
    """
    serving_dir = Path(serving_dir)
    entries: list[tuple[int, str]] = []
    for f in serving_dir.glob("replica-*.head"):
        try:
            replica_id = int(f.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        addr = f.read_text().strip()
        if addr and ":" in addr:
            entries.append((replica_id, addr))
    entries.sort(key=lambda e: e[0])
    return entries


def read_head_files(serving_dir: Path) -> list[str]:
    """Return replica ``host:port`` addresses, ordered by replica id.

    Args:
        serving_dir: Directory containing ``replica-<id>.head`` files, each
            holding a single ``host:port`` line.

    Returns:
        Addresses sorted ascending by the integer replica id encoded in the
        filename. Empty if no head files exist yet.
    """
    return [addr for _, addr in read_head_entries(serving_dir)]


def render_upstreams(
    serving_dir: Path,
    *,
    ray_enabled: bool,
    ray_dashboard_port: int,
    ray_port: int,
    upstream_keepalive: int = 0,
    router_address: str | None = None,
) -> str:
    """Render the nginx ``upstream`` blocks for the current replicas.

    Reproduces the upstream generation previously embedded in ``lb.sh.j2``.

    Args:
        serving_dir: Directory containing ``replica-*.head`` files.
        ray_enabled: Whether to also emit the Ray dashboard/control upstreams
            (host taken from the lowest-id replica with a valid address
            (typically replica 0)).
        ray_dashboard_port: Ray dashboard port for the ``ray`` upstream.
        ray_port: Ray client port for the ``ray_control`` upstream.
        upstream_keepalive: Idle connections each nginx worker caches for the
            ``llm`` upstream. 0 disables connection reuse, so every proxied
            request opens a fresh connection to a replica.
        router_address: ``host:port`` of a local router. When set, the ``llm``
            upstream holds only this server and the router balances across
            replicas; Ray upstreams still target the replica head.

    Returns:
        nginx configuration text defining the ``llm`` upstream (and Ray
        upstreams when enabled).
    """
    addrs = read_head_files(serving_dir)
    if router_address:
        lines = ["upstream llm {", "  zone llm 1m;", f"  server {router_address};"]
    else:
        lines = ["upstream llm {", "  zone llm 1m;", "  least_conn;"]
        lines.extend(f"  server {addr} max_fails=2 fail_timeout=10s;" for addr in addrs)
    if upstream_keepalive > 0:
        lines.append(f"  keepalive {upstream_keepalive};")
    lines.append("}")

    if ray_enabled and addrs:
        ray_host = addrs[0].split(":", 1)[0]
        lines += [
            "upstream ray {",
            f"  server {ray_host}:{ray_dashboard_port};",
            "  keepalive 8;",
            "}",
            "upstream ray_control {",
            f"  server {ray_host}:{ray_port};",
            "  keepalive 8;",
            "}",
        ]
    return "\n".join(lines) + "\n"


def write_if_changed(target: Path, content: str) -> bool:
    """Atomically write ``content`` to ``target`` only if it differs.

    Writes to a sibling temp file and ``os.replace`` so readers never observe a
    partial file.

    Args:
        target: Destination path.
        content: Text to write.

    Returns:
        True if the file was created or its content changed; False if the
        existing content already matched.
    """
    target = Path(target)
    if target.exists() and target.read_text() == content:
        return False
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, target)
    return True


UPSTREAMS_FILENAME = "00-upstreams.conf"
REPLICAS_FILENAME = "replicas.txt"
TARGETS_FILENAME = "targets.json"
GPU_TARGETS_FILENAME = "gpu_targets.json"
RAY_TARGETS_FILENAME = "ray_targets.json"


def render_replicas(serving_dir: Path) -> str:
    """Render the router-neutral replica list.

    Args:
        serving_dir: Directory containing ``replica-*.head`` files.

    Returns:
        One ``host:port`` per line, ordered by replica id, each newline
        terminated. Empty string when no head files exist.
    """
    return "".join(f"{addr}\n" for addr in read_head_files(serving_dir))


def render_targets(serving_dir: Path) -> str:
    """Render Prometheus file_sd targets JSON for the current replicas.

    Args:
        serving_dir: Directory containing ``replica-*.head`` files.

    Returns:
        A JSON string with one file_sd entry per replica, so each target also
        carries its ordinal ``replica`` label (Prometheus file_sd labels apply
        per entry, not per target) in addition to ``job=vllm``.
    """
    entries = read_head_entries(serving_dir)
    groups = [
        {"targets": [addr], "labels": {"job": "vllm", "replica": str(replica_id)}}
        for replica_id, addr in entries
    ]
    return json.dumps(groups) + "\n"


def read_gpu_target_entries(serving_dir: Path) -> list[tuple[str, str]]:
    """Return ``(host, "host:port")`` for each node announce file, deduped
    by host.

    Reads ``gpu-<host>.target`` files (each a single ``host:port`` line).
    Multiple files for the same host (replicas sharing a node) collapse to
    one entry.
    """
    serving_dir = Path(serving_dir)
    by_host: dict[str, str] = {}
    for f in serving_dir.glob("gpu-*.target"):
        addr = f.read_text().strip()
        if not addr or ":" not in addr:
            continue
        host = addr.split(":", 1)[0]
        by_host.setdefault(host, addr)
    return sorted(by_host.items(), key=lambda kv: kv[0])


def render_gpu_targets(serving_dir: Path) -> str:
    """Render Prometheus file_sd targets for per-node GPU exporters (deduped)."""
    groups = [
        {"targets": [addr], "labels": {"job": "gpu"}}
        for _host, addr in read_gpu_target_entries(serving_dir)
    ]
    return json.dumps(groups) + "\n"


def read_ray_target_entries(serving_dir: Path) -> list[tuple[str, str]]:
    """Return ``(host, "host:port")`` for each Ray node announce file, deduped
    by host.

    Reads ``ray-<host>.target`` files (each a single ``host:port`` line).
    Multiple files for the same host (replicas sharing a node) collapse to
    one entry.
    """
    serving_dir = Path(serving_dir)
    by_host: dict[str, str] = {}
    for f in serving_dir.glob("ray-*.target"):
        addr = f.read_text().strip()
        if not addr or ":" not in addr:
            continue
        host = addr.split(":", 1)[0]
        by_host.setdefault(host, addr)
    return sorted(by_host.items(), key=lambda kv: kv[0])


def render_ray_targets(serving_dir: Path) -> str:
    """Render Prometheus file_sd targets for per-node Ray metrics agents (deduped)."""
    groups = [
        {"targets": [addr], "labels": {"job": "ray"}}
        for _host, addr in read_ray_target_entries(serving_dir)
    ]
    return json.dumps(groups) + "\n"


GPU_OWNERSHIP_FILENAME = "gpu_ownership.prom"
_OWNER_RE = re.compile(r"gpu-owner-(?P<replica>\d+)(?:-.+)?\.txt$")


def render_gpu_ownership(serving_dir: Path) -> str:
    """Render the uuid→replica join metric in Prometheus text format.

    Reads ``gpu-owner-<replica>.txt`` files (one GPU UUID per line) and
    emits one ``dswarm_gpu_owner{uuid,replica} 1`` series per owned GPU.
    """
    serving_dir = Path(serving_dir)
    rows: list[tuple[str, str]] = []  # (replica, uuid)
    for f in serving_dir.glob("gpu-owner-*.txt"):
        m = _OWNER_RE.search(f.name)
        if not m:
            continue
        replica = m.group("replica")
        for line in f.read_text().splitlines():
            uuid = line.strip()
            if uuid:
                rows.append((replica, uuid))
    rows.sort()
    out = [
        "# HELP dswarm_gpu_owner Which replica owns a GPU (join key: uuid for "
        "nvidia_smi exporter, UUID for dcgm exporter).",
        "# TYPE dswarm_gpu_owner gauge",
    ]
    out += [
        f'dswarm_gpu_owner{{uuid="{uuid}",UUID="{uuid}",replica="{replica}"}} 1'
        for replica, uuid in rows
    ]
    return "\n".join(out) + "\n"


@dataclass
class SupervisorOptions:
    """Runtime options for the LB supervisor reconcile loop.

    Attributes:
        serving_dir: Directory holding ``replica-*.head`` and generated conf.d files.
        ray_enabled: Whether to emit Ray upstreams.
        ray_dashboard_port: Ray dashboard port.
        ray_port: Ray client port.
        emit_targets: Whether to also write the Prometheus file_sd targets.json.
        emit_gpu_targets: Whether to also write gpu_targets.json and gpu_ownership.prom.
        emit_ray_targets: Whether to also write the Prometheus file_sd ray_targets.json.
        upstream_keepalive: Idle upstream connections cached per nginx worker.
        router_address: ``host:port`` of a local router to front the ``llm`` upstream.
    """

    serving_dir: Path
    ray_enabled: bool
    ray_dashboard_port: int
    ray_port: int
    emit_targets: bool = False
    emit_gpu_targets: bool = False
    emit_ray_targets: bool = False
    upstream_keepalive: int = 0
    router_address: str | None = None


def reconcile_once(opts: SupervisorOptions) -> bool:
    """Regenerate the upstreams file and targets from the current head files.

    Writes ``00-upstreams.conf`` and ``replicas.txt`` (and ``targets.json`` when
    ``opts.emit_targets`` is set) to ``opts.serving_dir``. Does **not** reload
    nginx — reload is performed host-side by ``lb.sh.j2``.

    Args:
        opts: Supervisor options.

    Returns:
        True if the upstreams file changed, else False.
    """
    conf = render_upstreams(
        opts.serving_dir,
        ray_enabled=opts.ray_enabled,
        ray_dashboard_port=opts.ray_dashboard_port,
        ray_port=opts.ray_port,
        upstream_keepalive=opts.upstream_keepalive,
        router_address=opts.router_address,
    )
    changed = write_if_changed(opts.serving_dir / UPSTREAMS_FILENAME, conf)
    write_if_changed(opts.serving_dir / REPLICAS_FILENAME, render_replicas(opts.serving_dir))
    if opts.emit_targets:
        write_if_changed(opts.serving_dir / TARGETS_FILENAME, render_targets(opts.serving_dir))
    if opts.emit_gpu_targets:
        write_if_changed(
            opts.serving_dir / GPU_TARGETS_FILENAME, render_gpu_targets(opts.serving_dir)
        )
        write_if_changed(
            opts.serving_dir / GPU_OWNERSHIP_FILENAME, render_gpu_ownership(opts.serving_dir)
        )
    if opts.emit_ray_targets:
        write_if_changed(
            opts.serving_dir / RAY_TARGETS_FILENAME, render_ray_targets(opts.serving_dir)
        )
    return changed


def run_loop(opts: SupervisorOptions, interval: int) -> int:
    """Continuously reconcile upstreams with head files until terminated.

    Args:
        opts: Supervisor options.
        interval: Seconds to sleep between reconcile passes.

    Returns:
        Process exit code (0 on clean shutdown via SIGTERM/SIGINT).
    """
    stop = {"flag": False}

    def _handle(signum, frame):
        print(f"lb_supervisor: received signal {signum}, stopping", file=sys.stderr)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(
        f"lb_supervisor: watching {opts.serving_dir} (interval={interval}s, "
        f"ray_enabled={opts.ray_enabled})",
        file=sys.stderr,
    )
    while not stop["flag"]:
        try:
            reconcile_once(opts)
        except Exception as e:  # never let a transient error kill the loop
            print(f"lb_supervisor: reconcile error: {e!r}", file=sys.stderr)
        for _ in range(interval):
            if stop["flag"]:
                break
            time.sleep(1)
    return 0


def parse_args(argv: list[str] | None = None) -> tuple[SupervisorOptions, bool, int]:
    """Parse CLI arguments into supervisor options.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A tuple of (options, run_once, interval_seconds).
    """
    p = argparse.ArgumentParser(description="Domyn-Swarm LB supervisor.")
    p.add_argument("--serving-dir", required=True)
    p.add_argument("--ray-enabled", action="store_true")
    p.add_argument("--ray-dashboard-port", type=int, default=8265)
    p.add_argument("--ray-port", type=int, default=6379)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--once", action="store_true")
    p.add_argument("--emit-targets", action="store_true")
    p.add_argument("--emit-gpu-targets", action="store_true")
    p.add_argument("--emit-ray-targets", action="store_true")
    p.add_argument("--upstream-keepalive", type=int, default=0)
    p.add_argument("--router-address", default=None, metavar="HOST:PORT")
    a = p.parse_args(argv)
    opts = SupervisorOptions(
        serving_dir=Path(a.serving_dir),
        ray_enabled=a.ray_enabled,
        ray_dashboard_port=a.ray_dashboard_port,
        ray_port=a.ray_port,
        emit_targets=a.emit_targets,
        emit_gpu_targets=a.emit_gpu_targets,
        emit_ray_targets=a.emit_ray_targets,
        upstream_keepalive=a.upstream_keepalive,
        router_address=a.router_address,
    )
    return opts, a.once, a.interval


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the LB supervisor."""
    opts, once, interval = parse_args(argv)
    if once:
        reconcile_once(opts)
        return 0
    return run_loop(opts, interval)


if __name__ == "__main__":
    raise SystemExit(main())
