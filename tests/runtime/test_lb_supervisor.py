# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from domyn_swarm.runtime import lb_supervisor as lbs
from domyn_swarm.runtime.lb_supervisor import read_head_files, render_upstreams, write_if_changed


def _write_heads(serving_dir: Path, addrs: dict[int, str]) -> None:
    for replica_id, addr in addrs.items():
        (serving_dir / f"replica-{replica_id}.head").write_text(addr + "\n")


def test_read_head_files_sorted_by_replica_id(tmp_path: Path):
    _write_heads(tmp_path, {2: "h2:9000", 0: "h0:9000", 1: "h1:9000"})
    assert read_head_files(tmp_path) == ["h0:9000", "h1:9000", "h2:9000"]


def test_read_head_files_empty_when_none(tmp_path: Path):
    assert read_head_files(tmp_path) == []


def test_render_upstreams_no_ray(tmp_path: Path):
    _write_heads(tmp_path, {0: "h0:9000", 1: "h1:9000"})
    conf = render_upstreams(tmp_path, ray_enabled=False, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream llm {" in conf
    assert "least_conn;" in conf
    assert "zone llm 1m;" in conf
    assert "server h0:9000 max_fails=2 fail_timeout=10s;" in conf
    assert "server h1:9000 max_fails=2 fail_timeout=10s;" in conf
    assert "upstream ray" not in conf


def test_render_upstreams_omits_keepalive_by_default(tmp_path: Path):
    _write_heads(tmp_path, {0: "h0:9000"})
    conf = render_upstreams(tmp_path, ray_enabled=False, ray_dashboard_port=8265, ray_port=6379)
    assert "keepalive" not in conf


def test_render_upstreams_emits_keepalive_after_servers(tmp_path: Path):
    _write_heads(tmp_path, {0: "h0:9000", 1: "h1:9000"})
    conf = render_upstreams(
        tmp_path,
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
        upstream_keepalive=32,
    )
    lines = [ln.strip() for ln in conf.splitlines()]
    assert "keepalive 32;" in lines
    assert lines.index("keepalive 32;") > lines.index(
        "server h1:9000 max_fails=2 fail_timeout=10s;"
    )
    assert lines.index("keepalive 32;") < lines.index("}")


def test_render_upstreams_with_ray_uses_replica0_host(tmp_path: Path):
    _write_heads(tmp_path, {0: "rayhost:9000"})
    conf = render_upstreams(tmp_path, ray_enabled=True, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream ray {" in conf
    assert "server rayhost:8265;" in conf
    assert "upstream ray_control {" in conf
    assert "server rayhost:6379;" in conf


def test_render_upstreams_ray_enabled_empty_dir_has_no_ray_block(tmp_path: Path):
    conf = render_upstreams(tmp_path, ray_enabled=True, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream llm {" in conf
    assert "upstream ray" not in conf


def test_write_if_changed_creates_and_reports_true(tmp_path: Path):
    target = tmp_path / "00-upstreams.conf"
    assert write_if_changed(target, "abc\n") is True
    assert target.read_text() == "abc\n"


def test_write_if_changed_noop_when_same(tmp_path: Path):
    target = tmp_path / "00-upstreams.conf"
    write_if_changed(target, "abc\n")
    assert write_if_changed(target, "abc\n") is False


def test_write_if_changed_no_partial_temp_left(tmp_path: Path):
    target = tmp_path / "00-upstreams.conf"
    write_if_changed(target, "abc\n")
    assert list(tmp_path.glob("*.tmp")) == []


def test_reconcile_writes_upstreams_on_change(tmp_path: Path):
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "replica-0.head").write_text("h0:9000")

    opts = lbs.SupervisorOptions(
        serving_dir=serving,
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
    )
    # First reconcile: file created -> returns True
    assert lbs.reconcile_once(opts) is True
    assert (serving / "00-upstreams.conf").read_text().count("server h0:9000") == 1
    # Second reconcile, no change -> returns False
    assert lbs.reconcile_once(opts) is False


def test_parse_args_builds_options():
    opts, once, interval = lbs.parse_args(
        [
            "--serving-dir",
            "/srv",
            "--ray-enabled",
            "--ray-dashboard-port",
            "8265",
            "--ray-port",
            "6379",
            "--interval",
            "5",
            "--once",
        ]
    )
    assert opts.serving_dir == Path("/srv")
    assert opts.ray_enabled is True
    assert opts.ray_dashboard_port == 8265
    assert opts.ray_port == 6379
    assert interval == 5
    assert once is True


def test_parse_args_reads_upstream_keepalive():
    opts, _, _ = lbs.parse_args(["--serving-dir", "/tmp/s", "--upstream-keepalive", "64"])
    assert opts.upstream_keepalive == 64


def test_parse_args_upstream_keepalive_defaults_to_zero():
    opts, _, _ = lbs.parse_args(["--serving-dir", "/tmp/s"])
    assert opts.upstream_keepalive == 0


def test_parse_args_defaults_ray_disabled():
    opts, once, interval = lbs.parse_args(["--serving-dir", "/srv"])
    assert opts.ray_enabled is False
    assert once is False
    assert interval == 5
    assert opts.emit_targets is False


def test_parse_args_emit_targets_flag():
    opts, _once, _interval = lbs.parse_args(["--serving-dir", "/srv", "--emit-targets"])
    assert opts.emit_targets is True


def test_render_targets_json(tmp_path: Path):
    (tmp_path / "replica-0.head").write_text("h0:9000")
    (tmp_path / "replica-1.head").write_text("h1:9000")
    payload = json.loads(lbs.render_targets(tmp_path))
    # One entry per replica so each target carries its ordinal `replica` label.
    assert payload == [
        {"targets": ["h0:9000"], "labels": {"job": "vllm", "replica": "0"}},
        {"targets": ["h1:9000"], "labels": {"job": "vllm", "replica": "1"}},
    ]


def test_render_targets_empty(tmp_path: Path):
    assert json.loads(lbs.render_targets(tmp_path)) == []


def test_reconcile_writes_targets_when_enabled(tmp_path: Path):
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "replica-0.head").write_text("h0:9000")
    opts = lbs.SupervisorOptions(
        serving_dir=serving,
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
        emit_targets=True,
    )
    lbs.reconcile_once(opts)
    assert (serving / "targets.json").exists()


def test_reconcile_no_targets_when_disabled(tmp_path: Path):
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "replica-0.head").write_text("h0:9000")
    opts = lbs.SupervisorOptions(
        serving_dir=serving,
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
        emit_targets=False,
    )
    lbs.reconcile_once(opts)
    assert not (serving / "targets.json").exists()


def test_render_gpu_targets_dedup_by_host(tmp_path: Path):
    (tmp_path / "gpu-h0.target").write_text("h0:9835\n")
    (tmp_path / "gpu-h1.target").write_text("h1:9835\n")
    # a duplicate announce for the same host (e.g. two replicas on one node)
    (tmp_path / "gpu-h1b.target").write_text("h1:9835\n")
    payload = json.loads(lbs.render_gpu_targets(tmp_path))
    assert payload == [
        {"targets": ["h0:9835"], "labels": {"job": "gpu"}},
        {"targets": ["h1:9835"], "labels": {"job": "gpu"}},
    ]


def test_render_gpu_targets_empty(tmp_path: Path):
    assert json.loads(lbs.render_gpu_targets(tmp_path)) == []


def test_render_gpu_ownership(tmp_path: Path):
    (tmp_path / "gpu-owner-0.txt").write_text("GPU-aaa\nGPU-bbb\n")
    (tmp_path / "gpu-owner-1.txt").write_text("GPU-ccc\n")
    text = lbs.render_gpu_ownership(tmp_path)
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert lines == [
        'dswarm_gpu_owner{uuid="GPU-aaa",UUID="GPU-aaa",replica="0"} 1',
        'dswarm_gpu_owner{uuid="GPU-bbb",UUID="GPU-bbb",replica="0"} 1',
        'dswarm_gpu_owner{uuid="GPU-ccc",UUID="GPU-ccc",replica="1"} 1',
    ]
    assert "# TYPE dswarm_gpu_owner gauge" in text
    assert all('uuid="' in ln and 'UUID="' in ln for ln in lines)


def test_render_gpu_ownership_empty(tmp_path: Path):
    text = lbs.render_gpu_ownership(tmp_path)
    assert "dswarm_gpu_owner{" not in text
    assert "# TYPE dswarm_gpu_owner gauge" in text  # header always present


def test_parse_args_emit_gpu_targets_flag():
    opts, _once, _interval = lbs.parse_args(["--serving-dir", "/srv", "--emit-gpu-targets"])
    assert opts.emit_gpu_targets is True


def test_reconcile_writes_gpu_files_when_enabled(tmp_path: Path):
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "replica-0.head").write_text("h0:9000")
    (serving / "gpu-h0.target").write_text("h0:9835")
    (serving / "gpu-owner-0.txt").write_text("GPU-aaa\n")
    opts = lbs.SupervisorOptions(
        serving_dir=serving,
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
        emit_targets=True,
        emit_gpu_targets=True,
    )
    lbs.reconcile_once(opts)
    assert (serving / "gpu_targets.json").exists()
    assert (serving / "gpu_ownership.prom").exists()


def test_reconcile_no_gpu_files_when_disabled(tmp_path: Path):
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "replica-0.head").write_text("h0:9000")
    (serving / "gpu-h0.target").write_text("h0:9835")
    opts = lbs.SupervisorOptions(
        serving_dir=serving,
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
        emit_targets=True,
        emit_gpu_targets=False,
    )
    lbs.reconcile_once(opts)
    assert not (serving / "gpu_targets.json").exists()
    assert not (serving / "gpu_ownership.prom").exists()


def test_gpu_ownership_aggregates_per_node_files_under_replica(tmp_path):
    (tmp_path / "gpu-owner-0-nodeA.txt").write_text("GPU-aaa\nGPU-bbb\n")
    (tmp_path / "gpu-owner-0-nodeB.txt").write_text("GPU-ccc\n")
    out = lbs.render_gpu_ownership(tmp_path)
    assert 'replica="0"' in out
    for uuid in ("GPU-aaa", "GPU-bbb", "GPU-ccc"):
        assert f'uuid="{uuid}"' in out
    assert 'replica="0-nodeA"' not in out  # suffix must not leak into the label


def test_gpu_ownership_backward_compatible_single_file(tmp_path):
    (tmp_path / "gpu-owner-2.txt").write_text("GPU-zzz\n")
    out = lbs.render_gpu_ownership(tmp_path)
    assert 'uuid="GPU-zzz"' in out
    assert 'replica="2"' in out


def test_render_ray_targets_dedups_by_host(tmp_path):
    import json

    from domyn_swarm.runtime import lb_supervisor as lbs

    (tmp_path / "ray-nodeA.target").write_text("nodeA:8090\n")
    (tmp_path / "ray-nodeB.target").write_text("nodeB:8090\n")
    payload = json.loads(lbs.render_ray_targets(tmp_path))
    hosts = sorted(t for g in payload for t in g["targets"])
    assert hosts == ["nodeA:8090", "nodeB:8090"]
    assert all(g["labels"]["job"] == "ray" for g in payload)


def test_render_ray_targets_empty(tmp_path):
    import json

    from domyn_swarm.runtime import lb_supervisor as lbs

    assert json.loads(lbs.render_ray_targets(tmp_path)) == []


def test_reconcile_writes_ray_targets_when_enabled(tmp_path):
    from domyn_swarm.runtime import lb_supervisor as lbs

    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "ray-nodeA.target").write_text("nodeA:8090\n")
    opts = lbs.SupervisorOptions(
        serving_dir=serving,
        ray_enabled=True,
        ray_dashboard_port=8265,
        ray_port=6379,
        emit_ray_targets=True,
    )
    lbs.reconcile_once(opts)
    assert (serving / "ray_targets.json").is_file()


def test_parse_args_emit_ray_targets_flag():
    from domyn_swarm.runtime import lb_supervisor as lbs

    opts, _once, _interval = lbs.parse_args(["--serving-dir", "/srv", "--emit-ray-targets"])
    assert opts.emit_ray_targets is True


def test_render_replicas_orders_by_replica_id(tmp_path: Path):
    _write_heads(tmp_path, {1: "n2:8001", 0: "n1:8000"})
    assert lbs.render_replicas(tmp_path) == "n1:8000\nn2:8001\n"


def test_render_replicas_empty_without_head_files(tmp_path: Path):
    assert lbs.render_replicas(tmp_path) == ""


def test_upstreams_point_at_router_when_address_given(tmp_path: Path):
    _write_heads(tmp_path, {0: "n1:8000"})
    conf = render_upstreams(
        tmp_path,
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
        upstream_keepalive=32,
        router_address="127.0.0.1:9200",
    )
    assert "server 127.0.0.1:9200;" in conf
    assert "n1:8000" not in conf
    assert "least_conn" not in conf
    assert "max_fails" not in conf
    assert "keepalive 32;" in conf


def test_reconcile_writes_replicas_file(tmp_path: Path):
    _write_heads(tmp_path, {0: "n1:8000"})
    lbs.reconcile_once(
        lbs.SupervisorOptions(
            serving_dir=tmp_path, ray_enabled=False, ray_dashboard_port=8265, ray_port=6379
        )
    )
    assert (tmp_path / "replicas.txt").read_text() == "n1:8000\n"


def test_parse_args_router_address():
    opts, _, _ = lbs.parse_args(["--serving-dir", "/s", "--router-address", "127.0.0.1:9200"])
    assert opts.router_address == "127.0.0.1:9200"
