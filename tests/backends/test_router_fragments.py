# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess

import jinja2
import pytest

from domyn_swarm.config.router import RamjetRouter, VllmRouter
from domyn_swarm.config.swarm import DomynLLMSwarmConfig

TPL_DIR = Path(__file__).resolve().parents[2] / "src" / "domyn_swarm" / "templates"


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _cfg(router: dict, tmp_path: Path = Path("/tmp")) -> DomynLLMSwarmConfig:
    return DomynLLMSwarmConfig.model_validate(
        {
            "name": "s1",
            "image": "/img/vllm.sif",
            "model": "m",
            "replicas": 2,
            "home_directory": tmp_path,
            "backend": {
                "type": "slurm",
                "partition": "p",
                "account": "a",
                "qos": "q",
                "endpoint": {"nginx_image": "/img/nginx.sif", "router": router},
            },
        }
    )


def _render_lb(router: dict) -> str:
    return (
        _env()
        .get_template("lb.sh.j2")
        .render(
            cfg=_cfg(router),
            job_name="job",
            dep_jobid=111,
            replicas=2,
            swarm_directory="/swarm/s1",
            collector_script_path="/opt/watchdog_collector.py",
            supervisor_script_path="/opt/lb_supervisor.py",
        )
    )


ROUTER = "http://127.0.0.1:9200"


def _apply_pass(
    tmp_path: Path, applied: Path, new: list[str], reply_once: tuple[str, str, str] | None = None
) -> int:
    """Run one `router_apply_replicas` pass against a stub `curl`.

    The stub logs each URL it is asked for and answers `ok` with HTTP 200.
    `reply_once` is `(url_part, status, body)`: the first call whose URL contains
    `url_part` gets that answer instead, and an empty status stands for a client
    timeout (curl exit 28, no output).
    """
    fragment = tmp_path / "router.sh"
    bin_dir = tmp_path / "bin"
    log = tmp_path / "curl.log"
    reply = tmp_path / "reply-once"
    if not fragment.exists():
        fragment.write_text(
            _env()
            .get_template("routers/vllm-router.sh.j2")
            .render(router=VllmRouter(binary="vllm-router"))
        )
        bin_dir.mkdir()
        stub = bin_dir / "curl"
        stub.write_text(
            "#!/bin/bash\n"
            'url="${@: -1}"\n'
            f'echo "$url" >> {log}\n'
            f'if [ -s {reply} ] && [[ "$url" == *"$(sed -n 1p {reply})"* ]]; then\n'
            f'    status="$(sed -n 2p {reply})" body="$(sed -n 3p {reply})"\n'
            f"    rm {reply}\n"
            '    [ -n "$status" ] || exit 28\n'
            '    printf \'%s\\n%s\' "$body" "$status"\n'
            "    exit 0\n"
            "fi\n"
            "printf 'ok\\n200'\n"
        )
        stub.chmod(0o755)
    if reply_once:
        reply.write_text("\n".join(reply_once) + "\n")
    new_file = tmp_path / "new.txt"
    new_file.write_text("".join(f"{a}\n" for a in new))
    return subprocess.run(
        [
            "bash",
            "-euc",
            f'PATH="{bin_dir}:$PATH"; HOST_DIR={tmp_path}; source {fragment}; '
            f"router_apply_replicas {applied} {new_file}",
        ],
    ).returncode


def _function_body(script: str, name: str) -> str:
    start = script.index(f"{name}() {{\n") + len(f"{name}() {{\n")
    return script[start : script.index("\n}\n", start)]


def _curl_calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "curl.log"
    return log.read_text().splitlines() if log.exists() else []


def _run_apply(tmp_path: Path, old: list[str], new: list[str]) -> list[str]:
    applied = tmp_path / "applied.txt"
    applied.write_text("".join(f"{a}\n" for a in old))
    assert _apply_pass(tmp_path, applied, new) == 0
    return _curl_calls(tmp_path)


def _run_start(tmp_path: Path, router_body: str, health_check: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    router_bin = bin_dir / "fake-router"
    router_bin.write_text(f"#!/bin/bash\n{router_body}\n")
    router_bin.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text(f"#!/bin/bash\n{health_check}\n")
    curl.chmod(0o755)
    (tmp_path / "logs").mkdir()
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "replicas.txt").write_text("n1:8000\nn2:8001\n")
    fragment = tmp_path / "router.sh"
    fragment.write_text(
        _env()
        .get_template("routers/vllm-router.sh.j2")
        .render(router=VllmRouter(binary=str(router_bin), env={"FAKE_ENV": "a b"}))
    )
    return subprocess.run(
        [
            "bash",
            "-euc",
            f'PATH="{bin_dir}:$PATH"; HOST_DIR={tmp_path}; SERVING_DIR={tmp_path}/serving; '
            f"WAIT_SEC=5; source {fragment}; router_start; "
            'kill -0 "$ROUTER_PID"; router_stop; router_stop',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_vllm_router_start_passes_replicas_and_env(tmp_path):
    result = _run_start(
        tmp_path,
        f'printf "%s\\n" "$FAKE_ENV" "$@" > {tmp_path}/argv; exec sleep 30',
        health_check=f"test -s {tmp_path}/argv",
    )
    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[0] == "a b"
    assert argv[argv.index("--worker-urls") + 1 : argv.index("--worker-urls") + 3] == [
        "http://n1:8000",
        "http://n2:8001",
    ]
    assert "--api-key" not in argv


def test_vllm_router_start_exits_when_the_router_dies(tmp_path):
    result = _run_start(tmp_path, 'echo "bad flag" >&2; exit 2', health_check="exit 7")
    assert result.returncode == 1
    assert "vllm-router did not become healthy" in result.stderr
    assert "bad flag" in result.stderr


def test_nginx_router_renders_no_router_process():
    out = _render_lb({"type": "nginx"})
    assert "--router-address" not in out
    assert "vllm-router" not in out


def test_vllm_router_start_command():
    out = _render_lb(
        {"type": "vllm-router", "binary": "/opt/bin/vllm-router", "policy": "cache_aware"}
    )
    assert "/opt/bin/vllm-router --host 127.0.0.1 --port $ROUTER_PORT" in out
    assert "--policy cache_aware" in out
    assert "--prometheus-port 9201" in out
    assert "--router-address 127.0.0.1:$ROUTER_PORT" in out


def test_vllm_router_image_runs_through_singularity():
    out = _render_lb({"type": "vllm-router", "image": "/img/vr.sif"})
    assert 'singularity exec --bind "$HOST_DIR" /img/vr.sif vllm-router' in out


def test_router_starts_before_nginx_and_stops_on_exit():
    out = _render_lb({"type": "vllm-router", "binary": "vllm-router"})
    assert out.index("router_start") < out.index("singularity instance start")
    assert "router_stop" in out.split("trap", 1)[1]


def test_vllm_router_passes_both_supervisor_runs_the_router_address():
    out = _render_lb({"type": "vllm-router", "binary": "vllm-router"})
    assert out.count("--router-address 127.0.0.1:$ROUTER_PORT") == 2
    assert "ROUTER_PORT=9200" in out


def test_vllm_router_quotes_program_policy_json():
    out = _render_lb({"type": "vllm-router", "binary": "vllm-router", "policy": "program"})
    assert (
        "--program-scheduling-config-json "
        """'{"program_scheduling_enable_key":"auto","global_queue":true}'"""
    ) in out


def test_vllm_router_exports_env_and_appends_args():
    out = _render_lb(
        {
            "type": "vllm-router",
            "binary": "vllm-router",
            "env": {"RUST_LOG": "info debug"},
            "args": ["--request-timeout-secs", "600"],
        }
    )
    assert "export RUST_LOG='info debug'" in out
    assert "--request-timeout-secs 600" in out


def test_vllm_router_api_key_is_optional():
    out = _render_lb({"type": "vllm-router", "binary": "vllm-router"})
    assert '${LB_API_KEY:+--api-key "$LB_API_KEY"}' in out


def test_vllm_router_apply_moves_a_requeued_replica(tmp_path):
    log = _run_apply(tmp_path, old=["n1:8000", "n2:8001"], new=["n1:8000", "n3:8001"])
    assert log == [
        f"{ROUTER}/remove_worker?url=http://n2:8001",
        f"{ROUTER}/add_worker?url=http://n3:8001",
    ]


def test_vllm_router_apply_unchanged_set_makes_no_calls(tmp_path):
    assert _run_apply(tmp_path, old=["n1:8000"], new=["n1:8000"]) == []


def test_vllm_router_apply_from_empty_old_list_adds_every_replica(tmp_path):
    log = _run_apply(tmp_path, old=[], new=["n1:8000", "n2:8001"])
    assert log == [
        f"{ROUTER}/add_worker?url=http://n1:8000",
        f"{ROUTER}/add_worker?url=http://n2:8001",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        ("remove_worker?url=http://n2:8001", "500", "boom"),
        ("add_worker?url=http://n3:8001", "500", "boom"),
        ("add_worker?url=http://n3:8001", "", ""),
        ("add_worker?url=http://n3:8001", "400", "Invalid URL"),
    ],
    ids=["remove-500", "add-500", "add-timeout", "add-other-400"],
)
def test_vllm_router_apply_retries_only_the_failed_call(tmp_path, failure):
    applied = tmp_path / "applied.txt"
    applied.write_text("n1:8000\nn2:8001\n")

    assert _apply_pass(tmp_path, applied, ["n1:8000", "n3:8001"], reply_once=failure) != 0
    first_pass = len(_curl_calls(tmp_path))
    assert _apply_pass(tmp_path, applied, ["n1:8000", "n3:8001"]) == 0

    assert _curl_calls(tmp_path)[first_pass:] == [f"{ROUTER}/{failure[0]}"]
    assert sorted(applied.read_text().splitlines()) == ["n1:8000", "n3:8001"]


def test_vllm_router_apply_counts_an_existing_worker_as_added(tmp_path):
    applied = tmp_path / "applied.txt"
    applied.write_text("n1:8000\n")
    already = ("add_worker?url=http://n2:8001", "400", "Worker http://n2:8001 already exists")

    assert _apply_pass(tmp_path, applied, ["n1:8000", "n2:8001"], reply_once=already) == 0

    assert applied.read_text().splitlines() == ["n1:8000", "n2:8001"]


def test_vllm_router_apply_creates_a_missing_record(tmp_path):
    applied = tmp_path / "run" / "applied.txt"
    applied.parent.mkdir()

    assert _apply_pass(tmp_path, applied, ["n1:8000"]) == 0

    assert applied.read_text().splitlines() == ["n1:8000"]


def test_ramjet_image_uses_singularity_run_and_rj_env():
    out = _render_lb({"type": "ramjet", "image": "/img/ramjet.sif"})
    assert 'singularity run --bind "$HOST_DIR" /img/ramjet.sif' in out
    assert "RJ_API_ADDR=127.0.0.1:$ROUTER_PORT" in out
    assert "RJ_METRICS_ADDR=127.0.0.1:9201" in out
    assert "RJ_AFFINITY=prefix" in out


def test_ramjet_apply_restarts_the_router():
    out = _render_lb({"type": "ramjet", "image": "/r.sif"})
    body = _function_body(out, "router_apply_replicas")
    assert body.index("router_stop") < body.index("router_start")


def test_ramjet_apply_returns_non_zero_when_the_router_never_gets_healthy(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    router_bin = bin_dir / "fake-router"
    router_bin.write_text("#!/bin/bash\nexec sleep 30\n")
    router_bin.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/bash\nexit 7\n")
    curl.chmod(0o755)
    for name in ("logs", "run", "serving"):
        (tmp_path / name).mkdir()
    (tmp_path / "serving" / "replicas.txt").write_text("n1:8000\n")
    fragment = tmp_path / "router.sh"
    fragment.write_text(
        _env()
        .get_template("routers/ramjet.sh.j2")
        .render(router=RamjetRouter(binary=str(router_bin)))
    )

    result = subprocess.run(
        [
            "bash",
            "-euc",
            f'PATH="{bin_dir}:$PATH"; HOST_DIR={tmp_path}; SERVING_DIR={tmp_path}/serving; '
            f"WAIT_SEC=1; source {fragment}; "
            "if router_apply_replicas /dev/null /dev/null; then echo applied; "
            'else echo "failed rc=$?"; fi; echo still-running',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["failed rc=1", "still-running"]
    assert "ramjet did not become healthy" in result.stderr
    assert not (tmp_path / "run" / "router.pid").exists()


def test_rendered_lb_scripts_are_valid_bash(tmp_path):
    for router in (
        {"type": "nginx"},
        {"type": "vllm-router", "image": "/img/vr.sif"},
        {"type": "ramjet", "image": "/img/ramjet.sif"},
    ):
        script = tmp_path / f"{router['type']}.sh"
        script.write_text(_render_lb(router))
        subprocess.run(["bash", "-n", str(script)], check=True)
