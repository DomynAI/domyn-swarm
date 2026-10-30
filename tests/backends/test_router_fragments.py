# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import jinja2
import pytest

import domyn_swarm
from domyn_swarm.config.router import THUNDERAGENT_POLICY, RamjetRouter, VllmRouter
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.helpers.data import get_device_slices
from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.runtime import watchdog_args

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


def _render_replicas(router: dict) -> str:
    cfg = _cfg(router)
    return (
        _env()
        .get_template(cfg.backend.template_path.name)
        .render(
            cfg=cfg,
            job_name="job",
            path_exists=path_exists,
            is_folder=is_folder,
            cuda_visible_devices=get_device_slices(cfg.gpus_per_node, cfg.gpus_per_replica),
            swarm_directory="/swarm/s1",
            watchdog_script_path="/opt/watchdog.py",
            build_watchdog_args=watchdog_args.build_watchdog_args,
            args_to_str=watchdog_args.args_to_str,
            dswarm_agent_version=domyn_swarm.__version__,
            dcgm_core_counters_path="/dcgm/core.csv",
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
    for name in ("logs", "run", "serving"):
        (tmp_path / name).mkdir()
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
            'kill -0 "$(cat "$ROUTER_PID_FILE")"; router_stop; router_stop',
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
    assert not (tmp_path / "run" / "router.pid").exists()


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


def test_shutdown_stops_the_applier_before_the_router():
    out = _render_lb({"type": "ramjet", "binary": "ramjet"})
    shutdown = out[out.rindex("trap '") :]
    assert shutdown.index("kill ${APPLY_PID:-}") < shutdown.index("router_stop")


def test_shutdown_stops_the_router_when_the_applier_already_exited(tmp_path):
    out = _render_lb({"type": "ramjet", "binary": "ramjet"})
    shutdown = re.search(r"^trap '(echo \"\[lb\] shutting down\";.*)' EXIT SIGTERM$", out, re.M)
    assert shutdown is not None
    stopped = tmp_path / "stopped"
    script = f"""
set -eu
router_stop() {{ touch {stopped}; }}
singularity() {{ :; }}
APPLY_PID=$(sh -c 'echo $$')
COLLECTOR_PID=$APPLY_PID SUPERVISOR_PID=$APPLY_PID INSTANCE_NAME=lb
{shutdown.group(1)}
"""
    subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10)
    assert stopped.exists()


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
        {"type": "dynamo", "policy": "thunderagent", "venv": "/shared/dyn-venv"},
    ):
        script = tmp_path / f"{router['type']}.sh"
        script.write_text(_render_lb(router))
        subprocess.run(["bash", "-n", str(script)], check=True)


def _run_auth_block(
    tmp_path: Path,
    key: str,
    env_name: str = "VLLM_API_KEY",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out = _render_lb({"type": "vllm-router", "binary": "vllm-router"})
    start = out.index("# begin api key check")
    end = out.index("# end api key check")
    script = tmp_path / "auth.sh"
    script.write_text("set -eu\n" + out[start:end])
    (tmp_path / "serving").mkdir(exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "SERVING_DIR": str(tmp_path / "serving"),
        env_name: key,
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, check=False
    )


def test_router_mode_writes_auth_map_and_rejects_unsafe_keys():
    out = _render_lb({"type": "vllm-router", "binary": "vllm-router"})
    assert "05-auth.conf" in out
    assert "umask 077" in out
    assert "^[A-Za-z0-9._~+/=-]+$" in out


def test_nginx_mode_writes_no_auth_map():
    assert "05-auth.conf" not in _render_lb({"type": "nginx"})


@pytest.mark.parametrize("key", ['ab"cd', "ab$cd", "ab cd", "ab;cd", "ab}cd", "ab\\cd"])
def test_unsafe_api_key_stops_the_job(tmp_path, key):
    result = _run_auth_block(tmp_path, key)
    assert result.returncode == 1
    assert "API key holds characters" in result.stderr
    assert not (tmp_path / "serving" / "05-auth.conf").exists()


def test_safe_api_key_writes_bearer_map(tmp_path):
    result = _run_auth_block(tmp_path, "sk-abc.DEF_123")
    assert result.returncode == 0
    conf = tmp_path / "serving" / "05-auth.conf"
    assert '"~^Bearer sk-abc\\.DEF_123$" 1;' in conf.read_text()
    assert conf.stat().st_mode & 0o077 == 0


def test_empty_api_key_writes_open_map(tmp_path):
    result = _run_auth_block(tmp_path, "")
    assert result.returncode == 0
    text = (tmp_path / "serving" / "05-auth.conf").read_text()
    assert "default 1;" in text
    assert "Bearer" not in text


def _written_pattern(tmp_path: Path) -> str:
    text = (tmp_path / "serving" / "05-auth.conf").read_text()
    return re.search(r'"~(.*?)" 1;', text).group(1)


@pytest.mark.parametrize(
    ("key", "accepted", "rejected"),
    [
        (
            "sk-abc.DEF_123",
            ["Bearer sk-abc.DEF_123"],
            [
                "bearer SK-ABC.def_123",
                "Bearer sk-abcXDEF_123",
                "Bearer sk-abc.DEF_1234",
                "Bearer sk-abc.DEF_12",
            ],
        ),
        ("a+b.c", ["Bearer a+b.c"], ["Bearer aab.c", "Bearer aaab.c", "bearer a+b.c"]),
    ],
)
def test_bearer_pattern_matches_only_the_exact_key(tmp_path, key, accepted, rejected):
    _run_auth_block(tmp_path, key)
    pattern = _written_pattern(tmp_path)
    assert all(re.search(pattern, value) for value in accepted)
    assert not any(re.search(pattern, value) for value in rejected)


def test_singularity_env_key_wins_over_vllm_api_key(tmp_path):
    _run_auth_block(
        tmp_path, "from-singularity", "SINGULARITYENV_VLLM_API_KEY", {"VLLM_API_KEY": "from-vllm"}
    )
    text = (tmp_path / "serving" / "05-auth.conf").read_text()
    assert "from-singularity" in text
    assert "from-vllm" not in text


def _bind_list(script: str, command: str) -> list[str]:
    """Return the `--bind` entries of the `singularity exec` that runs `command`."""
    head = script[: script.index(command)]
    exec_line = head[head.rindex("singularity exec") :].split("\n", 1)[0]
    bind = re.search(r"--bind (\S+)", exec_line).group(1)
    return [entry.strip("\"'") for entry in bind.split(",")]


def test_vllm_engine_render_is_unchanged():
    out = _render_replicas({"type": "nginx"})
    assert "vllm serve" in out
    assert "dynamo.vllm" not in out


def test_dynamo_worker_command():
    out = _render_replicas({"type": "dynamo"})
    assert "python3 -m dynamo.vllm --model" in out
    assert "--namespace job" in out
    assert "--discovery-backend file --request-plane tcp --event-plane zmq" in out
    assert "SINGULARITYENV_PYTHONHASHSEED=0" in out
    assert "tcp://*:$(( 20080 + global_id ))" in out
    assert "vllm serve" not in out


def test_dynamo_worker_serves_health_on_the_replica_port():
    out = _render_replicas({"type": "dynamo"})
    assert "SINGULARITYENV_DYN_SYSTEM_PORT=$port" in out
    assert 'SINGULARITYENV_DYN_FILE_KV="$SWARM_DIR/dynamo-kv"' in out
    assert 'mkdir -p "$SWARM_DIR/dynamo-kv"' in out
    assert 'echo "$endpoint" > "$head_file"' in out


def _start_dynamo_replica(tmp_path: Path, getent_output: str) -> subprocess.CompletedProcess:
    """Run `start_vllm_replica 3 8003 node7` against stub `srun` and `getent`."""
    out = _render_replicas({"type": "dynamo"})
    start = out.index("function start_vllm_replica() {")
    function = out[start : out.index("\n}\n", start) + 3]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "srun").write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" > {tmp_path}/argv\nenv > {tmp_path}/env\n'
    )
    (bin_dir / "getent").write_text(f"#!/bin/bash\nprintf '{getent_output}'\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            f'PATH="{bin_dir}:$PATH"; LOG_DIR={tmp_path}/logs; SWARM_DIR={tmp_path}; '
            "GPUS_PER_NODE=4 GPUS_PER_REPLICA=4 MOUNTS=/m VLLM_IMAGE=/img.sif "
            "COLLECTOR_HOST=c COLLECTOR_PORT=1 DSWARM_AGENT_VERSION=0; "
            f'{function}start_vllm_replica 3 8003 node7; wait "$!"',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_dynamo_worker_environment_and_kv_events_endpoint(tmp_path):
    result = _start_dynamo_replica(
        tmp_path, "10.2.1.45       STREAM node7\\n10.2.1.45       DGRAM\\n"
    )
    assert result.returncode == 0, result.stderr
    env = dict(
        line.split("=", 1) for line in (tmp_path / "env").read_text().splitlines() if "=" in line
    )
    assert env["SINGULARITYENV_DYN_SYSTEM_PORT"] == "8003"
    assert env["SINGULARITYENV_DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] == '["generate"]'
    assert env["SINGULARITYENV_DYN_TCP_RPC_HOST"] == "10.2.1.45"
    assert env["SINGULARITYENV_DYN_FILE_KV"] == f"{tmp_path}/dynamo-kv"
    assert (tmp_path / "dynamo-kv").is_dir()
    argv = (tmp_path / "argv").read_text().splitlines()
    kv_events = json.loads(argv[argv.index("--kv-events-config") + 1])
    assert kv_events == {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "topic": "",
        "endpoint": "tcp://*:20083",
    }


def test_dynamo_worker_takes_the_rpc_host_from_an_ipv4_address():
    out = _render_replicas({"type": "dynamo"})
    assert 'getent ahostsv4 "$node"' in out
    assert "getent hosts" not in out


def test_dynamo_worker_stops_without_an_ipv4_address(tmp_path):
    result = _start_dynamo_replica(tmp_path, "")
    assert result.returncode == 1
    assert "node7 has no IPv4 address" in result.stderr
    assert not (tmp_path / "argv").exists()


def test_dynamo_worker_uses_the_venv():
    out = _render_replicas({"type": "dynamo", "venv": "/shared/dyn-venv"})
    assert "/shared/dyn-venv/bin/python -m dynamo.vllm" in out
    assert "/shared/dyn-venv" in _bind_list(out, "-m dynamo.vllm")


def test_dynamo_round_robin_skips_kv_events():
    out = _render_replicas({"type": "dynamo", "policy": "round_robin"})
    assert "-m dynamo.vllm" in out
    assert "--kv-events-config" not in out


def test_rendered_replica_scripts_are_valid_bash(tmp_path):
    for name, router in (
        ("nginx", {"type": "nginx"}),
        ("dynamo", {"type": "dynamo", "venv": "/shared/dyn-venv"}),
        ("dynamo-rr", {"type": "dynamo", "policy": "round_robin"}),
    ):
        script = tmp_path / f"{name}.sh"
        script.write_text(_render_replicas(router))
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_dynamo_frontend_command():
    out = _render_lb({"type": "dynamo", "policy": "thunderagent"})
    assert "-m dynamo.frontend --http-host 127.0.0.1 --http-port $ROUTER_PORT" in out
    assert "--namespace job" in out
    assert "--discovery-backend file --request-plane tcp --event-plane zmq" in out
    assert "DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS=1" in out
    assert "dynamo_thunderagent.yaml" in out
    assert "env -u DYN_SYSTEM_PORT" in out
    assert "--router-address 127.0.0.1:$ROUTER_PORT" in out


def test_dynamo_frontend_binds_the_venv_and_the_policy_file():
    out = _render_lb({"type": "dynamo", "policy": "thunderagent", "venv": "/shared/dyn-venv"})
    assert "/shared/dyn-venv/bin/python -m dynamo.frontend" in out
    assert _bind_list(out, "-m dynamo.frontend") == [
        "$HOST_DIR",
        "/shared/dyn-venv",
        str(THUNDERAGENT_POLICY),
    ]


def test_dynamo_frontend_without_policy_file_binds_only_the_swarm_directory():
    out = _render_lb({"type": "dynamo", "policy": "kv"})
    assert "python3 -m dynamo.frontend" in out
    assert _bind_list(out, "-m dynamo.frontend") == ["$HOST_DIR"]


def test_dynamo_apply_restarts_the_frontend():
    out = _render_lb({"type": "dynamo"})
    body = _function_body(out, "router_apply_replicas")
    assert body.index("router_stop") < body.index("( router_start )")


def _run_dynamo_frontend(tmp_path: Path, health_check: str, script: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    singularity = bin_dir / "singularity"
    singularity.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" > {tmp_path}/argv\nenv > {tmp_path}/env\nexec sleep 30\n'
    )
    singularity.chmod(0o755)
    curl = bin_dir / "curl"
    curl.write_text(f"#!/bin/bash\n{health_check}\n")
    curl.chmod(0o755)
    for name in ("logs", "run"):
        (tmp_path / name).mkdir()
    cfg = _cfg({"type": "dynamo", "env": {"DYN_LOG": "info debug"}})
    fragment = tmp_path / "router.sh"
    fragment.write_text(
        _env()
        .get_template("routers/dynamo.sh.j2")
        .render(router=cfg.backend.endpoint.router, cfg=cfg, job_name="job")
    )
    return subprocess.run(
        [
            "bash",
            "-euc",
            f'PATH="{bin_dir}:$PATH"; HOST_DIR={tmp_path}; WAIT_SEC=1; '
            f"export DYN_SYSTEM_PORT=8300; source {fragment}; {script}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_dynamo_frontend_start_sets_the_frontend_environment(tmp_path):
    result = _run_dynamo_frontend(
        tmp_path,
        health_check=f"test -s {tmp_path}/env",
        script='router_start; test -s "$ROUTER_PID_FILE"; router_stop',
    )
    assert result.returncode == 0, result.stderr
    env = dict(
        line.split("=", 1) for line in (tmp_path / "env").read_text().splitlines() if "=" in line
    )
    assert env["DYN_FILE_KV"] == f"{tmp_path}/dynamo-kv"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS"] == "1"
    assert env["DYN_LOG"] == "info debug"
    assert "DYN_SYSTEM_PORT" not in env
    assert (tmp_path / "dynamo-kv").is_dir()
    assert not (tmp_path / "run" / "router.pid").exists()
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[:3] == ["exec", "--bind", str(tmp_path)]


def test_dynamo_apply_returns_non_zero_when_the_frontend_never_gets_healthy(tmp_path):
    result = _run_dynamo_frontend(
        tmp_path,
        health_check="exit 7",
        script=(
            "if router_apply_replicas /dev/null /dev/null; then echo applied; "
            'else echo "failed rc=$?"; fi; echo still-running'
        ),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["failed rc=1", "still-running"]
    assert "dynamo frontend did not become healthy" in result.stderr
    assert not (tmp_path / "run" / "router.pid").exists()


def _snapshot(tmp_path: Path, router: dict) -> str:
    """Run the rendered lb.sh `replica_snapshot` with the router's fragment sourced."""
    out = _render_lb(router)
    fragment_start = out.index("router_registrations() { :; }")
    fragment = out[fragment_start : out.index("replica_snapshot() {")]
    script = tmp_path / "snapshot.sh"
    script.write_text(
        f"{fragment}replica_snapshot() {{\n{_function_body(out, 'replica_snapshot')}\n}}\n"
    )
    return subprocess.run(
        [
            "bash",
            "-euc",
            f"HOST_DIR={tmp_path}; SERVING_DIR={tmp_path}/serving; "
            f"source {script}; replica_snapshot",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _instances(tmp_path: Path) -> Path:
    instances = tmp_path / "dynamo-kv" / "v1" / "instances"
    instances.mkdir(parents=True)
    return instances


def test_dynamo_registrations_list_generate_instances_sorted(tmp_path):
    instances = _instances(tmp_path)
    for name in (
        "job%2Fbackend%2Fgenerate%2Fbbbb",
        "job%2Fbackend%2Fclear%5Fkv%5Fblocks%2Faaaa",
        "job%2Fbackend%2Fgenerate%2Faaaa",
    ):
        (instances / name).touch()
    result = _run_dynamo_frontend(
        tmp_path, health_check="exit 0", script="router_registrations; echo rc=$?"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "job%2Fbackend%2Fgenerate%2Faaaa",
        "job%2Fbackend%2Fgenerate%2Fbbbb",
        "rc=0",
    ]


def test_dynamo_registrations_print_nothing_without_a_store(tmp_path):
    result = _run_dynamo_frontend(
        tmp_path, health_check="exit 0", script="router_registrations; echo rc=$?"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "rc=0\n"


def test_a_new_dynamo_instance_changes_the_snapshot_of_an_unchanged_replica_list(tmp_path):
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "replicas.txt").write_text("n1:8000\nn2:8001\n")
    instances = _instances(tmp_path)
    (instances / "job%2Fbackend%2Fgenerate%2Faaaa").touch()
    before = _snapshot(tmp_path, {"type": "dynamo"})

    (instances / "job%2Fbackend%2Fgenerate%2Fcccc").touch()
    after = _snapshot(tmp_path, {"type": "dynamo"})

    assert before == "n1:8000\nn2:8001\njob%2Fbackend%2Fgenerate%2Faaaa\n"
    assert after != before
    assert after.startswith("n1:8000\nn2:8001\n")


@pytest.mark.parametrize(
    "router",
    [
        {"type": "nginx"},
        {"type": "vllm-router", "binary": "vllm-router"},
        {"type": "ramjet", "image": "/r.sif"},
    ],
    ids=["nginx", "vllm-router", "ramjet"],
)
def test_snapshot_is_the_replica_list_for_routers_without_registrations(tmp_path, router):
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "replicas.txt").write_text("n1:8000\nn2:8001\n")
    _instances(tmp_path)

    assert _snapshot(tmp_path, router) == "n1:8000\nn2:8001\n"


def test_applier_and_initial_record_use_the_snapshot():
    out = _render_lb({"type": "dynamo"})
    assert 'replica_snapshot >"$HOST_DIR/run/replicas.applied"' in out
    assert 'replica_snapshot >"$APPLIED_FILE.next"' in out
    assert 'cmp -s "$APPLIED_FILE.next" "$APPLIED_FILE"' in out
    assert 'cp "$REPLICAS_FILE"' not in out


def _run_applier(
    tmp_path: Path,
    router: dict,
    router_body: str,
    script: str,
    stubs: dict[str, str] | None = None,
):
    """Run `script` with the router fragment, `replica_snapshot` and `applier_pass` sourced.

    `router_body` runs as the router process, both for a `binary` router and,
    through a stub `singularity`, for Dynamo. The stub `curl` reports the router
    healthy at once. `stubs` adds more commands to PATH, by name and body.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {"fake-router": router_body, "singularity": router_body, "curl": "exit 0"}
    for name, body in {**commands, **(stubs or {})}.items():
        (bin_dir / name).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    for name in ("logs", "run", "serving"):
        (tmp_path / name).mkdir()
    (tmp_path / "serving" / "replicas.txt").write_text("n1:8000\nn2:8001\n")
    if router["type"] != "dynamo":
        router = {**router, "binary": str(bin_dir / "fake-router")}
    out = _render_lb(router)
    fragment = out[out.index("router_registrations() { :; }") : out.index("replica_snapshot() {")]
    functions = "".join(
        f"{name}() {{\n{_function_body(out, name)}\n}}\n"
        for name in ("replica_snapshot", "applier_pass")
    )
    lib = tmp_path / "applier.sh"
    lib.write_text(fragment + functions)
    return subprocess.run(
        [
            "bash",
            "-euc",
            f'PATH="{bin_dir}:$PATH"; HOST_DIR={tmp_path}; SERVING_DIR={tmp_path}/serving; '
            "WAIT_SEC=5; REPLICAS_FILE=$SERVING_DIR/replicas.txt; "
            f"APPLIED_FILE=$HOST_DIR/run/replicas.applied; source {lib}; {script}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("router_type", ["vllm-router", "ramjet", "dynamo"])
def test_applier_restarts_a_router_that_exited(tmp_path, router_type):
    starts = tmp_path / "starts"
    result = _run_applier(
        tmp_path,
        {"type": router_type},
        router_body=(
            f'echo start >> {starts}\n[ "$(wc -l < {starts})" -ge 2 ] && exec sleep 30\nexit 0'
        ),
        script=(
            'router_start; pid="$(cat "$ROUTER_PID_FILE")"; '
            'for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done; '
            "echo n3:8002 >> $REPLICAS_FILE; applier_pass; "
            'kill -0 "$(cat "$ROUTER_PID_FILE")"; router_stop'
        ),
    )

    assert result.returncode == 0, result.stderr
    assert starts.read_text().splitlines() == ["start", "start"]
    assert f"[lb] WARNING: {router_type} exited; restarting" in result.stderr
    assert (tmp_path / "run" / "replicas.applied").read_text() == "n1:8000\nn2:8001\nn3:8002\n"
    assert not (tmp_path / "run" / "router.pid").exists()


def test_applier_leaves_a_live_router_running(tmp_path):
    starts = tmp_path / "starts"
    result = _run_applier(
        tmp_path,
        {"type": "ramjet"},
        router_body=f"echo start >> {starts}\nexec sleep 30",
        script=(
            "router_start; replica_snapshot > $APPLIED_FILE; applier_pass; "
            'kill -0 "$(cat "$ROUTER_PID_FILE")"; router_stop'
        ),
    )

    assert result.returncode == 0, result.stderr
    assert starts.read_text().splitlines() == ["start"]
    assert "exited" not in result.stderr


def test_nginx_applier_does_not_watch_a_router_process():
    body = _function_body(_render_lb({"type": "nginx"}), "applier_pass")
    assert "ROUTER_PID_FILE" not in body
    assert "router_start" not in body


def test_nginx_applier_does_not_report_a_router_apply():
    assert "applied replica set" not in _render_lb({"type": "nginx"})
    assert "applied replica set to the router" in _render_lb({"type": "ramjet", "binary": "r"})


@pytest.mark.parametrize("router_type", ["vllm-router", "ramjet", "dynamo"])
def test_initial_snapshot_is_recorded_before_the_first_router_start(router_type):
    router = {"type": router_type}
    if router_type != "dynamo":
        router["binary"] = router_type
    out = _render_lb(router)
    assert out.index('replica_snapshot >"$HOST_DIR/run/replicas.applied"') < out.index(
        "\nrouter_start\n"
    )


@pytest.mark.parametrize("router_type", ["vllm-router", "ramjet", "dynamo"])
def test_router_stop_kills_a_router_that_ignores_sigterm(tmp_path, router_type):
    real_sleep = shutil.which("sleep")
    result = _run_applier(
        tmp_path,
        {"type": router_type},
        router_body=f"trap '' TERM\nexec {real_sleep} 30",
        script=(
            'router_start; pid="$(cat "$ROUTER_PID_FILE")"; router_stop; '
            f'for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; {real_sleep} 0.1; done; '
            'if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid"; echo alive; '
            "else echo stopped; fi"
        ),
        stubs={"sleep": "exit 0"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "stopped"
    assert not (tmp_path / "run" / "router.pid").exists()
