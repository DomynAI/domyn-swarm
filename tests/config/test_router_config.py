# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pydantic import TypeAdapter, ValidationError
import pytest

from domyn_swarm.config.router import (
    THUNDERAGENT_POLICY,
    DynamoRouter,
    NginxRouter,
    RamjetRouter,
    RouterConfig,
    VllmRouter,
)
from domyn_swarm.config.slurm import SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig


def _swarm_payload(*, gpus_per_replica, gpus_per_node, router):
    return {
        "name": "s",
        "model": "m",
        "image": "v.sif",
        "replicas": 2,
        "gpus_per_replica": gpus_per_replica,
        "gpus_per_node": gpus_per_node,
        "backend": {
            "type": "slurm",
            "account": "a",
            "partition": "p",
            "qos": "q",
            "endpoint": {"nginx_image": "nginx.sif", "router": router},
        },
    }


def test_endpoint_router_defaults_to_nginx():
    ep = SlurmEndpointConfig(nginx_image="/x/nginx.sif")
    assert ep.router.type == "nginx"
    assert ep.router.serve_engine == "vllm"


def test_old_endpoint_payload_without_router_loads_as_nginx():
    ep = SlurmEndpointConfig.model_validate({"nginx_image": "/x/nginx.sif", "port": 9000})
    assert isinstance(ep.router, NginxRouter)


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "vllm-router"},
        {"type": "vllm-router", "image": "/a.sif", "binary": "vllm-router"},
    ],
)
def test_external_router_needs_exactly_one_of_image_or_binary(payload):
    with pytest.raises(ValidationError, match="exactly one of `image` or `binary`"):
        TypeAdapter(RouterConfig).validate_python(payload)


@pytest.mark.parametrize(
    ("policy", "args"),
    [
        ("cache_aware", ["--policy", "cache_aware"]),
        ("consistent_hash", ["--policy", "consistent_hash"]),
        (
            "program",
            [
                "--policy",
                "cache_aware",
                "--enable-program-scheduling",
                "--program-scheduling-config-json",
                '{"program_scheduling_enable_key":"auto","global_queue":true}',
            ],
        ),
        ("round_robin", ["--policy", "round_robin"]),
    ],
)
def test_vllm_router_policy_args(policy, args):
    assert VllmRouter(binary="vllm-router", policy=policy).policy_args() == args


def test_ramjet_policy_env():
    assert RamjetRouter(image="/r.sif").policy_env() == {"RJ_AFFINITY": "prefix"}
    assert RamjetRouter(image="/r.sif", policy="load").policy_env() == {"RJ_AFFINITY": "load"}


def test_dynamo_frontend_args_per_policy():
    assert DynamoRouter(policy="round_robin").frontend_args() == ["--router-mode", "round-robin"]
    assert DynamoRouter(policy="kv").frontend_args() == [
        "--router-mode",
        "kv",
        "--router-kv-events",
    ]
    assert DynamoRouter().frontend_args() == [
        "--router-mode",
        "kv",
        "--router-kv-events",
        "--router-session-affinity-mode",
        "soft",
        "--router-session-affinity-ttl-secs",
        "1800",
    ]
    ta = DynamoRouter(policy="thunderagent", session_affinity_ttl_s=600).frontend_args()
    assert ta[-2:] == ["--router-policy-config", str(THUNDERAGENT_POLICY)]
    assert "600" in ta


def test_dynamo_user_policy_config_replaces_shipped_file(tmp_path):
    own = tmp_path / "p.yaml"
    router = DynamoRouter(policy="thunderagent", policy_config=own)
    assert router.resolved_policy_config() == str(own)


@pytest.mark.parametrize(
    ("policy", "loaded"),
    [("thunderagent", True), ("kv_session_affinity", True), ("kv", False), ("round_robin", False)],
)
def test_dynamo_policy_config_reaches_only_the_session_policies(tmp_path, policy, loaded):
    own = str(tmp_path / "p.yaml")
    args = DynamoRouter(policy=policy, policy_config=own).frontend_args()
    assert (own in args) is loaded


@pytest.mark.parametrize(
    "router", [{"type": "vllm-router"}, {"type": "ramjet"}, {"type": "dynamo"}]
)
@pytest.mark.parametrize("key", ["RUST-LOG", "1LOG", "A B", "A=B", ""])
def test_router_env_keys_must_be_shell_identifiers(router, key):
    payload = {**router, "env": {key: "x"}}
    if router["type"] != "dynamo":
        payload["binary"] = "r"
    with pytest.raises(ValidationError, match="env"):
        TypeAdapter(RouterConfig).validate_python(payload)


def test_router_env_accepts_shell_identifiers():
    router = TypeAdapter(RouterConfig).validate_python(
        {"type": "dynamo", "env": {"DYN_LOG": "debug", "_x1": "y"}}
    )
    assert router.env == {"DYN_LOG": "debug", "_x1": "y"}


def test_thunderagent_policy_file_ships():
    assert THUNDERAGENT_POLICY.is_file()


def test_metrics_targets():
    assert NginxRouter().metrics_target() is None
    assert VllmRouter(binary="vllm-router").metrics_target() == ("127.0.0.1:9201", "/metrics")
    assert DynamoRouter(port=9300).metrics_target() == ("127.0.0.1:9300", "/metrics")


def test_dynamo_with_ray_is_rejected():
    payload = _swarm_payload(gpus_per_replica=8, gpus_per_node=4, router={"type": "dynamo"})
    with pytest.raises(ValidationError, match="Dynamo routing does not support Ray"):
        DomynLLMSwarmConfig.model_validate(payload)


def test_dynamo_without_ray_is_accepted():
    payload = _swarm_payload(gpus_per_replica=4, gpus_per_node=4, router={"type": "dynamo"})
    assert DomynLLMSwarmConfig.model_validate(payload).backend.endpoint.router.type == "dynamo"


@pytest.mark.parametrize(
    "args",
    [
        "--enable-auto-tool-choice --max-model-len 32768",
        "--max-model-len 32768 --tool-call-parser hermes",
        "--tool-call-parser=hermes",
        "--enable_auto_tool_choice",
        "--tool_call_parser=hermes",
    ],
)
def test_dynamo_rejects_vllm_tool_call_args(args):
    payload = _swarm_payload(gpus_per_replica=4, gpus_per_node=4, router={"type": "dynamo"})
    payload["args"] = args
    with pytest.raises(ValidationError, match="--dyn-tool-call-parser") as excinfo:
        DomynLLMSwarmConfig.model_validate(payload)
    assert "--dyn-reasoning-parser" in str(excinfo.value)


def test_dynamo_accepts_its_own_tool_call_args():
    payload = _swarm_payload(gpus_per_replica=4, gpus_per_node=4, router={"type": "dynamo"})
    payload["args"] = "--dyn-tool-call-parser hermes --dyn-reasoning-parser qwen3"
    assert DomynLLMSwarmConfig.model_validate(payload).args == payload["args"]


def test_vllm_tool_call_args_stay_valid_without_dynamo():
    payload = _swarm_payload(gpus_per_replica=4, gpus_per_node=4, router={"type": "nginx"})
    payload["args"] = "--enable-auto-tool-choice --tool-call-parser hermes"
    assert DomynLLMSwarmConfig.model_validate(payload).args == payload["args"]
