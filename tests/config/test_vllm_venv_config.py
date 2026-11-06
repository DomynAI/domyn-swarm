# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError
import pytest

from domyn_swarm.config.preflight import check_config_paths
from domyn_swarm.config.swarm import DomynLLMSwarmConfig


def _cfg(tmp_path, *, image: str | None = None, **overrides) -> DomynLLMSwarmConfig:
    nginx_image = tmp_path / "nginx.sif"
    nginx_image.touch()
    data = {
        "name": "s",
        "model": "org/model",
        "backend": {
            "type": "slurm",
            "account": "a",
            "partition": "p",
            "qos": "q",
            "endpoint": {"nginx_image": str(nginx_image)},
            **overrides.pop("backend", {}),
        },
        **overrides,
    }
    if image is not None:
        data["image"] = image
    return DomynLLMSwarmConfig.model_validate(data)


def _venv(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "vllm").touch()
    return venv


def test_slurm_without_image_or_venv_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match=r"backend\.vllm_venv"):
        _cfg(tmp_path)


def test_slurm_venv_does_not_need_an_image(tmp_path):
    cfg = _cfg(tmp_path, backend={"vllm_venv": str(_venv(tmp_path))})

    assert cfg.image is None
    assert check_config_paths(cfg) == []


def test_venv_takes_precedence_over_an_image_from_defaults(tmp_path):
    """A site-wide default image must not block a bare-metal deployment, nor be checked."""
    cfg = _cfg(
        tmp_path,
        image="/site/images/missing.sif",
        backend={"vllm_venv": str(_venv(tmp_path))},
    )

    assert check_config_paths(cfg) == []


def test_venv_without_vllm_is_reported(tmp_path):
    venv = tmp_path / "empty-venv"
    venv.mkdir()

    problems = check_config_paths(_cfg(tmp_path, backend={"vllm_venv": str(venv)}))

    assert [p.field for p in problems] == ["backend.vllm_venv"]
    assert problems[0].value == str(venv / "bin" / "vllm")


def test_venv_rejects_replicas_spanning_several_nodes(tmp_path):
    with pytest.raises(ValidationError, match="spanning several nodes"):
        _cfg(
            tmp_path,
            gpus_per_replica=8,
            gpus_per_node=4,
            replicas=1,
            backend={"vllm_venv": str(_venv(tmp_path))},
        )
