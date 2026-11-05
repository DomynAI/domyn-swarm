# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Settings must be read when used, not captured at import time."""

import pytest

from domyn_swarm.config.settings import Settings, get_settings, reload_settings_cache


def test_api_token_set_after_import_reaches_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token exported after `domyn_swarm` is imported must still authenticate.

    Regression test for a module-scope ``settings = get_settings()`` binding,
    which freezes the token as of first import.
    """
    import domyn_swarm.jobs.api.base as base
    from domyn_swarm.jobs.api.chat_completion import ChatCompletionJob

    captured: dict = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(base, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setenv("DOMYN_SWARM_API_TOKEN", "token-set-after-import")
    reload_settings_cache()

    ChatCompletionJob(model="some-model", endpoint="http://localhost:8000")

    assert captured["default_headers"]["Authorization"] == "Bearer token-set-after-import"


def test_resolved_api_token_prefers_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """``api_token`` wins over the vLLM-compatible aliases."""
    monkeypatch.setenv("DOMYN_SWARM_API_TOKEN", "primary")
    monkeypatch.setenv("VLLM_API_KEY", "secondary")
    reload_settings_cache()

    token = get_settings().resolved_api_token

    assert token is not None
    assert token.get_secret_value() == "primary"


def test_resolved_api_token_accepts_apptainer_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOMYN_SWARM_API_TOKEN", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("SINGULARITYENV_VLLM_API_KEY", raising=False)
    monkeypatch.setenv("APPTAINERENV_VLLM_API_KEY", "apptainer-token")
    reload_settings_cache()

    token = get_settings().resolved_api_token

    assert token is not None
    assert token.get_secret_value() == "apptainer-token"


def test_resolved_api_token_keeps_singularity_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOMYN_SWARM_API_TOKEN", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("APPTAINERENV_VLLM_API_KEY", raising=False)
    monkeypatch.setenv("SINGULARITYENV_VLLM_API_KEY", "singularity-token")
    reload_settings_cache()

    token = get_settings().resolved_api_token

    assert token is not None
    assert token.get_secret_value() == "singularity-token"


def test_resolved_api_token_is_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No token configured resolves to ``None`` rather than an empty secret."""
    monkeypatch.delenv("DOMYN_SWARM_API_TOKEN", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("SINGULARITYENV_VLLM_API_KEY", raising=False)
    monkeypatch.delenv("APPTAINERENV_VLLM_API_KEY", raising=False)
    reload_settings_cache()

    assert Settings().resolved_api_token is None


def test_no_module_scope_settings_bindings() -> None:
    """No module may capture ``get_settings()`` at import time.

    A module-scope binding cannot be refreshed by ``reload_settings_cache()``,
    so it silently freezes configuration as of first import. This guard has a
    history: the binding pattern was removed once and came back.
    """
    import ast
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parents[2] / "src" / "domyn_swarm"
    offenders: list[str] = []

    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # module scope only, not nested bodies
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "get_settings":
                offenders.append(f"{path.relative_to(src_root)}:{node.lineno}")

    assert offenders == [], (
        "get_settings() must be called at the point of use, not bound at module "
        f"scope. Offending bindings: {offenders}"
    )
