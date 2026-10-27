# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``up``-time filesystem path preflight."""

import pytest

from domyn_swarm.config.preflight import PathProblem, check_config_paths
from domyn_swarm.config.swarm import DomynLLMSwarmConfig


def _cfg(tmp_path, **overrides):
    """Build a validated Slurm swarm config whose paths all exist by default.

    Args:
        tmp_path: pytest tmp dir used to materialise the default image files.
        **overrides: Top-level config keys to replace; ``backend`` is merged
            one level deep so a case can override just ``endpoint``.

    Returns:
        A validated :class:`DomynLLMSwarmConfig`.
    """
    image = tmp_path / "vllm.sif"
    image.touch()
    nginx_image = tmp_path / "nginx.sif"
    nginx_image.touch()

    backend = {
        "type": "slurm",
        "account": "a",
        "partition": "p",
        "qos": "q",
        "endpoint": {"nginx_image": str(nginx_image)},
    }
    backend.update(overrides.pop("backend", {}))

    data = {
        "name": "s",
        "model": "nvidia/Nemotron-Nano-9B-v2",
        "image": str(image),
        "backend": backend,
    }
    data.update(overrides)
    return DomynLLMSwarmConfig.model_validate(data)


def _fields(problems: list[PathProblem]) -> list[str]:
    """Return just the dotted field names of a problem list."""
    return [p.field for p in problems]


# --- happy path -------------------------------------------------------------


def test_config_with_existing_paths_has_no_problems(tmp_path):
    assert check_config_paths(_cfg(tmp_path)) == []


# --- images -----------------------------------------------------------------


def test_missing_image_is_reported(tmp_path):
    cfg = _cfg(tmp_path, image="/shared/images/vllm.sif")

    problems = check_config_paths(cfg)

    assert _fields(problems) == ["image"]
    assert problems[0].value == "/shared/images/vllm.sif"
    assert "does not exist" in problems[0].reason


def test_problem_hint_names_deepest_existing_parent(tmp_path):
    """The hint must pinpoint which path segment first diverges from reality."""
    cfg = _cfg(tmp_path, image=str(tmp_path / "images" / "vllm.sif"))

    (problem,) = check_config_paths(cfg)

    assert problem.hint is not None
    assert str(tmp_path) in problem.hint
    assert "images" not in problem.hint


def test_missing_nginx_image_is_reported(tmp_path):
    cfg = _cfg(tmp_path, backend={"endpoint": {"nginx_image": "/shared/images/nginx.sif"}})

    assert _fields(check_config_paths(cfg)) == ["backend.endpoint.nginx_image"]


@pytest.mark.parametrize(
    "ref",
    [
        "docker://vllm/vllm-openai:latest",
        "oras://registry.example.com/vllm:0.28.0",
        "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04",
        "vllm.sif",
    ],
)
def test_non_path_image_refs_are_skipped(tmp_path, ref):
    """Registry refs and bare names are not filesystem paths and must not fail."""
    assert check_config_paths(_cfg(tmp_path, image=ref)) == []


def test_unexpandable_env_var_in_path_is_skipped(tmp_path, monkeypatch):
    """A var set only on the compute node cannot be resolved here, so don't guess."""
    monkeypatch.delenv("SITE_IMAGES", raising=False)

    assert check_config_paths(_cfg(tmp_path, image="$SITE_IMAGES/vllm.sif")) == []


def test_env_var_in_path_is_expanded_before_checking(tmp_path, monkeypatch):
    monkeypatch.setenv("SITE_IMAGES", str(tmp_path))

    cfg = _cfg(tmp_path, image="$SITE_IMAGES/missing.sif")
    (problem,) = check_config_paths(cfg)

    assert problem.value == str(tmp_path / "missing.sif")


# --- monitoring sidecars ----------------------------------------------------


def test_monitoring_images_are_skipped_when_monitoring_disabled(tmp_path):
    cfg = _cfg(
        tmp_path,
        backend={
            "endpoint": {
                "nginx_image": str(tmp_path / "nginx.sif"),
                "monitoring": {
                    "enabled": False,
                    "prometheus_image": "/shared/images/prometheus.sif",
                },
            }
        },
    )

    assert check_config_paths(cfg) == []


def test_monitoring_images_are_checked_when_enabled(tmp_path):
    cfg = _cfg(
        tmp_path,
        backend={
            "endpoint": {
                "nginx_image": str(tmp_path / "nginx.sif"),
                "monitoring": {
                    "enabled": True,
                    "mode": "container",
                    "prometheus_image": "/shared/images/prometheus.sif",
                    "nginx_exporter_image": "/shared/images/exporter.sif",
                },
            }
        },
    )

    assert _fields(check_config_paths(cfg)) == [
        "backend.endpoint.monitoring.prometheus_image",
        "backend.endpoint.monitoring.nginx_exporter_image",
    ]


def test_monitoring_images_are_skipped_in_binary_mode(tmp_path):
    cfg = _cfg(
        tmp_path,
        backend={
            "endpoint": {
                "nginx_image": str(tmp_path / "nginx.sif"),
                "monitoring": {
                    "enabled": True,
                    "mode": "binary",
                    "prometheus_image": "/shared/images/prometheus.sif",
                },
            }
        },
    )

    assert check_config_paths(cfg) == []


def test_gpu_exporter_image_is_checked_when_enabled(tmp_path):
    cfg = _cfg(
        tmp_path,
        backend={
            "endpoint": {
                "nginx_image": str(tmp_path / "nginx.sif"),
                "monitoring": {
                    "enabled": True,
                    "mode": "container",
                    "prometheus_image": str(tmp_path / "vllm.sif"),
                    "nginx_exporter_image": str(tmp_path / "vllm.sif"),
                    "gpu_exporter": {"enabled": True, "image": "/shared/images/gpu.sif"},
                },
            }
        },
    )

    assert _fields(check_config_paths(cfg)) == ["backend.endpoint.monitoring.gpu_exporter.image"]


# --- model ------------------------------------------------------------------


def test_hugging_face_repo_id_model_is_not_checked(tmp_path):
    assert check_config_paths(_cfg(tmp_path, model="nvidia/Nemotron-Nano-9B-v2")) == []


def test_missing_local_model_directory_is_reported(tmp_path):
    cfg = _cfg(tmp_path, model="/shared/models/models--nvidia--Nemotron-Nano-9B-v2")

    assert _fields(check_config_paths(cfg)) == ["model"]


def test_existing_local_model_directory_passes(tmp_path):
    model_dir = tmp_path / "models--nvidia--Nemotron-Nano-9B-v2"
    model_dir.mkdir()

    assert check_config_paths(_cfg(tmp_path, model=str(model_dir))) == []


# --- mounts and venv --------------------------------------------------------


def test_missing_mount_source_is_reported(tmp_path):
    cfg = _cfg(tmp_path, backend={"mounts": ["/shared/data:/data:ro"]})

    (problem,) = check_config_paths(cfg)

    assert problem.field == "backend.mounts[0]"
    assert problem.value == "/shared/data"


def test_existing_mount_source_passes(tmp_path):
    assert check_config_paths(_cfg(tmp_path, backend={"mounts": [str(tmp_path)]})) == []


def test_system_mounts_are_not_checked(tmp_path):
    """They are site defaults, skipped-with-warning on the node by design."""
    cfg = _cfg(tmp_path, backend={"system_mounts": ["/shared/prod/opt"]})

    assert check_config_paths(cfg) == []


def test_missing_venv_path_is_reported(tmp_path):
    cfg = _cfg(tmp_path, backend={"venv_path": "/shared/venv"})

    assert _fields(check_config_paths(cfg)) == ["backend.venv_path"]


def test_unset_venv_path_is_skipped(tmp_path):
    cfg = _cfg(tmp_path)

    assert cfg.backend.venv_path is None
    assert check_config_paths(cfg) == []


# --- other backends ---------------------------------------------------------


def test_lepton_backend_is_not_path_checked(monkeypatch):
    """Lepton images are Docker refs and its paths live on platform volumes."""
    cfg = DomynLLMSwarmConfig.model_validate(
        {
            "name": "s",
            "model": "/shared/models/nemotron",
            "image": "/shared/images/vllm.sif",
            "backend": {
                "type": "lepton",
                "resource_shape": "gpu.1xh200",
            },
        }
    )

    assert check_config_paths(cfg) == []


def test_config_without_a_backend_attribute_is_tolerated():
    """The CLI passes whatever the loader returned; never crash on a stub."""

    class Stub:
        pass

    assert check_config_paths(Stub()) == []


# --- reporting --------------------------------------------------------------


def test_all_problems_are_reported_together(tmp_path):
    """One run should list every bad path, not just the first."""
    cfg = _cfg(
        tmp_path,
        image="/shared/images/vllm.sif",
        model="/shared/models/nemotron",
        backend={"endpoint": {"nginx_image": "/shared/images/nginx.sif"}},
    )

    assert set(_fields(check_config_paths(cfg))) == {
        "image",
        "model",
        "backend.endpoint.nginx_image",
    }


# --- unreadable and unresolvable paths --------------------------------------


def test_unreadable_parent_directory_is_skipped_not_reported(tmp_path):
    """`Path.exists()` raises EACCES; a path we cannot check is not a path we can fault."""
    private = tmp_path / "private"
    private.mkdir()
    (private / "vllm.sif").touch()
    private.chmod(0o000)
    try:
        assert check_config_paths(_cfg(tmp_path, image=str(private / "vllm.sif"))) == []
    finally:
        private.chmod(0o755)


def test_hint_skips_over_unreadable_ancestors(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    private.chmod(0o000)
    try:
        cfg = _cfg(tmp_path, model=str(tmp_path / "gone" / "model"))
        (problem,) = check_config_paths(cfg)
        assert problem.hint == f"deepest existing parent: {tmp_path}"
    finally:
        private.chmod(0o755)


def test_env_var_set_but_empty_is_skipped(tmp_path, monkeypatch):
    """`expandvars` turns an empty var into '', inventing a path the user never wrote."""
    monkeypatch.setenv("SITE_IMAGES", "")

    assert check_config_paths(_cfg(tmp_path, image="$SITE_IMAGES/images/vllm.sif")) == []


def test_braced_env_var_form_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("SITE_IMAGES", str(tmp_path))

    cfg = _cfg(tmp_path, image="${SITE_IMAGES}/missing.sif")
    (problem,) = check_config_paths(cfg)

    assert problem.value == str(tmp_path / "missing.sif")


def test_relative_paths_are_skipped(tmp_path):
    """Relative values resolve against the submit CWD, not where the job runs."""
    assert check_config_paths(_cfg(tmp_path, image="./images/vllm.sif")) == []
    assert check_config_paths(_cfg(tmp_path, image="../images/vllm.sif")) == []


# --- container images are not necessarily files -----------------------------


def test_singularity_sandbox_directory_image_is_accepted(tmp_path):
    """`singularity exec` takes a sandbox directory as happily as a .sif."""
    sandbox = tmp_path / "vllm-sandbox"
    (sandbox / ".singularity.d").mkdir(parents=True)

    assert check_config_paths(_cfg(tmp_path, image=str(sandbox))) == []


def test_sandbox_directory_is_accepted_for_nginx_image(tmp_path):
    sandbox = tmp_path / "nginx-sandbox"
    (sandbox / ".singularity.d").mkdir(parents=True)

    cfg = _cfg(tmp_path, backend={"endpoint": {"nginx_image": str(sandbox)}})

    assert check_config_paths(cfg) == []


# --- symlinks ---------------------------------------------------------------


def test_broken_symlink_says_so(tmp_path):
    link = tmp_path / "vllm.sif.link"
    link.symlink_to(tmp_path / "gone.sif")

    (problem,) = check_config_paths(_cfg(tmp_path, image=str(link)))

    assert "broken symlink" in problem.reason


def test_venv_path_pointing_at_a_file_is_reported(tmp_path):
    not_a_venv = tmp_path / "venv-file"
    not_a_venv.touch()

    cfg = _cfg(tmp_path, backend={"venv_path": str(not_a_venv)})
    (problem,) = check_config_paths(cfg)

    assert problem.field == "backend.venv_path"
    assert "expected a directory" in problem.reason


def test_existing_venv_path_passes(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()

    assert check_config_paths(_cfg(tmp_path, backend={"venv_path": str(venv)})) == []


# --- report -----------------------------------------------------------------


def test_report_is_ascii_so_it_survives_a_C_locale_stream(tmp_path):
    from domyn_swarm.config.preflight import format_path_problems

    problems = check_config_paths(_cfg(tmp_path, image="/shared/images/vllm.sif"))
    report = format_path_problems(problems, source="cfg.yaml")

    report.encode("ascii")  # raises UnicodeEncodeError if a fancy glyph crept in
    assert "cfg.yaml" in report
    assert "/shared/images/vllm.sif" in report


# --- routers ----------------------------------------------------------------


def _router_cfg(tmp_path, router):
    return _cfg(
        tmp_path,
        backend={"endpoint": {"nginx_image": str(tmp_path / "nginx.sif"), "router": router}},
    )


def test_router_bare_binary_is_not_a_path_problem(tmp_path):
    cfg = _router_cfg(tmp_path, {"type": "vllm-router", "binary": "vllm-router"})
    assert check_config_paths(cfg) == []


def test_router_missing_binary_path_is_reported(tmp_path):
    cfg = _router_cfg(tmp_path, {"type": "vllm-router", "binary": "/missing/vllm-router"})
    assert _fields(check_config_paths(cfg)) == ["backend.endpoint.router.binary"]


def test_router_missing_image_is_reported(tmp_path):
    cfg = _router_cfg(tmp_path, {"type": "ramjet", "image": "/missing/r.sif"})
    assert _fields(check_config_paths(cfg)) == ["backend.endpoint.router.image"]


def test_dynamo_venv_without_pyvenv_cfg_is_reported(tmp_path):
    venv = tmp_path / "v"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    cfg = _router_cfg(tmp_path, {"type": "dynamo", "venv": str(venv)})
    problems = check_config_paths(cfg)
    assert _fields(problems) == ["backend.endpoint.router.venv"]
    assert problems[0].value == str(venv / "pyvenv.cfg")


def test_dynamo_venv_with_pyvenv_cfg_passes_when_bin_python_points_into_the_image(tmp_path):
    venv = tmp_path / "v"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "bin" / "python").symlink_to("/opt/image-only/bin/python3")
    cfg = _router_cfg(tmp_path, {"type": "dynamo", "venv": str(venv)})
    assert check_config_paths(cfg) == []


def test_dynamo_missing_policy_config_is_reported(tmp_path):
    cfg = _router_cfg(tmp_path, {"type": "dynamo", "policy_config": str(tmp_path / "missing.yaml")})
    assert _fields(check_config_paths(cfg)) == ["backend.endpoint.router.policy_config"]
