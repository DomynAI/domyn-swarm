# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Checks that a swarm config's filesystem paths are usable, before deploying.

Not pydantic validators: a config is persisted and re-read to operate an
already-running swarm, so gating schema validation on path existence would
break ``down``/``status`` whenever an image moved.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import re
from typing import Any, Literal

_Kind = Literal["file", "dir"] | None

#: Matches ``$NAME`` and ``${NAME}`` in the *original* text. Inspecting the
#: expansion instead would miss a set-but-empty variable, which
#: ``os.path.expandvars`` turns into "" rather than leaving in place.
_ENV_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


@dataclass(frozen=True)
class PathProblem:
    """A config field pointing at a path that cannot be used.

    Attributes:
        field: Dotted config key, e.g. ``backend.endpoint.nginx_image``.
        value: The path after environment-variable and ``~`` expansion.
        reason: Why the path is unusable, in user-facing wording.
        hint: Optional extra context, such as the deepest existing ancestor.
    """

    field: str
    value: str
    reason: str
    hint: str | None = None


# The errors that `Path.exists` treated as "missing" up to Python 3.13. From
# 3.14 it returns False for every OSError, so an unreadable parent would look
# like a missing path.
_MISSING_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})


def _exists(path: Path) -> bool | None:
    """Report whether ``path`` exists, or ``None`` when that cannot be told.

    A path we may not look at, for example under an unreadable parent, is not
    one we can fault.

    Args:
        path: The path to probe.

    Returns:
        ``True``/``False``, or ``None`` if the filesystem refused the question.
    """
    try:
        path.stat()
    except OSError as exc:
        return False if exc.errno in _MISSING_ERRNOS else None
    except ValueError:
        return False
    return True


def _resolve_local_path(value: Any) -> str | None:
    """Expand a config value to a local path, or ``None`` if it is not one.

    Registry references, Hugging Face repo ids, bare names, relative paths
    (which resolve against the submit-host CWD, not the job's) and anything
    naming a variable unset or empty here are left alone: guessing at a value
    this host cannot resolve would block a valid launch.

    Args:
        value: Raw config value.

    Returns:
        The expanded absolute path, or ``None`` if it is not checkable here.
    """
    if value is None:
        return None

    text = os.fspath(value).strip() if isinstance(value, os.PathLike) else str(value).strip()
    if not text or "://" in text:
        return None

    if any(not os.environ.get(name) for name in _ENV_VAR.findall(text)):
        return None

    expanded = os.path.expanduser(os.path.expandvars(text))
    if not expanded.startswith("/"):
        return None
    return expanded


def _deepest_existing_parent(path: Path) -> str | None:
    """Describe the deepest ancestor of ``path`` that exists.

    This points at the segment where the path stops matching reality.
    Unprobeable ancestors are stepped over.

    Args:
        path: The missing path.

    Returns:
        A ready-to-print hint, or ``None`` if no ancestor could be confirmed.
    """
    for parent in path.parents:
        if _exists(parent):
            return f"deepest existing parent: {parent}"
    return None


def _check(
    problems: list[PathProblem],
    field: str,
    value: Any,
    *,
    kind: _Kind = None,
) -> None:
    """Append a :class:`PathProblem` for ``value`` if it is not usable.

    Args:
        problems: Accumulator the problem is appended to.
        field: Dotted config key being checked.
        value: Raw config value.
        kind: Require a ``file`` or a ``dir``; ``None`` only requires
            existence. Images use ``None``: Singularity takes a sandbox
            directory as readily as a ``.sif``.
    """
    resolved = _resolve_local_path(value)
    if resolved is None:
        return

    path = Path(resolved)
    exists = _exists(path)
    if exists is None:
        return

    if not exists:
        if path.is_symlink():
            problems.append(
                PathProblem(field, resolved, f"is a broken symlink to {os.readlink(path)}")
            )
        else:
            problems.append(
                PathProblem(field, resolved, "does not exist", _deepest_existing_parent(path))
            )
        return

    if kind == "file" and path.is_dir():
        problems.append(PathProblem(field, resolved, "is a directory, expected a file"))
    elif kind == "dir" and not path.is_dir():
        problems.append(PathProblem(field, resolved, "is a file, expected a directory"))


def _check_router_paths(problems: list[PathProblem], router: Any) -> None:
    """Check the router's image, binary, policy file and virtualenv.

    The virtualenv check looks for ``pyvenv.cfg``. ``bin/python`` is often a
    symlink to an interpreter that exists only inside the image.
    """
    prefix = "backend.endpoint.router"
    _check(problems, f"{prefix}.image", getattr(router, "image", None))
    binary = getattr(router, "binary", None)
    if binary is not None and "/" in str(binary):
        _check(problems, f"{prefix}.binary", binary)
    _check(problems, f"{prefix}.policy_config", getattr(router, "policy_config", None))
    venv = getattr(router, "venv", None)
    if venv is not None:
        _check(problems, f"{prefix}.venv", Path(str(venv)) / "pyvenv.cfg", kind="file")


def check_config_paths(cfg: Any) -> list[PathProblem]:
    """Collect every unusable filesystem path a swarm config refers to.

    Slurm only: Lepton images are Docker references and its paths live on
    platform volumes, neither resolvable from the submit host.

    Args:
        cfg: A ``DomynLLMSwarmConfig``. Anything without a Slurm backend yields
            no problems, so callers may pass whatever their loader returned.

    Returns:
        Every problem found, in config order. Empty when the config is fine.
    """
    from domyn_swarm.config.slurm import SlurmConfig

    backend = getattr(cfg, "backend", None)
    if not isinstance(backend, SlurmConfig):
        return []

    problems: list[PathProblem] = []

    _check(problems, "image", getattr(cfg, "image", None))
    _check(problems, "model", getattr(cfg, "model", None))

    endpoint = backend.endpoint
    _check(problems, "backend.endpoint.nginx_image", endpoint.nginx_image)

    _check_router_paths(problems, endpoint.router)

    monitoring = endpoint.monitoring
    if monitoring.enabled and monitoring.mode == "container":
        prefix = "backend.endpoint.monitoring"
        _check(problems, f"{prefix}.prometheus_image", monitoring.prometheus_image)
        _check(problems, f"{prefix}.nginx_exporter_image", monitoring.nginx_exporter_image)
        if monitoring.gpu_exporter.enabled:
            _check(
                problems,
                f"{prefix}.gpu_exporter.image",
                monitoring.gpu_exporter.resolved_image(mode=monitoring.mode),
            )

    for index, mount in enumerate(backend.mounts):
        _check(problems, f"backend.mounts[{index}]", mount.split(":")[0])

    _check(problems, "backend.venv_path", backend.venv_path, kind="dir")

    return problems


def raise_on_path_problems(cfg: Any) -> None:
    """Check ``cfg`` and raise if any of its paths are unusable.

    Args:
        cfg: A ``DomynLLMSwarmConfig``.

    Raises:
        SwarmConfigPathError: If at least one path cannot be used.
    """
    from domyn_swarm.exceptions import SwarmConfigPathError

    problems = check_config_paths(cfg)
    if problems:
        raise SwarmConfigPathError(problems)


def format_path_problems(problems: list[PathProblem], *, source: str | None = None) -> str:
    """Render problems as a plain-ASCII terminal report.

    Args:
        problems: The problems to render; must not be empty.
        source: Name of the config file they came from, when known.

    Returns:
        A multi-line report with no trailing newline.
    """
    count = len(problems)
    subject = "path" if count == 1 else "paths"
    origin = f" in {source}" if source else ""
    lines = [f"Config check failed: {count} {subject}{origin} cannot be used", ""]

    for problem in problems:
        lines.append(f"  {problem.field}")
        lines.append(f"      {problem.value}")
        detail = f"      -> {problem.reason}"
        if problem.hint:
            detail += f" ({problem.hint})"
        lines.append(detail)
        lines.append("")

    lines.append(
        "Nothing was submitted. Fix the paths above, or pass --skip-preflight to launch anyway."
    )
    return "\n".join(lines)
