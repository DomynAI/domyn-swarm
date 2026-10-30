# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Generate the configuration reference from the Pydantic config models.

Rendering from ``model_fields`` rather than ``model_json_schema()`` keeps the
real annotation and required-ness instead of ``$ref``/``anyOf`` indirection, so
the tables read the way a human would write them.

Output is written to ``<srcdir>/_generated/`` on ``builder-inited``. That
directory is in ``exclude_patterns``; pages pull the fragments in with
``{include}``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import types
import typing

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

CONFIG_MODELS: tuple[tuple[str, str], ...] = (
    ("domyn_swarm.config.swarm", "DomynLLMSwarmConfig"),
    ("domyn_swarm.config.backend", "BackendsConfig"),
    ("domyn_swarm.config.slurm", "SlurmConfig"),
    ("domyn_swarm.config.slurm", "SlurmEndpointConfig"),
    ("domyn_swarm.config.router", "NginxRouter"),
    ("domyn_swarm.config.router", "VllmRouter"),
    ("domyn_swarm.config.router", "RamjetRouter"),
    ("domyn_swarm.config.router", "DynamoRouter"),
    ("domyn_swarm.config.slurm", "MonitoringConfig"),
    ("domyn_swarm.config.slurm", "GpuExporterConfig"),
    ("domyn_swarm.config.slurm", "RayMetricsConfig"),
    ("domyn_swarm.config.lepton", "LeptonConfig"),
    ("domyn_swarm.config.lepton", "LeptonEndpointConfig"),
    ("domyn_swarm.config.lepton", "LeptonJobConfig"),
    ("domyn_swarm.config.watchdog", "WatchdogConfig"),
    ("domyn_swarm.config.watchdog", "WatchdogRayConfig"),
    ("domyn_swarm.config.pool", "SwarmPoolConfig"),
    ("domyn_swarm.config.pool", "SwarmPoolElement"),
)

SETTINGS_MODEL: tuple[str, str] = ("domyn_swarm.config.settings", "Settings")


def render_annotation(annotation: object) -> str:
    """Render a type annotation as a short, human-readable string."""
    if annotation is None or annotation is type(None):
        return "None"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Literal:
        return " | ".join(f'"{arg}"' if isinstance(arg, str) else str(arg) for arg in args)
    if origin in (types.UnionType, typing.Union):
        return " | ".join(render_annotation(arg) for arg in args)
    if origin is not None:
        base = getattr(origin, "__name__", str(origin))
        if args:
            inner = ", ".join(render_annotation(arg) for arg in args)
            return f"{base}[{inner}]"
        return base
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def render_default(field: FieldInfo) -> str:
    """Render a field's default as a table cell."""
    if field.is_required():
        return "**required**"
    if field.default_factory is not None:
        return "*computed*"
    return f"`{field.default!r}`"


def render_model_table(model: type[BaseModel], level: int = 3) -> str:
    """Render one Pydantic model as a MyST section containing a field table.

    Args:
        model: The Pydantic model to render.
        level: Markdown heading level for the model's section. Pages that have no
            intermediate ``##`` heading pass ``2`` to keep heading levels
            consecutive, which MyST warns about otherwise.
    """
    lines: list[str] = [f"{'#' * level} `{model.__name__}`", ""]

    doc = (model.__doc__ or "").strip()
    if doc:
        lines += [" ".join(doc.split()), ""]

    lines += [
        "| Field | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for name, field in model.model_fields.items():
        label = field.alias or name
        description = " ".join((field.description or "").split()).replace("|", r"\|")
        lines.append(
            f"| `{label}` | `{render_annotation(field.annotation)}` "
            f"| {render_default(field)} | {description} |"
        )
    lines.append("")
    return "\n".join(lines)


def env_var_name(name: str, field: FieldInfo, prefix: str) -> str:
    """Return the environment variable that populates a settings field.

    An explicit alias wins and is used verbatim; otherwise the model's
    ``env_prefix`` is prepended to the field name. Either way the result is
    upper-cased, which is how the variables are written in practice.
    """
    alias = field.alias or field.validation_alias
    if isinstance(alias, str):
        return alias.upper()
    return f"{prefix}{name}".upper()


def render_settings_table(model: type[BaseModel], level: int = 2) -> str:
    """Render a ``BaseSettings`` model as a table keyed by environment variable.

    Unlike :func:`render_model_table` this leads with the variable name rather
    than the field name, because that is what a reader actually sets.
    """
    prefix = str(model.model_config.get("env_prefix", ""))

    lines: list[str] = [f"{'#' * level} `{model.__name__}`", ""]

    doc = (model.__doc__ or "").strip()
    if doc:
        lines += [" ".join(doc.split()), ""]

    lines += [
        "| Variable | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for name, field in model.model_fields.items():
        description = " ".join((field.description or "").split()).replace("|", r"\|")
        lines.append(
            f"| `{env_var_name(name, field, prefix)}` "
            f"| `{render_annotation(field.annotation)}` "
            f"| {render_default(field)} | {description} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load(module_path: str, name: str) -> type[BaseModel]:
    return getattr(importlib.import_module(module_path), name)


def _write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def generate(app: Sphinx) -> None:
    """Write the generated reference fragments into ``<srcdir>/_generated``."""
    out_dir = Path(app.srcdir) / "_generated"

    tables = [render_model_table(_load(mod, name)) for mod, name in CONFIG_MODELS]
    _write_if_changed(out_dir / "config-tables.md", "\n".join(tables))

    _write_if_changed(
        out_dir / "settings-table.md",
        render_settings_table(_load(*SETTINGS_MODEL), level=2),
    )

    changelog = Path(app.srcdir).parent / "CHANGELOG.md"
    if changelog.is_file():
        _write_if_changed(out_dir / "changelog.md", changelog.read_text(encoding="utf-8"))
    else:
        logger.warning("CHANGELOG.md not found at %s", changelog)


def setup(app: Sphinx) -> dict[str, object]:
    app.connect("builder-inited", generate)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
