# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Every config field that appears in the generated reference must be described.

This locks in the description backfill: a new field added without a
``description=`` fails here rather than silently rendering an empty table row.
"""

from __future__ import annotations

import types
import typing

from pydantic import BaseModel
import pytest

from domyn_swarm.config.backend import BackendsConfig
from domyn_swarm.config.lepton import (
    LeptonConfig,
    LeptonEndpointConfig,
    LeptonJobConfig,
)
from domyn_swarm.config.pool import SwarmPoolConfig, SwarmPoolElement
from domyn_swarm.config.router import DynamoRouter, NginxRouter, RamjetRouter, VllmRouter
from domyn_swarm.config.settings import Settings
from domyn_swarm.config.slurm import (
    GpuExporterConfig,
    MonitoringConfig,
    RayMetricsConfig,
    SlurmConfig,
    SlurmEndpointConfig,
)
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.config.watchdog import WatchdogConfig, WatchdogRayConfig

DOCUMENTED_MODELS: tuple[type[BaseModel], ...] = (
    DomynLLMSwarmConfig,
    BackendsConfig,
    SlurmConfig,
    SlurmEndpointConfig,
    NginxRouter,
    VllmRouter,
    RamjetRouter,
    DynamoRouter,
    MonitoringConfig,
    GpuExporterConfig,
    RayMetricsConfig,
    LeptonConfig,
    LeptonEndpointConfig,
    LeptonJobConfig,
    WatchdogConfig,
    WatchdogRayConfig,
    SwarmPoolConfig,
    SwarmPoolElement,
    Settings,
)


def _undescribed(model: type[BaseModel]) -> list[str]:
    return [
        name for name, field in model.model_fields.items() if not (field.description or "").strip()
    ]


@pytest.mark.parametrize("model", DOCUMENTED_MODELS, ids=lambda m: m.__name__)
def test_every_field_has_a_description(model: type[BaseModel]) -> None:
    missing = _undescribed(model)
    assert not missing, (
        f"{model.__name__} fields without Field(description=...): {missing}. "
        "Add a description; it feeds the generated configuration reference."
    )


# Every model that is the root of a config document, rather than nested inside
# one: a swarm config, a pool config, and the environment settings.
CONFIG_ROOTS: tuple[type[BaseModel], ...] = (
    DomynLLMSwarmConfig,
    BackendsConfig,
    SwarmPoolConfig,
    Settings,
)


def _unwrap(annotation: object) -> list[object]:
    """Flatten Annotated, unions, optionals and containers into their members."""
    if hasattr(annotation, "__metadata__"):  # Annotated[X, ...] -> X
        return _unwrap(typing.get_args(annotation)[0])
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType, list, dict, tuple, set, frozenset):
        found: list[object] = []
        for arg in typing.get_args(annotation):
            found.extend(_unwrap(arg))
        return found
    return [annotation]


def _nested_models(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Every config model reachable from ``root`` through its field annotations."""
    seen: set[type[BaseModel]] = set()
    queue: list[type[BaseModel]] = [root]
    while queue:
        model = queue.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            queue.extend(
                candidate
                for candidate in _unwrap(field.annotation)
                if isinstance(candidate, type) and issubclass(candidate, BaseModel)
            )
    return seen


def test_the_field_graph_reaches_every_documented_model() -> None:
    """Guards the guard: a walk that reaches nothing would pass vacuously."""
    reachable = set().union(*(_nested_models(root) for root in CONFIG_ROOTS))
    unreachable = sorted(m.__name__ for m in set(DOCUMENTED_MODELS) - reachable)
    assert not unreachable, (
        f"documented models the field walk cannot reach: {unreachable}. "
        "Either they are no longer part of the config, or _unwrap needs to "
        "understand a new annotation form."
    )


def test_no_nested_config_model_is_undocumented() -> None:
    """A new nested model must join DOCUMENTED_MODELS, not slip through.

    The list above is hand-maintained, so it drifts silently: a feature branch
    that adds a config model gets no description coverage and renders empty rows
    in the generated reference. Walking the field graph is what notices.
    """
    reachable = set().union(*(_nested_models(root) for root in CONFIG_ROOTS))
    missing = sorted(m.__name__ for m in reachable - set(DOCUMENTED_MODELS))
    assert not missing, (
        f"config models reachable from a root config but not documented: {missing}. "
        "Add Field(description=...) to their fields, then list them in "
        "DOCUMENTED_MODELS and in gen_config_reference.CONFIG_MODELS."
    )
