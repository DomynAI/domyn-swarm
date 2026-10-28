# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def get_version() -> str:
    try:
        return version("domyn-swarm")
    except PackageNotFoundError:
        # Fallback when running from source without an installed dist
        data = tomllib.loads(Path("pyproject.toml").read_text())
        return (data.get("project") or {}).get("version", "0.0.0+unknown")


if __name__ == "__main__":
    print(get_version())
