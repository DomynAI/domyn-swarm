# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized environment configuration for domyn-swarm.

    Env var naming: DOMYN_SWARM_<FIELD_NAME> (custom aliases below).
    A .env file in CWD or ~/.domyn_swarm/.env is read automatically.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOMYN_SWARM_",
        env_file=(".env", "~/.domyn_swarm/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    log_level: str = Field(
        default="INFO",
        description="Global logging level, e.g. `DEBUG`, `INFO` or `WARNING`.",
    )
    home: Path = Field(
        default=Path("~/.domyn_swarm").expanduser(),
        description="Path to domyn-swarm home directory",
    )
    defaults_file: Path | None = Field(
        default_factory=lambda data: data["home"] / "defaults.yaml",
        alias="DOMYN_SWARM_DEFAULTS",
        description="YAML file containing project-wide configuration defaults.",
    )

    api_token: SecretStr | None = Field(
        default=None,
        description="API token for authenticating with the domyn-swarm vllm server",
    )  # DOMYN_SWARM_API_TOKEN
    vllm_api_key: SecretStr | None = Field(
        default_factory=lambda data: data.get("api_token"),
        alias="VLLM_API_KEY",
        description="Alternative env var for API token, used by vLLM",
    )
    apptainerenv_vllm_api_key: SecretStr | None = Field(
        default_factory=lambda data: data.get("vllm_api_key"),
        alias="APPTAINERENV_VLLM_API_KEY",
        description="Alternative API token passed into Apptainer containers.",
    )
    singularityenv_vllm_api_key: SecretStr | None = Field(
        default_factory=lambda data: data.get("vllm_api_key"),
        alias="SINGULARITYENV_VLLM_API_KEY",
        description="Alternative API token passed into Singularity containers.",
    )

    @property
    def resolved_api_token(self) -> SecretStr | None:
        """Return the first configured LLM API token, in precedence order.

        Returns:
            ``api_token`` if set, else ``vllm_api_key``, else
            ``apptainerenv_vllm_api_key``, else
            ``singularityenv_vllm_api_key``, else ``None``.
        """
        return (
            self.api_token
            or self.vllm_api_key
            or self.apptainerenv_vllm_api_key
            or self.singularityenv_vllm_api_key
        )

    mail_user: str | None = Field(
        description="Email address for Slurm job notifications (if enabled)",
        default=None,
    )  # DOMYN_SWARM_MAIL_USER

    lepton_api_token: SecretStr | None = Field(
        default=None,
        alias="LEPTONAI_API_TOKEN",
        description="API token for authenticating with Lepton AI",
    )
    lepton_workspace_id: str | None = Field(
        default=None,
        alias="LEPTON_WORKSPACE_ID",
        description="Workspace ID for Lepton AI",
    )

    ascii: bool = Field(
        default=False,
        description="Use ASCII glyphs instead of Unicode emojis in CLI output",
    )  # DOMYN_SWARM_ASCII

    skip_db_upgrade: bool = Field(
        default=False,
        description="If true, skip automatic database schema upgrades on CLI startup",
    )  # DOMYN_SWARM_SKIP_DB_UPGRADE


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor. Call this wherever you need settings.
    Tests can `cache_clear()` before reading to pick up monkeypatched env.
    """
    return Settings()


def reload_settings_cache() -> None:
    get_settings.cache_clear()  # type: ignore[attr-defined]
