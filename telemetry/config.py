from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from paths import data_path

load_dotenv()

DEFAULT_DB_PATH = "telemetry.db"
DEFAULT_ENVIRONMENT = "development"
DEFAULT_APP_VERSION = "0.1.0"


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    enabled: bool = True
    db_path: str = DEFAULT_DB_PATH
    environment: str = DEFAULT_ENVIRONMENT
    app_version: str = DEFAULT_APP_VERSION

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        return cls(
            enabled=_env_bool("TELEMETRY_ENABLED"),
            db_path=os.getenv("TELEMETRY_DB_PATH", str(data_path(DEFAULT_DB_PATH))),
            environment=os.getenv("ENVIRONMENT", DEFAULT_ENVIRONMENT),
            app_version=os.getenv("APP_VERSION", DEFAULT_APP_VERSION),
        )
