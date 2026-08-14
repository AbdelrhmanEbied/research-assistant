from __future__ import annotations

import logging
from contextvars import ContextVar

from telemetry.config import TelemetryConfig
from telemetry.storage import TelemetryStore, get_default_store
from telemetry.tracker import NullTelemetryTracker, TelemetryTracker

logger = logging.getLogger(__name__)

_NULL_TRACKER = NullTelemetryTracker()

_current_tracker: ContextVar[TelemetryTracker] = ContextVar(
    "telemetry_tracker", default=_NULL_TRACKER
)


def init_telemetry(config: TelemetryConfig | None = None) -> bool:
    cfg = config or TelemetryConfig.from_env()
    if not cfg.enabled:
        logger.info("Telemetry is disabled (TELEMETRY_ENABLED=false).")
        return False
    try:
        get_default_store(cfg.db_path)
    except Exception as exc:
        logger.warning("Failed to initialize telemetry store: %s", exc)
        return False
    logger.info(
        "Telemetry initialized: db_path=%s environment=%s app_version=%s",
        cfg.db_path,
        cfg.environment,
        cfg.app_version,
    )
    return True


def start_request_tracking(
    *,
    route: str,
    request_id: str | None = None,
    conversation_id: int | str | None = None,
    model: str | None = None,
    embedding_model: str | None = None,
    config: TelemetryConfig | None = None,
    store: TelemetryStore | None = None,
) -> TelemetryTracker:
    tracker = TelemetryTracker(
        route=route,
        request_id=request_id,
        conversation_id=conversation_id,
        model=model,
        embedding_model=embedding_model,
        config=config,
        store=store,
    )
    _current_tracker.set(tracker)
    return tracker


def get_current_tracker() -> TelemetryTracker:
    return _current_tracker.get()


def clear_request_tracking() -> None:
    _current_tracker.set(_NULL_TRACKER)


__all__ = [
    "NullTelemetryTracker",
    "TelemetryTracker",
    "TelemetryConfig",
    "TelemetryStore",
    "init_telemetry",
    "start_request_tracking",
    "get_current_tracker",
    "clear_request_tracking",
]
