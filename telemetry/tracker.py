from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from telemetry.config import TelemetryConfig
from telemetry.storage import TelemetryStore, _naive_utc_now, get_default_store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Span:
    """A named, timed segment of a request (retrieve, rag_prepare, ...)."""

    name: str
    span_type: str
    duration_ms: float = 0.0


class TelemetryTracker:
    """Request-scoped telemetry sink backed by a local SQLite store.

    Every call is fail-safe: persistence errors are logged at debug level and
    never raised, so telemetry can never break the application.
    """

    def __init__(
        self,
        *,
        route: str,
        request_id: str | None = None,
        conversation_id: int | str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        config: TelemetryConfig | None = None,
        store: TelemetryStore | None = None,
    ) -> None:
        self._config = config or TelemetryConfig.from_env()
        self.request_id = request_id or str(uuid4())
        self._metrics: dict[str, float] = {}
        self._tags: dict[str, str] = {
            "request_id": self.request_id,
            "conversation_id": "" if conversation_id is None else str(conversation_id),
            "route": route,
            "model": model or "",
            "embedding_model": embedding_model or "",
            "environment": self._config.environment,
            "app_version": self._config.app_version,
        }
        self._spans: list[Span] = []
        self._store: TelemetryStore | None = None
        self._enabled = False
        self._started_at = _naive_utc_now()

        if not self._config.enabled:
            logger.debug("Telemetry disabled for request %s", self.request_id)
            return

        try:
            self._store = store if store is not None else get_default_store()
            self._enabled = True
        except Exception as exc:
            logger.debug("Failed to open telemetry store for request %s: %s", self.request_id, exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def add_metric(self, name: str, value: float | int) -> None:
        if not self._enabled:
            return
        try:
            self._metrics[name] = self._metrics.get(name, 0.0) + float(value)
        except (TypeError, ValueError) as exc:
            logger.debug("Dropped invalid telemetry metric %s=%r: %s", name, value, exc)

    def add_tag(self, name: str, value: float | int | str) -> None:
        if not self._enabled:
            return
        self._tags[name] = str(value)

    def metrics(self) -> dict[str, float]:
        """Snapshot of recorded metrics (used to render response details)."""
        return dict(self._metrics)

    def tags(self) -> dict[str, str]:
        """Snapshot of recorded tags (used to render response details)."""
        return dict(self._tags)

    def timed(self, metric: str) -> TimedSpan:
        return TimedSpan(self, metric=metric)

    def span(
        self,
        name: str,
        span_type: str = "UNKNOWN",
        latency_metric: str | None = None,
    ) -> TimedSpan:
        return TimedSpan(self, name=name, span_type=span_type, metric=latency_metric)

    def finish(self, *, success: bool = True, error_type: str | None = None) -> None:
        if not self._enabled:
            return
        finished_at = _naive_utc_now()
        self._tags.update(
            {
                "success": "true" if success else "false",
                "error_type": error_type or "",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        duration_ms = (finished_at - self._started_at).total_seconds() * 1000.0

        try:
            assert self._store is not None
            self._store.insert_event(
                {
                    "request_id": self.request_id,
                    "route": self._tags["route"],
                    "conversation_id": self._tags["conversation_id"] or None,
                    "model": self._tags.get("model") or None,
                    "embedding_model": self._tags.get("embedding_model") or None,
                    "success": success,
                    "error_type": error_type,
                    "environment": self._config.environment,
                    "app_version": self._config.app_version,
                    "started_at": self._started_at,
                    "finished_at": finished_at,
                    "duration_ms": duration_ms,
                    "metrics": dict(self._metrics),
                    "tags": dict(self._tags),
                    "spans": [
                        {"name": s.name, "span_type": s.span_type, "duration_ms": s.duration_ms}
                        for s in self._spans
                    ],
                }
            )
        except Exception as exc:
            logger.debug("Failed to persist telemetry event %s: %s", self.request_id, exc)
        finally:
            self._enabled = False

    def _record_latency(self, metric: str, elapsed_seconds: float) -> None:
        self.add_metric(metric, elapsed_seconds * 1000.0)


class NullTelemetryTracker(TelemetryTracker):
    """No-op tracker used when no request scope is active."""

    def __init__(self) -> None:
        self.request_id = ""
        self._config = TelemetryConfig(enabled=False)
        self._metrics: dict[str, float] = {}
        self._tags: dict[str, str] = {}
        self._spans: list[Span] = []
        self._store = None
        self._enabled = False
        self._started_at = _naive_utc_now()


class TimedSpan:
    """Timing + optional span record (sync and async)."""

    __slots__ = ("_tracker", "_name", "_span_type", "_metric", "_span", "_started_at")

    def __init__(
        self,
        tracker: TelemetryTracker,
        *,
        name: str | None = None,
        span_type: str = "UNKNOWN",
        metric: str | None = None,
    ) -> None:
        self._tracker = tracker
        self._name = name
        self._span_type = span_type
        self._metric = metric
        self._span: Span | None = None
        self._started_at: float | None = None

    def __enter__(self) -> Span | None:
        self._started_at = time.perf_counter()
        if self._name is not None and self._tracker._enabled:
            self._span = Span(name=self._name, span_type=self._span_type)
            self._tracker._spans.append(self._span)
        return self._span

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        elapsed = time.perf_counter() - self._started_at if self._started_at is not None else 0.0
        if self._metric is not None:
            self._tracker._record_latency(self._metric, elapsed)
        if self._span is not None:
            self._span.duration_ms = round(elapsed * 1000.0, 3)
        return False

    async def __aenter__(self) -> Span | None:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.__exit__(exc_type, exc_value, traceback)
        return False
