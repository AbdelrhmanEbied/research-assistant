from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, String, create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from telemetry.config import TelemetryConfig

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "telemetry.db"


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class StorageBase(DeclarativeBase):
    pass


class StoredEvent(StorageBase):
    __tablename__ = "telemetry_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    route: Mapped[str] = mapped_column(String(120), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    environment: Mapped[str] = mapped_column(String(40), default="development")
    app_version: Mapped[str] = mapped_column(String(40), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    tags: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    spans: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)


class TelemetryStore:
    """Local SQLite-backed store for completed telemetry events."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        engine: Engine | None = None,
    ) -> None:
        if engine is not None:
            self._engine = engine
        else:
            self._engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False},
            )
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        StorageBase.metadata.create_all(self._engine)

    @property
    def engine(self) -> Engine:
        return self._engine

    def insert_event(self, event: dict[str, Any]) -> int:
        with self._session_factory() as session:
            row = StoredEvent(**event)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        with self._session_factory() as session:
            row = session.get(StoredEvent, event_id)
            return self._to_dict(row) if row is not None else None

    def list_events(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            rows = (
                session.scalars(
                    select(StoredEvent)
                    .order_by(StoredEvent.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
                .all()
            )
            return [self._to_dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self._session_factory() as session:
            rows = session.execute(
                select(
                    StoredEvent.success,
                    StoredEvent.route,
                    StoredEvent.duration_ms,
                    StoredEvent.started_at,
                    StoredEvent.metrics,
                )
            ).all()

        total = len(rows)
        if total == 0:
            return {
                "total": 0,
                "success": 0,
                "failures": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "metric_averages": {},
                "routes": {},
                "timeline": [],
            }

        success_count = sum(1 for row in rows if row.success)
        durations = [row.duration_ms for row in rows if row.duration_ms]
        avg_duration = round(sum(durations) / len(durations), 2) if durations else 0.0

        metric_values: dict[str, list[float]] = {}
        for row in rows:
            for name, value in (row.metrics or {}).items():
                if isinstance(value, (int, float)):
                    metric_values.setdefault(name, []).append(float(value))
        metric_averages = {
            name: round(sum(values) / len(values), 2)
            for name, values in metric_values.items()
        }

        route_buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            bucket = route_buckets.setdefault(
                row.route,
                {"count": 0, "success": 0, "durations": []},
            )
            bucket["count"] += 1
            bucket["success"] += 1 if row.success else 0
            if row.duration_ms:
                bucket["durations"].append(row.duration_ms)
        routes: dict[str, dict[str, Any]] = {}
        for name, bucket in route_buckets.items():
            durations = bucket["durations"]
            routes[name] = {
                "count": bucket["count"],
                "success": bucket["success"],
                "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            }

        now = _naive_utc_now()
        hourly: dict[str, dict[str, Any]] = {}
        for row in rows:
            start = row.started_at
            if start is None:
                continue
            if (now - start).total_seconds() > 24 * 3600:
                continue
            hour = start.replace(minute=0, second=0, microsecond=0).isoformat()
            bucket = hourly.setdefault(hour, {"count": 0, "success": 0, "durations": []})
            bucket["count"] += 1
            bucket["success"] += 1 if row.success else 0
            if row.duration_ms:
                bucket["durations"].append(row.duration_ms)
        timeline: list[dict[str, Any]] = []
        for hour, bucket in sorted(hourly.items()):
            durations = bucket["durations"]
            timeline.append(
                {
                    "hour": hour,
                    "count": bucket["count"],
                    "success": bucket["success"],
                    "failures": bucket["count"] - bucket["success"],
                    "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
                }
            )

        return {
            "total": total,
            "success": success_count,
            "failures": total - success_count,
            "success_rate": round(success_count / total * 100.0, 2),
            "avg_duration_ms": avg_duration,
            "metric_averages": metric_averages,
            "routes": routes,
            "timeline": timeline,
        }

    def clear(self) -> int:
        with self._session_factory() as session:
            result = session.execute(delete(StoredEvent))
            session.commit()
            return result.rowcount or 0

    def close(self) -> None:
        self._engine.dispose()

    @staticmethod
    def _to_dict(row: StoredEvent) -> dict[str, Any]:
        return {
            "id": row.id,
            "request_id": row.request_id,
            "route": row.route,
            "conversation_id": row.conversation_id,
            "model": row.model,
            "embedding_model": row.embedding_model,
            "success": row.success,
            "error_type": row.error_type,
            "environment": row.environment,
            "app_version": row.app_version,
            "started_at": _iso(row.started_at),
            "finished_at": _iso(row.finished_at),
            "duration_ms": row.duration_ms,
            "metrics": row.metrics or {},
            "tags": row.tags or {},
            "spans": row.spans or [],
        }


_default_store: TelemetryStore | None = None


def get_default_store(db_path: str | None = None) -> TelemetryStore:
    """Return the process-wide store, creating it from env config on first use."""
    global _default_store
    if db_path is None:
        db_path = TelemetryConfig.from_env().db_path
    if _default_store is None:
        _default_store = TelemetryStore(db_path)
    return _default_store


def reset_default_store() -> None:
    global _default_store
    _default_store = None