"""Lightweight additive migrations for the local SQLite databases.

``create_all`` only creates missing tables; it never alters existing ones. This
helper adds columns that new models introduce so an already-initialized local
DB keeps working without a destructive rebuild. Only additive changes are
supported — renaming/dropping is out of scope for this local-first app.
"""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

#: table -> list of (column, DDL) additive migrations applied once.
_COLUMN_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "messages": [
        (
            "extra",
            "ALTER TABLE messages ADD COLUMN extra JSON",
        ),
    ],
}


def ensure_schema_migrations(engine: Engine) -> None:
    """Apply missing additive column migrations for known tables."""
    with engine.begin() as conn:
        for table, migrations in _COLUMN_MIGRATIONS.items():
            existing: set[str] = set()
            try:
                rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                existing = {row[1] for row in rows}
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Could not inspect table %s: %s", table, exc)
                continue

            for column, ddl in migrations:
                if column in existing:
                    continue
                try:
                    conn.execute(text(ddl))
                    logger.info("Applied migration: %s.%s", table, column)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Migration %s.%s failed (column may already exist): %s",
                        table,
                        column,
                        exc,
                    )


__all__ = ["ensure_schema_migrations"]