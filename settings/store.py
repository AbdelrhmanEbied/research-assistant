from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS_PATH = Path("settings.json")

#: Provider -> label, env var that seeds the key, and a curated list of
#: model identifiers. The list is a convenience for the Settings UI; users can
#: always type an arbitrary model name.
PROVIDERS: dict[str, dict[str, Any]] = {
    "google_genai": {
        "label": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "models": [
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.5-pro",
        ],
    },
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "o3",
            "o3-mini",
        ],
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-haiku-4-5",
        ],
    },
}

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "google_genai")

RETRIEVAL_KEYS = ("search_type", "limit", "rerank", "rerank_top_k", "search_depth")


class SettingsStore:
    """Local, single-user source of truth for runtime configuration.

    Settings that used to require editing ``.env`` or ``agent/llms.py`` are
    persisted to a JSON file next to the app. Env vars remain the bootstrap
    defaults so an untouched install behaves exactly as before.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
        self._data: dict[str, Any] = {}
        self._load()

    # --- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self._data, dict):
                self._data = {}
        except Exception as exc:
            logger.warning("Could not read settings file %s: %s", self.path, exc)
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except Exception as exc:
            logger.warning("Could not persist settings to %s: %s", self.path, exc)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    # --- LLM defaults (single source of truth) ------------------------------

    def get_llm(self) -> dict[str, str | None]:
        """Raw stored values; ``None`` means "not configured by the user"."""
        return {
            "model": self._data.get("llm_model"),
            "model_provider": self._data.get("llm_provider"),
        }

    def effective_llm(self) -> dict[str, str]:
        """Stored values falling back to the env-seeded defaults."""
        return {
            "model": self._data.get("llm_model") or DEFAULT_MODEL,
            "model_provider": self._data.get("llm_provider") or DEFAULT_PROVIDER,
        }

    def set_llm(self, model: str, model_provider: str) -> None:
        self.set("llm_model", model)
        self.set("llm_provider", model_provider)

    def _env_key_for(self, provider: str) -> str:
        info = PROVIDERS.get(provider, {})
        return os.getenv(info.get("env_key", "")) or ""

    def has_api_key(self, provider: str) -> bool:
        """True when a key is configured (stored or seeded from env)."""
        return bool(self.get_api_key(provider) or self._env_key_for(provider))

    def get_api_key(self, provider: str | None = None) -> str | None:
        """Return the user-configured key for ``provider`` (settings.json only)."""
        provider = provider or self.effective_llm()["model_provider"]
        stored = (self._data.get("api_keys") or {}).get(provider)
        return str(stored) if stored else None

    def set_api_key(self, provider: str, value: str | None) -> None:
        keys = dict(self._data.get("api_keys") or {})
        value = (value or "").strip()
        if value:
            keys[provider] = value
        else:
            keys.pop(provider, None)
        self.set("api_keys", keys)

    def clear_api_key(self, provider: str) -> None:
        self.set_api_key(provider, None)

    # --- retrieval defaults -------------------------------------------------

    def get_retrieval(self) -> dict[str, Any]:
        stored = self._data.get("retrieval") or {}
        return {
            "search_type": stored.get("search_type") or "hybrid",
            "limit": stored.get("limit") or 10,
            "rerank": True if stored.get("rerank") is None else bool(stored.get("rerank")),
            "rerank_top_k": stored.get("rerank_top_k") or 5,
            "search_depth": stored.get("search_depth") or "basic",
        }

    def set_retrieval(self, **kwargs: Any) -> None:
        current = self.get_retrieval()
        for key in RETRIEVAL_KEYS:
            if key in kwargs and kwargs[key] is not None:
                current[key] = kwargs[key]
        self.set("retrieval", current)

    # --- secrets safety ------------------------------------------------------

    def public_dict(self) -> dict[str, Any]:
        """Expose settings to the API without ever revealing key material."""
        llm = self.effective_llm()
        return {
            "llm": {
                "default_model": llm["model"],
                "default_provider": llm["model_provider"],
            },
            "providers": {
                name: {
                    "label": info["label"],
                    "models": info["models"],
                    "has_api_key": self.has_api_key(name),
                }
                for name, info in PROVIDERS.items()
            },
            "retrieval": self.get_retrieval(),
        }


_default_store: SettingsStore | None = None


def get_settings_store(path: str | Path | None = None) -> SettingsStore:
    global _default_store
    if path is not None:
        return SettingsStore(path)
    if _default_store is None:
        _default_store = SettingsStore()
    return _default_store


def reset_settings_store() -> None:
    """Drop the process-wide store (used by tests)."""
    global _default_store
    _default_store = None


__all__ = [
    "PROVIDERS",
    "SettingsStore",
    "get_settings_store",
    "reset_settings_store",
]