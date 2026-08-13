from __future__ import annotations

import os

import pytest

from settings import get_settings_store, reset_settings_store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    reset_settings_store()
    yield tmp_path / "settings.json"
    reset_settings_store()


def test_store_persists_llm_defaults(isolated_store):
    store = get_settings_store(isolated_store)
    store.set_llm("gpt-4o", "openai")

    reloaded = get_settings_store(isolated_store)
    assert reloaded.get_llm() == {"model": "gpt-4o", "model_provider": "openai"}


def test_store_api_key_never_leaks_into_public_dict(isolated_store, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    store = get_settings_store(isolated_store)
    store.set_api_key("openai", "sk-secret-value")

    public = store.public_dict()
    assert public["providers"]["openai"]["has_api_key"] is True
    dumped = str(public)
    assert "sk-secret-value" not in dumped

    assert store.get_api_key("openai") == "sk-secret-value"


def test_store_api_key_falls_back_to_env(isolated_store, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    store = get_settings_store(isolated_store)
    # env key is available but not stored in settings.json
    assert store.has_api_key("google_genai") is True
    assert store.get_api_key("google_genai") is None
    # stored key wins over env
    store.set_api_key("google_genai", "stored-key")
    assert store.get_api_key("google_genai") == "stored-key"


def test_store_retrieval_defaults_and_overrides(isolated_store):
    store = get_settings_store(isolated_store)
    defaults = store.get_retrieval()
    assert defaults["search_type"] == "hybrid"
    assert defaults["rerank"] is True
    assert 1 <= defaults["limit"] <= 50

    store.set_retrieval(search_type="dense", limit=6, rerank=False)
    reloaded = get_settings_store(isolated_store)
    assert reloaded.get_retrieval()["search_type"] == "dense"
    assert reloaded.get_retrieval()["limit"] == 6
    assert reloaded.get_retrieval()["rerank"] is False


def test_settings_endpoints(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.backend.routers.settings_router import router
    from settings.store import SettingsStore

    monkeypatch.setenv("GEMINI_API_KEY", "env-key")

    store = SettingsStore(tmp_path / "settings.json")
    monkeypatch.setattr("app.backend.routers.settings_router.get_settings_store", lambda: store)

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        # keys are never exposed in responses
        res = client.get("/settings/").json()
        assert res["llm"]["default_provider"] == "google_genai"
        assert res["providers"]["google_genai"]["has_api_key"] is True
        assert "env-key" not in str(res)

        res = client.put("/settings/llm", json={"model": "gpt-4o", "model_provider": "openai"})
        assert res.status_code == 200
        assert client.get("/settings/").json()["llm"] == {
            "default_model": "gpt-4o",
            "default_provider": "openai",
        }

        res = client.put("/settings/api-keys", json={"provider": "openai", "api_key": "sk-x"})
        assert res.status_code == 200
        assert store.get_api_key("openai") == "sk-x"
        # still never echoed back
        assert "sk-x" not in str(client.get("/settings/").json())

        res = client.put(
            "/settings/api-keys",
            json={"provider": "openai", "api_key": ""},
        )
        assert res.status_code == 200
        assert store.get_api_key("openai") is None

        res = client.put(
            "/settings/retrieval",
            json={"search_type": "sparse", "limit": 8, "rerank": False},
        )
        assert res.status_code == 200
        assert client.get("/settings/").json()["retrieval"]["search_type"] == "sparse"