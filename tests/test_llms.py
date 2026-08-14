from types import SimpleNamespace

import pytest

from agent.llms import extract_llm_text, extract_usage_tokens, get_llms
from settings import reset_settings_store


@pytest.fixture(autouse=True)
def _isolated_settings():
    """Drop any process-wide settings store so env defaults are authoritative."""
    reset_settings_store()
    yield
    reset_settings_store()


def test_extract_llm_text_from_plain_string():
    response = SimpleNamespace(content="plain answer")
    assert extract_llm_text(response) == "plain answer"


def test_extract_llm_text_from_gemini_content_blocks():
    response = SimpleNamespace(
        content=[
            {"type": "text", "text": "first "},
            {"type": "text", "text": "second"},
        ]
    )
    assert extract_llm_text(response) == "first second"


def test_extract_llm_text_from_mixed_blocks():
    response = SimpleNamespace(
        content=[
            "str block ",
            {"type": "text", "text": "dict block"},
            {"type": "tool_use", "text": ""},
        ]
    )
    assert extract_llm_text(response) == "str block dict block"


def test_extract_llm_text_empty():
    assert extract_llm_text(SimpleNamespace(content=None)) == ""
    assert extract_llm_text(SimpleNamespace(content=[])) == ""


def test_extract_usage_tokens_gemini_style():
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    assert extract_usage_tokens(response) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_extract_usage_tokens_openai_style():
    response = SimpleNamespace(usage_metadata={"prompt_tokens": 20, "completion_tokens": 7})
    assert extract_usage_tokens(response) == {
        "input_tokens": 20,
        "output_tokens": 7,
        "total_tokens": 27,
    }


def test_extract_usage_tokens_missing():
    assert extract_usage_tokens(SimpleNamespace()) is None
    assert extract_usage_tokens(SimpleNamespace(usage_metadata={})) is None


def test_get_llms_caches_instances(monkeypatch):
    built = []

    def fake_build(model, provider, api_key):
        built.append((model, provider, api_key))
        return SimpleNamespace(model=model), SimpleNamespace(model=model)

    monkeypatch.setattr("agent.llms._build_llms", fake_build)

    gen1, cls1 = get_llms("model-a", "openai", "k1")
    gen2, cls2 = get_llms("model-a", "openai", "k1")

    assert gen1 is gen2
    assert cls1 is cls2
    assert len(built) == 1

    get_llms("model-b", "anthropic", None)
    assert len(built) == 2


def test_get_llms_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr("agent.llms.DEFAULT_MODEL", "env-model")
    monkeypatch.setattr("agent.llms.DEFAULT_PROVIDER", "google_genai")
    monkeypatch.setattr("agent.llms.DEFAULT_API_KEY", "env-key")

    built = []

    def fake_build(model, provider, api_key):
        built.append((model, provider, api_key))
        return SimpleNamespace(model=model), SimpleNamespace(model=model)

    monkeypatch.setattr("agent.llms._build_llms", fake_build)

    get_llms()
    assert built[-1] == ("env-model", "google_genai", "env-key")

    get_llms(model="custom", api_key="custom-key")
    assert built[-1] == ("custom", "google_genai", "custom-key")
