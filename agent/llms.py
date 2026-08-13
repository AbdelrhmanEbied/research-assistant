import os
from contextvars import ContextVar
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from agent.agent_schemas import RouteQuery
from settings import get_settings_store
from settings.store import PROVIDERS

load_dotenv()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "google_genai")
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY")

#: Request-scoped API key supplied from the frontend. Kept in a context var so
#: it never lands in the persisted graph state (checkpoints.db).
_request_api_key: ContextVar[str | None] = ContextVar("request_api_key", default=None)


def set_request_api_key(api_key: str | None) -> None:
    _request_api_key.set(api_key)


def get_request_api_key() -> str | None:
    return _request_api_key.get()


def _build_llms(model: str, model_provider: str, api_key: str | None = None):
    kwargs = {"model": model, "model_provider": model_provider}
    if api_key:
        kwargs["api_key"] = api_key

    llm = init_chat_model(**kwargs)
    classifier_llm = init_chat_model(**kwargs).with_structured_output(RouteQuery)
    return llm, classifier_llm


@lru_cache(maxsize=32)
def _cached_llms(
    model: str,
    model_provider: str,
    api_key: str | None,
):
    return _build_llms(model, model_provider, api_key)


def _default_llm_config() -> tuple[str, str]:
    """Resolve default model/provider from the settings store, else env.

    The settings store is the single source of truth for persistent defaults
    set from the Settings page; ``.env`` only seeds it on first run.
    """
    stored = get_settings_store().get_llm()
    model = stored.get("model") or DEFAULT_MODEL
    provider = stored.get("model_provider") or DEFAULT_PROVIDER
    return model, provider


def _resolve_default_api_key(provider: str) -> str | None:
    """Look up the API key for ``provider`` from settings, then env."""
    store = get_settings_store()
    key = store.get_api_key(provider)
    if key:
        return key
    if provider == "google_genai":
        return DEFAULT_API_KEY
    env_key = os.getenv(PROVIDERS.get(provider, {}).get("env_key", ""))
    return env_key or None


def get_llms(
    model: str | None = None,
    model_provider: str | None = None,
    api_key: str | None = None,
):
    """Return ``(generation_llm, classifier_llm)`` for the given configuration.

    Omitted values fall back to the persisted settings (seeded from ``.env``).
    Instances are cached per configuration so repeated calls reuse the same LLM
    objects instead of rebuilding them. No global state is mutated, so
    concurrent requests can each use their own model.
    """
    if model is None or model_provider is None:
        default_model, default_provider = _default_llm_config()
        model = model or default_model
        model_provider = model_provider or default_provider

    if api_key is None:
        api_key = _resolve_default_api_key(model_provider)

    return _cached_llms(model, model_provider, api_key)


def configure_llms(
    model: str,
    model_provider: str,
    api_key: str | None = None,
):
    """Compatibility alias kept for callers that used to mutate global state.

    It simply returns the cached LLM pair for the given configuration and no
    longer affects other requests.
    """
    return get_llms(model=model, model_provider=model_provider, api_key=api_key)


def get_generation_llm():
    return get_llms()[0]


def get_classifier_llm():
    return get_llms()[1]


def extract_llm_text(response) -> str:
    """Extract the text payload from an ``AIMessage``/``AIMessageChunk``.

    Different providers return ``content`` as a plain string (OpenAI,
    Anthropic) or as a list of content blocks (Google), and blocks may be
    strings or dicts. This handles all of those shapes.
    """
    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                if block:
                    parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(text)
        return "".join(parts)

    return getattr(response, "text", "") or ""


def extract_usage_tokens(response) -> dict | None:
    """Return ``{input_tokens, output_tokens, total_tokens}`` if available.

    Gemini exposes ``input_tokens``/``output_tokens`` while OpenAI-style
    providers expose ``prompt_tokens``/``completion_tokens``. Returns ``None``
    when the provider did not attach usage metadata.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")

    if input_tokens is None or output_tokens is None:
        return None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens") or input_tokens + output_tokens,
    }
