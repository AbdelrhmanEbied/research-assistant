# Research-Assistant

A local-first AI research assistant with a clean web UI for chatting with LLMs, uploading documents, and searching your knowledge base with hybrid RAG. Your chats, documents, and settings live on your own machine. Models, retrieval options, and API keys are configurable from a Settings page — no code editing required.

[![CI/CD](https://github.com/AbdelrhmanEbied/research-assistant/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/AbdelrhmanEbied/research-assistant/actions/workflows/ci-cd.yml)

## Demo

![Demo](assets/demo.gif)

## What it does

Research-Assistant gives you a simple GUI where you can:

* chat with Google Gemini, OpenAI, or Anthropic Claude
* switch models / providers and set API keys from the Settings page
* upload documents into a local knowledge base
* scope retrieval to specific documents or use live web search via Tavily
* force a mode (chat, summarize, compare, explain) or source (documents, web, chat) per request
* stream responses with per-response details (model, retrieval options, latency, token usage) and citations
* regenerate an answer, export a conversation (markdown or JSON), rename chats, and search across them
* view an analytics dashboard of requests, spans, and token usage
* keep chat history stored locally on your machine

Hybrid search is enabled by default, and dense-only and sparse-only retrieval are also available.

## Why I built it

This project was built to learn and practice:

* RAG systems
* AI agents
* FastAPI backend development
* LangChain and LangGraph orchestration
* local document search and retrieval
* frontend/backend communication for AI applications

## Features

* Multi-provider LLM support (Gemini, OpenAI, Claude)
* Settings page for LLM defaults, API keys, and retrieval defaults
* Per-request mode/source overrides and retrieval controls (search type, rerank, limit, web search depth)
* Local chat UI with streaming responses, citations, and per-response details
* Chat history stored locally with SQLite and SQLAlchemy (WAL mode)
* Conversation search, rename, export, and paginated history
* Regenerate answers
* Document upload and management, with compare and summarize
* Link documents to a conversation for scoped retrieval
* Hybrid RAG retrieval with Qdrant
* Live web search via Tavily (basic / advanced depth)
* FastEmbed embeddings and cross-encoder reranking
* Telemetry: local SQLite analytics store and in-app dashboard
* Dockerized: single-image container with pre-baked embedding models and health checks
* CI/CD pipeline: lint, format, tests, and automated image publishing to GHCR

## Tech Stack

* FastAPI
* SQLAlchemy
* SQLite
* LangChain
* LangGraph
* Google Gemini / OpenAI / Anthropic Claude
* Qdrant
* FastEmbed
* Tavily
* HTML, CSS, JavaScript
* Docker

## How it works

1. You open the website.
2. You send a message through the frontend (optionally picking a mode, source, and retrieval options).
3. The frontend sends the request to the FastAPI backend.
4. The LangGraph agent classifies the request — or honors your explicit overrides — retrieves context from your documents (Qdrant) or the web (Tavily) as needed, and streams the answer back.
5. Documents are embedded with FastEmbed and stored in Qdrant for hybrid search.
6. Chats, messages, and metadata are stored locally; every request is recorded in the local telemetry store.

## Project structure

```bash
research-assistant/
├── agent/
│   ├── agent_schemas.py
│   ├── graph.py
│   ├── llms.py
│   ├── nodes.py
│   ├── prompts.py
│   ├── web_service.py
│   └── __init__.py
├── app/
│   ├── backend/
│   │   ├── main.py
│   │   ├── lifespan.py
│   │   ├── __init__.py
│   │   ├── database/
│   │   │   ├── base.py
│   │   │   ├── database.py
│   │   │   ├── migrations.py
│   │   │   ├── models.py
│   │   │   └── repositories.py
│   │   ├── routers/
│   │   │   ├── chat_router.py
│   │   │   ├── document_router.py
│   │   │   ├── settings_router.py
│   │   │   └── telemetry_router.py
│   │   ├── schemas/
│   │   │   ├── chat.py
│   │   │   ├── conversation.py
│   │   │   └── document.py
│   │   └── services/
│   │       ├── chat_service.py
│   │       └── document_service.py
│   ├── frontend/
│   │   ├── index.html
│   │   ├── css/
│   │   │   └── styles.css
│   │   └── js/
│   │       ├── analytics.js
│   │       ├── chat.js
│   │       ├── conversations.js
│   │       ├── documents.js
│   │       ├── main.js
│   │       ├── markdown.js
│   │       ├── motion.js
│   │       ├── render.js
│   │       ├── settings.js
│   │       ├── state.js
│   │       └── utils.js
│   └── __init__.py
├── rag/
│   ├── builders.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── loader.py
│   ├── qdrant_manager.py
│   ├── rag_schemas.py
│   ├── rag_service.py
│   ├── reranker.py
│   ├── retriever.py
│   └── __init__.py
├── settings/
│   ├── store.py
│   └── __init__.py
├── telemetry/
│   ├── config.py
│   ├── storage.py
│   ├── tokens.py
│   ├── tracker.py
│   └── __init__.py
├── tests/
│   ├── test_agent_graph.py
│   ├── test_builders.py
│   ├── test_chat_service.py
│   ├── test_llms.py
│   ├── test_routers.py
│   ├── test_settings.py
│   └── test_telemetry.py
├── .github/
│   └── workflows/ci-cd.yml
├── .env.example
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── paths.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting started

You can run Research-Assistant either **with Docker** (recommended) or **locally with uv**.

### Option A — Run with Docker (recommended)

Requires [Docker](https://docs.docker.com/engine/install/) with the Compose plugin.

```bash
git clone https://github.com/AbdelrhmanEbied/research-assistant.git
cd research-assistant

cp .env.example .env
# fill in your API keys (see "Environment variables" below)

docker compose up --build -d
```

Open `http://localhost:8000`.

What the compose setup does:

* builds the image with `BAKE_MODELS=true`, pre-downloading the FastEmbed embedding, sparse (BM25), and reranker models into the image so the first run starts fast
* maps port `8000` on your host to the app
* loads your keys from `.env`
* persists data in two named volumes:
  * `research_data` — chats, documents, uploads, and the telemetry store (`/data`)
  * `fastembed_cache` — model downloads (`/home/appuser/.cache/fastembed`)

Useful commands:

```bash
docker compose logs -f            # follow the app logs
docker compose down               # stop the app (keeps your volumes)
docker compose down -v            # stop AND delete your data
docker compose pull               # pull a prebuilt image instead of building
```

> Building with `BAKE_MODELS=true` makes the image larger. Set it to `false` in `docker-compose.yml` to skip pre-downloading; models are then fetched on demand at runtime (the first retrieval will be slower).

### Option B — Run locally with uv

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AbdelrhmanEbied/research-assistant.git
cd research-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

uv sync

cp .env.example .env
# fill in your API keys (see "Environment variables" below)

uv run uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

> Prefer `uv` — the lockfile (`uv.lock`) pins exact versions. With pip you can install the project directly with `pip install .`.

Then open `http://127.0.0.1:8000` in your browser.

### Environment variables

Copy `.env.example` to `.env` and fill in your keys. None are strictly required to boot — set the keys for the providers you want to use:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `LLM_MODEL` | Default model (e.g. `gemini-3.5-flash-lite`) |
| `LLM_PROVIDER` | Default provider: `google_genai`, `openai`, or `anthropic` |
| `TAVILY_API_KEY` | Key for live web search |
| `TELEMETRY_ENABLED` | `true`/`false` to toggle the telemetry store |
| `TELEMETRY_DB_PATH` | Path to the telemetry SQLite file |
| `DATA_DIR` | Directory for runtime data (set to `/data` in Docker) |
| `ENVIRONMENT` | `development`, `staging`, `production`, … |
| `APP_VERSION` | Version tag recorded in telemetry |

Keys and defaults can also be changed at runtime from the **Settings** page — env vars only seed the defaults on first run.

## Configuration

Settings are managed from the **Settings** page in the UI and persisted to a local `settings.json` (which is gitignored because it may hold API keys). Environment variables only seed the defaults on first run.

* **LLM** — choose the default model and provider (Gemini, OpenAI, or Claude) and store the corresponding API keys. You can also override the model per request from the chat input row.
* **Retrieval** — set defaults for the search type (hybrid, dense, sparse), the number of documents retrieved, reranking, and the Tavily web search depth (basic, advanced).

## Running tests

```bash
uv run pytest
```

Tests live in `tests/` and cover the agent graph, RAG builders, chat streaming, routers, settings, and telemetry.

## CI/CD

A GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push to `main`, `v*` tags, and all pull requests:

* **Lint & Test** — sets up Python 3.14 with `uv`, installs the locked dependencies, then runs `ruff check`, `ruff format --check`, and `pytest`.
* **Build & Push** — on pushes only (after lint/tests pass), builds the Docker image with BuildKit caching and pushes it to the GitHub Container Registry (`ghcr.io/<owner>/research-assistant`) tagged with the short commit SHA, `latest` on the default branch, and semver tags for `v*` releases.

## Credits

* Frontend by [geopero123](https://github.com/geopero123)
* Backend, agent, and RAG pipeline by [AbdelrhmanEbied](https://github.com/AbdelrhmanEbied)

## License

This project is licensed under the Apache License 2.0.