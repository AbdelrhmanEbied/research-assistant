# Research-Assistant

A local-first AI research assistant with a clean web UI for chatting with LLMs, uploading documents, and searching your knowledge base with hybrid RAG. Your chats, documents, and settings live on your own machine. Models, retrieval options, and API keys are configurable from a Settings page — no code editing required.

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
* FastAPI backend
* Frontend built with HTML, CSS, and JavaScript

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
├── .env.example
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/AbdelrhmanEbied/research-assistant.git
cd research-assistant
```

### 2. Create a virtual environment

Requires Python 3.14+.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
# If you are using pip
pip install -r requirements.txt

# Or if you are using uv
uv sync
```

### 4. Configure your environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

On Windows:

```bash
copy .env.example .env
```

```env
# LLM — used as bootstrap defaults, overridable from the Settings page
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
LLM_MODEL=gemini-3.5-flash-lite
LLM_PROVIDER=google_genai

# Web search
TAVILY_API_KEY=your_tavily_api_key_here

# Telemetry
TELEMETRY_ENABLED=true
TELEMETRY_DB_PATH=telemetry.db
ENVIRONMENT=development
APP_VERSION=0.1.0
```

### 5. Run the app

```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

> If you use [uv](https://github.com/astral-sh/uv) as your package manager, you can run it instead with:
> ```bash
> uv run uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
> ```

Then open `http://127.0.0.1:8000` in your browser.

## Configuration

Settings are managed from the **Settings** page in the UI and persisted to a local `settings.json` (which is gitignored because it may hold API keys). Environment variables only seed the defaults on first run.

* **LLM** — choose the default model and provider (Gemini, OpenAI, or Claude) and store the corresponding API keys. You can also override the model per request from the chat input row.
* **Retrieval** — set defaults for the search type (hybrid, dense, sparse), the number of documents retrieved, reranking, and the Tavily web search depth (basic, advanced).

## Running tests

```bash
pytest
```

## Credits

* Frontend by [geopero123](https://github.com/geopero123)
* Backend, agent, and RAG pipeline by [AbdelrhmanEbied](https://github.com/AbdelrhmanEbied)

## License

This project is licensed under the Apache License 2.0.