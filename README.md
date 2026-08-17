# Research-Assistant

A local-first AI research assistant with a clean web UI for chatting with LLMs, uploading documents, and searching your knowledge base with hybrid RAG. Your chats, documents, and settings live on your own machine. Models, retrieval options, and API keys are configurable from a Settings page — no code editing required.

[![CI/CD](https://github.com/AbdelrhmanEbied/research-assistant/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/AbdelrhmanEbied/research-assistant/actions/workflows/ci-cd.yaml)

## Demo

![Demo](assets/demo.gif)

> The demo GIF is from an early release — the current version adds Thinking mode, agent tools, and much more (see the features below).

## What it does

Research-Assistant gives you a simple GUI where you can:

* chat with Google Gemini, OpenAI, or Anthropic Claude
* switch models / providers and set API keys from the Settings page
* upload documents into a local knowledge base
* scope retrieval to specific documents or use live web search via Tavily
* force a mode (chat, summarize, compare, explain) or source (documents, web, chat) per request
* **pick a response mode per request — Fast for low latency, or Thinking for a Claude-style reasoning agent** that streams its chain of thought and runs local tools (calculator, Python sandbox, document reading) before answering
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

### Agent & chat

* Multi-provider LLM support (Gemini, OpenAI, Claude)
* **Thinking / Fast modes** — a per-request toggle next to the composer:
  * **Fast** — single-pass generation for low latency (Gemini 3 models are configured with `thinking_level="minimal"`)
  * **Thinking** — a LangGraph ReAct loop that reasons step by step (Gemini 3 uses `thinking_level="high"` + `include_thoughts=True`), can call tools, and streams a collapsible **Thinking** panel above the final answer
* **Live thinking stream** — the provider's actual thought content (Gemini exposes it as `thinking` content blocks) streams progressively into the panel while generating, with live tool statuses such as *Running Python...*, *Reading documents...*, and *Calling calculator...*
* **Local-first agent tools** (no extra dependencies):
  * `calculator` — safe AST-whitelisted arithmetic using `math`/`statistics`
  * `python_code_executor` — runs Python in an isolated subprocess (`sys.executable -I`) in a per-conversation workspace with a 30s timeout
  * `list_documents` / `read_document` — discover and read the documents scoped to the current conversation
* Settings page for LLM defaults, API keys, and retrieval defaults
* Per-request mode/source overrides and retrieval controls (search type, rerank, limit, web search depth)
* Local chat UI with streaming responses, citations, and per-response details
* **Single send/stop button** — the composer button toggles between sending a message and stopping generation while the model is replying
* Chat history stored locally with SQLite and SQLAlchemy (WAL mode)
* Conversation search, rename, export, and paginated history
* Regenerate answers

### Retrieval & documents

* Document upload and management, with compare and summarize
* **Fair multi-document retrieval** — compare queries are grouped per document (each in-scope document is queried separately with `ceil(limit / n)` chunks) and the reranker greedily guarantees ≥1 chunk per document before filling by relevance, so one document can't dominate the results
* Link documents to a conversation for scoped retrieval
* Hybrid RAG retrieval with Qdrant (dense, sparse, and hybrid search types)
* Live web search via Tavily (basic / advanced depth)
* FastEmbed embeddings and cross-encoder reranking

### Operations

* Telemetry: local SQLite analytics store and in-app dashboard
* Dockerized: single-image container with pre-baked embedding models and health checks
* Kubernetes-ready: manifests under `k8s/` deploy a single-replica app with a persistent volume, health probes, and ingress
* CI/CD pipeline: lint, format, tests, image publishing to GHCR, and auto-deploy to Kubernetes

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
5. In **Thinking** mode the request is routed through a ReAct loop: the model reasons (streaming its thoughts), optionally calls local tools, then produces a final answer. The backend tags each streamed event so the frontend can separate thinking content from the final answer.
6. Documents are embedded with FastEmbed and stored in Qdrant for hybrid search.
7. Chats, messages, and metadata are stored locally; every request is recorded in the local telemetry store.

### Thinking-mode streaming pipeline

```
Gemini API → LangChain (thinking_level=high, include_thoughts=true)
           → LangGraph agent_reason ↔ execute_tools loop
           → FastAPI stream (thought blocks + tool statuses + final answer)
           → frontend Thinking panel (collapsible) + answer bubble
```

Gemini 3 models expose their reasoning as `{"type": "thinking"}` content blocks. The backend streams those blocks live, adds human-readable tool statuses from `on_tool_start` events, and buffers the answer text until the call ends — so thinking and answer events can never be mixed. Models that don't expose thinking simply stream their answer with no panel.

## Project structure

```bash
research-assistant/
├── agent/
│   ├── agent_schemas.py
│   ├── graph.py
│   ├── llms.py
│   ├── nodes.py
│   ├── prompts.py
│   ├── tools.py            # calculator, python_code_executor, document tools
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
│   ├── test_retrieval.py
│   ├── test_routers.py
│   ├── test_settings.py
│   ├── test_telemetry.py
│   └── test_tools.py
├── .github/
│   └── workflows/ci-cd.yaml
├── k8s/
│   ├── configmap.yaml
│   ├── deployment.yaml
│   ├── ingress.yaml
│   ├── namespace.yaml
│   ├── pvc.yaml
│   └── service.yaml
├── scripts/
│   └── deploy.sh
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

### Using Thinking mode

1. In the composer toolbar, toggle **Fast / Thinking** (Thinking is recommended for analysis, math, comparisons, and anything that benefits from step-by-step reasoning).
2. Send your message. The assistant streams its reasoning into a collapsible **Thinking** panel above the final answer, including live tool activity (Python execution, document reads, calculator calls).
3. Click the panel header to expand or collapse it. The thinking content is saved with the message and remains expandable when you reopen the conversation.

> Thinking mode works best with Gemini 3 models (which support `thinking_level` and `include_thoughts`). On other providers/models it degrades gracefully to a tool-enabled agent loop without a thought stream.

## Running tests

```bash
uv run pytest
```

Tests live in `tests/` and cover the agent graph (including the thinking tool loop, thinking-config wiring, and fast-mode bypass), RAG builders, per-document retrieval and diversified reranking, local tool execution, chat streaming (thinking markers, tool statuses, error handling), routers, settings, and telemetry.

```bash
uv run ruff check .
uv run ruff format --check .
```

## RAG evaluation

The `evaluation/` package scores the real retrieval and generation pipeline (`RAGService.prepare`) against labeled datasets. There are two layers:

* **Retrieval metrics** — offline, deterministic, no LLM calls: **hit rate@k, recall@k, precision@k, MRR, NDCG@k** against ground-truth chunk or document ids.
* **LLM-as-judge metrics** — **faithfulness** (is the answer grounded in the retrieved context?), **answer relevance**, and **context relevance**, scored by the app's own LLM providers via structured output. Requires configured API keys.

### Dataset format

A JSONL file with one `EvalItem` per line:

```json
{"query": "How does hybrid retrieval work?", "relevant_document_ids": ["hybrid_retrieval"]}
{"query": "What is attention?", "relevant_chunk_ids": ["c1", "c3"], "reference_answer": "..."}
```

Provide `relevant_chunk_ids`, `relevant_document_ids`, or both. `reference_answer` is optional and unused by the current judges. A runnable sample lives in `evaluation/sample_data/`.

### Running

```bash
# retrieval-only against a fresh corpus (indexed into a temp store)
uv run python scripts/evaluate_rag.py \
  --dataset evaluation/sample_data/example.jsonl \
  --corpus evaluation/sample_data/corpus \
  --search-types dense,sparse,hybrid --k 3

# against an existing indexed store
uv run python scripts/evaluate_rag.py --dataset my-data.jsonl --db-path ./qdrant_db

# add LLM-as-judge metrics (needs API keys; 1 generation + 3 judge calls per query)
uv run python scripts/evaluate_rag.py --dataset my-data.jsonl --corpus ./docs --judge \
  --judge-model gemini-3.5-flash-lite

# save the full JSON report
uv run python scripts/evaluate_rag.py --dataset data.jsonl --corpus ./docs --output report.json
```

Run `uv run python scripts/evaluate_rag.py --help` for all options.

### Baseline results

Run on `evaluation/sample_data/` (3 documents, one clearly relevant per query, top-k 3, no rerank):

| search type | hit_rate | recall@3 | precision@3 | mrr | ndcg@3 |
|---|---|---|---|---|---|
| dense | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 |
| sparse | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 |

Perfect hit/recall/MRR/NDCG confirm the pipeline (chunking, embedding, Qdrant search) works end-to-end. Treat these as a **smoke test**, not real quality signal: the sample corpus has one clearly relevant document per query, so it does not stress retrieval discrimination. `precision@3` is capped at ~0.33 because only one of the three returned documents is relevant; sparse scores higher because BM25 returns only exact-match chunks. Build a larger dataset with overlapping topics and chunk-level labels for meaningful numbers.

## Deploying to Kubernetes

Kubernetes manifests live in `k8s/` and deploy a single-replica app backed by a persistent volume — the app keeps all state (SQLite, settings, and the embedded Qdrant store) on a 1Gi PVC mounted at `/data`.

* `namespace.yaml` — the `research-assistant` namespace
* `configmap.yaml` — non-secret defaults (model, provider, telemetry, environment)
* `deployment.yaml` — one replica with a `Recreate` strategy (required for the ReadWriteOnce volume), liveness/readiness probes on `/health`, and resource requests/limits
* `service.yaml` — internal ClusterIP service on port 8000
* `ingress.yaml` — routes your domain to the app through the nginx ingress controller, with long proxy timeouts and buffering disabled for response streaming
* `pvc.yaml` — 1Gi `ReadWriteOnce` persistent volume for `/data`

### Secrets

API keys are never committed. `scripts/deploy.sh` reads `.env` (gitignored), builds a Kubernetes Secret from the four `*_API_KEY` values, applies it, and deletes the temporary file afterwards. The Deployment injects them via `envFrom.secretRef`. Note that a Secret is only base64-encoded, not encrypted — anyone with cluster read access can decode it. Use Sealed Secrets or an external secrets operator for production hardening.

### Deploy

```bash
# first time: creates the Secret from .env and applies all manifests
bash scripts/deploy.sh

# use a specific image instead of :latest
IMAGE="ghcr.io/<owner>/research-assistant:<tag>" bash scripts/deploy.sh

# preview without ingress
kubectl port-forward svc/research-assistant 8000:8000 -n research-assistant
```

For local kind/minikube there is no cloud load balancer, so point the ingress host at the cluster with `/etc/hosts` and port-forward the nginx controller service.

## CI/CD

A GitHub Actions workflow (`.github/workflows/ci-cd.yaml`) runs on every push to `main`, `v*` tags, and all pull requests:

* **Lint & Test** — sets up Python 3.14 with `uv`, installs the locked dependencies, then runs `ruff check`, `ruff format --check`, and `pytest`.
* **RAG Evaluation** — after tests pass, runs the retrieval-only evaluation on the sample dataset/corpus and fails the build if `hit_rate` or `mrr` drop below 0.7.
* **Build & Push** — on pushes only (after lint/tests pass), builds the Docker image with BuildKit caching and `BAKE_MODELS=true` (so the embedding models ship inside the image) and pushes it to the GitHub Container Registry (`ghcr.io/<owner>/research-assistant`) tagged with the short commit SHA, `latest` on the default branch, and semver tags for `v*` releases.
* **Deploy** — on pushes to `main` only (after the image is built), deploys the new image to Kubernetes from a self-hosted runner using `scripts/deploy.sh`. The runner must have `kubectl` access to the cluster and a `.env` file with your API keys; secrets are built from it at deploy time and never stored in the repository.

## Credits

* Frontend by [geopero123](https://github.com/geopero123)
* Backend, agent, and RAG pipeline by [AbdelrhmanEbied](https://github.com/AbdelrhmanEbied)

## License

This project is licensed under the Apache License 2.0.
