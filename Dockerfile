FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DATA_DIR=/data \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser

ARG BAKE_MODELS=false
RUN <<'EOF'
if [ "$BAKE_MODELS" = "true" ]; then
  uv run python - <<'PYEOF'
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

cache_dir = str(Path.home() / ".cache" / "fastembed")
SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=cache_dir)
TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=cache_dir)
TextCrossEncoder("Xenova/ms-marco-MiniLM-L-12-v2", cache_dir=cache_dir)
PYEOF
fi
EOF

EXPOSE 8000
VOLUME /data

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]