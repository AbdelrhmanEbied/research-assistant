#!/usr/bin/env python
"""Evaluate the RAG pipeline against a JSONL dataset.

Retrieval-only (no LLM calls, no API keys):

    uv run python scripts/evaluate_rag.py --dataset evaluation/sample_data/example.jsonl

Retrieval against a fresh corpus (indexed into a temp store):

    uv run python scripts/evaluate_rag.py --dataset data.jsonl --corpus ./docs

Retrieval + LLM-as-judge (needs API keys configured):

    uv run python scripts/evaluate_rag.py --dataset data.jsonl --corpus ./docs --judge

Run ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from evaluation.dataset import load_dataset
from evaluation.report import build_report, render_markdown
from evaluation.runner import RAGEvaluator, build_service, index_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline against a JSONL dataset."
    )
    parser.add_argument("--dataset", required=True, help="JSONL dataset (EvalItem per line)")
    parser.add_argument(
        "--db-path",
        default="./qdrant_db",
        help="Existing Qdrant store to evaluate against (default: ./qdrant_db)",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional directory of documents to index into a temp store first",
    )
    parser.add_argument(
        "--search-types",
        default="hybrid",
        help="Comma-separated search types: dense,sparse,hybrid (default: hybrid)",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for ranked metrics (default: 5)")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument(
        "--judge", action="store_true", help="Run LLM-as-judge metrics (needs API keys)"
    )
    parser.add_argument("--judge-model", default=None, help="Override the LLM used for judging")
    parser.add_argument(
        "--judge-provider", default=None, help="Override the provider used for judging"
    )
    parser.add_argument("--output", default=None, help="Write the full JSON report to this file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    items = load_dataset(args.dataset)
    search_types = [s.strip() for s in args.search_types.split(",") if s.strip()]
    if not search_types:
        raise SystemExit("--search-types must contain at least one of: dense,sparse,hybrid")

    tmp_dir: tempfile.TemporaryDirectory | None = None
    db_path = args.db_path
    if args.corpus:
        tmp_dir = tempfile.TemporaryDirectory(prefix="rag-eval-")
        db_path = str(Path(tmp_dir.name) / "qdrant_db")
    else:
        Path(db_path).mkdir(parents=True, exist_ok=True)

    try:
        service = build_service(db_path, rerank=args.rerank)
        if args.corpus:
            index_corpus(service, args.corpus)

        evaluator = RAGEvaluator(service, k=args.k)
        retrieval = evaluator.evaluate_retrieval(items, search_types, rerank=args.rerank)

        quality = []
        if args.judge:
            quality = evaluator.evaluate_quality(
                items,
                search_types[0],
                rerank=args.rerank,
                model=args.judge_model,
                model_provider=args.judge_provider,
            )
    finally:
        service.qdrant_manager.close_client()
        if tmp_dir is not None:
            tmp_dir.cleanup()

    report = build_report(
        config={
            "dataset": args.dataset,
            "db_path": args.db_path if not args.corpus else "(temp corpus)",
            "k": args.k,
            "search_types": search_types,
            "rerank": args.rerank,
            "judge": args.judge,
            "items": len(items),
        },
        retrieval=retrieval,
        quality=quality,
    )

    print(render_markdown(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to {args.output}")


if __name__ == "__main__":
    main()
