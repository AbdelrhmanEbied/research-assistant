from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from evaluation.retrieval_metrics import average_metrics
from evaluation.runner import QualityResult, RetrievalResult


def _config_key(search_type: str, rerank: bool) -> str:
    return f"{search_type}/rerank" if rerank else f"{search_type}/no-rerank"


def aggregate_retrieval(results: list[RetrievalResult]) -> dict[str, dict[str, float]]:
    """Average retrieval metrics per (search type, rerank, ground-truth type)."""
    buckets: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)

    for result in results:
        key = _config_key(result.search_type, result.rerank)
        for label, metrics in (
            ("chunk", result.chunk_metrics),
            ("document", result.document_metrics),
        ):
            if metrics is not None:
                buckets[(key, label)].append(metrics)

    aggregated: dict[str, dict[str, float]] = {}
    for (key, label), rows in sorted(buckets.items()):
        average = average_metrics(rows)
        if average is not None:
            aggregated[f"{key}:{label}"] = {**average, "n": len(rows)}

    return aggregated


def aggregate_quality(results: list[QualityResult]) -> dict[str, float]:
    """Average LLM-as-judge scores across the evaluated queries."""
    fields = ("faithfulness", "answer_relevance", "context_relevance")
    values: dict[str, list[float]] = {field: [] for field in fields}

    for result in results:
        for field in fields:
            value = getattr(result, field)
            if value is not None:
                values[field].append(value)

    aggregated: dict[str, float] = {}
    for field, samples in values.items():
        if samples:
            aggregated[field] = round(sum(samples) / len(samples), 4)
            aggregated[f"{field}_n"] = len(samples)

    return aggregated


def build_report(
    *,
    config: dict[str, Any],
    retrieval: list[RetrievalResult],
    quality: list[QualityResult],
) -> dict[str, Any]:
    return {
        "config": config,
        "retrieval": aggregate_retrieval(retrieval),
        "quality": aggregate_quality(quality),
        "per_query_retrieval": [asdict(result) for result in retrieval],
        "per_query_quality": [asdict(result) for result in quality],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# RAG Evaluation Report", ""]

    config = report["config"]
    lines.extend(
        [
            "## Config",
            "",
            f"- Dataset: `{config.get('dataset')}`",
            f"- Store: `{config.get('db_path')}`",
            f"- Top-k: `{config.get('k')}`",
            f"- Search types: `{', '.join(config.get('search_types', []))}`",
            f"- Rerank: `{config.get('rerank')}`",
            f"- LLM judge: `{config.get('judge')}`",
            "",
        ]
    )

    retrieval = report["retrieval"]
    if retrieval:
        lines.append("## Retrieval metrics (average)")
        lines.append("")
        lines.append("| config | hit_rate | recall | precision | mrr | ndcg | n |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, metrics in retrieval.items():
            lines.append(
                "| {key} | {hit_rate:.3f} | {recall:.3f} | {precision:.3f} | {mrr:.3f} | {ndcg:.3f} | {n} |".format(
                    key=key, **metrics
                )
            )
        lines.append("")

    quality = report["quality"]
    if quality:
        lines.append("## LLM-as-judge metrics (average)")
        lines.append("")
        lines.append("| metric | score | n |")
        lines.append("|---|---|---|")
        for field in ("faithfulness", "answer_relevance", "context_relevance"):
            if field in quality:
                lines.append(f"| {field} | {quality[field]:.3f} | {quality[f'{field}_n']} |")
        lines.append("")

    return "\n".join(lines)
