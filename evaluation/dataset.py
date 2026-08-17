from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    """One evaluation sample: a query plus its ground truth.

    ``relevant_chunk_ids`` are used for chunk-level retrieval metrics;
    ``relevant_document_ids`` for document-level metrics. At least one of the
    two should be provided. ``reference_answer`` is optional and is only used
    by LLM-as-judge metrics that need a gold answer.
    """

    query: str
    reference_answer: str | None = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    conversation_id: str | None = None


def load_dataset(path: str | Path) -> list[EvalItem]:
    """Load a JSONL dataset (one ``EvalItem`` per line, blank lines ignored)."""
    items: list[EvalItem] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(EvalItem.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid dataset line {line_no}: {exc}") from exc

    if not items:
        raise ValueError(f"No evaluation items found in {path}")

    return items
