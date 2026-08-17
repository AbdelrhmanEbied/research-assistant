from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agent.llms import build_structured_llm

logger = logging.getLogger(__name__)


class FaithfulnessScore(BaseModel):
    """Share of the answer's factual claims supported by the retrieved context."""

    score: float = Field(
        ge=0,
        le=1,
        description="Fraction of the answer's claims that are supported by the context (0 to 1).",
    )
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class RelevanceScore(BaseModel):
    """Generic 0-1 relevance judgement with a short justification."""

    score: float = Field(ge=0, le=1)
    reasoning: str = Field(default="", description="One sentence explaining the score.")


FAITHFULNESS_PROMPT = """You are evaluating a RAG answer for faithfulness.

Question: {question}

Retrieved context:
{context}

Answer:
{answer}

Task:
1. Split the answer into individual factual claims.
2. For each claim, decide whether it is SUPPORTED by the retrieved context,
   contradicts it, or is neither present nor contradicted.
3. score = (number of supported claims) / (total number of claims).
   Claims not present in the context are NOT supported (the model may be
   inventing them), so they count against the score.
4. Put the supported claims in supported_claims and the rest in
   unsupported_claims. If the answer has no claims, score is 1."""

ANSWER_RELEVANCE_PROMPT = """You are evaluating how well an answer addresses a question.

Question: {question}

Answer:
{answer}

Score how completely and directly the answer responds to the question.
- 1.0: fully answers the question, no missing key points.
- 0.5: partially answers; some key aspects are missing or off-topic.
- 0.0: the answer is irrelevant or does not address the question.
Return the score and a one-sentence reasoning."""

CONTEXT_RELEVANCE_PROMPT = """You are evaluating how relevant a retrieved context is to a question.

Question: {question}

Retrieved context:
{context}

Score the fraction of the retrieved context that actually helps answer the
question. Ignore boilerplate and metadata.
- 1.0: most of the context directly answers the question.
- 0.5: roughly half is useful.
- 0.0: the context is irrelevant to the question.
Return the score and a one-sentence reasoning."""


def _score(
    schema: type[BaseModel],
    prompt: str,
    *,
    model: str | None,
    model_provider: str | None,
    api_key: str | None,
) -> BaseModel | None:
    llm = build_structured_llm(schema, model, model_provider, api_key, temperature=0)
    try:
        result = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - judges must be fail-safe
        logger.warning("Judge LLM call failed: %s", exc)
        return None
    return schema.model_validate(result)


def score_faithfulness(
    question: str,
    context: str,
    answer: str,
    *,
    model: str | None = None,
    model_provider: str | None = None,
    api_key: str | None = None,
) -> FaithfulnessScore | None:
    prompt = FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
    result = _score(
        FaithfulnessScore, prompt, model=model, model_provider=model_provider, api_key=api_key
    )
    return result if isinstance(result, FaithfulnessScore) else None


def score_answer_relevance(
    question: str,
    answer: str,
    *,
    model: str | None = None,
    model_provider: str | None = None,
    api_key: str | None = None,
) -> RelevanceScore | None:
    prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    result = _score(
        RelevanceScore, prompt, model=model, model_provider=model_provider, api_key=api_key
    )
    return result if isinstance(result, RelevanceScore) else None


def score_context_relevance(
    question: str,
    context: str,
    *,
    model: str | None = None,
    model_provider: str | None = None,
    api_key: str | None = None,
) -> RelevanceScore | None:
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
    result = _score(
        RelevanceScore, prompt, model=model, model_provider=model_provider, api_key=api_key
    )
    return result if isinstance(result, RelevanceScore) else None
