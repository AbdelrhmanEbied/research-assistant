import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator

from fastapi.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from agent.llms import (
    extract_llm_text,
    get_llms,
    get_request_api_key,
    set_request_api_key,
)
from app.backend.database.database import SessionLocal
from app.backend.database.repositories import (
    ConversationRepository,
    DocumentRepository,
    MessageRepository,
)
from app.backend.schemas.chat import ChatRequest, RegenerateRequest
from settings import get_settings_store
from telemetry import clear_request_tracking, start_request_tracking

logger = logging.getLogger(__name__)

#: Terminator appended to the streamed response, followed by the JSON sources.
#: The frontend splits on these markers so citations/details never leak into
#: the markdown body.
SOURCES_MARKER = "@@RESEARCH_SOURCES@@"
DETAILS_MARKER = "@@RESEARCH_DETAILS@@"
#: Sent at the end of the stream when generation failed so the frontend can
#: surface the real error instead of a generic "generation stopped" message.
ERROR_MARKER = "@@RESEARCH_ERROR@@"

#: Provider label used in response details.
PROVIDER_LABELS = {
    "google_genai": "Google Gemini",
    "openai": "OpenAI",
    "anthropic": "Anthropic Claude",
}


class ChatService:
    def __init__(self, graph, checkpointer, rag=None):
        self.graph = graph
        self.checkpointer = checkpointer
        self.rag = rag

    def _embedding_model_name(self) -> str | None:
        try:
            return getattr(self.rag.embedder.dense_model, "model_name", None)
        except Exception:
            return None

    def _generation_model_name(self, llm_config: dict | None) -> str | None:
        if llm_config and llm_config.get("model"):
            return llm_config["model"]
        try:
            generation_llm = get_llms()[0]
            return (
                getattr(generation_llm, "model_name", None)
                or getattr(generation_llm, "model", None)
            )
        except Exception:
            return None

    def _generation_provider(self, llm_config: dict | None) -> str | None:
        if llm_config and llm_config.get("model_provider"):
            return llm_config["model_provider"]
        return get_settings_store().effective_llm()["model_provider"]

    @staticmethod
    def _fallback_title(query: str, max_length: int = 50) -> str:
        title = re.sub(r"\s+", " ", query.strip())
        title = title.rstrip(".,!?;:")

        if not title:
            return "New Chat"

        if len(title) <= max_length:
            return title

        truncated = title[:max_length]

        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]

        return truncated + "..."

    @staticmethod
    def _is_placeholder_title(title: str) -> bool:
        """True when the LLM echoed the default 'New chat' placeholder.

        Models sometimes answer short greetings with literally "New Chat"
        instead of a real title, which would leave the sidebar looking
        unchanged after the auto-titling ran. Treat those as a failed
        generation so the query-derived fallback is used instead.
        """
        normalized = re.sub(r"[\W_]+", " ", title).strip().lower()
        return normalized in {
            "new chat",
            "new chat title",
            "a new chat",
            "chat title",
        }

    async def generate_title(
        self,
        query: str,
        llm_config: dict | None = None,
        max_length: int = 50,
    ) -> str:
        """Ask the LLM for a short title, falling back to query truncation."""
        try:
            cfg = llm_config or {}
            llm = get_llms(
                model=cfg.get("model"),
                model_provider=cfg.get("model_provider"),
                api_key=get_request_api_key(),
            )[0]

            prompt = (
                "Generate a short, concise title (under 6 words) for a chat "
                "that starts with the user's message below. Reply with only "
                "the title, no quotes, no punctuation.\n\n"
                f'Message: "{query}"\n\nTitle:'
            )

            response = await run_in_threadpool(llm.invoke, prompt)
            title = extract_llm_text(response).strip().strip('"').strip()
            title = re.sub(r"\s+", " ", title)

            if title and not self._is_placeholder_title(title):
                if len(title) > max_length:
                    truncated = title[:max_length]
                    if " " in truncated:
                        truncated = truncated.rsplit(" ", 1)[0]
                    return truncated + "..."
                return title
        except Exception as exc:
            logger.warning("LLM title generation failed, using fallback: %s", exc)

        return self._fallback_title(query, max_length)

    # --- database helpers (fresh session per op, off the event loop) ---

    @staticmethod
    async def _run_db(fn):
        return await run_in_threadpool(fn)

    async def _get_conversation(self, conversation_id: int) -> dict | None:
        def _do():
            db = SessionLocal()
            try:
                conv = ConversationRepository(db).get_by_id(conversation_id)
                return {"id": conv.id, "title": conv.title} if conv else None
            finally:
                db.close()

        return await self._run_db(_do)

    async def _get_message_history(self, conversation_id: int) -> list[dict]:
        messages = await self._get_messages(conversation_id)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    async def _get_messages(self, conversation_id: int) -> list[dict]:
        def _do():
            db = SessionLocal()
            try:
                messages = MessageRepository(db).list_for_history(conversation_id)
                return [
                    {"id": m.id, "role": m.role, "content": m.content}
                    for m in messages
                ]
            finally:
                db.close()

        return await self._run_db(_do)

    async def _persist_message(self, conversation_id: int, role: str, content: str) -> int:
        def _do():
            db = SessionLocal()
            try:
                message = MessageRepository(db).add_message(conversation_id, role, content)
                return message.id
            finally:
                db.close()

        return await self._run_db(_do)

    async def _update_message_metadata(self, message_id: int, metadata: dict | None):
        def _do():
            db = SessionLocal()
            try:
                return MessageRepository(db).update_metadata(message_id, metadata)
            finally:
                db.close()

        return await self._run_db(_do)

    async def _delete_messages_after(self, conversation_id: int, after_id: int) -> int:
        def _do():
            db = SessionLocal()
            try:
                return MessageRepository(db).delete_after_id(conversation_id, after_id)
            finally:
                db.close()

        return await self._run_db(_do)

    async def _set_title(self, conversation_id: int, title: str):
        def _do():
            db = SessionLocal()
            try:
                return ConversationRepository(db).update_title(conversation_id, title)
            finally:
                db.close()

        return await self._run_db(_do)

    async def _attach_document_names(self, sources: list[dict]) -> list[dict]:
        doc_ids = {s.get("document_id") for s in sources if s.get("document_id")}
        names: dict[str, str] = {}
        if doc_ids:
            def _do():
                db = SessionLocal()
                try:
                    return {str(doc.id): doc.name for doc in DocumentRepository(db).list_all()}
                finally:
                    db.close()

            all_docs = await self._run_db(_do)
            names = {key: value for key, value in all_docs.items() if key in doc_ids}

        enriched = []
        for source in sources:
            label = source.get("label")
            doc_id = source.get("document_id")
            if not label and doc_id:
                label = names.get(doc_id) or f"Document {doc_id}"
            enriched.append({**source, "label": label})
        return enriched

    @staticmethod
    def _sanitize_llm_config(request_llm_config) -> dict | None:
        """Strip the API key out of the graph state; it stays in a context var."""
        if request_llm_config is None:
            return None
        return {
            "model": request_llm_config.model,
            "model_provider": request_llm_config.model_provider,
        }

    # --- request entry points -------------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str]:
        async for chunk in self._generate(
            conversation_id=request.conversation_id,
            query=request.query,
            llm_config=self._sanitize_llm_config(request.llm_config),
            request_api_key=request.llm_config.api_key if request.llm_config else None,
            mode=request.mode.value if request.mode else None,
            source=request.source.value if request.source else None,
            retrieval=request.retrieval.model_dump() if request.retrieval else None,
            history=None,
            persist_user=True,
            generate_title=True,
        ):
            yield chunk

    async def regenerate(self, request: RegenerateRequest) -> AsyncGenerator[str]:
        """Re-run the answer for the conversation's last user message.

        Trailing assistant messages are dropped (so the old answer is not
        duplicated), the previous turn history is reused, and the user message
        is not persisted again.
        """
        conversation = await self._get_conversation(request.conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {request.conversation_id} not found")

        messages = await self._get_messages(request.conversation_id)
        if not messages:
            raise ValueError("Conversation has no messages to regenerate")

        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            raise ValueError("No user message to regenerate")

        last_user = messages[last_user_idx]

        trailing_ids = [m["id"] for m in messages[last_user_idx + 1:] if m["id"] > last_user["id"]]
        if trailing_ids:
            await self._delete_messages_after(request.conversation_id, last_user["id"])

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[:last_user_idx]
        ]

        async for chunk in self._generate(
            conversation_id=request.conversation_id,
            query=last_user["content"],
            llm_config=self._sanitize_llm_config(request.llm_config),
            request_api_key=request.llm_config.api_key if request.llm_config else None,
            mode=request.mode.value if request.mode else None,
            source=request.source.value if request.source else None,
            retrieval=request.retrieval.model_dump() if request.retrieval else None,
            history=history,
            persist_user=False,
            generate_title=False,
        ):
            yield chunk

    # --- shared streaming core ------------------------------------------------

    async def _generate(
        self,
        *,
        conversation_id: int,
        query: str,
        llm_config: dict | None,
        request_api_key: str | None,
        mode: str | None,
        source: str | None,
        retrieval: dict | None,
        history: list[dict] | None,
        persist_user: bool,
        generate_title: bool,
    ) -> AsyncGenerator[str]:
        set_request_api_key(request_api_key)

        tracker = None
        conversation = None
        assistant_buffer: list[str] = []
        sources: list[dict] = []
        assistant_message_id: int | None = None

        try:
            conversation = await self._get_conversation(conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation {conversation_id} not found")
            if generate_title and not conversation["title"]:
                title = await self.generate_title(query, llm_config)
                if title:
                    await self._set_title(conversation["id"], title)

            if history is None:
                history = await self._get_message_history(conversation_id)

            if persist_user:
                await self._persist_message(conversation_id, "user", query)

            tracker = start_request_tracking(
                route="/chat/",
                conversation_id=conversation_id,
                model=self._generation_model_name(llm_config),
                embedding_model=self._embedding_model_name(),
            )

            state = {
                "query": query,
                "conversation_id": str(conversation_id),
                "history": history,
                "llm_config": llm_config,
                "mode_override": mode,
                "source_override": source,
                "retrieval_config": retrieval,
            }

            config = {
                "configurable": {
                    "thread_id": str(conversation_id),
                }
            }

            async with tracker.span(
                "chat_request",
                span_type="AGENT",
                latency_metric="agent_latency_ms",
            ):
                async for event in self.graph.astream_events(
                    state,
                    config=config,
                    version="v2",
                ):
                    if event["event"] == "on_chat_model_stream":
                        if event.get("metadata", {}).get("langgraph_node") != "generate_answer":
                            continue

                        chunk = event["data"]["chunk"]
                        content = chunk.content

                        if isinstance(content, str):
                            if content:
                                assistant_buffer.append(content)
                                yield content
                            continue

                        for block in content:
                            if isinstance(block, str):
                                if block:
                                    assistant_buffer.append(block)
                                    yield block
                            elif isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    assistant_buffer.append(text)
                                    yield text

                    elif event["event"] == "on_chain_end":
                        if event.get("metadata", {}).get("langgraph_node") != "prepare_prompt":
                            continue
                        output = event["data"].get("output")
                        if isinstance(output, dict) and output.get("sources"):
                            sources = output["sources"]

            details = self._build_details(
                tracker,
                llm_config,
                sources,
                model=self._generation_model_name(llm_config),
                provider=self._generation_provider(llm_config),
            )

            if tracker is not None:
                tracker.finish(success=True)

        except ClientDisconnect:
            if tracker is not None:
                tracker.finish(success=False, error_type="ClientDisconnect")
            raise
        except asyncio.CancelledError:
            if tracker is not None:
                tracker.finish(success=False, error_type="Cancelled")
            raise
        except Exception as exc:
            # Don't let a backend failure kill the stream halfway: the client
            # would read a truncated body and report a bogus "generation
            # stopped". Instead, surface the real error as a marker so the
            # frontend can render it, then finish this stream normally.
            if tracker is not None:
                tracker.finish(success=False, error_type=type(exc).__name__)
            logger.warning("Chat generation failed for conversation %s: %s", conversation_id, exc)
            error_payload = {"message": str(exc) or type(exc).__name__}
            prefix = "\n\n" if assistant_buffer else ""
            yield f"{prefix}{ERROR_MARKER}\n{json.dumps(error_payload)}\n"
            return
        finally:
            if tracker is not None:
                clear_request_tracking()
            set_request_api_key(None)

        full_answer = "".join(assistant_buffer).strip()
        if full_answer:
            assistant_message_id = await self._persist_message(
                conversation_id, "assistant", full_answer
            )

        if assistant_message_id is not None and (sources or details):
            final_sources = await self._attach_document_names(sources)
            await self._update_message_metadata(
                assistant_message_id,
                {"sources": final_sources or None, "details": details or None},
            )

        if sources:
            final_sources = await self._attach_document_names(sources)
            if final_sources:
                yield f"\n\n{SOURCES_MARKER}\n{json.dumps(final_sources)}\n"

        if details:
            yield f"{DETAILS_MARKER}\n{json.dumps(details)}\n"

    # --- response details ------------------------------------------------------

    @staticmethod
    def _build_details(
        tracker,
        llm_config: dict | None,
        sources: list[dict],
        *,
        model: str | None,
        provider: str | None,
    ) -> dict:
        """Collect already-measured telemetry into a compact details payload.

        Reads from the request tracker in memory; nothing is re-stored.
        """
        metrics = tracker.metrics() if tracker is not None else {}
        tags = tracker.tags() if tracker is not None else {}

        return {
            "model": model or (llm_config or {}).get("model"),
            "provider": PROVIDER_LABELS.get(provider, provider),
            "source": tags.get("source"),
            "mode": tags.get("mode"),
            "search_type": tags.get("search_type"),
            "rerank": tags.get("rerank"),
            "retrieval_limit": tags.get("retrieval_limit"),
            "search_depth": tags.get("search_depth"),
            "retrieved_documents": metrics.get("retrieved_documents"),
            "reranked_documents": metrics.get("reranked_documents"),
            "source_count": len(sources),
            "latencies": {
                "agent_latency_ms": metrics.get("agent_latency_ms"),
                "rag_latency_ms": metrics.get("rag_latency_ms"),
                "web_search_latency_ms": metrics.get("web_search_latency_ms"),
                "retrieval_latency_ms": metrics.get("retrieval_latency_ms"),
                "reranker_latency_ms": metrics.get("reranker_latency_ms"),
                "generation_latency_ms": metrics.get("llm_latency_ms"),
            },
            "tokens": {
                "input_tokens": metrics.get("input_tokens"),
                "output_tokens": metrics.get("output_tokens"),
                "total_tokens": metrics.get("total_tokens"),
            },
        }

    # --- export ----------------------------------------------------------------

    async def export_conversation(self, conversation_id: int, fmt: str) -> str:
        conversation = await self._get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        messages = await self._get_messages(conversation_id)

        if fmt == "json":
            return json.dumps(
                {
                    "id": conversation["id"],
                    "title": conversation["title"],
                    "messages": [
                        {
                            "role": m["role"],
                            "content": m["content"],
                        }
                        for m in messages
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )

        if fmt == "markdown":
            title = conversation["title"] or "Conversation"
            lines = [f"# {title}", ""]
            for m in messages:
                lines.append(f"## {m['role'].capitalize()}")
                lines.append("")
                lines.append(m["content"])
                lines.append("")
            return "\n".join(lines)

        raise ValueError(f"Unsupported export format: {fmt}")