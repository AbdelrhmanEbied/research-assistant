from agent.agent_schemas import ChatMessage, PromptMode
from rag.rag_schemas import Context, RetrievedDocuments


class ContextBuilder:
    def __init__(
        self,
        separator: str = "\n\n" + "-" * 80 + "\n\n",
        include_metadata: bool = True,
        include_scores: bool = False,
    ):
        self.separator = separator
        self.include_metadata = include_metadata
        self.include_scores = include_scores

    def build(
        self,
        documents: list[RetrievedDocuments],
    ) -> Context:

        context_parts = []
        sources = []

        for index, document in enumerate(documents, start=1):
            lines = [f"[Document {index}]"]

            if self.include_metadata:
                for key, value in document.metadata.items():
                    if key in {
                        "chunk_id",
                        "document_id",
                        "created_at",
                        "conversation_id",
                        "chunk_index",
                        "total_chunks",
                        "source",
                    }:
                        continue

                    lines.append(f"{key.title()}: {value}")

            if self.include_scores:
                lines.append(f"Score: {document.score:.4f}")

            lines.append("")
            lines.append(document.text)

            context_parts.append("\n".join(lines))

            sources.append(
                {
                    "source": document.metadata.get("source"),
                    "page": document.metadata.get("page"),
                    "chunk_id": document.metadata.get("chunk_id"),
                }
            )

        return Context(
            text=self.separator.join(context_parts),
            sources=sources,
        )


class PromptBuilder:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    def build(
        self,
        mode: PromptMode | str,
        question: str,
        context: Context | None = None,
        history: list[ChatMessage] | None = None,
    ):
        mode_str = getattr(mode, "value", str(mode)).lower()

        handlers = {
            "chat": self._build_chat,
            "summarize": self._build_summarize,
            "compare": self._build_compare,
            "explain": self._build_explain,
        }

        handler = handlers.get(mode_str)
        if not handler:
            raise ValueError(f"Unsupported prompt mode: {mode} (resolved to '{mode_str}')")

        return handler(question, context, history)

    def _base_prompt(
        self,
        *,
        instructions: str,
        question: str,
        context: Context | None,
        history: list[ChatMessage],
    ) -> str:

        sections = [self.system_prompt]

        if history:
            sections.append("Conversation History:")
            sections.extend(str(message) for message in history)

        if context and context.text:
            sections.append("Retrieved Context:")
            sections.append(context.text)

        sections.append("Instructions:")
        sections.append(instructions)

        sections.append("User Question:")
        sections.append(question)

        sections.append("Assistant:")

        return "\n\n".join(sections)

    def _build_chat(
        self,
        question: str,
        context: Context | None,
        history: list[ChatMessage],
    ) -> str:

        return self._base_prompt(
            instructions=(
                "Answer the user's question naturally. "
                "Use the retrieved context when it is relevant. "
                "If the context does not contain the answer, say you do not know instead of inventing information."
            ),
            question=question,
            context=context,
            history=history,
        )

    def _build_summarize(
        self,
        question: str,
        context: Context | None,
        history: list[ChatMessage],
    ) -> str:

        return self._base_prompt(
            instructions=(
                "Produce a concise but complete summary of the retrieved context. "
                "Do not introduce information that is not present."
            ),
            question=question,
            context=context,
            history=history,
        )

    def _build_compare(
        self,
        question: str,
        context: Context | None,
        history: list[ChatMessage],
    ) -> str:

        return self._base_prompt(
            instructions=(
                "Compare the relevant information from the retrieved context. "
                "Highlight similarities, differences, advantages, disadvantages, and trade-offs."
            ),
            question=question,
            context=context,
            history=history,
        )

    def _build_explain(
        self,
        question: str,
        context: Context | None,
        history: list[ChatMessage],
    ) -> str:

        return self._base_prompt(
            instructions=(
                "Explain the topic clearly and accurately. "
                "Use examples when helpful. "
                "Do not fabricate missing information."
            ),
            question=question,
            context=context,
            history=history,
        )
