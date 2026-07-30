from schemas import RetrievedDocuments,Context

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
    def __init__(
            self,
            system_prompt: str,
    ):
        self.system_prompt = system_prompt

    def build(
            self,
            question:str,
            context:str,
    ):
        return f"""
        {self.system_prompt}

        Context:
        --------------------
        {context.text}
        --------------------

        Question:
        {question}

        Answer:
        """.strip()