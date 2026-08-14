"""Local-first tools for the thinking agent loop.

Everything here runs locally: a safe arithmetic evaluator, an isolated
subprocess for arbitrary Python, and document discovery/reading scoped to the
current conversation. No new dependencies are required.
"""

import ast
import math
import statistics
import subprocess
import sys

from langchain_core.tools import tool
from qdrant_client import models

from agent.llms import get_request_conversation_id
from paths import data_path

EXEC_TIMEOUT_SECONDS = 30

#: ``math``/``statistics`` names available to the calculator. ``__builtins__``
#: and anything underscore-prefixed are intentionally excluded.
_SAFE_FUNCTIONS = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
_SAFE_FUNCTIONS.update(
    {name: getattr(statistics, name) for name in dir(statistics) if not name.startswith("_")}
)

_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)


def _is_safe_node(node: ast.AST) -> bool:
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Call,
        ast.keyword,
        ast.BinOp,
        ast.UnaryOp,
        ast.List,
        ast.Tuple,
        ast.Load,
        *_ALLOWED_BIN_OPS,
        *_ALLOWED_UNARY_OPS,
    )
    return isinstance(node, allowed)


def _evaluate(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)) or node.value is None:
            return node.value
        raise ValueError(f"Unsupported literal: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            raise ValueError("Only whitelisted math/statistics functions may be called")
        args = [_evaluate(arg) for arg in node.args]
        kwargs = {kw.arg: _evaluate(kw.value) for kw in node.keywords}
        return _SAFE_FUNCTIONS[node.func.id](*args, **kwargs)
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
    if isinstance(node, ast.List):
        return [_evaluate(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(element) for element in node.elts)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_expression(expression: str):
    """Evaluate ``expression`` against the safe AST whitelist.

    Rejects imports, attribute access, lambdas, comprehensions and any call
    outside the whitelisted ``math``/``statistics`` functions.
    """
    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not _is_safe_node(node):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _evaluate(tree.body)


def _conversation_filter(conversation_id: str | None) -> models.Filter | None:
    if conversation_id is None:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="conversation_id",
                match=models.MatchValue(value=str(conversation_id)),
            )
        ]
    )


def _conversation_workspace() -> str:
    conversation_id = get_request_conversation_id() or "default"
    workspace = data_path("workspace") / str(conversation_id)
    workspace.mkdir(parents=True, exist_ok=True)
    return str(workspace)


def build_agent_tools(rag):
    """Build the tools exposed to the thinking agent for ``rag``.

    The tools close over the RAG service and read the request-scoped
    ``conversation_id`` context var at call time, so one shared set can be
    built at graph construction and reused across conversations.
    """

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression and return the result.

        Supports the basic operators plus a whitelist of ``math`` and
        ``statistics`` functions (e.g. ``sqrt(2)``, ``mean([1, 2, 3])``).
        Never evaluates imports, attributes, lambdas, or arbitrary code.
        """
        try:
            return str(evaluate_expression(expression))
        except Exception as exc:
            return f"Calculator error: {exc}"

    @tool
    def python_code_executor(code: str) -> str:
        """Execute Python code in an isolated sandbox and return stdout/stderr.

        Use for arithmetic, algorithms, data analysis, and file manipulation.
        Runs with the standard interpreter's isolation flags in this
        conversation's private workspace directory (``data/workspace/<id>``),
        with a 30 second timeout.
        """
        try:
            process = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=_conversation_workspace(),
                timeout=EXEC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: execution timed out after {EXEC_TIMEOUT_SECONDS} seconds."

        output = process.stdout or ""
        if process.stderr:
            output += "\n[stderr]\n" + process.stderr
        return output.strip() or "(no output)"

    @tool
    def list_documents() -> str:
        """List the documents available in the current conversation.

        Returns the name and id of every indexed document this conversation
        can retrieve from, or a note when none are available.
        """
        qdrant_filter = _conversation_filter(get_request_conversation_id())
        documents = rag.qdrant_manager.list_document_ids(qdrant_filter)

        if not documents:
            return "No documents are available in this conversation."

        return "\n".join(
            f"- {name or document_id} (id: {document_id})" for document_id, name in documents
        )

    @tool
    def read_document(name: str) -> str:
        """Read the full text of an indexed document by name.

        Pass the exact document name from ``list_documents`` (e.g.
        ``report.pdf``). The full text, with chunks in reading order, is
        returned.
        """
        qdrant_filter = _conversation_filter(get_request_conversation_id())
        documents = rag.qdrant_manager.list_document_ids(qdrant_filter)

        matches = [doc_id for doc_id, doc_name in documents if doc_name == name]
        if not matches:
            available = (
                ", ".join((doc_name or doc_id) for doc_id, doc_name in documents) or "(none)"
            )
            return (
                f"Document '{name}' not found in this conversation. "
                f"Available documents: {available}"
            )

        points = rag.qdrant_manager.get_points_by_document(
            matches[0], qdrant_filter=qdrant_filter, limit=1000
        )

        chunks = [point.payload.get("text", "") for point in points]
        chunks = [text for text in chunks if text.strip()]

        if not chunks:
            return f"Document '{name}' has no readable text."

        return "\n\n".join(chunks)

    return [calculator, python_code_executor, list_documents, read_document]
