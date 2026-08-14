from types import SimpleNamespace

import pytest
from qdrant_client import models

import agent.tools as tools_mod
from agent.llms import set_request_conversation_id
from agent.tools import build_agent_tools


class FakeQdrant:
    def __init__(self, documents, points):
        self.documents = documents
        self.points = points
        self.list_calls = []
        self.points_calls = []

    def list_document_ids(self, qdrant_filter=None):
        self.list_calls.append(qdrant_filter)
        return self.documents

    def get_points_by_document(self, document_id, qdrant_filter=None, limit=100):
        combined = models.Filter(must=list((qdrant_filter.must or []) if qdrant_filter else []))
        combined.must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        )
        self.points_calls.append((document_id, combined, limit))
        return [SimpleNamespace(payload=payload) for payload in self.points.get(document_id, [])]


def make_tools(qdrant=None):
    qdrant = qdrant or FakeQdrant(
        documents=[("1", "a.pdf"), ("2", "b.pdf")],
        points={"1": [{"text": "alpha one"}, {"text": "alpha two"}]},
    )
    rag = SimpleNamespace(qdrant_manager=qdrant)
    tools = {tool.name: tool for tool in build_agent_tools(rag)}
    return tools, qdrant


# --- calculator ------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("2 ** 10", "1024"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("-5 + 3", "-2"),
        ("sqrt(16)", "4.0"),
        ("fabs(-3.5)", "3.5"),
        ("gcd(12, 8)", "4"),
        ("mean([1, 2, 3, 4])", "2.5"),
    ],
)
def test_calculator_math(expression, expected):
    tools, _ = make_tools()
    assert tools["calculator"].invoke({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "1.__class__",
        "(lambda: 1)()",
        "open('/etc/passwd')",
        "os.getcwd()",
        "getattr(math, 'sqrt')",
        "{}",
        "[x for x in range(3)]",
    ],
)
def test_calculator_rejects_unsafe_constructs(expression):
    tools, _ = make_tools()
    result = tools["calculator"].invoke({"expression": expression})
    assert "error" in result.lower()


# --- code executor ---------------------------------------------------------


def test_code_executor_returns_stdout():
    tools, _ = make_tools()
    result = tools["python_code_executor"].invoke({"code": "print(2 + 2)"})
    assert result == "4"


def test_code_executor_surfaces_stderr():
    tools, _ = make_tools()
    result = tools["python_code_executor"].invoke({"code": "1 / 0"})
    assert "ZeroDivisionError" in result


def test_code_executor_times_out(monkeypatch):
    monkeypatch.setattr(tools_mod, "EXEC_TIMEOUT_SECONDS", 1)
    tools, _ = make_tools()
    result = tools["python_code_executor"].invoke({"code": "import time; time.sleep(5)"})
    assert "timed out" in result


# --- document tools --------------------------------------------------------


def test_read_document_returns_full_text_in_chunk_order():
    tools, qdrant = make_tools()
    set_request_conversation_id("9")
    try:
        result = tools["read_document"].invoke({"name": "a.pdf"})
    finally:
        set_request_conversation_id(None)

    assert result == "alpha one\n\nalpha two"
    assert qdrant.points_calls[0][0] == "1"


def test_read_document_scopes_retrieval_to_conversation():
    tools, qdrant = make_tools()
    set_request_conversation_id("9")
    try:
        tools["read_document"].invoke({"name": "a.pdf"})
    finally:
        set_request_conversation_id(None)

    scope_filter = qdrant.list_calls[0]
    assert scope_filter is not None
    assert scope_filter.must[0].key == "conversation_id"
    assert scope_filter.must[0].match.value == "9"

    _, doc_filter, _ = qdrant.points_calls[0]
    keys = {condition.key for condition in doc_filter.must}
    assert keys == {"conversation_id", "document_id"}


def test_read_document_unknown_name_lists_available():
    tools, _ = make_tools()
    set_request_conversation_id("9")
    try:
        result = tools["read_document"].invoke({"name": "missing.pdf"})
    finally:
        set_request_conversation_id(None)

    assert "not found" in result
    assert "a.pdf" in result
    assert "b.pdf" in result


def test_list_documents_discovery():
    tools, _ = make_tools()
    set_request_conversation_id("3")
    try:
        result = tools["list_documents"].invoke({})
    finally:
        set_request_conversation_id(None)

    assert "a.pdf" in result
    assert "id: 1" in result


def test_list_documents_empty():
    qdrant = FakeQdrant(documents=[], points={})
    tools, _ = make_tools(qdrant)
    result = tools["list_documents"].invoke({})
    assert "No documents" in result


def test_code_executor_runs_in_conversation_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    tools, _ = make_tools()
    set_request_conversation_id("77")
    try:
        result = tools["python_code_executor"].invoke({"code": "import os; print(os.getcwd())"})
    finally:
        set_request_conversation_id(None)

    assert str(tmp_path / "workspace" / "77") == result
