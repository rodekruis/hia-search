"""Graph-level tests for agents/rag_agent.py using an in-memory checkpointer.

conftest replaces ``agents.rag_agent`` in sys.modules with a mock (so the app
never touches Postgres); load the real module under a private name here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

_spec = importlib.util.spec_from_file_location(
    "rag_agent_real",
    Path(__file__).resolve().parent.parent / "agents" / "rag_agent.py",
)
rag_agent = importlib.util.module_from_spec(_spec)
# get_type_hints(RagState) resolves names via sys.modules[__module__]
sys.modules[_spec.name] = rag_agent
_spec.loader.exec_module(rag_agent)


class FakeLLM:
    """Scripted chat model: returns queued AIMessages in order."""

    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.prompts: list[list] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _tool_call_msg(query: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "retrieve", "args": {"query": query}, "id": "call-1"}],
    )


DOCS = [Document(page_content="doc A"), Document(page_content="doc B")]
CONFIG = {
    "configurable": {
        "thread_id": "t1",
        "googleSheetId": "sheet-xyz",
        "system_prompt": "You are HIA.",
    }
}


@pytest.fixture()
def vector_store():
    store = MagicMock()
    store.similarity_search.return_value = DOCS
    with patch.object(rag_agent, "get_vector_store", return_value=store) as gvs:
        yield gvs


def test_retrieval_turn_keeps_docs_in_state_and_prunes_scaffolding(vector_store):
    llm = FakeLLM([_tool_call_msg("housing"), AIMessage(content="final answer")])
    graph = rag_agent.build_graph(MemorySaver())

    with patch.object(rag_agent, "_get_llm", return_value=llm):
        result = graph.invoke({"messages": [("human", "where can I live?")]}, CONFIG)

    # sheet id comes from config, not from the LLM's tool args
    vector_store.assert_called_once_with("sheet-xyz", check_if_exists=True)
    assert result["retrieved_docs"] == DOCS
    assert result["messages"][-1].content == "final answer"
    # persisted history holds only the human/AI exchange
    assert [m.type for m in result["messages"]] == ["human", "ai"]

    # system prompt from config; docs injected into the generate prompt
    assert llm.prompts[0][0].content == "You are HIA."
    assert "doc A" in llm.prompts[1][0].content and "doc B" in llm.prompts[1][0].content
    assert llm.prompts[1][0].content.startswith("You are HIA.")


def test_direct_answer_resets_retrieved_docs(vector_store):
    llm = FakeLLM(
        [
            _tool_call_msg("housing"),
            AIMessage(content="answer 1"),
            AIMessage(content="hello!"),
        ]
    )
    graph = rag_agent.build_graph(MemorySaver())

    with patch.object(rag_agent, "_get_llm", return_value=llm):
        graph.invoke({"messages": [("human", "where can I live?")]}, CONFIG)
        result = graph.invoke({"messages": [("human", "thanks")]}, CONFIG)

    assert result["retrieved_docs"] == []
    assert [m.type for m in result["messages"]] == ["human", "ai", "human", "ai"]
    # second turn's prompt carries prior conversation but no tool scaffolding
    assert [m.type for m in llm.prompts[2][1:]] == ["human", "ai", "human"]


def test_conversation_window_keeps_last_ten_turns():
    from langchain_core.messages import HumanMessage

    messages = [
        AIMessage(content=f"m{i}") if i % 2 else HumanMessage(content=f"m{i}")
        for i in range(30)
    ]
    window = rag_agent._conversation({"messages": messages})
    assert len(window) == 10
    assert window[-1].content == "m29"


def test_retrieve_tool_schema_hides_sheet_id():
    """The LLM only sees `query`; the sheet id is injected from config."""
    assert set(rag_agent.retrieve.args) == {"query"}


class TestAgentLifecycle:
    """Postgres wiring: URL-encoded creds, sslmode, pool health checks, init/close."""

    @pytest.fixture()
    def wired(self, monkeypatch):
        monkeypatch.setenv("CHECKPOINT_DB_USER", "user@host")
        monkeypatch.setenv("CHECKPOINT_DB_PASSWORD", "p#a$s&w*o^rd")
        monkeypatch.setenv("CHECKPOINT_DB_HOST", "db.example:5432/dbname")
        pool_cls = MagicMock(name="ConnectionPool")
        pool_cls.check_connection = rag_agent.ConnectionPool.check_connection
        saver_cls = MagicMock(name="PostgresSaver")
        build_graph = MagicMock(name="build_graph")
        monkeypatch.setattr(rag_agent, "ConnectionPool", pool_cls)
        monkeypatch.setattr(rag_agent, "PostgresSaver", saver_cls)
        monkeypatch.setattr(rag_agent, "build_graph", build_graph)
        monkeypatch.setattr(rag_agent, "_checkpointer_pool", None)
        monkeypatch.setattr(rag_agent, "_rag_agent", None)
        return pool_cls, saver_cls, build_graph

    def test_connection_uri_is_encoded_and_encrypted(self, wired):
        pool_cls, saver_cls, build_graph = wired

        agent = rag_agent.init_rag_agent()

        conninfo = pool_cls.call_args.kwargs["conninfo"]
        assert conninfo == (
            "postgresql://user%40host:p%23a%24s%26w%2Ao%5Erd@db.example:5432/dbname"
            "?sslmode=require"
        )
        # stale connections are detected on checkout
        assert pool_cls.call_args.kwargs["check"] is rag_agent.ConnectionPool.check_connection
        assert pool_cls.call_args.kwargs["kwargs"] == {"autocommit": True, "prepare_threshold": 0}
        saver_cls.assert_called_once_with(pool_cls.return_value)
        saver_cls.return_value.setup.assert_called_once()
        build_graph.assert_called_once_with(saver_cls.return_value)
        assert agent is build_graph.return_value

    def test_host_with_existing_query_string_appends_sslmode(self, wired, monkeypatch):
        pool_cls, _, _ = wired
        monkeypatch.setenv("CHECKPOINT_DB_HOST", "db.example/db?application_name=hia")

        rag_agent.init_rag_agent()

        assert pool_cls.call_args.kwargs["conninfo"].endswith(
            "db.example/db?application_name=hia&sslmode=require"
        )

    def test_get_before_init_raises(self, wired):
        with pytest.raises(RuntimeError, match="not initialized"):
            rag_agent.get_rag_agent()

    def test_init_is_idempotent_and_get_returns_it(self, wired):
        pool_cls, _, build_graph = wired

        first = rag_agent.init_rag_agent()
        second = rag_agent.init_rag_agent()

        assert first is second is rag_agent.get_rag_agent() is build_graph.return_value
        pool_cls.assert_called_once()

    def test_close_releases_pool_and_agent(self, wired):
        pool_cls, _, _ = wired
        rag_agent.init_rag_agent()

        rag_agent.close_rag_agent()

        pool_cls.return_value.close.assert_called_once()
        with pytest.raises(RuntimeError):
            rag_agent.get_rag_agent()

    def test_close_without_init_is_a_noop(self, wired):
        rag_agent.close_rag_agent()

    def test_bad_db_config_fails_at_init(self, wired, monkeypatch):
        """Startup must surface DB misconfiguration instead of the first chat turn."""
        pool_cls, _, build_graph = wired
        pool_cls.side_effect = RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            rag_agent.init_rag_agent()
        build_graph.assert_not_called()


def test_get_llm_uses_azure_env(monkeypatch):
    llm_cls = MagicMock(name="AzureChatOpenAI")
    monkeypatch.setattr(rag_agent, "AzureChatOpenAI", llm_cls)
    monkeypatch.setattr(rag_agent, "_llm", None)
    monkeypatch.setenv("OPENAI_ENDPOINT", "https://oai.example")
    monkeypatch.setenv("MODEL_CHAT", "gpt-test")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-06-01")

    first = rag_agent._get_llm()
    second = rag_agent._get_llm()

    llm_cls.assert_called_once_with(
        azure_endpoint="https://oai.example",
        azure_deployment="gpt-test",
        openai_api_version="2024-06-01",
        temperature=0.2,
    )
    assert first is second
