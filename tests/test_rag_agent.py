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
