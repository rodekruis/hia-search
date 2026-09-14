from __future__ import annotations
from langchain_core.documents import Document
from typing_extensions import Annotated, List
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langchain.messages import SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import os
from urllib.parse import quote
from utils.vector_store import get_vector_store
from dotenv import load_dotenv

load_dotenv()

_llm = None
_rag_agent = None
_checkpointer_pool = None

# Number of recent human/AI messages passed to the LLM
_HISTORY_WINDOW = 10


class RagState(MessagesState):
    """Conversation state: `messages` holds only human/AI turns; docs live apart."""

    retrieved_docs: List[Document]
    # the query the model wrote for the retrieve tool this turn ("" when no retrieval)
    search_query: str


def format_context(docs: List[Document]) -> str:
    """The retrieved documents exactly as they are fed to the model."""
    return "\n\n".join(f"Document: {doc.page_content}" for doc in docs)


def _get_llm():
    """Lazily initialize the LLM client."""
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_endpoint=os.environ["OPENAI_ENDPOINT"],
            azure_deployment=os.environ["MODEL_CHAT"],
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            temperature=0.2,
        )
    return _llm


def _system_prompt(config: RunnableConfig) -> str:
    return config["configurable"]["system_prompt"]


def _llm_config(config: RunnableConfig) -> RunnableConfig:
    """Tracing callbacks for the LLM calls only.

    Passing callbacks to the graph run would also record every node, edge and
    tool wrapper; only the two model calls are worth an observation.
    """
    return {"callbacks": config["configurable"].get("llm_callbacks") or []}


def _conversation(state: RagState) -> list:
    """Recent human/AI turns, excluding tool-call scaffolding."""
    return [
        message
        for message in state["messages"]
        if message.type == "human" or (message.type == "ai" and not message.tool_calls)
    ][-_HISTORY_WINDOW:]


# Retrieval tool. googleSheetId comes from the run config, not from the LLM,
# so the model cannot hallucinate a sheet id.
@tool(response_format="content_and_artifact")
def retrieve(
    query: Annotated[str, "The search query to retrieve relevant documents."],
    config: RunnableConfig,
) -> tuple[str, List[Document]]:
    """Retrieve information related to a query."""
    google_sheet_id = config["configurable"]["googleSheetId"]
    vector_store = get_vector_store(google_sheet_id)
    retrieved_docs = vector_store.similarity_search(query, k=10)
    return format_context(retrieved_docs), retrieved_docs


# Define retrieve-or-respond node
def query_or_respond(state: RagState, config: RunnableConfig) -> dict:
    """Generate tool call for retrieval or respond."""
    llm_with_tools = _get_llm().bind_tools([retrieve])
    prompt = [SystemMessage(_system_prompt(config))] + _conversation(state)
    response = llm_with_tools.invoke(prompt, config=_llm_config(config))

    # MessagesState appends messages to state instead of overwriting;
    # retrieved_docs/search_query are reset so a direct answer never reports stale context.
    return {"messages": [response], "retrieved_docs": [], "search_query": ""}


# Generate a response using the retrieved content.
def generate(state: RagState, config: RunnableConfig):
    """Generate answer."""

    # Collect this turn's tool-call scaffolding (AI tool call + tool results)
    scaffolding = []
    for message in reversed(state["messages"]):
        scaffolding.append(message)
        if message.type == "ai" and message.tool_calls:
            break
    scaffolding.reverse()

    docs: List[Document] = []
    search_query = ""
    for message in scaffolding:
        if message.type == "tool":
            docs.extend(message.artifact or [])
        elif message.type == "ai" and message.tool_calls:
            search_query = str(message.tool_calls[0]["args"].get("query", ""))

    system_prompt = f"{_system_prompt(config)}.\n\n{format_context(docs)}"
    prompt = [SystemMessage(system_prompt)] + _conversation(state)

    response = _get_llm().invoke(prompt, config=_llm_config(config))

    # Drop the scaffolding from persisted history: docs are kept in retrieved_docs
    # for this turn only, so the checkpoint does not grow by 20 documents per turn.
    removals = [RemoveMessage(id=message.id) for message in scaffolding]
    return {
        "messages": removals + [response],
        "retrieved_docs": docs,
        "search_query": search_query,
    }


def init_rag_agent():
    """Open the checkpoint pool and compile the graph."""
    global _rag_agent, _checkpointer_pool
    if _rag_agent is not None:
        return _rag_agent

    db_user = quote(os.environ["CHECKPOINT_DB_USER"], safe="")
    db_password = quote(os.environ["CHECKPOINT_DB_PASSWORD"], safe="")
    db_host = os.environ["CHECKPOINT_DB_HOST"]
    separator = "&" if "?" in db_host else "?"
    db_uri = f"postgresql://{db_user}:{db_password}@{db_host}{separator}sslmode=require"

    # Use a connection pool that validates connections on checkout so that
    # stale/closed connections (idle timeouts, DB restarts, network blips)
    # are transparently recreated instead of raising "the connection is closed".
    _checkpointer_pool = ConnectionPool(
        conninfo=db_uri,
        max_size=20,
        open=True,
        check=ConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )

    checkpointer = PostgresSaver(_checkpointer_pool)
    checkpointer.setup()
    _rag_agent = build_graph(checkpointer)
    return _rag_agent


def close_rag_agent():
    """Close the checkpoint pool (app shutdown)."""
    global _rag_agent, _checkpointer_pool
    if _checkpointer_pool is not None:
        _checkpointer_pool.close()
    _checkpointer_pool = None
    _rag_agent = None


def build_graph(checkpointer):
    """Compile the RAG graph on top of the given checkpointer."""
    tools = ToolNode([retrieve])
    graph_builder = StateGraph(RagState)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile(checkpointer=checkpointer)


def get_rag_agent():
    """Return the agent built by init_rag_agent()."""
    if _rag_agent is None:
        raise RuntimeError(
            "RAG agent not initialized; init_rag_agent() must run at startup"
        )
    return _rag_agent
