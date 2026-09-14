from __future__ import annotations
import atexit
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

# Lazy-initialized globals
_llm = None
_rag_agent = None
_checkpointer_pool = None

# Number of recent human/AI messages passed to the LLM
_HISTORY_WINDOW = 10


class RagState(MessagesState):
    """Conversation state: `messages` holds only human/AI turns; docs live apart."""

    retrieved_docs: List[Document]


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
    vector_store = get_vector_store(google_sheet_id, check_if_exists=True)
    retrieved_docs = vector_store.similarity_search(query, k=20)
    serialized = "\n\n".join(f"Document: {doc.page_content}" for doc in retrieved_docs)
    return serialized, retrieved_docs


# Define retrieve-or-respond node
def query_or_respond(state: RagState, config: RunnableConfig) -> dict:
    """Generate tool call for retrieval or respond."""
    llm_with_tools = _get_llm().bind_tools([retrieve])
    prompt = [SystemMessage(_system_prompt(config))] + _conversation(state)
    response = llm_with_tools.invoke(prompt)

    # MessagesState appends messages to state instead of overwriting;
    # retrieved_docs is reset so a direct answer never reports stale context.
    return {"messages": [response], "retrieved_docs": []}


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
    for message in scaffolding:
        if message.type == "tool":
            docs.extend(message.artifact or [])
    docs_content = "\n\n".join(f"Document: {doc.page_content}" for doc in docs)

    system_prompt = f"{_system_prompt(config)}.\n\n{docs_content}"
    prompt = [SystemMessage(system_prompt)] + _conversation(state)

    response = _get_llm().invoke(prompt)

    # Drop the scaffolding from persisted history: docs are kept in retrieved_docs
    # for this turn only, so the checkpoint does not grow by 20 documents per turn.
    removals = [RemoveMessage(id=message.id) for message in scaffolding]
    return {"messages": removals + [response], "retrieved_docs": docs}


def _cleanup_checkpointer():
    """Close the connection pool on shutdown."""
    if _checkpointer_pool is not None:
        _checkpointer_pool.close()


def _build_agent():
    """Build and return the RAG agent graph (called once on first use)."""
    global _checkpointer_pool
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
    atexit.register(_cleanup_checkpointer)

    checkpointer = PostgresSaver(_checkpointer_pool)
    checkpointer.setup()
    return build_graph(checkpointer)


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
    """Lazily initialize and return the RAG agent."""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = _build_agent()
    return _rag_agent
