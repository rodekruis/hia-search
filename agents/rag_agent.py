from __future__ import annotations
from langchain_core.documents import Document
from typing_extensions import Annotated, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langchain.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from openai import BadRequestError
import os
from urllib.parse import quote
from utils.vector_store import get_vector_store
from utils.prompt_loader import get_direct_answer_prompt, get_language_prompt, get_wellbeing_prompt
from utils.translator import detect_language, translate
from utils.constants import DocumentMetadata as dm
from dotenv import load_dotenv

load_dotenv()

_llm = None
_rag_agent = None
_checkpointer_pool = None

# Number of recent human/AI messages passed to the LLM
_HISTORY_WINDOW = 10

# Twilio drops a webhook reply after 15 s and a turn makes up to two sequential
# LLM calls, so each call must fail fast enough for the fallback message to be
# delivered. Retries are off by default: the client honours Retry-After (up to
# 60 s) and retries timeouts, either of which would exceed that budget.
_DEFAULT_LLM_TIMEOUT_S = 6.0
_DEFAULT_LLM_MAX_RETRIES = 0

# OpenAI's recommended effort for chat assistants that use tools; "none" is
# faster, higher efforts are too slow for the Twilio window.
_DEFAULT_LLM_REASONING_EFFORT = "low"

# Documents (one per Q&A or offer) retrieved per search (RETRIEVAL_K)
_DEFAULT_RETRIEVAL_K = 10

# Fixed replies (in English, translated to the user's language) for messages that
# Azure's content filter blocks before the model sees them. {emergency} is filled
# with the instance's emergency contact.
FILTERED_REPLIES = {
    "jailbreak": (
        "Sorry, I can't help with that. I can only share information about the "
        "support available through this service. What would you like to know?"
    ),
    "self_harm": (
        "I'm really sorry you're going through this. You don't have to face it alone. "
        "If you are in danger or thinking about ending your life, please {emergency} now. "
        "You can also talk to a doctor or someone you trust. I'm here if you want to "
        "ask about the support available to you through this service."
    ),
}
DEFAULT_FILTERED_REPLY = (
    "Sorry, I can't respond to that message. If you or someone else is in danger, "
    "please {emergency}. You can ask me about the support available to you."
)
# when a message trips several filters, the first match picks the reply
_FILTER_PRIORITY = ("jailbreak", "self_harm")


class RagState(MessagesState):
    """Conversation state: `messages` holds only human/AI turns; docs live apart."""

    retrieved_docs: List[Document]
    # the query the model wrote for the retrieve tool this turn ("" when no retrieval)
    search_query: str
    # content-filter category that blocked this turn's message ("" when not blocked)
    blocked_category: str


def format_context(docs: List[Document]) -> str:
    """The retrieved documents exactly as they are fed to the model, numbered in rank order."""
    return "\n\n".join(f"[{index}] {doc.page_content}" for index, doc in enumerate(docs, start=1))


def _azure_v1_base_url(endpoint: str) -> str:
    """Azure OpenAI v1 API root (no api-version) for a resource endpoint."""
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/openai/v1"):
        endpoint = f"{endpoint}/openai/v1"
    return f"{endpoint}/"


def _get_llm():
    """Lazily initialize the LLM client.

    Azure OpenAI v1 API through the plain OpenAI client, on the Responses API:
    GPT-6 reasoning models only call tools with reasoning there (Chat Completions
    allows tools only with reasoning effort "none"). No temperature: reasoning
    models reject sampling parameters. store=False keeps conversations from being
    stored by the Responses API.
    """
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=_azure_v1_base_url(os.environ["OPENAI_ENDPOINT"]),
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.environ["MODEL_CHAT"],
            use_responses_api=True,
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT", _DEFAULT_LLM_REASONING_EFFORT),
            store=False,
            timeout=float(os.environ.get("LLM_TIMEOUT_S", _DEFAULT_LLM_TIMEOUT_S)),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", _DEFAULT_LLM_MAX_RETRIES)),
        )
    return _llm


def _system_prompt(config: RunnableConfig) -> str:
    return config["configurable"]["system_prompt"]


def _support_contacts(config: RunnableConfig) -> dict[str, str]:
    return config["configurable"].get("support_contacts") or {}


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


def _prompt_filter_category(exc: BadRequestError) -> str | None:
    """Category of a user prompt blocked by Azure's content filter; None for other errors."""
    if exc.code != "content_filter":
        return None
    body = exc.body if isinstance(exc.body, dict) else {}
    filters = body.get("content_filters") or []
    if filters:
        prompt_filters = [f for f in filters if f.get("source_type") == "prompt" and f.get("blocked")]
    elif body.get("param") == "prompt":
        prompt_filters = [{}]
    else:
        prompt_filters = []
    if not prompt_filters:
        return None
    hits = [
        category
        for f in prompt_filters
        for category, result in (f.get("content_filter_results") or {}).items()
        if result.get("filtered")
    ]
    for category in _FILTER_PRIORITY:
        if category in hits:
            return category
    return hits[0] if hits else "unknown"


def _in_user_language(text: str, user_message: str) -> str:
    language = detect_language(user_message)
    if language == "en":
        return text
    return translate(from_lang="en", to_lang=language, text=text)


def filtered_reply(category: str, support_contacts: dict[str, str]) -> str:
    """Fixed English reply for a turn the content filter blocked."""
    emergency = support_contacts.get("emergency")
    action = (
        f"contact emergency services ({emergency})"
        if emergency
        else "contact your local emergency services"
    )
    return FILTERED_REPLIES.get(category, DEFAULT_FILTERED_REPLY).format(emergency=action)


# Blocked output is not an error: Azure returns an incomplete response, and LangChain
# drops its content_filters extension, so the category is unknown.
OUTPUT_FILTERED = "output"


def _output_filtered(response: AIMessage) -> bool:
    details = response.response_metadata.get("incomplete_details") or {}
    return details.get("reason") == "content_filter"


def _blocked_turn(
    state: RagState,
    config: RunnableConfig,
    category: str,
    scaffolding: list | None = None,
    hide_message: bool = True,
) -> dict:
    """Fixed reply for a turn the content filter blocked.

    A blocked user message is replaced in history by a placeholder: kept, it would
    be sent again with every later turn and block the whole conversation until it
    left the history window. When only the output was blocked, the message stays.
    """
    user_message = next(m for m in reversed(state["messages"]) if m.type == "human")
    reply = filtered_reply(category, _support_contacts(config))
    messages = [RemoveMessage(id=m.id) for m in (scaffolding or [])]
    if hide_message:
        messages += [
            RemoveMessage(id=user_message.id),
            HumanMessage(f"[Message hidden by the content filter: {category}]"),
        ]
    messages.append(AIMessage(_in_user_language(reply, user_message.text)))
    return {
        "messages": messages,
        "retrieved_docs": [],
        "search_query": "",
        "blocked_category": category,
    }


_RETRIEVE_DESCRIPTION = (
    "Search the HIA documents for information to answer the user's latest message. "
    "Call this for every question about information, services or help, including "
    "short follow-ups that refer back to earlier turns. When in doubt, call it."
)


# Retrieval tool. googleSheetId comes from the run config, not from the LLM,
# so the model cannot hallucinate a sheet id.
@tool(response_format="content_and_artifact", description=_RETRIEVE_DESCRIPTION)
def retrieve(
    query: Annotated[
        str,
        "A standalone search query in English that can be understood without the "
        "conversation history. Resolve references and pronouns in the user's latest "
        "message (e.g. 'are you sure?', 'and then?', 'what about children?') into an "
        "explicit question using the earlier turns. If the latest message is already "
        "self-contained, use it unchanged. Do not answer the question.",
    ],
    config: RunnableConfig,
) -> tuple[str, List[Document]]:
    """Retrieve documents for a query; the LLM-facing description is _RETRIEVE_DESCRIPTION."""
    google_sheet_id = config["configurable"]["googleSheetId"]
    vector_store = get_vector_store(google_sheet_id)
    k = int(os.environ.get("RETRIEVAL_K", _DEFAULT_RETRIEVAL_K))
    retrieved_docs = []
    for doc, score in vector_store.similarity_search_with_score(query, k=k):
        doc.metadata[dm.SCORE] = score
        retrieved_docs.append(doc)
    return format_context(retrieved_docs), retrieved_docs


# Define retrieve-or-respond node
def query_or_respond(state: RagState, config: RunnableConfig) -> dict:
    """Generate tool call for retrieval or respond."""
    llm_with_tools = _get_llm().bind_tools([retrieve])
    # This call answers directly when it skips retrieval, so it also gets the
    # direct-answer rules; generate() must not, as it answers from documents.
    system_prompt = "\n\n".join(
        [
            _system_prompt(config).rstrip(),
            get_direct_answer_prompt(),
            get_wellbeing_prompt(_support_contacts(config)),
            get_language_prompt(),
        ]
    )
    prompt = [SystemMessage(system_prompt)] + _conversation(state)
    try:
        response = llm_with_tools.invoke(prompt, config=_llm_config(config))
    except BadRequestError as exc:
        category = _prompt_filter_category(exc)
        if category is None:
            raise
        return _blocked_turn(state, config, category)
    if _output_filtered(response):
        return _blocked_turn(state, config, OUTPUT_FILTERED, hide_message=False)

    # MessagesState appends messages to state instead of overwriting;
    # retrieved_docs/search_query are reset so a direct answer never reports stale context.
    return {
        "messages": [response],
        "retrieved_docs": [],
        "search_query": "",
        "blocked_category": "",
    }


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

    # "(none)" makes an empty search explicit, so the prompt's "no documents"
    # rule applies instead of the model answering from memory. The language
    # rules come last: English documents otherwise pull the reply into English.
    context = format_context(docs) or "(none)"
    system_prompt = "\n\n".join(
        [
            _system_prompt(config).rstrip(),
            get_wellbeing_prompt(_support_contacts(config)),
            f"Context documents:\n{context}",
            get_language_prompt(),
        ]
    )
    prompt = [SystemMessage(system_prompt)] + _conversation(state)

    try:
        response = _get_llm().invoke(prompt, config=_llm_config(config))
    except BadRequestError as exc:
        category = _prompt_filter_category(exc)
        if category is None:
            raise
        return _blocked_turn(state, config, category, scaffolding)
    if _output_filtered(response):
        return _blocked_turn(state, config, OUTPUT_FILTERED, scaffolding, hide_message=False)

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
