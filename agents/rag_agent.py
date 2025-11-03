from __future__ import annotations
from dataclasses import dataclass
from langchain_core.documents import Document
from typing_extensions import List
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import os
from utils.vector_store import get_vector_store
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ContextSchema:
    vector_store_id: str


# Initialize the LLM client
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["MODEL_CHAT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0.2,
)


class RetrieveInput(BaseModel):
    """Input schema for retrieval tool."""

    query: str = Field(description="The search query to retrieve relevant documents.")
    vector_store_id: str = Field(description="The ID of the vector store to search.")


# Retrieval tool
@tool(response_format="content_and_artifact", args_schema=RetrieveInput)
def retrieve(query: str, vector_store_id: str) -> tuple[str, List[Document]]:
    """Retrieve information related to a query."""
    vector_store = get_vector_store(vector_store_id=vector_store_id)
    retrieved_docs = vector_store.similarity_search(query, k=10)
    serialized = "\n\n".join(f"Document: {doc.page_content}" for doc in retrieved_docs)
    return serialized, retrieved_docs


# Load RAG agent's prompt
with open("config/rag_agent_prompt.txt", "r") as f:
    rag_agent_prompt = f.read()


# Define retrieve-or-respond node
def query_or_respond(state: MessagesState) -> dict:
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    prompt = [SystemMessage(f"{rag_agent_prompt}")] + state["messages"]
    response = llm_with_tools.invoke(prompt)
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""

    # Get all docs recently retrieved
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    # Format into prompt
    system_message_content = f"{rag_agent_prompt}.\n\n{docs_content}"

    # Pass also recent conversation messages
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}


# Define and build the agent graph
tools = ToolNode([retrieve])
graph_builder = StateGraph(MessagesState)
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

memory = MemorySaver()  # in-memory checkpointer
rag_agent = graph_builder.compile(checkpointer=memory)
