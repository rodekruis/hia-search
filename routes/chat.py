from __future__ import annotations

from fastapi import (
    Depends,
    Request,
    APIRouter,
    HTTPException,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from utils.vector_store import googleid_to_vectorstoreid
from agents.rag_agent import rag_agent
import os

router = APIRouter()

key_query_scheme = APIKeyHeader(name="Authorization")


class QuestionPayload(BaseModel):
    question: str = Field(
        ...,
        description="""
        Text of the question""",
    )
    googleSheetId: str = Field(
        ...,
        description="""HIA Google sheet ID""",
    )


@router.post("/chat")
async def chat(
    payload: QuestionPayload, request: Request, api_key: str = Depends(key_query_scheme)
):
    """Ask something to the chatbot and get an answer."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # use client host as memory thread ID
    client_host = request.client.host
    config = {"configurable": {"thread_id": client_host}}
    vector_store_id = googleid_to_vectorstoreid(payload.googleSheetId)

    # invoke the agent graph with the question
    response = rag_agent.invoke(
        {
            "messages": [
                {"role": "system", "content": f"vector_store_id is {vector_store_id}"},
                {"role": "user", "content": payload.question},
            ]
        },
        config=config,
    )
    answer = response["messages"][-1].content

    return answer
