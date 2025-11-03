from __future__ import annotations

import pandas as pd
from fastapi import (
    Depends,
    Request,
    APIRouter,
    HTTPException,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from __future__ import annotations
from agents import rag_agent
import os

router = APIRouter()

key_query_scheme = APIKeyHeader(name="Authorization")


class QuestionPayload(BaseModel):
    question: str = Field(
        ...,
        description="""
        Text of the question""",
    )


@router.post("/chat")
async def ask_question(
    payload: QuestionPayload, request: Request, api_key: str = Depends(key_query_scheme)
):
    """Ask something to the chatbot and get an answer."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # use client host as memory thread ID
    client_host = request.client.host
    config = {"configurable": {"thread_id": client_host}}

    # invoke the agent graph with the question
    response = rag_agent.invoke(
        {"messages": [{"role": "user", "content": payload.question}]}, config=config
    )
    answer = response["messages"][-1].content

    return answer
