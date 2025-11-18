from __future__ import annotations

from fastapi import (
    Request,
    Response,
    APIRouter,
)
from fastapi.security import APIKeyHeader
from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel, Field
from utils.vector_store import get_vector_store
from agents.rag_agent import rag_agent

router = APIRouter()

key_query_scheme = APIKeyHeader(name="Authorization")


@router.post("/chat-twilio-webhook", tags=["chat"])
async def chat_twilio_webhook(
    googleSheetId: str,
    request: Request,
):
    """Chat endpoint for [Twilio Incoming Messaging Webhooks](https://www.twilio.com/docs/usage/webhooks/messaging-webhooks#incoming-message-webhook)"""
    form_data = await request.form()
    message = form_data.get("Body", None)
    if message is None:
        return Response(content="No message provided", status_code=400)

    # use the phone number or channel address that sent this message as memory thread ID
    config = {"configurable": {"thread_id": form_data.get("From")}}

    # check if vector store exists for the given googleSheetId
    _ = get_vector_store(googleSheetId)

    # invoke the agent graph with the question
    response = rag_agent.invoke(
        {
            "messages": [
                {"role": "system", "content": f"googleSheetId is {googleSheetId}"},
                {"role": "user", "content": message},
            ]
        },
        config=config,
    )

    resp = MessagingResponse()
    resp.message(response["messages"][-1].content)
    return Response(content=str(resp), media_type="application/xml")


class MessagePayload(BaseModel):
    message: str = Field(
        ...,
        description="""
        Text of the message.""",
    )


@router.post("/chat-dummy", tags=["chat"])
async def chat_dummy(
    payload: MessagePayload,
    request: Request,
    googleSheetId: str = "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04",
):
    """Dummy chat endpoint for testing"""

    # use client host as memory thread ID
    client_host = request.client.host
    config = {"configurable": {"thread_id": client_host}}

    # check if vector store exists for the given googleSheetId
    _ = get_vector_store(googleSheetId)

    # invoke the agent graph with the question
    response = rag_agent.invoke(
        {
            "messages": [
                {"role": "system", "content": f"googleSheetId is {googleSheetId}"},
                {"role": "user", "content": payload.message},
            ]
        },
        config=config,
    )

    return {"response": response["messages"][-1].content}
