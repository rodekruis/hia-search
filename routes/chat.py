from __future__ import annotations

from typing import Annotated
from fastapi import (
    Request,
    Response,
    APIRouter,
)
from fastapi.security import APIKeyHeader
from twilio.twiml.messaging_response import MessagingResponse
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

    # parse the incoming message from Twilio
    message_form = await request.form()
    message = message_form.get("Body")

    # use the phone number or channel address that sent this message as memory thread ID
    config = {"configurable": {"thread_id": message_form.get("From")}}

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
