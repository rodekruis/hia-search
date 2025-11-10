from __future__ import annotations

from typing import Annotated
from fastapi import (
    Depends,
    Request,
    Form,
    Response
    APIRouter,
    HTTPException,
)
from fastapi.security import APIKeyHeader
from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel, Field
from utils.vector_store import googleid_to_vectorstoreid
from agents.rag_agent import rag_agent
import os

router = APIRouter()

key_query_scheme = APIKeyHeader(name="Authorization")


# class MessagePayload(BaseModel):
#     message: str = Field(
#         ...,
#         description="""
#         Text of the message""",
#     )


@router.post("/twilio_chat")
async def chat(
    google_sheet_id: str,
    request: Request,
    # api_key: str = Depends(key_query_scheme),
):
    """Chat endpoint for Twilio Messaging Webhook."""
    form_data = await request.form()
    message = form_data.get("Body", None)
    print("Received message:", message)
    print("Google Sheet ID:", google_sheet_id)

    resp = MessagingResponse()
    resp.message("The Robots are coming! Head for the hills!")

    # if api_key != os.environ["API_KEY"]:
    #     raise HTTPException(status_code=401, detail="Unauthorized")

    # message_form = await request.form()
    # message = message_form.get("Body")
    #
    # # use client host as memory thread ID
    # client_host = request.client.host
    # config = {"configurable": {"thread_id": client_host}}
    # vector_store_id = googleid_to_vectorstoreid(google_sheet_id)
    #
    # # invoke the agent graph with the question
    # response = rag_agent.invoke(
    #     {
    #         "messages": [
    #             {"role": "system", "content": f"vector_store_id is {vector_store_id}"},
    #             {"role": "user", "content": message},
    #         ]
    #     },
    #     config=config,
    # )
    # answer = response["messages"][-1].content

    return Response(content=str(resp), media_type="application/xml")
