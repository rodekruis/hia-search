from __future__ import annotations

from fastapi import Response, APIRouter, Depends, Form, Query
from twilio.twiml.messaging_response import MessagingResponse
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from utils.vector_store import get_vector_store
from agents.rag_agent import get_rag_agent
from utils.auth import require_read_key, require_twilio_signature
from utils.logger import logger
from utils.messaging import number_chunks, split_message
from utils.prompt_loader import get_system_prompt
import hashlib
import uuid
from utils.translator import translate, detect_language

router = APIRouter()

# Sent to the end user when generation fails; Twilio would otherwise deliver nothing.
FALLBACK_MESSAGE = "Sorry, something went wrong on our side. Please try again in a few minutes."


def chat(
    threadId: str, googleSheetId: str, message: str, include_context: bool = False
) -> dict:
    """Core chat function used by multiple endpoints."""

    # ensure the vector store exists (created from the sheet if not); cached after the
    # first call, so this is the only existence check per turn
    _ = get_vector_store(googleSheetId, check_if_exists=True)

    # translate message to English if needed
    detected_lang = detect_language(message)
    if detected_lang != "en":
        message = translate(from_lang=detected_lang, to_lang="en", text=message)

    prompt = get_system_prompt(googleSheetId)

    # invoke the agent graph with the question; prompt and sheet id travel in the
    # run config so they are not persisted into the conversation history
    response = get_rag_agent().invoke(
        {"messages": [HumanMessage(message)]},
        config={
            "configurable": {
                "thread_id": threadId,
                "googleSheetId": googleSheetId,
                "system_prompt": prompt,
            }
        },
    )
    response_text = response["messages"][-1].content

    retrieved_context = None
    if include_context:
        retrieved_context = [
            doc.page_content for doc in response.get("retrieved_docs") or []
        ]

    # translate response back to original language if needed
    if detected_lang != "en":
        response_text = translate(
            from_lang="en", to_lang=detected_lang, text=response_text
        )

    if include_context:
        return {"response": response_text, "context": retrieved_context}
    return {"response": response_text}


@router.post(
    "/chat-twilio-webhook",
    tags=["chat"],
    dependencies=[Depends(require_twilio_signature)],
)
def chat_twilio_webhook(
    googleSheetId: str,
    message: str | None = Form(None, alias="Body"),
    sender: str | None = Form(None, alias="From"),
):
    """Chat endpoint for [Twilio Incoming Messaging Webhooks](https://www.twilio.com/docs/usage/webhooks/messaging-webhooks#incoming-message-webhook).

    Requests must carry a valid `X-Twilio-Signature` (validated with `TWILIO_AUTH_TOKEN`)."""
    if message is None:
        return Response(content="No message provided", status_code=400)

    if sender is None:
        return Response(content="No sender provided", status_code=400)

    # use the hashed phone number or channel address that sent this message as memory thread ID
    threadId = hashlib.sha256(sender.encode()).hexdigest()
    extra_logs = {"googleSheetId": googleSheetId, "threadId": threadId}

    try:
        result = chat(threadId, googleSheetId, message)
        response_text = result["response"]
        # log user message and assistant response
        logger.info(f"user: {message}, assistant: {response_text}", extra=extra_logs)
    except Exception:
        # A 5xx would leave the user with no reply at all
        logger.exception("Twilio chat turn failed", extra=extra_logs)
        response_text = FALLBACK_MESSAGE

    # return TwiML response; long answers go out as several numbered messages,
    # since Twilio drops any single <Message> of 1600+ characters
    resp = MessagingResponse()
    for chunk in number_chunks(split_message(response_text)):
        resp.message(chunk)
    return Response(content=str(resp), media_type="application/xml")


class MessagePayload(BaseModel):
    message: str = Field(
        ...,
        description="""
        Text of the message.""",
    )


@router.post(
    "/chat-dummy", tags=["chat"], dependencies=[Depends(require_read_key)]
)
def chat_dummy(
    payload: MessagePayload,
    googleSheetId: str = "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04",
    threadId: str | None = Query(
        None,
        min_length=1,
        max_length=200,
        description="Conversation thread ID. Omit to start a new conversation; "
        "reuse the `threadId` returned in the response to continue it.",
    ),
    include_context: bool = False,
):
    """Dummy chat endpoint for testing. Protected with `API_KEY`."""

    if threadId is None:
        threadId = str(uuid.uuid4())

    result = chat(
        threadId, googleSheetId, payload.message, include_context=include_context
    )
    result["threadId"] = threadId

    return result
