from __future__ import annotations

from fastapi import Response, APIRouter, Depends, Form, Query
from twilio.twiml.messaging_response import MessagingResponse
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from utils.vector_store import get_vector_store
from agents.rag_agent import get_rag_agent
from utils.auth import require_read_key, require_twilio_signature
from utils.logger import logger
from utils.prompt_loader import PromptLoader
import hashlib
import uuid
from utils.translator import translate, detect_language
from pathlib import Path

router = APIRouter()


def chat(
    threadId: str, googleSheetId: str, message: str, include_context: bool = False
) -> dict:
    """Core chat function used by multiple endpoints."""

    # check if vector store exists for the given googleSheetId (if it doesn't, it will be created)
    _ = get_vector_store(googleSheetId, check_if_exists=True)

    # translate message to English if needed
    detected_lang = detect_language(message)
    if detected_lang != "en":
        message = translate(from_lang=detected_lang, to_lang="en", text=message)

    # get system prompt
    prompt_loader = PromptLoader(
        document_type="googlesheet",
        document_id=googleSheetId,
    )
    prompt = prompt_loader.get_prompt()
    if prompt == "":
        # use default prompt
        prompt_path = (
            Path(__file__).resolve().parent.parent / "config" / "rag_agent_prompt.txt"
        )
        with open(prompt_path, "r") as f:
            prompt = f.read()

    # invoke the agent graph with the question
    response = get_rag_agent().invoke(
        {
            "messages": [
                SystemMessage(prompt + f" googleSheetId is {googleSheetId}."),
                HumanMessage(message),
            ]
        },
        config={"configurable": {"thread_id": threadId}},
    )
    response_text = response["messages"][-1].content

    # extract retrieved context from tool messages if requested
    retrieved_context = None
    if include_context:
        # Each tool message contains concatenated docs separated by "\n\nDocument: "
        # Split them into individual documents and deduplicate
        all_docs = []
        seen = set()
        for msg in response["messages"]:
            if msg.type == "tool":
                # Split on the "Document: " prefix pattern
                parts = msg.content.split("\n\nDocument: ")
                for i, part in enumerate(parts):
                    # First part already starts with "Document: ", others don't
                    doc_text = (
                        part if part.startswith("Document: ") else f"Document: {part}"
                    )
                    doc_text = doc_text.strip()
                    if doc_text and doc_text not in seen:
                        seen.add(doc_text)
                        all_docs.append(doc_text)
        retrieved_context = all_docs

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

    result = chat(threadId, googleSheetId, message)

    # log user message and assistant response
    extra_logs = {"googleSheetId": googleSheetId, "threadId": threadId}
    logger.info(f"user: {message}, assistant: {result['response']}", extra=extra_logs)

    # return TwiML response
    resp = MessagingResponse()
    resp.message(result["response"])
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
