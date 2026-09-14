from __future__ import annotations

from fastapi import Response, APIRouter, Depends, Form, HTTPException, Query
from twilio.twiml.messaging_response import MessagingResponse
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from utils.vector_store import get_vector_store
from agents.rag_agent import format_context, get_rag_agent
from utils.auth import require_read_key, require_twilio_signature
from utils.messaging import number_chunks, split_message
from utils.prompt_loader import get_system_prompt
from utils.tracing import get_langfuse, langchain_callbacks, observe
import hashlib
import logging
import uuid
from time import perf_counter
from utils.translator import translate, detect_language

logger = logging.getLogger(__name__)

router = APIRouter()

# Sent to the end user when generation fails; Twilio would otherwise deliver nothing.
FALLBACK_MESSAGE = "Sorry, something went wrong on our side. Please try again in a few minutes."


def _conversation_history(messages: list) -> str:
    """Prior turns as a plain transcript; the current turn is the span's input/output."""
    turns = [m for m in messages if getattr(m, "type", None) in ("human", "ai")]
    role = {"human": "user", "ai": "assistant"}
    return "\n".join(f"{role[m.type]}: {m.content}" for m in turns[:-2])


def chat(
    threadId: str,
    googleSheetId: str,
    message: str,
    include_context: bool = False,
    channel: str = "dummy",
) -> dict:
    """Core chat function used by multiple endpoints."""
    started = perf_counter()
    original_message = message

    # ensure the vector store exists (created from the sheet if not); cached after the
    # first call, so this is the only existence check per turn
    _ = get_vector_store(googleSheetId, check_if_exists=True)

    # detected before the span opens so the language can be a trace tag
    detected_lang = detect_language(message)
    tags = [f"sheet:{googleSheetId}", f"channel:{channel}", f"lang:{detected_lang}"]

    trace_id = None
    with observe(
        "chat-turn",
        project="chat",
        input=original_message,
        tags=tags,
        session_id=threadId,
        user_id=threadId,
    ) as span:
        # translate message to English if needed
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
                },
                "callbacks": langchain_callbacks("chat"),
            },
        )
        answer_en = response["messages"][-1].content
        docs = response.get("retrieved_docs") or []
        response_text = answer_en

        # translate response back to original language if needed
        if detected_lang != "en":
            response_text = translate(from_lang="en", to_lang=detected_lang, text=answer_en)

        if span is not None:
            # Same keys as the search span: one evaluator mapping serves both channels.
            # retrieval_used gates RAG judges so they skip small talk; *_en fields let
            # judges compare answer and context in the same language.
            metadata = {
                "retrieval_used": bool(docs),
                "n_docs": len(docs),
                "detected_lang": detected_lang,
                "message_en": message,
                "answer_en": answer_en,
                "conversation_history": _conversation_history(response.get("messages") or []),
            }
            if docs:
                metadata["retrieved_context"] = format_context(docs)
                metadata["search_query"] = response.get("search_query") or message
            span.update(output=response_text, metadata=metadata)
            trace_id = span.trace_id

    # Metadata only: conversation content is recorded in the LLM observability
    # tool, never in application logs.
    logger.info(
        "chat turn",
        extra={
            "googleSheetId": googleSheetId,
            "threadId": threadId,
            "channel": channel,
            "detected_lang": detected_lang,
            "retrieval_used": bool(docs),
            "n_docs": len(docs),
            "message_chars": len(original_message),
            "response_chars": len(response_text),
            "duration_ms": round((perf_counter() - started) * 1000),
        },
    )

    result = {"response": response_text, "traceId": trace_id}
    if include_context:
        result["context"] = [doc.page_content for doc in docs]
    return result


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
        response_text = chat(threadId, googleSheetId, message, channel="twilio")["response"]
    except Exception:
        # A 5xx would leave the user with no reply at all
        logger.exception("Twilio chat turn failed", extra=extra_logs)
        response_text = FALLBACK_MESSAGE

    # return TwiML response; long answers go out as several numbered messages,
    # since Twilio drops any single <Message> of 1600+ characters
    chunks = number_chunks(split_message(response_text))
    logger.info("twilio reply", extra={**extra_logs, "n_chunks": len(chunks)})
    resp = MessagingResponse()
    for chunk in chunks:
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
    """Dummy chat endpoint for testing. Protected with `API_KEY`.

    Returns `traceId` (null when tracing is disabled) for use with `/feedback`."""

    if threadId is None:
        threadId = str(uuid.uuid4())

    result = chat(
        threadId, googleSheetId, payload.message, include_context=include_context
    )
    result["threadId"] = threadId

    return result


class FeedbackPayload(BaseModel):
    traceId: str = Field(..., min_length=1, max_length=64, description="`traceId` of a chat answer")
    positive: bool = Field(..., description="Thumbs up (true) or down (false)")
    comment: str | None = Field(None, max_length=2000)


@router.post(
    "/feedback", tags=["chat"], status_code=202, dependencies=[Depends(require_read_key)]
)
def submit_feedback(payload: FeedbackPayload):
    """Attach a user-feedback score to a previously generated chat answer."""
    langfuse = get_langfuse("chat")
    if langfuse is None:
        raise HTTPException(status_code=503, detail="Feedback collection is not enabled")
    langfuse.create_score(
        trace_id=payload.traceId,
        name="user-feedback",
        value=1.0 if payload.positive else 0.0,
        data_type="NUMERIC",
        comment=payload.comment,
    )
    return {"message": "Feedback accepted."}
