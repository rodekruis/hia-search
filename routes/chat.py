from __future__ import annotations

from fastapi import Request, Response, APIRouter
from twilio.twiml.messaging_response import MessagingResponse
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from utils.vector_store import get_vector_store
from agents.rag_agent import rag_agent
from utils.logger import logger
from utils.prompt_loader import PromptLoader
import hashlib
from utils.translator import translate, detect_language

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
        with open("config/rag_agent_prompt.txt", "r") as f:
            prompt = f.read()

    # invoke the agent graph with the question
    response = rag_agent.invoke(
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

    # use the hashed phone number or channel address that sent this message as memory thread ID
    threadId = hashlib.sha256(form_data.get("From").encode()).hexdigest()

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


@router.post("/chat-dummy", tags=["chat"])
async def chat_dummy(
    payload: MessagePayload,
    request: Request,
    googleSheetId: str = "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04",
    threadId: str = None,
    include_context: bool = False,
):
    """Dummy chat endpoint for testing"""

    # if thread ID is not provided, use hashed client host
    if threadId is None:
        threadId = hashlib.sha256(str(request.client.host).encode()).hexdigest()

    result = chat(
        threadId, googleSheetId, payload.message, include_context=include_context
    )

    return result
