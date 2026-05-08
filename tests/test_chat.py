"""Tests for the /chat-dummy and /chat-twilio-webhook endpoints."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_response(text: str, tool_contents: list[str] | None = None):
    """Build a fake rag_agent.invoke() return value."""
    messages = []
    # optional tool messages (retrieved context)
    for content in tool_contents or []:
        msg = MagicMock()
        msg.type = "tool"
        msg.content = content
        messages.append(msg)
    # final AI message
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = text
    messages.append(ai_msg)
    return {"messages": messages}


# ---------------------------------------------------------------------------
# /chat-dummy
# ---------------------------------------------------------------------------


class TestChatDummy:
    """Tests for the POST /chat-dummy endpoint."""

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_basic_response(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "You are helpful."
        mock_agent.invoke.return_value = _make_agent_response("Hello there!")

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123"},
            json={"message": "Hi"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert body["response"] == "Hello there!"
        assert "context" not in body

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_include_context_false(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent.invoke.return_value = _make_agent_response(
            "answer", tool_contents=["doc1", "doc2"]
        )

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123", "include_context": False},
            json={"message": "question"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert "context" not in body

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_include_context_true(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent.invoke.return_value = _make_agent_response(
            "answer", tool_contents=["ctx1", "ctx2"]
        )

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123", "include_context": True},
            json={"message": "question"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "answer"
        assert body["context"] == ["ctx1", "ctx2"]

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_include_context_no_tool_messages(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When include_context=True but agent returned no tool messages,
        context should be an empty list."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent.invoke.return_value = _make_agent_response(
            "answer", tool_contents=[]
        )

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123", "include_context": True},
            json={"message": "hi"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["context"] == []

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_uses_default_google_sheet_id(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When no googleSheetId is provided, the default is used."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent.invoke.return_value = _make_agent_response("ok")

        resp = client.post("/chat-dummy", json={"message": "hello"})

        assert resp.status_code == 200
        # PromptLoader should have been called with the default sheet ID
        call_kwargs = mock_prompt_loader.call_args
        assert (
            call_kwargs[1]["document_id"]
            == "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04"
        )

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="it")
    @patch("routes.chat.translate")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_translation_roundtrip(
        self,
        mock_agent,
        mock_prompt_loader,
        mock_translate,
        mock_detect,
        mock_vs,
        client,
    ):
        """Non-English messages should be translated to EN, and the response back."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_translate.side_effect = lambda from_lang, to_lang, text: (
            "translated-to-en" if to_lang == "en" else "translated-back"
        )
        mock_agent.invoke.return_value = _make_agent_response("english answer")

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123"},
            json={"message": "Ciao"},
        )

        assert resp.status_code == 200
        assert resp.json()["response"] == "translated-back"
        # translate should have been called twice (to EN + back)
        assert mock_translate.call_count == 2

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_fallback_prompt(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When PromptLoader returns empty string, the default prompt file is used."""
        mock_prompt_loader.return_value.get_prompt.return_value = ""
        mock_agent.invoke.return_value = _make_agent_response("answer")

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read.return_value = "default prompt"

            resp = client.post(
                "/chat-dummy",
                params={"googleSheetId": "sheet123"},
                json={"message": "hi"},
            )

        assert resp.status_code == 200

    def test_missing_message_field(self, client):
        """Omitting the required 'message' field should return 422."""
        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123"},
            json={},
        )
        assert resp.status_code == 422

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_custom_thread_id(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When threadId is explicitly provided, it should be forwarded to the agent."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent.invoke.return_value = _make_agent_response("ok")

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123", "threadId": "my-thread-42"},
            json={"message": "hello"},
        )

        assert resp.status_code == 200
        # verify thread_id was passed to agent config
        config = (
            mock_agent.invoke.call_args[1].get("config")
            or mock_agent.invoke.call_args[0][1]
        )
        assert config["configurable"]["thread_id"] == "my-thread-42"


# ---------------------------------------------------------------------------
# /chat-twilio-webhook
# ---------------------------------------------------------------------------


class TestChatTwilioWebhook:
    """Tests for the POST /chat-twilio-webhook endpoint."""

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_basic_twilio_response(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent.invoke.return_value = _make_agent_response("Hello from bot!")

        resp = client.post(
            "/chat-twilio-webhook",
            params={"googleSheetId": "sheet123"},
            data={"Body": "Hi", "From": "+31612345678"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert "Hello from bot!" in resp.text

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_twilio_missing_body(
        self, mock_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When Body is missing from the form data, return 400."""
        resp = client.post(
            "/chat-twilio-webhook",
            params={"googleSheetId": "sheet123"},
            data={"From": "+31612345678"},
        )

        assert resp.status_code == 400

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="nl")
    @patch("routes.chat.translate")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.rag_agent")
    def test_twilio_with_translation(
        self,
        mock_agent,
        mock_prompt_loader,
        mock_translate,
        mock_detect,
        mock_vs,
        client,
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_translate.side_effect = (
            lambda from_lang, to_lang, text: f"[{to_lang}]{text}"
        )
        mock_agent.invoke.return_value = _make_agent_response("english reply")

        resp = client.post(
            "/chat-twilio-webhook",
            params={"googleSheetId": "sheet123"},
            data={"Body": "Hallo", "From": "+31600000000"},
        )

        assert resp.status_code == 200
        # response should contain the translated-back text
        assert "[nl]english reply" in resp.text
