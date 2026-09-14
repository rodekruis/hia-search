"""Tests for the /chat-dummy and /chat-twilio-webhook endpoints."""

from __future__ import annotations

import json
import uuid
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from twilio.request_validator import RequestValidator

from main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

READ_KEY_HEADER = {"Authorization": "test-api-key"}
TWILIO_TOKEN = "test-twilio-token"


@pytest.fixture()
def client():
    """TestClient that sends the read API key by default."""
    return TestClient(app, headers=READ_KEY_HEADER)


def _twilio_headers(path: str, params: dict, form: dict, token: str = TWILIO_TOKEN):
    """Sign a webhook request the way Twilio does."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"http://testserver{path}?{query}"
    return {"X-Twilio-Signature": RequestValidator(token).compute_signature(url, form)}


def _make_agent_response(text: str, doc_contents: list[str] | None = None):
    """Build a fake rag_agent.invoke() return value."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = text
    docs = [Document(page_content=content) for content in doc_contents or []]
    return {"messages": [ai_msg], "retrieved_docs": docs}


# ---------------------------------------------------------------------------
# /chat-dummy
# ---------------------------------------------------------------------------


class TestChatDummy:
    """Tests for the POST /chat-dummy endpoint."""

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_basic_response(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "You are helpful."
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("Hello there!")
        mock_get_agent.return_value = mock_agent

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
    @patch("routes.chat.get_rag_agent")
    def test_include_context_false(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response(
            "answer", doc_contents=["doc1", "doc2"]
        )
        mock_get_agent.return_value = mock_agent

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
    @patch("routes.chat.get_rag_agent")
    def test_include_context_true(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response(
            "answer", doc_contents=["ctx1", "ctx2"]
        )
        mock_get_agent.return_value = mock_agent

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
    @patch("routes.chat.get_rag_agent")
    def test_include_context_no_tool_messages(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When include_context=True but the agent answered without retrieving,
        context should be an empty list."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response(
            "answer", doc_contents=[]
        )
        mock_get_agent.return_value = mock_agent

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
    @patch("routes.chat.get_rag_agent")
    def test_uses_default_google_sheet_id(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When no googleSheetId is provided, the default is used."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        resp = client.post("/chat-dummy", json={"message": "hello"})

        assert resp.status_code == 200
        # PromptLoader should have been called with the default sheet ID
        call_kwargs = mock_prompt_loader.call_args
        assert (
            call_kwargs[1]["document_id"]
            == "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04"
        )

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_prompt_and_sheet_id_travel_in_config_not_messages(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "You are helpful."
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123"},
            json={"message": "hello"},
        )

        assert resp.status_code == 200
        state_input, kwargs = mock_agent.invoke.call_args
        configurable = kwargs["config"]["configurable"]
        assert configurable["system_prompt"] == "You are helpful."
        assert configurable["googleSheetId"] == "sheet123"
        assert [m.type for m in state_input[0]["messages"]] == ["human"]

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="it")
    @patch("routes.chat.translate")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_translation_roundtrip(
        self,
        mock_get_agent,
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
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("english answer")
        mock_get_agent.return_value = mock_agent

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
    @patch("routes.chat.get_rag_agent")
    def test_fallback_prompt(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When PromptLoader returns empty string, the default prompt file is used."""
        mock_prompt_loader.return_value.get_prompt.return_value = ""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("answer")
        mock_get_agent.return_value = mock_agent

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

    def test_requires_read_key(self):
        resp = TestClient(app).post(
            "/chat-dummy", params={"googleSheetId": "sheet123"}, json={"message": "hi"}
        )
        assert resp.status_code == 401

    def test_rejects_wrong_read_key(self):
        resp = TestClient(app, headers={"Authorization": "wrong"}).post(
            "/chat-dummy", params={"googleSheetId": "sheet123"}, json={"message": "hi"}
        )
        assert resp.status_code == 401

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_custom_thread_id(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When threadId is explicitly provided, it should be forwarded to the agent."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

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
        assert resp.json()["threadId"] == "my-thread-42"

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_generates_fresh_thread_id_when_omitted(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Without threadId each request gets its own random thread, never a shared one."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        ids = []
        for _ in range(2):
            resp = client.post(
                "/chat-dummy",
                params={"googleSheetId": "sheet123"},
                json={"message": "hello"},
            )
            assert resp.status_code == 200
            thread_id = resp.json()["threadId"]
            uuid.UUID(thread_id)  # raises if not a valid uuid
            config = mock_agent.invoke.call_args[1]["config"]
            assert config["configurable"]["thread_id"] == thread_id
            ids.append(thread_id)
        assert ids[0] != ids[1]

    def test_rejects_overlong_thread_id(self, client):
        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123", "threadId": "x" * 201},
            json={"message": "hello"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /chat-twilio-webhook
# ---------------------------------------------------------------------------


class TestChatTwilioWebhook:
    """Tests for the POST /chat-twilio-webhook endpoint."""

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_basic_twilio_response(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("Hello from bot!")
        mock_get_agent.return_value = mock_agent

        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "+31612345678"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert "Hello from bot!" in resp.text

    def test_twilio_missing_signature(self, client):
        resp = client.post(
            "/chat-twilio-webhook",
            params={"googleSheetId": "sheet123"},
            data={"Body": "Hi", "From": "+31612345678"},
        )
        assert resp.status_code == 401

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_replies_with_fallback_when_chat_fails(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Twilio gets valid TwiML (200) so the user is told something went wrong."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("boom")
        mock_get_agent.return_value = mock_agent

        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "+31612345678"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/xml"
        assert "something went wrong" in resp.text
        assert "boom" not in resp.text

    def test_twilio_invalid_signature(self, client):
        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "+31612345678"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers(
                "/chat-twilio-webhook", params, form, token="other-token"
            ),
        )
        assert resp.status_code == 401

    def test_twilio_missing_token_rejects(self, client, monkeypatch):
        monkeypatch.delenv("TWILIO_AUTH_TOKEN")
        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "+31612345678"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )
        assert resp.status_code == 401

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_per_instance_tokens(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client, monkeypatch
    ):
        """Each googleSheetId is validated against its own Twilio account token."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent
        monkeypatch.setenv(
            "TWILIO_AUTH_TOKENS", json.dumps({"sheetA": "tok-A", "sheetB": "tok-B"})
        )
        form = {"Body": "Hi", "From": "+31612345678"}

        def post(sheet: str, token: str):
            params = {"googleSheetId": sheet}
            return client.post(
                "/chat-twilio-webhook",
                params=params,
                data=form,
                headers=_twilio_headers("/chat-twilio-webhook", params, form, token),
            )

        assert post("sheetA", "tok-A").status_code == 200
        assert post("sheetB", "tok-B").status_code == 200
        # account A must not be able to drive instance B
        assert post("sheetB", "tok-A").status_code == 401
        # sheet without an entry is rejected even though TWILIO_AUTH_TOKEN is set
        assert post("sheetC", TWILIO_TOKEN).status_code == 401

    def test_twilio_invalid_tokens_json_rejects(self, client, monkeypatch):
        monkeypatch.setenv("TWILIO_AUTH_TOKENS", "{not json")
        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "+31612345678"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )
        assert resp.status_code == 401

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_signature_honours_forwarded_proto(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Twilio signs the public https URL even when the proxy forwards http."""
        mock_prompt_loader.return_value.get_prompt.return_value = "prompt"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "+31612345678"}
        url = "https://testserver/chat-twilio-webhook?googleSheetId=sheet123"
        signature = RequestValidator(TWILIO_TOKEN).compute_signature(url, form)
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers={"X-Twilio-Signature": signature, "X-Forwarded-Proto": "https"},
        )
        assert resp.status_code == 200

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_missing_body(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When Body is missing from the form data, return 400."""
        params = {"googleSheetId": "sheet123"}
        form = {"From": "+31612345678"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )

        assert resp.status_code == 400

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="nl")
    @patch("routes.chat.translate")
    @patch("routes.chat.PromptLoader")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_with_translation(
        self,
        mock_get_agent,
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
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("english reply")
        mock_get_agent.return_value = mock_agent

        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hallo", "From": "+31600000000"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )

        assert resp.status_code == 200
        # response should contain the translated-back text
        assert "[nl]english reply" in resp.text
