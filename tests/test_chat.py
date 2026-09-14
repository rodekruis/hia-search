"""Tests for the /chat-dummy and /chat-twilio-webhook endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
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


def _logged_text(caplog) -> str:
    """Everything that reached the application loggers, messages and extra fields alike."""
    return " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)


def _record(caplog, message: str) -> logging.LogRecord:
    return next(r for r in caplog.records if r.getMessage() == message)


@pytest.fixture()
def app_logger(caplog):
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture()
def client():
    """TestClient that sends the read API key by default."""
    return TestClient(app, headers=READ_KEY_HEADER)


def _twilio_headers(path: str, params: dict, form: dict, token: str = TWILIO_TOKEN):
    """Sign a webhook request the way Twilio does."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"http://testserver{path}?{query}"
    return {"X-Twilio-Signature": RequestValidator(token).compute_signature(url, form)}


def _make_agent_response(text: str, doc_contents: list[str] | None = None, search_query: str = ""):
    """Build a fake rag_agent.invoke() return value."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = text
    docs = [Document(page_content=content) for content in doc_contents or []]
    return {"messages": [ai_msg], "retrieved_docs": docs, "search_query": search_query}


# ---------------------------------------------------------------------------
# /chat-dummy
# ---------------------------------------------------------------------------


class TestChatDummy:
    """Tests for the POST /chat-dummy endpoint."""

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt")
    @patch("routes.chat.get_rag_agent")
    def test_basic_response(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        mock_prompt_loader.return_value = "You are helpful."
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
        assert body["traceId"] is None  # tracing disabled in tests

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_include_context_false(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_include_context_true(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_include_context_no_tool_messages(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When include_context=True but the agent answered without retrieving,
        context should be an empty list."""
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_uses_default_google_sheet_id(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When no googleSheetId is provided, the default is used."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        resp = client.post("/chat-dummy", json={"message": "hello"})

        assert resp.status_code == 200
        mock_prompt_loader.assert_called_once_with(
            "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04"
        )

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="You are helpful.")
    @patch("routes.chat.get_rag_agent")
    def test_prompt_and_sheet_id_travel_in_config_not_messages(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_custom_thread_id(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """When threadId is explicitly provided, it should be forwarded to the agent."""
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_generates_fresh_thread_id_when_omitted(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Without threadId each request gets its own random thread, never a shared one."""
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

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="uk")
    @patch("routes.chat.translate", side_effect=lambda from_lang, to_lang, text: text)
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_logs_turn_metadata_but_never_content(
        self, mock_get_agent, mock_prompt, mock_translate, mock_detect, mock_vs, client, app_logger
    ):
        """Conversation content belongs to the LLM observability tool, not App Insights."""
        user_text = "My asylum interview is tomorrow and I am scared"
        bot_text = "Here is what will happen at the interview"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response(bot_text, doc_contents=["d1", "d2"])
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/chat-dummy",
            params={"googleSheetId": "sheet123", "threadId": "t-1"},
            json={"message": user_text},
        )
        assert resp.status_code == 200

        logged = _logged_text(app_logger)
        assert user_text not in logged
        assert bot_text not in logged
        assert "d1" not in logged

        turn = _record(app_logger, "chat turn").__dict__
        assert turn["googleSheetId"] == "sheet123"
        assert turn["threadId"] == "t-1"
        assert turn["detected_lang"] == "uk"
        assert turn["retrieval_used"] is True and turn["n_docs"] == 2
        assert turn["message_chars"] == len(user_text)
        assert turn["response_chars"] == len(bot_text)
        assert isinstance(turn["duration_ms"], int)


# ---------------------------------------------------------------------------
# /chat-twilio-webhook
# ---------------------------------------------------------------------------


class TestChatTwilioWebhook:
    """Tests for the POST /chat-twilio-webhook endpoint."""

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_basic_twilio_response(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
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
        assert resp.text.count("<Message>") == 1
        assert "(1/" not in resp.text

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_logs_no_content_or_phone_number(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client, app_logger
    ):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("Go to the town hall.")
        mock_get_agent.return_value = mock_agent

        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Where do I register?", "From": "whatsapp:+31612345678"}
        client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )

        logged = _logged_text(app_logger)
        assert "Where do I register?" not in logged
        assert "Go to the town hall." not in logged
        assert "31612345678" not in logged
        reply = _record(app_logger, "twilio reply").__dict__
        assert reply["n_chunks"] == 1
        assert reply["threadId"] == hashlib.sha256(b"whatsapp:+31612345678").hexdigest()

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_long_answer_is_sent_as_numbered_messages(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Twilio rejects a single <Message> of 1600+ chars, so long answers are split."""
        import xml.etree.ElementTree as ET

        answer = "\n\n".join(f"Paragraph {i}: " + "detail " * 60 for i in range(10))
        assert len(answer) > 1600
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response(answer)
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
        bodies = [m.text for m in ET.fromstring(resp.text).findall("Message")]
        assert len(bodies) >= 3
        assert all(len(b) < 1600 for b in bodies)
        assert [b.split(" ")[0] for b in bodies] == [
            f"({i}/{len(bodies)})" for i in range(1, len(bodies) + 1)
        ]
        assert "Paragraph 0:" in bodies[0] and "Paragraph 9:" in bodies[-1]

    def test_twilio_missing_signature(self, client):
        resp = client.post(
            "/chat-twilio-webhook",
            params={"googleSheetId": "sheet123"},
            data={"Body": "Hi", "From": "+31612345678"},
        )
        assert resp.status_code == 401

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_replies_with_fallback_when_chat_fails(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Twilio gets valid TwiML (200) so the user is told something went wrong."""
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_per_instance_tokens(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client, monkeypatch
    ):
        """Each googleSheetId is validated against its own Twilio account token."""
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_signature_honours_forwarded_proto(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Twilio signs the public https URL even when the proxy forwards http."""
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
    @patch("routes.chat.get_system_prompt", return_value="prompt")
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

    def test_twilio_missing_sender(self, client):
        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi"}
        resp = client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )
        assert resp.status_code == 400
        assert "sender" in resp.text

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_thread_is_hashed_sender(
        self, mock_get_agent, mock_prompt_loader, mock_detect, mock_vs, client
    ):
        """Phone numbers never reach the checkpoint store in clear text."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "whatsapp:+31612345678"}
        client.post(
            "/chat-twilio-webhook",
            params=params,
            data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )

        thread_id = mock_agent.invoke.call_args.kwargs["config"]["configurable"]["thread_id"]
        assert thread_id == hashlib.sha256(b"whatsapp:+31612345678").hexdigest()
        assert "31612345678" not in thread_id

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="nl")
    @patch("routes.chat.translate")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
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


# ---------------------------------------------------------------------------
# Langfuse: chat-turn root span and /feedback
# ---------------------------------------------------------------------------


class TestChatTracing:
    @pytest.fixture()
    def chat_span(self, monkeypatch):
        import utils.tracing as tracing

        lf_client = MagicMock(name="chat-client")
        span = MagicMock(name="span")
        span.trace_id = "trace-abc"
        lf_client.start_as_current_observation.return_value.__enter__.return_value = span
        monkeypatch.setattr(tracing, "_clients", {"chat": lf_client})
        monkeypatch.setattr(tracing, "_public_keys", {"chat": "pk-chat"})
        monkeypatch.setattr(tracing, "propagate_attributes", MagicMock())
        callbacks = [MagicMock(name="callback-handler")]
        monkeypatch.setattr("routes.chat.langchain_callbacks", lambda project: callbacks)
        return span, lf_client, tracing.propagate_attributes, callbacks

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="uk")
    @patch("routes.chat.translate", side_effect=lambda from_lang, to_lang, text: f"[{to_lang}]{text}")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_retrieval_turn_span_has_evaluator_fields(
        self, mock_get_agent, mock_prompt, mock_translate, mock_detect, mock_vs, client, chat_span
    ):
        span, lf_client, propagate, callbacks = chat_span
        prior_user, prior_bot = MagicMock(type="human", content="earlier q"), MagicMock(type="ai", content="earlier a")
        current_user, answer = MagicMock(type="human", content="[en]Де лікар?"), MagicMock(type="ai", content="At the GP.")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [prior_user, prior_bot, current_user, answer],
            "retrieved_docs": [Document(page_content="doc A"), Document(page_content="doc B")],
            "search_query": "where is a doctor",
        }
        mock_get_agent.return_value = mock_agent

        resp = client.post(
            "/chat-dummy", params={"googleSheetId": "sheetX", "threadId": "t-9"}, json={"message": "Де лікар?"}
        )

        assert resp.status_code == 200
        assert resp.json()["traceId"] == "trace-abc"
        propagate.assert_called_once_with(
            session_id="t-9", user_id="t-9", tags=["sheet:sheetX", "channel:dummy", "lang:uk"]
        )
        # input is the user's message verbatim, in their language
        lf_client.start_as_current_observation.assert_called_once_with(
            as_type="span", name="chat-turn", input="Де лікар?"
        )
        # graph spans nest under the root via the project's callback handler
        assert mock_agent.invoke.call_args.kwargs["config"]["configurable"]["llm_callbacks"] == callbacks

        update = span.update.call_args.kwargs
        assert update["output"] == "[uk]At the GP."
        assert update["metadata"] == {
            "retrieval_used": True,
            "n_docs": 2,
            "detected_lang": "uk",
            "message_en": "[en]Де лікар?",
            "answer_en": "At the GP.",
            "conversation_history": "user: earlier q\nassistant: earlier a",
            "retrieved_context": "Document: doc A\n\nDocument: doc B",
            "search_query": "where is a doctor",
        }

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_small_talk_span_has_no_retrieval_fields(
        self, mock_get_agent, mock_prompt, mock_detect, mock_vs, client, chat_span
    ):
        span, *_ = chat_span
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("You're welcome!")
        mock_get_agent.return_value = mock_agent

        client.post("/chat-dummy", params={"googleSheetId": "s"}, json={"message": "thanks"})

        metadata = span.update.call_args.kwargs["metadata"]
        assert metadata["retrieval_used"] is False and metadata["n_docs"] == 0
        assert "retrieved_context" not in metadata and "search_query" not in metadata

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_twilio_turns_are_tagged_by_channel(
        self, mock_get_agent, mock_prompt, mock_detect, mock_vs, client, chat_span
    ):
        _, _, propagate, _ = chat_span
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        params = {"googleSheetId": "sheet123"}
        form = {"Body": "Hi", "From": "+31612345678"}
        client.post(
            "/chat-twilio-webhook", params=params, data=form,
            headers=_twilio_headers("/chat-twilio-webhook", params, form),
        )

        tags = propagate.call_args.kwargs["tags"]
        assert "channel:twilio" in tags
        thread = hashlib.sha256(b"+31612345678").hexdigest()
        assert propagate.call_args.kwargs["session_id"] == thread

    @patch("routes.chat.get_vector_store")
    @patch("routes.chat.detect_language", return_value="en")
    @patch("routes.chat.get_system_prompt", return_value="prompt")
    @patch("routes.chat.get_rag_agent")
    def test_no_callbacks_when_untraced(self, mock_get_agent, mock_prompt, mock_detect, mock_vs, client):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = _make_agent_response("ok")
        mock_get_agent.return_value = mock_agent

        client.post("/chat-dummy", params={"googleSheetId": "s"}, json={"message": "hi"})

        assert mock_agent.invoke.call_args.kwargs["config"]["configurable"]["llm_callbacks"] == []


class TestFeedback:
    def test_requires_read_key(self):
        resp = TestClient(app).post("/feedback", json={"traceId": "t", "positive": True})
        assert resp.status_code == 401

    def test_unavailable_when_chat_tracing_disabled(self, client):
        resp = client.post("/feedback", json={"traceId": "t", "positive": True})
        assert resp.status_code == 503

    @pytest.mark.parametrize("positive,value", [(True, 1.0), (False, 0.0)])
    def test_scores_trace_in_chat_project(self, client, monkeypatch, positive, value):
        import utils.tracing as tracing

        lf_client = MagicMock()
        monkeypatch.setattr(tracing, "_clients", {"chat": lf_client, "search": MagicMock()})

        resp = client.post(
            "/feedback", json={"traceId": "trace-abc", "positive": positive, "comment": "useful"}
        )

        assert resp.status_code == 202
        lf_client.create_score.assert_called_once_with(
            trace_id="trace-abc", name="user-feedback", value=value, data_type="NUMERIC", comment="useful"
        )
        tracing._clients["search"].create_score.assert_not_called()

    def test_validates_payload(self, client):
        assert client.post("/feedback", json={"positive": True}).status_code == 422
        assert client.post("/feedback", json={"traceId": "t"}).status_code == 422
