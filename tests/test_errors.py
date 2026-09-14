"""Tests for request-id propagation and structured error responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from azure.core.exceptions import HttpResponseError
from fastapi.testclient import TestClient
from openai import APIConnectionError

from main import app

AUTH = {"Authorization": "test-api-key"}


def _client():
    # Let the generic 500 handler answer instead of re-raising into the test
    return TestClient(app, headers=AUTH, raise_server_exceptions=False)


def _chat_failing_with(exc: Exception):
    """Return patches making /chat-dummy raise `exc` from the agent."""
    agent = MagicMock()
    agent.invoke.side_effect = exc
    return (
        patch("routes.chat.get_vector_store"),
        patch("routes.chat.detect_language", return_value="en"),
        patch("routes.chat.PromptLoader"),
        patch("routes.chat.get_rag_agent", return_value=agent),
    )


def _post_chat(client):
    return client.post(
        "/chat-dummy", params={"googleSheetId": "s"}, json={"message": "hi"}
    )


class TestRequestId:
    def test_every_response_has_request_id_header(self, client):
        resp = client.get("/health")
        assert resp.headers["x-request-id"].startswith("req_")

    def test_request_ids_are_unique(self, client):
        ids = {client.get("/health").headers["x-request-id"] for _ in range(3)}
        assert len(ids) == 3


class TestErrorHandlers:
    def test_external_service_error_returns_502(self):
        exc = APIConnectionError(request=MagicMock())
        p1, p2, p3, p4 = _chat_failing_with(exc)
        with p1, p2, p3, p4:
            resp = _post_chat(_client())

        assert resp.status_code == 502
        body = resp.json()["error"]
        assert body["code"] == "external_service_error"
        assert body["request_id"] == resp.headers["x-request-id"]
        # no upstream details leak to the client
        assert "APIConnectionError" not in resp.text

    def test_azure_error_returns_502(self):
        p1, p2, p3, p4 = _chat_failing_with(HttpResponseError("index gone"))
        with p1, p2, p3, p4:
            resp = _post_chat(_client())

        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "external_service_error"
        assert "index gone" not in resp.text

    def test_unexpected_error_returns_500_without_details(self):
        p1, p2, p3, p4 = _chat_failing_with(RuntimeError("secret internals"))
        with p1, p2, p3, p4:
            resp = _post_chat(_client())

        assert resp.status_code == 500
        body = resp.json()["error"]
        assert body["code"] == "internal_error"
        assert body["request_id"] == resp.headers["x-request-id"]
        assert "secret internals" not in resp.text

    def test_value_error_returns_422(self):
        p1, p2, p3, p4 = _chat_failing_with(ValueError("bad input"))
        with p1, p2, p3, p4:
            resp = _post_chat(_client())

        assert resp.status_code == 422
        assert resp.json()["error"] == {
            "code": "validation_error",
            "message": "bad input",
            "request_id": resp.headers["x-request-id"],
        }

    def test_http_exceptions_keep_default_shape(self, client):
        resp = TestClient(app).post(
            "/chat-dummy", params={"googleSheetId": "s"}, json={"message": "hi"}
        )
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Unauthorized"}
