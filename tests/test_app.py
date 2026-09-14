"""Tests for the main FastAPI app (root, models, CORS, lifespan)."""

from __future__ import annotations
import os
import sys

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from main import app


class TestLifespan:
    def test_agent_initialized_on_startup_and_closed_on_shutdown(self):
        # conftest replaces agents.rag_agent with a mock; main imported from it
        rag_agent = sys.modules["agents.rag_agent"]
        rag_agent.init_rag_agent.reset_mock()
        rag_agent.close_rag_agent.reset_mock()

        with TestClient(app):
            rag_agent.init_rag_agent.assert_called_once()
            rag_agent.close_rag_agent.assert_not_called()
        rag_agent.close_rag_agent.assert_called_once()

    def test_tracing_initialized_and_shut_down(self, monkeypatch):
        import main as main_module

        init, shutdown = MagicMock(), MagicMock()
        monkeypatch.setattr(main_module, "init_tracing", init)
        monkeypatch.setattr(main_module, "shutdown_tracing", shutdown)

        with TestClient(app):
            init.assert_called_once()
            shutdown.assert_not_called()
        shutdown.assert_called_once()

    def test_startup_failure_aborts_boot(self):
        rag_agent = sys.modules["agents.rag_agent"]
        rag_agent.init_rag_agent.side_effect = RuntimeError("db down")
        try:
            with pytest.raises(RuntimeError, match="db down"):
                with TestClient(app):
                    pass
        finally:
            rag_agent.init_rag_agent.side_effect = None


class TestAppRoot:

    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert "/docs" in resp.headers["location"]

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_get_models(self, client):
        resp = client.get("/get-models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "OpenAI"
        assert "model embeddings" in body
        assert "model chat" in body

    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "hia-search"
        # verify all expected paths exist
        paths = schema["paths"]
        assert "/search" in paths
        assert "/chat-dummy" in paths
        assert "/chat-twilio-webhook" in paths
        assert "/create-vector-store" in paths
        assert "/delete-vector-store" in paths
        assert "/get-models" in paths
