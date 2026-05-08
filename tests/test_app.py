"""Tests for the main FastAPI app (root, models, CORS)."""

from __future__ import annotations
import os


class TestAppRoot:

    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert "/docs" in resp.headers["location"]

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
