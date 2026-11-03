"""Shared fixtures and mocks for all tests."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

# Set dummy env vars BEFORE any app modules are imported
_ENV_DEFAULTS = {
    "PORT": "8000",
    "API_KEY": "test-api-key",
    "API_KEY_WRITE": "test-write-key",
    "VECTOR_STORE_ADDRESS": "https://fake-search.search.windows.net",
    "VECTOR_STORE_PASSWORD": "fake-password",
    "CHECKPOINT_DB_USER": "user",
    "CHECKPOINT_DB_PASSWORD": "pass",
    "CHECKPOINT_DB_HOST": "localhost",
    "SUBSCRIPTION_DB_HOST": "unused.invalid",
    "SUBSCRIPTION_DB_NAME": "unused_test",
    "SUBSCRIPTION_DB_USER": "unused_test",
    "SUBSCRIPTION_DB_PASSWORD": "unused-test-only",
    "OPENAI_API_TYPE": "azure",
    "OPENAI_ENDPOINT": "https://fake-openai.openai.azure.com/",
    "OPENAI_API_KEY": "fake-key",
    "OPENAI_API_VERSION": "2024-02-01",
    "MODEL_EMBEDDINGS": "text-embedding-ada-002",
    "MODEL_CHAT": "gpt-4",
    # empty: App Insights export must stay disabled in tests (never a real key from the shell)
    "APPLICATIONINSIGHTS_CONNECTION_STRING": "",
    "LANGFUSE_SEARCH_PUBLIC_KEY": "",
    "LANGFUSE_SEARCH_SECRET_KEY": "",
    "LANGFUSE_CHAT_PUBLIC_KEY": "",
    "LANGFUSE_CHAT_SECRET_KEY": "",
    "MSCOGNITIVE_KEY": "fake-cognitive-key",
    "MSCOGNITIVE_LOCATION": "westeurope",
    "TWILIO_AUTH_TOKEN": "test-twilio-token",
    # empty (not unset, or load_dotenv would fill it from .env): it overrides TWILIO_AUTH_TOKEN
    "TWILIO_AUTH_TOKENS": "",
}

# Force-set so values injected by the shell/.env never leak into tests
# Never inherit subscription credentials from the shell or a developer's .env.
for _name in tuple(os.environ):
    if _name.startswith("SUBSCRIPTION_DB_"):
        del os.environ[_name]
os.environ.update(_ENV_DEFAULTS)

# ---------------------------------------------------------------------------
# Patch heavy modules that connect to external services on import
# ---------------------------------------------------------------------------

# Mock the rag_agent module so it doesn't connect to PostgreSQL / OpenAI
_mock_rag_agent = MagicMock()
_mock_get_rag_agent = MagicMock(return_value=_mock_rag_agent)
sys.modules.setdefault(
    "agents.rag_agent",
    MagicMock(
        get_rag_agent=_mock_get_rag_agent,
        # pure helper, mirrored here so route tests see the real context format
        format_context=lambda docs: "\n\n".join(
            f"[{index}] {d.page_content}" for index, d in enumerate(docs, start=1)
        ),
    ),
)

# ---------------------------------------------------------------------------
# Now it's safe to import the FastAPI app
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient

from main import app


@pytest.fixture()
def client():
    """FastAPI TestClient with mocked externals."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def isolate_subscription_store(monkeypatch):
    """Existing regression tests must never open a subscription database."""
    monkeypatch.setattr(
        app.state, "subscription_store_factory", lambda: nullcontext(MagicMock())
    )


@pytest.fixture(autouse=True)
def support_contacts():
    """Route tests stub get_system_prompt; the support contacts come from the same Chat sheet."""
    with patch("routes.chat.get_support_contacts", return_value={}) as mock:
        yield mock


@pytest.fixture()
def mock_rag_agent():
    """Return the mocked rag_agent so tests can configure its .invoke()."""
    return _mock_rag_agent
