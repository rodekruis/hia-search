"""Shared fixtures and mocks for all tests."""

from __future__ import annotations

import os
import sys
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
    "OPENAI_API_TYPE": "azure",
    "OPENAI_ENDPOINT": "https://fake-openai.openai.azure.com/",
    "OPENAI_API_KEY": "fake-key",
    "OPENAI_API_VERSION": "2024-02-01",
    "MODEL_EMBEDDINGS": "text-embedding-ada-002",
    "MODEL_CHAT": "gpt-4",
    # empty: App Insights export must stay disabled in tests (never a real key from the shell)
    "APPLICATIONINSIGHTS_CONNECTION_STRING": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "MSCOGNITIVE_KEY": "fake-cognitive-key",
    "MSCOGNITIVE_LOCATION": "westeurope",
    "TWILIO_AUTH_TOKEN": "test-twilio-token",
}

# Force-set so values injected by the shell/.env never leak into tests
os.environ.update(_ENV_DEFAULTS)

# ---------------------------------------------------------------------------
# Patch heavy modules that connect to external services on import
# ---------------------------------------------------------------------------

# Mock the rag_agent module so it doesn't connect to PostgreSQL / OpenAI
_mock_rag_agent = MagicMock()
_mock_get_rag_agent = MagicMock(return_value=_mock_rag_agent)
sys.modules.setdefault("agents.rag_agent", MagicMock(get_rag_agent=_mock_get_rag_agent))

# ---------------------------------------------------------------------------
# Now it's safe to import the FastAPI app
# ---------------------------------------------------------------------------
from main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    """FastAPI TestClient with mocked externals."""
    return TestClient(app)


@pytest.fixture()
def mock_rag_agent():
    """Return the mocked rag_agent so tests can configure its .invoke()."""
    return _mock_rag_agent
