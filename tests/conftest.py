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
    "AISAFETY_ENDPOINT": "https://fake-safety.cognitiveservices.azure.com",
    "AISAFETY_API_KEY": "fake-safety-key",
    "AISAFETY_API_VERSION": "2024-06-01",
    "MODEL_EMBEDDINGS": "text-embedding-ada-002",
    "MODEL_CHAT": "gpt-4",
    "MODEL_GROUNDEDNESS": "gpt-4",
    "APPLICATIONINSIGHTS_CONNECTION_STRING": "InstrumentationKey=00000000-0000-0000-0000-000000000000",
    "MSCOGNITIVE_KEY": "fake-cognitive-key",
    "MSCOGNITIVE_LOCATION": "westeurope",
}

for k, v in _ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

# ---------------------------------------------------------------------------
# Patch heavy modules that connect to external services on import
# ---------------------------------------------------------------------------

# Mock the Azure Monitor logger so it doesn't try to connect
_mock_logger = MagicMock()
sys.modules.setdefault("utils.logger", MagicMock(logger=_mock_logger))

# Mock the rag_agent module so it doesn't connect to PostgreSQL / OpenAI
_mock_rag_agent = MagicMock()
sys.modules.setdefault("agents.rag_agent", MagicMock(rag_agent=_mock_rag_agent))

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
