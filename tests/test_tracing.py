"""Tests for utils/tracing.py (shared Langfuse client, optional root span)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import utils.tracing as tracing


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch):
    monkeypatch.setattr(tracing, "_langfuse", None)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


class TestInit:
    def test_disabled_without_keys(self, monkeypatch):
        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)

        assert tracing.init_tracing() is None
        assert tracing.get_langfuse() is None
        langfuse_cls.assert_not_called()

    def test_disabled_with_only_one_key(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setattr(tracing, "Langfuse", MagicMock())
        assert tracing.init_tracing() is None

    def test_builds_once_from_env(self, monkeypatch):
        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://lf.example")
        monkeypatch.setenv("ENVIRONMENT", "dev")

        first = tracing.init_tracing()
        second = tracing.init_tracing()

        assert first is second is langfuse_cls.return_value is tracing.get_langfuse()
        langfuse_cls.assert_called_once_with(
            public_key="pk", secret_key="sk", base_url="https://lf.example", environment="dev"
        )

    def test_defaults_to_prod_environment_and_cloud_url(self, monkeypatch):
        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "")

        tracing.init_tracing()

        kwargs = langfuse_cls.call_args.kwargs
        assert kwargs["base_url"] is None and kwargs["environment"] == "prod"

    def test_shutdown_flushes_and_resets(self, monkeypatch):
        client = MagicMock()
        monkeypatch.setattr(tracing, "_langfuse", client)

        tracing.shutdown_tracing()

        client.shutdown.assert_called_once()
        assert tracing.get_langfuse() is None

    def test_shutdown_without_client_is_noop(self):
        tracing.shutdown_tracing()


class TestObserve:
    def test_yields_none_when_disabled(self):
        with tracing.observe("search", input="q", tags=["x"]) as span:
            assert span is None

    def test_opens_root_span_with_propagated_attributes(self, monkeypatch):
        client = MagicMock()
        span = MagicMock(name="span")
        client.start_as_current_observation.return_value.__enter__.return_value = span
        monkeypatch.setattr(tracing, "_langfuse", client)
        propagate = MagicMock()
        monkeypatch.setattr(tracing, "propagate_attributes", propagate)

        with tracing.observe(
            "search", input="q", tags=["sheet:s"], session_id="t1", user_id="u1"
        ) as got:
            assert got is span

        propagate.assert_called_once_with(session_id="t1", user_id="u1", tags=["sheet:s"])
        client.start_as_current_observation.assert_called_once_with(
            as_type="span", name="search", input="q"
        )
