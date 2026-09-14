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
        langfuse_cls.assert_called_once()
        kwargs = langfuse_cls.call_args.kwargs
        assert kwargs["public_key"] == "pk" and kwargs["secret_key"] == "sk"
        assert kwargs["base_url"] == "https://lf.example" and kwargs["environment"] == "dev"

    def test_uses_a_dedicated_tracer_provider(self, monkeypatch):
        """Langfuse must not share the global provider that App Insights exports from."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

        tracing.init_tracing()

        provider = langfuse_cls.call_args.kwargs["tracer_provider"]
        assert isinstance(provider, TracerProvider)
        assert provider is not trace.get_tracer_provider()

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


class TestProviderIsolation:
    """Offline end-to-end check of the split between App Insights and Langfuse spans.

    Uses in-memory exporters on both sides; no network. Guards the privacy
    contract: content-bearing Langfuse spans never reach the global provider.
    """

    def test_spans_are_partitioned_and_langfuse_span_is_app_root(self):
        from langfuse import Langfuse
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        # "App Insights" side: a provider standing in for the global one
        azure_exporter = InMemorySpanExporter()
        global_provider = TracerProvider()
        global_provider.add_span_processor(SimpleSpanProcessor(azure_exporter))
        http_tracer = global_provider.get_tracer("opentelemetry.instrumentation.fastapi")

        # Langfuse side: isolated provider, exporter captured in memory
        lf_exporter = InMemorySpanExporter()
        langfuse = Langfuse(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            tracer_provider=TracerProvider(),
            span_exporter=lf_exporter,
        )
        try:
            with http_tracer.start_as_current_span("POST /search") as server_span:
                with langfuse.start_as_current_observation(
                    as_type="span", name="search", input="user text"
                ) as root:
                    with http_tracer.start_as_current_span("HTTP POST azure-search"):
                        pass
                    root.update(output="retrieved text", metadata={"k": "user text"})
            langfuse.flush()
            global_provider.force_flush()
        finally:
            langfuse.shutdown()

        lf = {s.name: s for s in lf_exporter.get_finished_spans()}
        az = {s.name: s for s in azure_exporter.get_finished_spans()}

        assert set(lf) == {"search"}
        assert set(az) == {"POST /search", "HTTP POST azure-search"}
        # the Langfuse span has a foreign parent but is marked as application root
        search = lf["search"]
        assert search.parent.span_id == server_span.context.span_id
        assert search.attributes.get("langfuse.internal.is_app_root") is True
        # one trace id across both backends, so requests can be correlated by hand
        assert len({s.context.trace_id for s in [*lf.values(), *az.values()]}) == 1
        # nothing content-bearing crossed over
        assert not any(
            "user text" in str(v) or "retrieved text" in str(v)
            for s in az.values()
            for v in s.attributes.values()
        )
