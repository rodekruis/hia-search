"""Tests for utils/tracing.py (per-project Langfuse clients, optional root span)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import utils.tracing as tracing


@pytest.fixture(autouse=True)
def _reset_clients(monkeypatch):
    monkeypatch.setattr(tracing, "_clients", {})
    monkeypatch.setattr(tracing, "_public_keys", {})
    for project in ("SEARCH", "CHAT"):
        monkeypatch.delenv(f"LANGFUSE_{project}_PUBLIC_KEY", raising=False)
        monkeypatch.delenv(f"LANGFUSE_{project}_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


def _set_keys(monkeypatch, project: str, suffix: str = ""):
    monkeypatch.setenv(f"LANGFUSE_{project.upper()}_PUBLIC_KEY", f"pk-{project}{suffix}")
    monkeypatch.setenv(f"LANGFUSE_{project.upper()}_SECRET_KEY", f"sk-{project}{suffix}")


class TestInit:
    def test_disabled_without_keys(self, monkeypatch):
        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)

        assert tracing.init_tracing() == {}
        assert tracing.get_langfuse("search") is None
        assert tracing.get_langfuse("chat") is None
        langfuse_cls.assert_not_called()

    def test_disabled_with_only_one_key(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_CHAT_PUBLIC_KEY", "pk")
        monkeypatch.setattr(tracing, "Langfuse", MagicMock())
        assert tracing.init_tracing() == {}

    def test_one_client_per_project_with_distinct_keys_same_url(self, monkeypatch):
        langfuse_cls = MagicMock(side_effect=lambda **kw: MagicMock(name=kw["public_key"]))
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
        _set_keys(monkeypatch, "search")
        _set_keys(monkeypatch, "chat")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://lf.example")
        monkeypatch.setenv("ENVIRONMENT", "dev")

        clients = tracing.init_tracing()

        assert set(clients) == {"search", "chat"}
        assert clients["search"] is not clients["chat"]
        by_key = {c.kwargs["public_key"]: c.kwargs for c in langfuse_cls.call_args_list}
        assert by_key["pk-search"]["secret_key"] == "sk-search"
        assert by_key["pk-chat"]["secret_key"] == "sk-chat"
        assert all(k["base_url"] == "https://lf.example" and k["environment"] == "dev" for k in by_key.values())
        assert tracing.get_langfuse("chat") is clients["chat"]

    def test_only_configured_project_is_traced(self, monkeypatch):
        monkeypatch.setattr(tracing, "Langfuse", MagicMock())
        _set_keys(monkeypatch, "chat")

        clients = tracing.init_tracing()

        assert set(clients) == {"chat"}
        assert tracing.get_langfuse("search") is None

    def test_init_is_idempotent(self, monkeypatch):
        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
        _set_keys(monkeypatch, "chat")

        first = tracing.init_tracing()["chat"]
        second = tracing.init_tracing()["chat"]

        assert first is second
        langfuse_cls.assert_called_once()

    def test_each_client_gets_its_own_dedicated_tracer_provider(self, monkeypatch):
        """Neither shares the global provider that App Insights exports from, nor each other's."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
        _set_keys(monkeypatch, "search")
        _set_keys(monkeypatch, "chat")

        tracing.init_tracing()

        providers = [c.kwargs["tracer_provider"] for c in langfuse_cls.call_args_list]
        assert all(isinstance(p, TracerProvider) for p in providers)
        assert providers[0] is not providers[1]
        assert all(p is not trace.get_tracer_provider() for p in providers)

    def test_defaults_to_prod_environment_and_cloud_url(self, monkeypatch):
        langfuse_cls = MagicMock()
        monkeypatch.setattr(tracing, "Langfuse", langfuse_cls)
        _set_keys(monkeypatch, "search")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "")

        tracing.init_tracing()

        kwargs = langfuse_cls.call_args.kwargs
        assert kwargs["base_url"] is None and kwargs["environment"] == "prod"

    def test_shutdown_flushes_and_resets(self, monkeypatch):
        search, chat = MagicMock(), MagicMock()
        monkeypatch.setattr(tracing, "_clients", {"search": search, "chat": chat})
        monkeypatch.setattr(tracing, "_public_keys", {"search": "a", "chat": "b"})

        tracing.shutdown_tracing()

        search.shutdown.assert_called_once()
        chat.shutdown.assert_called_once()
        assert tracing.get_langfuse("search") is None and tracing.get_langfuse("chat") is None

    def test_shutdown_without_clients_is_noop(self):
        tracing.shutdown_tracing()


class TestLangchainCallbacks:
    def test_empty_when_project_untraced(self):
        assert tracing.langchain_callbacks("chat") == []

    def test_handler_bound_to_project_public_key(self, monkeypatch):
        monkeypatch.setattr(tracing, "_clients", {"chat": MagicMock()})
        monkeypatch.setattr(tracing, "_public_keys", {"chat": "pk-chat"})
        with patch("langfuse.langchain.CallbackHandler") as handler_cls:
            callbacks = tracing.langchain_callbacks("chat")
        handler_cls.assert_called_once_with(public_key="pk-chat")
        assert callbacks == [handler_cls.return_value]


class TestObserve:
    def test_yields_none_when_disabled(self):
        with tracing.observe("search", project="search", input="q", tags=["x"]) as span:
            assert span is None

    def test_yields_none_for_untraced_project_even_if_other_is_traced(self, monkeypatch):
        monkeypatch.setattr(tracing, "_clients", {"search": MagicMock()})
        with tracing.observe("chat-turn", project="chat", input="q", tags=[]) as span:
            assert span is None

    def test_opens_root_span_on_the_projects_client(self, monkeypatch):
        search_client, chat_client = MagicMock(), MagicMock()
        span = MagicMock(name="span")
        chat_client.start_as_current_observation.return_value.__enter__.return_value = span
        monkeypatch.setattr(tracing, "_clients", {"search": search_client, "chat": chat_client})
        propagate = MagicMock()
        monkeypatch.setattr(tracing, "propagate_attributes", propagate)

        with tracing.observe(
            "chat-turn", project="chat", input="q", tags=["sheet:s"], session_id="t1", user_id="u1"
        ) as got:
            assert got is span

        propagate.assert_called_once_with(session_id="t1", user_id="u1", tags=["sheet:s"])
        chat_client.start_as_current_observation.assert_called_once_with(
            as_type="span", name="chat-turn", input="q"
        )
        search_client.start_as_current_observation.assert_not_called()


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
