"""Tests for utils/telemetry.py (Azure Monitor distro wiring, environment stamping)."""

from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.instrumentation.logging.handler import LoggingHandler

import utils.telemetry as telemetry


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(telemetry, "_configured", False)
    monkeypatch.setattr(telemetry, "ENVIRONMENT", "staging")
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in root.handlers:
        if handler not in before:
            root.removeHandler(handler)


class TestEnvironmentStamping:
    def test_filter_sets_environment_on_record(self):
        record = logging.LogRecord("some.lib", logging.INFO, __file__, 1, "hello", None, None)
        assert telemetry.EnvironmentFilter().filter(record)
        assert record.environment == "staging"

    def test_span_processor_sets_environment_attribute(self):
        span = MagicMock()
        telemetry.EnvironmentSpanProcessor().on_start(span)
        span.set_attribute.assert_called_once_with("environment", "staging")


class TestConfigureTelemetry:
    def test_without_connection_string_only_console_logging(self, monkeypatch):
        monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
        with patch("azure.monitor.opentelemetry.configure_azure_monitor") as configure:
            telemetry.configure_telemetry()
        configure.assert_not_called()

        console = next(
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, LoggingHandler)
            and any(isinstance(f, telemetry.EnvironmentFilter) for f in h.filters)
        )
        buffer = io.StringIO()
        console.setStream(buffer)
        logging.getLogger("child.logger").info("propagated")
        assert " : INFO : staging : propagated" in buffer.getvalue()

    def test_with_connection_string_configures_distro(self, monkeypatch):
        monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=abc")

        def fake_configure(**kwargs):
            # the real distro attaches an OTel LoggingHandler to the root logger
            logging.getLogger().addHandler(LoggingHandler(logger_provider=MagicMock()))

        with patch(
            "azure.monitor.opentelemetry.configure_azure_monitor", side_effect=fake_configure
        ) as configure, patch(
            "opentelemetry.instrumentation.psycopg.PsycopgInstrumentor"
        ) as psycopg_instr:
            telemetry.configure_telemetry()

        kwargs = configure.call_args.kwargs
        assert kwargs["connection_string"] == "InstrumentationKey=abc"
        assert kwargs["resource"].attributes["service.name"] == "hia-search"
        assert kwargs["resource"].attributes["deployment.environment"] == "staging"
        assert any(isinstance(p, telemetry.EnvironmentSpanProcessor) for p in kwargs["span_processors"])
        psycopg_instr.return_value.instrument.assert_called_once()

        otel_handler = next(h for h in logging.getLogger().handlers if isinstance(h, LoggingHandler))
        assert any(isinstance(f, telemetry.EnvironmentFilter) for f in otel_handler.filters)

    def test_configure_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=abc")
        with patch("azure.monitor.opentelemetry.configure_azure_monitor") as configure, patch(
            "opentelemetry.instrumentation.psycopg.PsycopgInstrumentor"
        ):
            telemetry.configure_telemetry()
            telemetry.configure_telemetry()
        configure.assert_called_once()

    def test_noisy_loggers_are_quietened(self, monkeypatch):
        monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
        telemetry.configure_telemetry()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("azure").level == logging.WARNING


class TestInstrumentApp:
    def test_adds_otel_middleware_hook_and_excludes_health(self):
        with patch("utils.telemetry.FastAPIInstrumentor") as instrumentor:
            app = MagicMock()
            telemetry.instrument_app(app)
        instrumentor.instrument_app.assert_called_once_with(app, excluded_urls="health")
