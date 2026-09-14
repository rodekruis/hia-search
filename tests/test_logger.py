"""Tests for utils/logger.py.

conftest replaces ``utils.logger`` with a mock so the app never talks to Azure
Monitor; load the real file here with the exporter patched out.
"""

from __future__ import annotations

import importlib.util
import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

LOGGER_PY = Path(__file__).resolve().parent.parent / "utils" / "logger.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LOGGER_PY)
    module = importlib.util.module_from_spec(spec)
    with patch(
        "azure.monitor.opentelemetry.exporter.AzureMonitorLogExporter", MagicMock()
    ), patch("opentelemetry._logs.set_logger_provider"), patch("dotenv.load_dotenv"):
        spec.loader.exec_module(module)
    return module


@pytest.fixture()
def real_logger_module(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    root = logging.getLogger()
    before = list(root.handlers)
    module = _load("utils_logger_real")
    yield module
    # do not leave the module's handlers on the root logger for other tests
    for handler in root.handlers:
        if handler not in before:
            root.removeHandler(handler)


class TestEnvironmentInLogs:
    def test_handlers_stamp_environment_on_every_record(self, real_logger_module):
        module = real_logger_module
        record = logging.LogRecord("some.lib", logging.INFO, __file__, 1, "hello", None, None)

        for handler in (module.otel_handler, module.console_handler):
            assert handler.filter(record)
            assert record.environment == "staging"

    def test_console_format_includes_environment(self, real_logger_module):
        module = real_logger_module
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
        module.console_handler.filter(record)

        line = module.console_handler.format(record)

        assert " : INFO : staging : hello" in line

    def test_applies_to_child_loggers_via_handlers(self, real_logger_module):
        """Filters live on the handlers, so propagated records from any logger get stamped."""
        buffer = io.StringIO()
        real_logger_module.console_handler.setStream(buffer)

        logging.getLogger("child.logger").info("propagated")

        assert " : staging : propagated" in buffer.getvalue()

    def test_environment_defaults_to_prod(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        root = logging.getLogger()
        before = list(root.handlers)
        try:
            assert _load("utils_logger_real2").ENVIRONMENT == "prod"
        finally:
            for handler in root.handlers:
                if handler not in before:
                    root.removeHandler(handler)
