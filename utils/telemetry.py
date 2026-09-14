"""Application Insights via the Azure Monitor OpenTelemetry distro.

Two steps, because Starlette builds its middleware stack on the first ASGI call
(the lifespan event) and never rebuilds it:

- instrument_app(app): at import time, right after the app is created. Adds the
  OpenTelemetry middleware; no exporter needed yet (spans go through a proxy
  tracer until a provider is set).
- configure_telemetry(): once per worker process in the lifespan, after fork.
  Sets the *global* TracerProvider/LoggerProvider and exporters. Langfuse
  deliberately uses its own isolated provider (see utils.tracing) so prompts
  and completions never reach Application Insights and HTTP spans never reach
  Langfuse.
"""

from __future__ import annotations

import logging
import os
import sys

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor

from utils.logger import logger

ENVIRONMENT = os.environ.get("ENVIRONMENT", "prod")
SERVICE_NAME = "hia-search"

_configured = False


class EnvironmentFilter(logging.Filter):
    """Stamp every record with the deployment environment (App Insights custom dimension)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.environment = ENVIRONMENT
        return True


class EnvironmentSpanProcessor(SpanProcessor):
    """Stamp every span (requests, dependencies) with the deployment environment."""

    def on_start(self, span, parent_context=None) -> None:
        span.set_attribute("environment", ENVIRONMENT)


def _configure_console_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(EnvironmentFilter())
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s : %(levelname)s : %(environment)s : %(message)s")
    )
    root.addHandler(console_handler)

    for noisy in (
        "requests", "openai", "httpcore", "httpx", "urllib3", "azure",
        "requests_oauthlib", "asyncio", "opentelemetry",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def instrument_app(app) -> None:
    """Add request tracing middleware; call right after creating the FastAPI app."""
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")


def configure_telemetry() -> None:
    """Console logging always; App Insights export when the connection string is set."""
    global _configured
    if _configured:
        return
    _configured = True

    _configure_console_logging()

    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING not set; App Insights export disabled")
        return

    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry.instrumentation.logging.handler import LoggingHandler
    from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

    configure_azure_monitor(
        connection_string=connection_string,
        resource=Resource.create(
            {"service.name": SERVICE_NAME, "deployment.environment": ENVIRONMENT}
        ),
        span_processors=[EnvironmentSpanProcessor()],
    )
    # the distro attached its LoggingHandler to the root logger; stamp its records too
    for handler in logging.getLogger().handlers:
        if isinstance(handler, LoggingHandler):
            handler.addFilter(EnvironmentFilter())

    # psycopg 3 is not in the distro's bundle; must run before the checkpoint pool opens
    PsycopgInstrumentor().instrument()
