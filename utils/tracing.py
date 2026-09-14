"""Langfuse tracing: one shared client per process, optional when keys are unset.

The Langfuse SDK keeps one process-wide instance per public key; a second
construction silently returns the first and drops its arguments. Build it once
in the app lifespan (after fork, per worker) and inject it via get_langfuse().

The client runs on its own OpenTelemetry TracerProvider, isolated from the global
one that Application Insights uses (utils.telemetry): Langfuse spans carry
prompts, completions and user text and must never be exported to App Insights;
HTTP/DB spans must not be exported to Langfuse. Both providers share the OTel
context, so a Langfuse root span opened inside a FastAPI request has a foreign
parent; the SDK marks such spans as application roots, which is what
observation-level evaluators filter on ("Is Root Observation").
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from langfuse import Langfuse, propagate_attributes
from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

# Search and chat are separate Langfuse projects (distinct keys, same base URL).
# Each client gets its own isolated TracerProvider, so routing is by provider,
# not by the SDK's experimental public-key attribute matching.
PROJECTS = ("search", "chat")
_clients: dict[str, Langfuse] = {}
_public_keys: dict[str, str] = {}


def _build_client(project: str) -> tuple[Langfuse, str] | None:
    prefix = f"LANGFUSE_{project.upper()}"
    public_key = os.environ.get(f"{prefix}_PUBLIC_KEY")
    secret_key = os.environ.get(f"{prefix}_SECRET_KEY")
    if not (public_key and secret_key):
        logger.info("%s keys not set; Langfuse tracing disabled for %s", prefix, project)
        return None
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=os.environ.get("LANGFUSE_BASE_URL") or None,
        environment=os.environ.get("ENVIRONMENT", "prod"),
        # never registered as the global provider: keeps content out of App Insights
        tracer_provider=TracerProvider(),
    )
    return client, public_key


def init_tracing() -> dict[str, Langfuse]:
    """Create one client per configured project; unconfigured projects stay untraced."""
    for project in PROJECTS:
        if project not in _clients:
            built = _build_client(project)
            if built is not None:
                _clients[project], _public_keys[project] = built
    return dict(_clients)


def shutdown_tracing() -> None:
    """Flush pending events and release the clients."""
    for client in _clients.values():
        client.shutdown()
    _clients.clear()
    _public_keys.clear()


def get_langfuse(project: str) -> Langfuse | None:
    return _clients.get(project)


def langchain_callbacks(project: str) -> list:
    """LangChain callbacks that nest LLM/tool spans under the current root span."""
    if project not in _clients:
        return []
    from langfuse.langchain import CallbackHandler

    # the public key selects the project's client, and with it its isolated provider
    return [CallbackHandler(public_key=_public_keys[project])]


@contextmanager
def observe(
    name: str,
    *,
    project: str,
    input: object,
    tags: list[str],
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[object | None]:
    """Root span for one request; yields None when tracing is disabled.

    Everything an evaluator needs must be written onto this span (input,
    output, metadata): observation-level evaluators do not see children.
    """
    client = _clients.get(project)
    if client is None:
        yield None
        return
    with propagate_attributes(
        session_id=session_id, user_id=user_id, tags=tags, trace_name=name
    ), client.start_as_current_observation(
        as_type="span", name=name, input=input
    ) as span:
        yield span
