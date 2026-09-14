"""Langfuse tracing: one shared client per process, optional when keys are unset.

The Langfuse SDK keeps one process-wide instance per public key; a second
construction silently returns the first and drops its arguments. Build it once
in the app lifespan (after fork, per worker) and inject it via get_langfuse().
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from langfuse import Langfuse, propagate_attributes

from utils.logger import logger

_langfuse: Langfuse | None = None


def init_tracing() -> Langfuse | None:
    """Create the shared client, or leave tracing disabled when keys are missing."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        logger.info("Langfuse keys not set; tracing disabled")
        return None
    _langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=os.environ.get("LANGFUSE_BASE_URL") or None,
        environment=os.environ.get("ENVIRONMENT", "prod"),
    )
    return _langfuse


def shutdown_tracing() -> None:
    """Flush pending events and release the client."""
    global _langfuse
    if _langfuse is not None:
        _langfuse.shutdown()
    _langfuse = None


def get_langfuse() -> Langfuse | None:
    return _langfuse


@contextmanager
def observe(
    name: str,
    *,
    input: object,
    tags: list[str],
    session_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[object | None]:
    """Root span for one request; yields None when tracing is disabled.

    Everything an evaluator needs must be written onto this span (input,
    output, metadata): observation-level evaluators do not see children.
    """
    client = _langfuse
    if client is None:
        yield None
        return
    with propagate_attributes(
        session_id=session_id, user_id=user_id, tags=tags
    ), client.start_as_current_observation(
        as_type="span", name=name, input=input
    ) as span:
        yield span
