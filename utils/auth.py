from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from twilio.request_validator import RequestValidator

from utils.logger import logger

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def _require_key(provided: str | None, env_var: str) -> None:
    expected = os.environ.get(env_var, "")
    if not provided or not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_read_key(api_key: str | None = Security(api_key_header)) -> None:
    """Require the read API key (`API_KEY`)."""
    _require_key(api_key, "API_KEY")


def require_write_key(api_key: str | None = Security(api_key_header)) -> None:
    """Require the write API key (`API_KEY_WRITE`)."""
    _require_key(api_key, "API_KEY_WRITE")


async def require_twilio_signature(request: Request) -> None:
    """Verify the request was signed by Twilio with `TWILIO_AUTH_TOKEN`."""
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not token:
        logger.error("TWILIO_AUTH_TOKEN is not set; rejecting webhook request")
        raise HTTPException(status_code=401, detail="Unauthorized")

    signature = request.headers.get("X-Twilio-Signature", "")
    form = await request.form()
    url = request.url
    # Behind a TLS-terminating proxy the app sees http; Twilio signed the public https URL.
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto and forwarded_proto != url.scheme:
        url = url.replace(scheme=forwarded_proto)

    if not RequestValidator(token).validate(str(url), dict(form), signature):
        raise HTTPException(status_code=401, detail="Invalid Twilio signature")
