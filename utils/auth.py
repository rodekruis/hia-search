from __future__ import annotations

import json
import logging
import os
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from twilio.request_validator import RequestValidator

logger = logging.getLogger(__name__)

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


def _twilio_token_for(google_sheet_id: str | None) -> str | None:
    """Token for this HIA instance.

    When TWILIO_AUTH_TOKENS (JSON {googleSheetId: token}) is set it is authoritative:
    a sheet without an entry is rejected. Otherwise TWILIO_AUTH_TOKEN applies to all.
    """
    tokens_json = os.environ.get("TWILIO_AUTH_TOKENS")
    if not tokens_json:
        return os.environ.get("TWILIO_AUTH_TOKEN") or None
    try:
        tokens = json.loads(tokens_json)
    except json.JSONDecodeError:
        logger.error("TWILIO_AUTH_TOKENS is not valid JSON")
        return None
    if not isinstance(tokens, dict):
        logger.error("TWILIO_AUTH_TOKENS must be a JSON object {googleSheetId: token}")
        return None
    return (tokens.get(google_sheet_id) if google_sheet_id else None) or None


async def require_twilio_signature(request: Request) -> None:
    """Verify the request was signed by the Twilio account bound to `googleSheetId`."""
    google_sheet_id = request.query_params.get("googleSheetId")
    token = _twilio_token_for(google_sheet_id)
    if not token:
        logger.error(
            "No Twilio auth token configured for googleSheetId=%s; rejecting webhook",
            google_sheet_id,
        )
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
