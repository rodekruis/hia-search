from __future__ import annotations

from azure.core.exceptions import AzureError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAIError
from psycopg import Error as PsycopgError

from utils.logger import logger

# Upstream failures we cannot recover from: surfaced as 502 without leaking details
EXTERNAL_SERVICE_ERRORS = (OpenAIError, AzureError, PsycopgError)


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    """Structured error body: {"error": {"code", "message", "request_id"}}."""
    request_id = request_id_of(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
        # 500s are sent by Starlette's outermost ServerErrorMiddleware, bypassing
        # RequestIdMiddleware's send wrapper, so set the header here as well.
        headers={"x-request-id": request_id} if request_id else None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return error_response(request, 422, "validation_error", str(exc))

    for exc_type in EXTERNAL_SERVICE_ERRORS:

        @app.exception_handler(exc_type)
        async def external_service_error_handler(request: Request, exc: Exception):
            logger.error(
                "External service error [%s] %s: %s",
                request_id_of(request),
                type(exc).__name__,
                exc,
            )
            return error_response(
                request,
                502,
                "external_service_error",
                "An upstream service failed; please retry later",
            )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error [%s]", request_id_of(request))
        return error_response(
            request, 500, "internal_error", "An unexpected error occurred"
        )
