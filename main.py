from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from agents.rag_agent import close_rag_agent, init_rag_agent
from routes import chat, data, search, subscriptions
from subscriptions.store import SubscriptionStore, open_subscription_store
from utils.errors import register_exception_handlers
from utils.middleware import RequestIdMiddleware
from utils.telemetry import configure_telemetry, instrument_app
from utils.tracing import init_tracing, shutdown_tracing

load_dotenv()

# load environment variables
if "PORT" not in os.environ:
    port = 8000
else:
    port = os.environ["PORT"]

description = """
Search and chat with [HIA](https://github.com/rodekruis/helpful-information).

Built with love by [NLRC 510](https://www.510.global/). See
[the project on GitHub](https://github.com/rodekruis/hia-search) or [contact us](mailto:support@510.global).
"""

tags_metadata = [
    {
        "name": "data",
        "description": "Create, update or delete _vector stores_, i.e. databases for embeddings.",
    },
    {
        "name": "search",
        "description": "Search HIA.",
    },
    {
        "name": "chat",
        "description": "Chat with HIA.",
    },
    {
        "name": "models",
        "description": "Get information on the AI models used.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once per worker process: the first user must not pay for pool setup,
    # and a bad DB config must fail the boot instead of the first chat turn.
    # Order matters: App Insights claims the global TracerProvider first, Langfuse
    # then attaches to its own; the DB pool opens after psycopg is instrumented.
    configure_telemetry()
    init_tracing()
    try:
        with app.state.subscription_store_factory() as store:
            app.state.subscription_store = store
            try:
                init_rag_agent()
                try:
                    yield
                finally:
                    close_rag_agent()
            finally:
                del app.state.subscription_store
    finally:
        shutdown_tracing()


router = APIRouter()


@router.get("/", include_in_schema=False)
async def docs_redirect():
    """Redirect base URL to docs."""
    return RedirectResponse(url="/docs")


@router.get("/health", include_in_schema=False)
async def health():
    """Liveness probe for container orchestration."""
    return {"status": "ok"}


@router.get("/get-models", tags=["models"])
async def get_models():
    """Get information on the AI models used."""
    return JSONResponse(
        status_code=200,
        content={
            "provider": "OpenAI",
            "model embeddings": os.environ["MODEL_EMBEDDINGS"],
            "model chat": os.environ["MODEL_CHAT"],
        },
    )


def create_app(
    *,
    subscription_store_factory: Callable[
        [], AbstractContextManager[SubscriptionStore]
    ] = open_subscription_store,
) -> FastAPI:
    app = FastAPI(
        title="hia-search",
        description=description,
        version="0.0.2",
        # Preserve the existing project's license and endpoint contracts.
        license_info={
            "name": "AGPL-3.0 license",
            "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
        },
        openapi_tags=[
            *tags_metadata,
            {"name": "subscriptions", "description": "Persist sheet subscriptions."},
        ],
        lifespan=lifespan,
    )
    app.state.subscription_store_factory = subscription_store_factory
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["POST"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)
    instrument_app(app)
    for api_router in (
        router,
        search.router,
        data.router,
        chat.router,
        subscriptions.router,
    ):
        app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
