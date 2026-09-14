from __future__ import annotations
import uvicorn
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
)
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routes import search, data, chat
from agents.rag_agent import init_rag_agent, close_rag_agent
from utils.errors import register_exception_handlers
from utils.middleware import RequestIdMiddleware
from utils.telemetry import configure_telemetry, instrument_app
from utils.tracing import init_tracing, shutdown_tracing
import os
from dotenv import load_dotenv

load_dotenv()

# load environment variables
if "PORT" not in os.environ.keys():
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
    init_rag_agent()
    yield
    shutdown_tracing()
    close_rag_agent()


# initialize FastAPI
app = FastAPI(
    title="hia-search",
    description=description,
    version="0.0.2",
    license_info={
        "name": "AGPL-3.0 license",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)
# must happen before the first ASGI call (the lifespan), see utils.telemetry
instrument_app(app)


@app.get("/", include_in_schema=False)
async def docs_redirect():
    """Redirect base URL to docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness probe for container orchestration."""
    return {"status": "ok"}


# Include routes
app.include_router(search.router)
app.include_router(data.router)
app.include_router(chat.router)


@app.get("/get-models", tags=["models"])
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
