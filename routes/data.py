from __future__ import annotations
from datetime import datetime
from typing import Literal
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from utils.auth import require_write_key
from utils.vector_store import (
    ensure_index_fresh,
    googleid_to_vectorstoreid,
    list_managed_indexes,
)
from utils.constants import DocumentMetadata
import os

dm = DocumentMetadata()

router = APIRouter()


class VectorStorePayload(BaseModel):
    googleSheetId: str = Field(
        ...,
        description="HIA Google sheet ID",
    )
    data: dict = Field(
        default_factory=dict,
        description=" JSON data from Google Sheet",
    )
    force: bool = Field(False, description="Reconcile even when the source hash is unchanged.")


class IndexStatus(BaseModel):
    index_name: str
    source_sheet_id: str
    source_type: Literal["googlesheet", "json"]
    created_at: datetime | None = None
    last_refreshed_at: datetime | None = None
    last_searched_at: datetime | None = None
    last_error: str | None = None
    failing_since: datetime | None = Field(
        None, description="Start of the current run of refresh failures, if any."
    )


@router.get(
    "/vector-stores", tags=["data"], dependencies=[Depends(require_write_key)],
    response_model=list[IndexStatus], status_code=200,
)
def list_vector_stores() -> list[dict]:
    """List this environment's managed indexes and their refresh metadata.

    Protected with `API_KEY_WRITE`; used by the scheduled refresh workflow."""
    return list_managed_indexes()


# Plain `def`, not `async def`: refreshing an index performs blocking I/O.
# In the threadpool that keeps
# the event loop free for chat/Twilio requests and gunicorn's worker heartbeat.
@router.post(
    "/create-vector-store", tags=["data"], dependencies=[Depends(require_write_key)],
    response_model=str, status_code=200,
)
def create_vector_store(payload: VectorStorePayload):
    """Create or incrementally refresh a vector store from a HIA instance.

    Protected with `API_KEY_WRITE`."""

    index_name = googleid_to_vectorstoreid(payload.googleSheetId)
    result = ensure_index_fresh(
        index_name, force=payload.force,
        source_sheet_id=payload.googleSheetId, document_data=payload.data or None,
    )

    if result["status"] == "unchanged":
        return JSONResponse(
            status_code=200,
            content=f"Vector store index {index_name} is unchanged "
            f"with {result['document_count']} documents.",
        )
    return JSONResponse(
        status_code=200,
        content=f"Created vector store index {index_name} "
        f"with {result['document_count']} documents.",
    )


@router.delete(
    "/delete-vector-store", tags=["data"], dependencies=[Depends(require_write_key)]
)
def delete_vector_store(payload: VectorStorePayload):
    """Delete a vector store. Protected with `API_KEY_WRITE`."""

    vector_store_id = googleid_to_vectorstoreid(payload.googleSheetId)
    azure_search_index_client = SearchIndexClient(
        os.environ["VECTOR_STORE_ADDRESS"],
        AzureKeyCredential(os.environ["VECTOR_STORE_PASSWORD"]),
    )

    try:
        _ = azure_search_index_client.delete_index(vector_store_id)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))

    return JSONResponse(
        status_code=200, content=f"Deleted vector store index {vector_store_id}."
    )
