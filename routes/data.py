from __future__ import annotations
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
    create_vector_store_index,
    googleid_to_vectorstoreid,
    invalidate_vector_store,
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
        {},
        description=" JSON data from Google Sheet",
    )


# Plain `def`, not `async def`: rebuilding an index loads the sheet, translates
# and embeds every entry (minutes of blocking I/O). In the threadpool that keeps
# the event loop free for chat/Twilio requests and gunicorn's worker heartbeat.
@router.post(
    "/create-vector-store", tags=["data"], dependencies=[Depends(require_write_key)]
)
def create_vector_store(payload: VectorStorePayload):
    """Create a vector store from a HIA instance. Replace all entries if it already exists.

    Protected with `API_KEY_WRITE`."""

    if payload.data:
        document_type = "json"
    else:
        document_type = "googlesheet"

    vector_store = create_vector_store_index(
        document_type=document_type,
        document_id=payload.googleSheetId,
        document_data=payload.data,
    )

    # count what was uploaded: the index's own document count lags a few
    # seconds behind indexing and reads 0 right after the index is recreated
    return JSONResponse(
        status_code=200,
        content=f"Created vector store index {googleid_to_vectorstoreid(payload.googleSheetId)} "
        f"with {vector_store.n_documents_added} documents.",
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
    invalidate_vector_store(payload.googleSheetId)

    return JSONResponse(
        status_code=200, content=f"Deleted vector store index {vector_store_id}."
    )
