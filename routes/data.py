from __future__ import annotations
from fastapi import (
    APIRouter,
    HTTPException,
    Request
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from utils.vector_store import create_vector_store_index, googleid_to_vectorstoreid
from utils.constants import DocumentMetadata
import os

dm = DocumentMetadata()

router = APIRouter()

key_query_scheme = APIKeyHeader(name="Authorization")


class VectorStorePayload(BaseModel):
    googleSheetId: str = Field(
        ...,
        description="HIA Google sheet ID",
    )
    data: dict = Field(
        {},
        description=" JSON data from Google Sheet",
    )


@router.post("/create-vector-store", tags=["data"])
async def create_vector_store(
    payload: VectorStorePayload, request: Request
):
    """Create a vector store from a HIA instance. Replace all entries if it already exists."""

    if payload.data:
        document_type = "json"
    else:
        document_type = "googlesheet"

    if document_type == "json":
        if "Authorization" not in request.headers or request.headers["Authorization"] != os.environ["API_KEY_WRITE"]:
            raise HTTPException(status_code=401, detail="Unauthorized")

    vector_store = create_vector_store_index(
        document_type=document_type,
        document_id=payload.googleSheetId,
        document_data=payload.data,
    )

    return JSONResponse(
        status_code=200,
        content=f"Created vector store index {googleid_to_vectorstoreid(payload.googleSheetId)} "
        f"with {vector_store.client.get_document_count()} documents.",
    )


@router.delete("/delete-vector-store", tags=["data"])
async def delete_vector_store(
    payload: VectorStorePayload
):
    """Delete a vector store."""

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
