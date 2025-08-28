from __future__ import annotations
from fastapi import (
    Depends,
    APIRouter,
    HTTPException,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from utils.document_loader import DocumentLoader
from utils.document_chunker import DocumentChunker
from utils.vector_store import VectorStore, googleid_to_vectorstoreid
from utils.constants import DocumentMetadata
from routes.search import azure_search_index_client
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


@router.post("/create-vector-store")
async def create_vector_store(
    payload: VectorStorePayload, api_key: str = Depends(key_query_scheme)
):
    """Create a vector store from a HIA instance. Replace all entries if it already exists."""

    if api_key != os.environ["API_KEY_WRITE"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    vector_store_id = googleid_to_vectorstoreid(payload.googleSheetId)

    if payload.data:
        document_type = "json"
    else:
        document_type = "googlesheet"

    # load documents from Google Sheet
    doc_loader = DocumentLoader(
        document_type=document_type,
        document_id=payload.googleSheetId,
        document_data=payload.data,
    )
    docs = doc_loader.load()

    document_chunker = DocumentChunker(
        chunking_strategy="TokenizedSentenceSplitting",
        kwargs={"chunk_overlap": 20, "chunk_size": 256},
    )
    docs = document_chunker.split_documents(documents=docs)

    # add documents to vector store
    vector_store = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_service="azuresearch",
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        embedding_source="OpenAI",
        embedding_model=os.environ["MODEL_EMBEDDINGS"],
        store_id=vector_store_id,
    )
    n_docs = vector_store.add_documents(docs)

    return JSONResponse(
        status_code=200,
        content=f"Created index {vector_store_id} with {n_docs} documents.",
    )


@router.delete("/delete-vector-store")
async def delete_vector_store(
    payload: VectorStorePayload, api_key: str = Depends(key_query_scheme)
):
    """Delete a vector store."""

    if api_key != os.environ["API_KEY_WRITE"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    vector_store_id = googleid_to_vectorstoreid(payload.googleSheetId)

    try:
        _ = azure_search_index_client.delete_index(vector_store_id)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex))

    return JSONResponse(status_code=200, content=f"Deleted index {vector_store_id}.")
