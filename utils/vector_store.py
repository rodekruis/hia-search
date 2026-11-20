from __future__ import annotations
import hashlib
import json
import math
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import List, Literal, TypedDict
from fastapi import HTTPException
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError, HttpResponseError, ResourceNotFoundError
from openai import OpenAIError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswParameters,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import IndexingResult, VectorizedQuery
from utils.constants import DocumentMetadata
from utils.document_loader import DocumentLoader
from utils.translator import TranslationError
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_HUGGING_FACE_MODEL = "sentence-transformers/all-mpnet-base-v2"
INDEX_METADATA_ID = "hia-index-metadata"
INDEX_PIPELINE_VERSION = "2"
_DOCUMENT_FILTER = f"id ne '{INDEX_METADATA_ID}'"
_METADATA_STRINGS = (
    "sheet_hash", "last_error", "source_sheet_id", "source_type",
    "index_owner", "index_environment", "pipeline_version", "embedding_model",
)
_METADATA_DATES = ("created_at", "last_refreshed_at", "last_searched_at")
_ERROR_SEPARATOR = " | "


dm = DocumentMetadata()


class IndexNotReady(Exception):
    """The index is missing or still being built; a background build is running."""


# Per-process state of first builds. Indexes become complete once (sheet_hash written)
# and stay so, so a completed check is cached; a deleted index is detected on search.
_builds_lock = threading.Lock()
_builds_running: set[str] = set()
_build_failures: dict[str, Exception] = {}
_ready_indexes: set[str] = set()


class RefreshResult(TypedDict):
    status: Literal["unchanged", "updated"]
    index_name: str
    document_count: int
    added: int
    removed: int


def _openai_embedder(model: str) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["OPENAI_ENDPOINT"], deployment=model,
        chunk_size=1, request_timeout=30,
    )


def _azuresearch_fields(
    vector_dimensions: int, vector_profile: str
) -> list[SearchField]:
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=vector_profile,
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
    ]
    fields.extend(
        SimpleField(name=name, type=SearchFieldDataType.String)
        for name in _METADATA_STRINGS
    )
    fields.extend(
        SimpleField(name=name, type=SearchFieldDataType.DateTimeOffset)
        for name in _METADATA_DATES
    )
    return fields


def _index_prefix() -> str:
    return f"hia-{os.environ.get('ENVIRONMENT', 'prod')}-"


def googleid_to_vectorstoreid(googleid: str) -> str:
    """Derive a stable Azure index name from the exact environment and sheet ID."""
    return _index_prefix() + hashlib.sha256(googleid.encode()).hexdigest()


def failing_since(last_error: str | None) -> datetime | None:
    """Start of the current run of failures, stored as the prefix of `last_error`."""
    stamp, separator, _ = (last_error or "").partition(_ERROR_SEPARATOR)
    if not separator:
        return None
    try:
        since = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return since if since.tzinfo else None


class VectorStore:
    """
    Search client for indexed Q&As and offers.
    """

    output: List[Document]

    def __init__(
        self,
        store_path: str,
        store_service: str,
        store_password: str = None,
        embedding_source: str = None,
        embedding_model: str = None,
        store_id: str = "document_embeddings",
        source_sheet_id: str | None = None,
    ):
        self.store_id = store_id
        self.embedding_source = embedding_source
        self.embedding_model = embedding_model
        self.store_service = store_service
        self.store_password = store_password
        self.store_path = store_path
        self.source_sheet_id = source_sheet_id
        self.embedder = self._set_embedder()
        self.client = self._set_client()

    def _set_embedder(self):
        """Sets the document embedder based on the input embedding model or embedding source.
        If no embedding model is given, a default model is used.
        """
        if self.embedding_source.lower() == "openai":
            return _openai_embedder(self.embedding_model)

        elif self.embedding_source.lower() == "huggingface":
            if self.embedding_model is None:
                self.embedding_model = DEFAULT_HUGGING_FACE_MODEL
            return HuggingFaceEmbeddings(model_name=self.embedding_model)

        else:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding source {self.embedding_source} not available. Only embedding models from 'HuggingFace' or 'OpenAI' are currently available.",
            )

    def _set_client(self):
        """Sets the vector store client"""
        if self.store_service.lower() == "azuresearch":
            return SearchClient(
                self.store_path,
                index_name=self.store_id,
                credential=AzureKeyCredential(self.store_password),
                connection_timeout=5,
                read_timeout=30,
                retry_total=2,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Vector store {self.store_service} not available. Only 'azuresearch' are currently available.",
            )

    def count_documents(self) -> int:
        """Count content documents, excluding the internal metadata record."""
        results = self.client.search(
            search_text="*",
            filter=_DOCUMENT_FILTER,
            select=["id"],
            include_total_count=True,
            top=1,
        )
        count = results.get_count()
        if count is None:
            raise RuntimeError(f"No document count returned for index {self.store_id}.")
        return count

    def get_documents(self) -> list[dict]:
        """Get all documents from the vector store"""
        return _read_documents(self.client)

    def _hybrid_search(
        self, query: str, embedding: list[float], k: int
    ) -> list[tuple[Document, float]]:
        results = self.client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(
                vector=embedding, k_nearest_neighbors=k, fields="content_vector",
            )],
            filter=_DOCUMENT_FILTER,
            select=["id", "content", "metadata"],
            top=k,
        )
        return [
            (
                Document(
                    page_content=result["content"],
                    metadata={"id": result["id"], **json.loads(result["metadata"])},
                ),
                float(result["@search.score"]),
            )
            for result in results
        ]

    def _record_search(self) -> None:
        """Best effort: a failed usage write must not fail the search."""
        try:
            _write_documents(self.client.merge_documents, [{
                "id": INDEX_METADATA_ID,
                "last_searched_at": datetime.now(UTC).isoformat(),
            }])
        except AzureError as error:
            logger.warning(
                "Could not record index usage",
                extra={"index_name": self.store_id, "error_type": type(error).__name__},
            )

    def _ensure_ready(self) -> None:
        """Raise IndexNotReady (starting a build) unless the index was fully built.

        Only `ensure_index_fresh` writes `sheet_hash`, as its last step, so an index
        without it is missing, half-built by a crashed build, or still being built.
        """
        if not self.source_sheet_id or self.store_id in _ready_indexes:
            return
        try:
            metadata = self.client.get_document(
                INDEX_METADATA_ID, selected_fields=["sheet_hash", "source_type"],
            )
        except ResourceNotFoundError:
            metadata = None
        if metadata and metadata.get("sheet_hash"):
            with _builds_lock:
                _ready_indexes.add(self.store_id)
            return
        if metadata and metadata.get("source_type") == "json":
            # JSON indexes are built by their upload request, never from a sheet.
            raise IndexNotReady(self.store_id)
        _start_build(self.store_id, self.source_sheet_id)
        raise IndexNotReady(self.store_id)

    def similarity_search_with_score(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Hybrid search.

        Raises IndexNotReady while the index is missing or incomplete; it is then
        built from its Google Sheet in the background.
        """
        self._ensure_ready()
        embedding = self.embedder.embed_query(query)
        try:
            results = self._hybrid_search(query, embedding, k)
        except ResourceNotFoundError:
            if not self.source_sheet_id:
                raise
            # deleted since it was found complete, e.g. by cleanup
            with _builds_lock:
                _ready_indexes.discard(self.store_id)
            _start_build(self.store_id, self.source_sheet_id)
            raise IndexNotReady(self.store_id) from None
        self._record_search()
        return results


def _start_build(index_name: str, source_sheet_id: str) -> None:
    """Build an index in a background thread, at most one build per index at a time.

    A failed build is raised once to the next search instead of silently starting
    another, so a broken sheet surfaces its error; the search after that retries.
    """
    with _builds_lock:
        failure = _build_failures.pop(index_name, None)
        if failure is None:
            if index_name in _builds_running:
                return
            _builds_running.add(index_name)
    if failure is not None:
        raise failure
    logger.info("Index not ready; building it in the background", extra={"index_name": index_name})
    threading.Thread(
        target=_run_build, args=(index_name, source_sheet_id),
        name=f"index-build-{index_name}", daemon=True,
    ).start()


def _run_build(index_name: str, source_sheet_id: str) -> None:
    try:
        ensure_index_fresh(index_name, source_sheet_id=source_sheet_id)
    except Exception as error:  # thread boundary: every failure must be recorded
        logger.exception("Background index build failed", extra={"index_name": index_name})
        with _builds_lock:
            _build_failures[index_name] = error
    else:
        with _builds_lock:
            _ready_indexes.add(index_name)
    finally:
        with _builds_lock:
            _builds_running.discard(index_name)


def _read_documents(client: SearchClient, select: list[str] | None = None) -> list[dict]:
    documents = []
    seen: set[str] = set()
    while True:
        # Equal-score results have no stable ordering, so avoid skip-based paging.
        query_filter = _DOCUMENT_FILTER
        if seen:
            query_filter += f" and not search.in(id, '{','.join(sorted(seen))}')"
        page = list(client.search(
            search_text="*", filter=query_filter, select=select, top=1000,
        ))
        keys = {doc["id"] for doc in page}
        if len(keys) != len(page) or keys & seen:
            raise HttpResponseError("Document scan returned duplicate keys.")
        seen.update(keys)
        documents.extend(page)
        if len(page) < 1000:
            return documents


def _write_documents(
    operation: Callable[[list[dict]], list[IndexingResult]], documents: list[dict]
) -> None:
    """Check individual results; the SDK handles HTTP retries and oversized batches."""
    for start in range(0, len(documents), 1000):
        pending = documents[start:start + 1000]
        for attempt in range(3):
            results = operation(pending)
            by_id = {document["id"]: document for document in pending}
            if len(results) != len(pending) or {r.key for r in results} != by_id.keys():
                raise HttpResponseError("Incomplete indexing response.")
            failed = [result for result in results if not result.succeeded]
            if not failed:
                break
            if attempt == 2 or any(
                result.status_code not in (409, 422, 429, 500, 502, 503, 504)
                for result in failed
            ):
                raise HttpResponseError("Indexing failed for one or more documents.")
            pending = [by_id[result.key] for result in failed]
            time.sleep(2 ** attempt)


def _source_documents(documents: list[Document]) -> dict[str, Document]:
    occurrences: Counter[str] = Counter()
    result = {}
    for document in documents:
        semantic_metadata = {
            key: value for key, value in document.metadata.items()
            if key not in (dm.GOOGLE_INDEX, dm.EMBEDDING_MODEL)
        }
        content = json.dumps(
            [INDEX_PIPELINE_VERSION, document.page_content, semantic_metadata],
            sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        )
        occurrence = occurrences[content]
        occurrences[content] += 1
        document_id = hashlib.sha256(f"{content}:{occurrence}".encode()).hexdigest()
        result[document_id] = document
    return result


def _create_index(index_client: SearchIndexClient, name: str, dimensions: int) -> None:
    index_client.create_index(SearchIndex(
        name=name,
        fields=_azuresearch_fields(dimensions, "myHnswProfile"),
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(
                name="Hnsw", kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4, ef_construction=400, ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )],
            profiles=[VectorSearchProfile(
                name="myHnswProfile", algorithm_configuration_name="Hnsw",
            )],
        ),
    ))


def _wait_for_documents(client: SearchClient, document_ids: set[str]) -> None:
    deadline = time.monotonic() + 30
    while not document_ids <= {doc["id"] for doc in _read_documents(client, ["id"])}:
        if time.monotonic() >= deadline:
            raise HttpResponseError("New documents did not become searchable within 30 seconds.")
        time.sleep(1)


def ensure_index_fresh(
    index_name: str,
    force: bool = False,
    *,
    source_sheet_id: str | None = None,
    document_data: dict | None = None,
) -> RefreshResult:
    """Reconcile a validated source snapshot without rebuilding an existing index."""
    credential = AzureKeyCredential(os.environ["VECTOR_STORE_PASSWORD"])
    endpoint = os.environ["VECTOR_STORE_ADDRESS"]
    with SearchIndexClient(
        endpoint, credential, connection_timeout=5, read_timeout=30, retry_total=2,
    ) as index_client, SearchClient(
        endpoint, index_name=index_name, credential=credential,
        connection_timeout=5, read_timeout=30, retry_total=2,
    ) as client:
        try:
            definition = index_client.get_index(index_name)
        except ResourceNotFoundError:
            definition = None
        try:
            metadata = client.get_document(INDEX_METADATA_ID) if definition else None
        except ResourceNotFoundError:
            metadata = None
        environment = os.environ.get("ENVIRONMENT", "prod")
        model = os.environ["MODEL_EMBEDDINGS"]
        fields = {field.name: field for field in definition.fields} if definition else {}
        if definition and not {
            "id", "content", "metadata", "content_vector", *_METADATA_STRINGS, *_METADATA_DATES,
        } <= fields.keys():
            raise HTTPException(status_code=409, detail="Index schema requires recreation.")
        if metadata:
            if (
                metadata.get("index_owner") != "hia-search"
                or metadata.get("index_environment") != environment
                or (source_sheet_id and metadata.get("source_sheet_id") != source_sheet_id)
            ):
                raise HTTPException(status_code=409, detail="Index source or ownership mismatch.")
            if metadata.get("embedding_model") != model:
                raise HTTPException(
                    status_code=409, detail="Embedding model changed; create a new index."
                )
            source_sheet_id = metadata["source_sheet_id"]
            if metadata.get("source_type") not in ("json", "googlesheet"):
                raise HTTPException(status_code=409, detail="Unknown index source type.")
            if metadata.get("source_type") == "json" and document_data is None:
                raise HTTPException(
                    status_code=409, detail="This index requires a new JSON payload."
                )
        if not source_sheet_id:
            raise HTTPException(status_code=404, detail="Index source is unknown.")
        source_type = "json" if document_data is not None else "googlesheet"
        try:
            loader = DocumentLoader(
                document_type=source_type, document_id=source_sheet_id,
                document_data=document_data,
            )
            snapshot = loader.load_source()
            desired = _source_documents(snapshot.documents)
            desired_metadata = {
                key: {**doc.metadata, dm.EMBEDDING_MODEL: model}
                for key, doc in desired.items()
            }
            existing = {
                doc["id"]: doc for doc in _read_documents(client, ["id", "metadata"])
            } if definition else {}
            if existing and not metadata:
                raise HTTPException(
                    status_code=409,
                    detail="Index exists but lacks metadata; delete and recreate it first.",
                )
            existing_metadata = {
                key: json.loads(doc["metadata"]) for key, doc in existing.items()
            }
            if (
                not force and metadata
                and metadata.get("sheet_hash") == snapshot.sheet_hash
                and metadata.get("pipeline_version") == INDEX_PIPELINE_VERSION
                and metadata.get("source_type") == source_type
                and existing_metadata == desired_metadata
            ):
                if metadata.get("last_error"):
                    _write_documents(client.merge_or_upload_documents, [{
                        "id": INDEX_METADATA_ID, "last_error": None,
                    }])
                return RefreshResult(
                    status="unchanged", index_name=index_name,
                    document_count=len(desired), added=0, removed=0,
                )
            added = sorted(desired.keys() - existing.keys())
            removed = sorted(existing.keys() - desired.keys())
            translated = [loader.translate_document(desired[key]) for key in added]
            vectors = (
                _openai_embedder(model).embed_documents([doc.page_content for doc in translated])
                if added else []
            )
            if len(vectors) != len(added) or any(
                not vector or len(vector) != len(vectors[0])
                or any(not math.isfinite(value) for value in vector)
                for vector in vectors
            ):
                raise HttpResponseError("Embedding API returned invalid vectors.")
            if definition:
                if vectors and fields["content_vector"].vector_search_dimensions != len(vectors[0]):
                    raise HTTPException(
                        status_code=409, detail="Vector dimensions changed; create a new index."
                    )
            else:
                dimensions = (
                    len(vectors[0]) if vectors else len(_openai_embedder(model).embed_query("Text"))
                )
                if dimensions < 1:
                    raise HttpResponseError("Embedding API returned an empty probe vector.")
                try:
                    _create_index(index_client, index_name, dimensions)
                except HttpResponseError:
                    # A concurrent first search may have just created the same index;
                    # it can take a moment before it is visible.
                    existing_definition = None
                    for attempt in range(5):
                        try:
                            existing_definition = index_client.get_index(index_name)
                            break
                        except ResourceNotFoundError:
                            time.sleep(attempt + 1)
                    vector_field = next(
                        (f for f in existing_definition.fields if f.name == "content_vector"),
                        None,
                    ) if existing_definition else None
                    if not vector_field or vector_field.vector_search_dimensions != dimensions:
                        raise
                    logger.info("Index created concurrently", extra={"index_name": index_name})
            if not metadata:
                initial_metadata = {
                    "id": INDEX_METADATA_ID, "source_sheet_id": source_sheet_id,
                    "source_type": source_type, "index_owner": "hia-search",
                    "index_environment": environment,
                    "created_at": datetime.now(UTC).isoformat(),
                    "pipeline_version": INDEX_PIPELINE_VERSION, "embedding_model": model,
                }
                _write_documents(client.merge_or_upload_documents, [initial_metadata])
                metadata = initial_metadata
            _write_documents(client.upload_documents, [
                {
                    "id": key, "content": doc.page_content, "content_vector": vector,
                    "metadata": json.dumps(
                        desired_metadata[key], ensure_ascii=False, allow_nan=False,
                    ),
                    "embedding_model": model,
                }
                for key, doc, vector in zip(added, translated, vectors, strict=True)
            ])
            _write_documents(client.merge_documents, [
                {"id": key, "metadata": json.dumps(meta, ensure_ascii=False, allow_nan=False)}
                for key, meta in desired_metadata.items()
                if key in existing_metadata and meta != existing_metadata[key]
            ])
            if added:
                _wait_for_documents(client, set(added))
            _write_documents(client.delete_documents, [{"id": key} for key in removed])
            _write_documents(client.merge_or_upload_documents, [{
                "id": INDEX_METADATA_ID, "source_type": source_type,
                "sheet_hash": snapshot.sheet_hash, "pipeline_version": INDEX_PIPELINE_VERSION,
                "embedding_model": model,
                "last_refreshed_at": datetime.now(UTC).isoformat(),
                "last_error": None,
            }])
            logger.info(
                "Index refreshed",
                extra={"index_name": index_name, "added": len(added), "removed": len(removed)},
            )
            return RefreshResult(
                status="updated", index_name=index_name, document_count=len(desired),
                added=len(added), removed=len(removed),
            )
        except (HTTPException, AzureError, OpenAIError, TranslationError, ValueError) as error:
            if metadata:
                message = (
                    f"{error.status_code}: {error.detail}"
                    if isinstance(error, HTTPException)
                    else f"{type(error).__name__}: index refresh failed; see service logs."
                )
                since = failing_since(metadata.get("last_error")) or datetime.now(UTC)
                try:
                    _write_documents(client.merge_or_upload_documents, [{
                        "id": INDEX_METADATA_ID,
                        "last_error": f"{since.isoformat()}{_ERROR_SEPARATOR}{message}",
                    }])
                except AzureError:
                    logger.error(
                        "Could not record index refresh failure", extra={"index_name": index_name},
                    )
            logger.error(
                "Index refresh failed",
                extra={"index_name": index_name, "error_type": type(error).__name__},
            )
            raise


def get_vector_store(google_sheet_id: str) -> VectorStore:
    """Build a search client for a HIA instance.

    A missing or incomplete index is built from the Google Sheet in the background
    on its first search, which raises IndexNotReady meanwhile.
    """
    return VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_service="azuresearch",
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        embedding_source="OpenAI",
        embedding_model=os.environ["MODEL_EMBEDDINGS"],
        store_id=googleid_to_vectorstoreid(google_sheet_id),
        source_sheet_id=google_sheet_id,
    )

def list_managed_indexes() -> list[dict]:
    """Metadata of this application's indexes in the current environment.

    Indexes without recognized metadata are skipped, never returned for refresh or cleanup.
    """
    environment = os.environ.get("ENVIRONMENT", "prod")
    credential = AzureKeyCredential(os.environ["VECTOR_STORE_PASSWORD"])
    endpoint = os.environ["VECTOR_STORE_ADDRESS"]
    fields = ["source_sheet_id", "source_type", *_METADATA_DATES, "last_error"]
    managed = []
    with SearchIndexClient(
        endpoint, credential, connection_timeout=5, read_timeout=30, retry_total=2,
    ) as index_client:
        names = sorted(
            name for name in index_client.list_index_names() if name.startswith(_index_prefix())
        )
    for name in names:
        with SearchClient(
            endpoint, index_name=name, credential=credential,
            connection_timeout=5, read_timeout=30, retry_total=2,
        ) as client:
            try:
                metadata = client.get_document(
                    INDEX_METADATA_ID,
                    selected_fields=[*fields, "index_owner", "index_environment"],
                )
            except ResourceNotFoundError:
                metadata = None
        if (
            not metadata
            or metadata.get("index_owner") != "hia-search"
            or metadata.get("index_environment") != environment
            or metadata.get("source_type") not in ("googlesheet", "json")
            or not metadata.get("source_sheet_id")
            or googleid_to_vectorstoreid(metadata["source_sheet_id"]) != name
        ):
            logger.warning("Skipping unrecognized index", extra={"index_name": name})
            continue
        managed.append({
            "index_name": name,
            **{field: metadata.get(field) for field in fields},
            "failing_since": failing_since(metadata.get("last_error")),
        })
    return managed
