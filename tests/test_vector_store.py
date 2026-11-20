"""Tests for utils/vector_store.py with Azure/OpenAI clients mocked."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents.indexes.models import SearchFieldDataType
from fastapi import HTTPException

import utils.vector_store as vs_module
from utils.vector_store import IndexNotReady, VectorStore, get_vector_store

_SEARCH_RESULT = {
    "id": "doc-1",
    "content": "question and answer",
    "metadata": '{"google_index": "QnAs1"}',
    "@search.score": 0.9,
}


@pytest.fixture(autouse=True)
def _build_state(monkeypatch):
    """Fresh per-process build state for every test."""
    monkeypatch.setattr(vs_module, "_builds_running", set())
    monkeypatch.setattr(vs_module, "_build_failures", {})
    monkeypatch.setattr(vs_module, "_ready_indexes", set())


def _wait_for_builds() -> None:
    deadline = time.monotonic() + 5
    while vs_module._builds_running:
        assert time.monotonic() < deadline, "background build did not finish"
        time.sleep(0.01)


@pytest.fixture()
def azure(monkeypatch):
    """Patch every external client used by VectorStore; yields the mocks."""
    mocks = {}
    for name in ("AzureOpenAIEmbeddings", "SearchClient", "SearchIndexClient"):
        mock = MagicMock(name=name)
        monkeypatch.setattr(vs_module, name, mock)
        mocks[name] = mock
    mocks["AzureOpenAIEmbeddings"].return_value.embed_query.return_value = [0.5] * 3
    client = mocks["SearchClient"].return_value
    client.search.return_value = iter([dict(_SEARCH_RESULT)])
    client.merge_documents.return_value = [
        SimpleNamespace(key=vs_module.INDEX_METADATA_ID, succeeded=True, status_code=200)
    ]
    yield mocks


def _store(**overrides) -> VectorStore:
    params = dict(
        store_path="https://search.example",
        store_service="azuresearch",
        store_password="pw",
        embedding_source="OpenAI",
        embedding_model="emb-model",
        store_id="idx",
        source_sheet_id="ABC",
    )
    params.update(overrides)
    return VectorStore(**params)


class TestConstruction:
    def test_wires_azure_clients_without_network_calls(self, azure):
        store = _store()
        assert azure["AzureOpenAIEmbeddings"].call_args.kwargs["deployment"] == "emb-model"
        assert azure["SearchClient"].call_args.kwargs["index_name"] == "idx"
        assert store.embedder is azure["AzureOpenAIEmbeddings"].return_value
        store.embedder.embed_query.assert_not_called()
        store.client.search.assert_not_called()
        azure["SearchIndexClient"].assert_not_called()

    def test_index_schema_includes_nonsearchable_metadata_fields(self):
        fields = {field.name: field for field in vs_module._azuresearch_fields(3, "profile")}
        assert {
            "source_sheet_id", "source_type", "index_owner", "index_environment",
            "created_at", "pipeline_version", "embedding_model", "sheet_hash",
            "last_refreshed_at", "last_error", "last_searched_at",
        } <= fields.keys()
        assert fields["id"].filterable
        assert not fields["source_sheet_id"].searchable
        assert fields["created_at"].type == SearchFieldDataType.DateTimeOffset
        assert fields["content_vector"].vector_search_dimensions == 3

    def test_unknown_embedding_source_raises_500(self, azure):
        with pytest.raises(HTTPException) as exc:
            _store(embedding_source="cohere")
        assert exc.value.status_code == 500

    def test_unknown_store_service_raises_500(self, azure):
        with pytest.raises(HTTPException) as exc:
            _store(store_service="pinecone")
        assert exc.value.status_code == 500

    def test_huggingface_defaults_model(self, azure, monkeypatch):
        hf = MagicMock()
        monkeypatch.setattr(vs_module, "HuggingFaceEmbeddings", hf)
        store = _store(embedding_source="huggingface", embedding_model=None)
        hf.assert_called_once_with(model_name=vs_module.DEFAULT_HUGGING_FACE_MODEL)
        assert store.embedding_model == vs_module.DEFAULT_HUGGING_FACE_MODEL


class TestReads:
    def test_count_and_get_documents_exclude_metadata(self, azure):
        store = _store()
        page = MagicMock()
        page.get_count.return_value = 5
        store.client.search.side_effect = [page, iter([{"id": "1"}, {"id": "2"}])]

        assert store.count_documents() == 5
        assert store.get_documents() == [{"id": "1"}, {"id": "2"}]
        assert store.client.search.call_args_list == [
            call(
                search_text="*", filter="id ne 'hia-index-metadata'",
                select=["id"], include_total_count=True, top=1,
            ),
            call(
                search_text="*", filter="id ne 'hia-index-metadata'",
                select=None, top=1000,
            ),
        ]

    def test_missing_document_count_is_an_error(self, azure):
        store = _store()
        store.client.search.return_value = MagicMock(**{"get_count.return_value": None})
        with pytest.raises(RuntimeError, match="No document count returned"):
            store.count_documents()


class TestSearch:
    def test_hybrid_query_excludes_metadata_and_maps_results(self, azure):
        store = _store()

        results = store.similarity_search_with_score("query", k=2)

        kwargs = store.client.search.call_args.kwargs
        assert kwargs["search_text"] == "query"
        assert kwargs["filter"] == "id ne 'hia-index-metadata'"
        assert kwargs["top"] == 2
        assert "content_vector" not in kwargs["select"]
        (vector_query,) = kwargs["vector_queries"]
        assert vector_query.vector == [0.5] * 3
        assert vector_query.k_nearest_neighbors == 2
        assert vector_query.fields == "content_vector"
        [(doc, score)] = results
        assert doc.page_content == "question and answer"
        assert doc.metadata == {"id": "doc-1", "google_index": "QnAs1"}
        assert score == 0.9

    def test_every_search_records_usage(self, azure):
        store = _store()
        store.client.search.side_effect = lambda **kwargs: iter([dict(_SEARCH_RESULT)])

        store.similarity_search_with_score("query", 2)
        store.similarity_search_with_score("query", 2)

        assert store.client.merge_documents.call_count == 2
        usage_document = store.client.merge_documents.call_args.args[0][0]
        assert set(usage_document) == {"id", "last_searched_at"}
        assert usage_document["id"] == vs_module.INDEX_METADATA_ID

    def test_failed_usage_write_does_not_fail_search(self, azure, caplog):
        store = _store()
        store.client.merge_documents.side_effect = HttpResponseError("unavailable")

        assert len(store.similarity_search_with_score("query", 2)) == 1
        assert "Could not record index usage" in caplog.text

    def test_failed_search_does_not_record_usage(self, azure):
        store = _store()
        store.client.search.side_effect = HttpResponseError()
        with pytest.raises(HttpResponseError):
            store.similarity_search_with_score("query", 2)
        store.client.merge_documents.assert_not_called()

    @patch("utils.vector_store.ensure_index_fresh")
    def test_missing_index_is_built_once_in_background_then_searched(self, refresh, azure):
        store = _store()
        store.client.get_document.side_effect = ResourceNotFoundError()
        release = threading.Event()
        refresh.side_effect = lambda *args, **kwargs: release.wait(5)

        for _ in range(2):
            with pytest.raises(IndexNotReady):
                store.similarity_search_with_score("query", 2)
        release.set()
        _wait_for_builds()

        assert len(store.similarity_search_with_score("query", 2)) == 1
        refresh.assert_called_once_with("idx", source_sheet_id="ABC")
        # no embedding or search is wasted while the index is not ready
        store.embedder.embed_query.assert_called_once_with("query")
        store.client.search.assert_called_once()

    @patch("utils.vector_store.ensure_index_fresh")
    def test_half_built_index_is_completed_in_background(self, refresh, azure):
        store = _store()
        store.client.get_document.return_value = {"source_type": "googlesheet"}

        with pytest.raises(IndexNotReady):
            store.similarity_search_with_score("query", 2)
        _wait_for_builds()

        refresh.assert_called_once_with("idx", source_sheet_id="ABC")
        store.client.search.assert_not_called()

    @patch("utils.vector_store.ensure_index_fresh")
    def test_complete_index_is_checked_once_per_process(self, refresh, azure):
        store = _store()
        store.client.get_document.return_value = {"sheet_hash": "h"}
        store.client.search.side_effect = lambda **kwargs: iter([dict(_SEARCH_RESULT)])

        store.similarity_search_with_score("query", 2)
        store.similarity_search_with_score("query", 2)

        store.client.get_document.assert_called_once()
        refresh.assert_not_called()

    @patch("utils.vector_store.ensure_index_fresh")
    def test_failed_build_is_raised_once_then_retried(self, refresh, azure):
        store = _store()
        store.client.get_document.side_effect = ResourceNotFoundError()
        refresh.side_effect = [HTTPException(status_code=502, detail="Could not read the Google Sheet."), None]

        with pytest.raises(IndexNotReady):
            store.similarity_search_with_score("query", 2)
        _wait_for_builds()
        with pytest.raises(HTTPException) as exc:
            store.similarity_search_with_score("query", 2)
        assert exc.value.status_code == 502
        with pytest.raises(IndexNotReady):
            store.similarity_search_with_score("query", 2)
        _wait_for_builds()

        assert refresh.call_count == 2
        store.client.merge_documents.assert_not_called()

    @patch("utils.vector_store.ensure_index_fresh")
    def test_unfinished_json_index_is_not_built_from_a_sheet(self, refresh, azure):
        store = _store()
        store.client.get_document.return_value = {"source_type": "json"}
        with pytest.raises(IndexNotReady):
            store.similarity_search_with_score("query", 2)
        refresh.assert_not_called()

    @patch("utils.vector_store.ensure_index_fresh")
    def test_index_deleted_after_check_is_rebuilt(self, refresh, azure):
        store = _store()
        store.client.get_document.return_value = {"sheet_hash": "h"}
        store.client.search.side_effect = ResourceNotFoundError()

        with pytest.raises(IndexNotReady):
            store.similarity_search_with_score("query", 2)
        _wait_for_builds()

        refresh.assert_called_once_with("idx", source_sheet_id="ABC")

    @patch("utils.vector_store.ensure_index_fresh")
    def test_missing_index_without_source_is_not_recreated(self, refresh, azure):
        store = _store(source_sheet_id=None)
        store.client.search.side_effect = ResourceNotFoundError()
        with pytest.raises(ResourceNotFoundError):
            store.similarity_search_with_score("query", 2)
        refresh.assert_not_called()


class TestFactory:
    def test_builds_a_fresh_store_per_call(self, azure):
        first = get_vector_store("ABC")
        second = get_vector_store("ABC")

        assert first is not second
        assert first.store_id == vs_module.googleid_to_vectorstoreid("ABC")
        assert first.source_sheet_id == "ABC"
        first.client.search.assert_not_called()
        azure["SearchIndexClient"].assert_not_called()
