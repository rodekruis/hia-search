"""Tests for utils/vector_store.py with Azure/OpenAI clients mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

import utils.vector_store as vs_module
from utils.vector_store import VectorStore, create_vector_store_index, get_vector_store


@pytest.fixture()
def azure(monkeypatch):
    """Patch every external client used by VectorStore; yields the mocks."""
    mocks = {}
    for name in ("AzureOpenAIEmbeddings", "SearchClient", "AzureSearch", "SearchIndexClient"):
        mock = MagicMock(name=name)
        monkeypatch.setattr(vs_module, name, mock)
        mocks[name] = mock
    mocks["AzureOpenAIEmbeddings"].return_value.embed_query.return_value = [0.0] * 3
    yield mocks


def _store(**overrides) -> VectorStore:
    params = dict(
        store_path="https://search.example",
        store_service="azuresearch",
        store_password="pw",
        embedding_source="OpenAI",
        embedding_model="emb-model",
        store_id="idx",
    )
    params.update(overrides)
    return VectorStore(**params)


def _chunk(google_index: str, nth: int, text: str = "t") -> Document:
    return Document(page_content=text, metadata={"google_index": google_index, "nth_chunk": nth})


class TestConstruction:
    def test_wires_azure_clients(self, azure):
        store = _store()
        azure["AzureOpenAIEmbeddings"].assert_called_once()
        assert azure["AzureOpenAIEmbeddings"].call_args.kwargs["deployment"] == "emb-model"
        azure["SearchClient"].assert_called_once()
        assert azure["SearchClient"].call_args.kwargs["index_name"] == "idx"
        assert azure["AzureSearch"].call_args.kwargs["index_name"] == "idx"
        assert store.embedder is azure["AzureOpenAIEmbeddings"].return_value

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


class TestAddDocuments:
    def test_empty_input_adds_nothing(self, azure):
        store = _store()
        assert not store.add_documents([])
        store.langchain_client.add_texts.assert_not_called()

    def test_adds_texts_with_ids_and_embedding_model_metadata(self, azure):
        store = _store()
        store.client.get_document_count.return_value = 0
        chunks = [_chunk("QnAs1", 0, "a"), _chunk("QnAs1", 1, "b"), _chunk("Offers3", 0, "c")]

        n = store.add_documents(chunks)

        assert n == 3
        kwargs = store.langchain_client.add_texts.call_args.kwargs
        assert kwargs["texts"] == ["a", "b", "c"]
        assert kwargs["ids"] == ["QnAs1_0", "QnAs1_1", "Offers3_0"]
        assert all(m["embedding_model"] == "emb-model" for m in kwargs["metadatas"])
        # caller's metadata is left untouched
        assert "embedding_model" not in chunks[0].metadata
        azure["SearchIndexClient"].return_value.delete_index.assert_not_called()

    def test_replaces_existing_index_when_not_empty(self, azure):
        store = _store()
        store.client.get_document_count.return_value = 42

        store.add_documents([_chunk("QnAs1", 0)])

        index_client = azure["SearchIndexClient"].return_value
        index_client.delete_index.assert_called_once_with("idx")
        created = index_client.create_index.call_args.args[0]
        assert created.name == "idx"
        assert {f.name for f in created.fields} == {"id", "content", "content_vector", "metadata"}
        vector_field = next(f for f in created.fields if f.name == "content_vector")
        assert vector_field.vector_search_dimensions == 3


class TestReads:
    def test_count_and_get_documents_delegate_to_search_client(self, azure):
        store = _store()
        store.client.get_document_count.return_value = 5
        store.client.search.return_value = iter([{"id": "1"}, {"id": "2"}])

        assert store.count_documents() == 5
        assert store.get_documents() == [{"id": "1"}, {"id": "2"}]
        store.client.search.assert_called_once_with(search_text="*")

    def test_similarity_search_delegates_to_langchain_client(self, azure):
        store = _store()
        store.similarity_search("q", k=3)
        store.langchain_client.similarity_search.assert_called_once_with(query="q", k=3)
        store.similarity_search_with_score("q", k=2)
        store.langchain_client.similarity_search_with_score.assert_called_once_with(query="q", k=2)


class TestFactories:
    @patch("utils.vector_store.DocumentLoader")
    def test_create_index_with_no_documents_raises_400(self, mock_loader, azure):
        mock_loader.return_value.load.return_value = []
        with pytest.raises(HTTPException) as exc:
            create_vector_store_index("googlesheet", "Sheet_1", {})
        assert exc.value.status_code == 400

    @patch("utils.vector_store.DocumentChunker")
    @patch("utils.vector_store.DocumentLoader")
    def test_create_index_loads_chunks_and_adds(self, mock_loader, mock_chunker, azure):
        docs = [Document(page_content="x", metadata={"google_index": "QnAs1"})]
        mock_loader.return_value.load.return_value = docs
        mock_chunker.return_value.split_documents.return_value = [_chunk("QnAs1", 0, "x")]
        azure["SearchClient"].return_value.get_document_count.return_value = 0

        store = create_vector_store_index("json", "Sheet_1", {"values": []})

        mock_loader.assert_called_once_with(
            document_type="json", document_id="Sheet_1", document_data={"values": []}
        )
        mock_chunker.return_value.split_documents.assert_called_once_with(documents=docs)
        assert store.store_id == "sheet1"
        store.langchain_client.add_texts.assert_called_once()

    @patch("utils.vector_store.create_vector_store_index")
    def test_get_vector_store_returns_existing(self, mock_create, azure):
        azure["SearchClient"].return_value.get_document_count.return_value = 10
        store = get_vector_store("ABC", check_if_exists=True)
        assert store.store_id == "abc"
        mock_create.assert_not_called()

    @patch("utils.vector_store.create_vector_store_index")
    def test_get_vector_store_creates_when_missing(self, mock_create, azure):
        azure["SearchClient"].return_value.get_document_count.return_value = 0
        store = get_vector_store("ABC", check_if_exists=True)
        mock_create.assert_called_once_with(
            document_type="googlesheet", document_id="ABC", document_data={}
        )
        assert store is mock_create.return_value

    @patch("utils.vector_store.create_vector_store_index")
    def test_get_vector_store_skips_check_by_default(self, mock_create, azure):
        get_vector_store("ABC")
        azure["SearchClient"].return_value.get_document_count.assert_not_called()
        mock_create.assert_not_called()
