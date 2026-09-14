"""Tests for utils/document_chunker.py (spaCy splitter is mocked; no model download needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from utils.document_chunker import DocumentChunker


def _bare_chunker() -> DocumentChunker:
    """Instance without running __init__ (avoids loading spaCy)."""
    return object.__new__(DocumentChunker)


class TestSetChunker:
    @patch("utils.document_chunker.SpacyTextSplitter")
    def test_sentence_splitting_passes_sizes(self, mock_splitter):
        DocumentChunker(
            chunking_strategy="SentenceSplitting",
            kwargs={"chunk_size": 100, "chunk_overlap": 10},
        )
        mock_splitter.assert_called_once_with(
            chunk_size=100, chunk_overlap=10, separator="\n\n", pipeline="en_core_web_sm"
        )

    @patch("utils.document_chunker.SpacyTextSplitter")
    def test_tokenized_splitting_uses_tiktoken_encoder(self, mock_splitter):
        DocumentChunker(
            chunking_strategy="TokenizedSentenceSplitting",
            kwargs={"chunk_size": 256, "chunk_overlap": 20, "encoding_name": "o200k_base"},
        )
        mock_splitter.from_tiktoken_encoder.assert_called_once_with(
            chunk_size=256,
            chunk_overlap=20,
            separator="\n\n",
            pipeline="en_core_web_sm",
            encoding_name="o200k_base",
        )

    def test_unknown_strategy_raises(self):
        with pytest.raises(NotImplementedError):
            DocumentChunker(chunking_strategy="magic", kwargs={})


class TestChunkMetadata:
    def test_nth_chunk_counts_within_each_document(self):
        chunks = [
            Document(page_content="a1", metadata={"google_index": "QnAs1"}),
            Document(page_content="a2", metadata={"google_index": "QnAs1"}),
            Document(page_content="a3", metadata={"google_index": "QnAs1"}),
            Document(page_content="b1", metadata={"google_index": "QnAs2"}),
            Document(page_content="c1", metadata={"google_index": "QnAs1"}),
        ]
        out = _bare_chunker()._add_chunk_metadata(chunks)
        assert [d.metadata["nth_chunk"] for d in out] == [0, 1, 2, 0, 0]
        assert [d.page_content for d in out] == ["a1", "a2", "a3", "b1", "c1"]

    def test_original_metadata_is_not_mutated(self):
        meta = {"google_index": "QnAs1"}
        _bare_chunker()._add_chunk_metadata([Document(page_content="x", metadata=meta)])
        assert "nth_chunk" not in meta

    def test_split_documents_delegates_then_annotates(self):
        chunker = _bare_chunker()
        chunker.chunker = MagicMock()
        chunker.chunker.split_documents.return_value = [
            Document(page_content="p1", metadata={"google_index": "QnAs7"}),
            Document(page_content="p2", metadata={"google_index": "QnAs7"}),
        ]
        docs = [Document(page_content="p1 p2", metadata={"google_index": "QnAs7"})]

        out = chunker.split_documents(docs)

        chunker.chunker.split_documents.assert_called_once_with(documents=docs)
        assert [d.metadata["nth_chunk"] for d in out] == [0, 1]


class TestUrlExtraction:
    def test_extracts_urls_joined_by_newline(self):
        doc = Document(
            page_content="See https://example.org/a and http://foo.bar/x?y=1 for more."
        )
        urls = _bare_chunker()._get_urls_from_page_content(doc)
        assert urls == "https://example.org/a\n http://foo.bar/x?y=1"

    def test_no_urls_returns_empty_string(self):
        assert _bare_chunker()._get_urls_from_page_content(Document(page_content="none")) == ""
