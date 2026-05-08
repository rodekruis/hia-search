"""Tests for the /search endpoint."""

from __future__ import annotations

from unittest.mock import patch, MagicMock, PropertyMock
import pytest


def _make_doc(metadata: dict, page_content: str = ""):
    doc = MagicMock()
    doc.metadata = metadata
    doc.page_content = page_content
    return doc


def _make_vector_store(docs_and_scores, all_docs_metadata):
    """Build a mock vector store."""
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = docs_and_scores
    vs.get_documents.return_value = [
        {"metadata": __import__("json").dumps(m)} for m in all_docs_metadata
    ]
    return vs


class TestSearch:
    """Tests for the POST /search endpoint."""

    @patch("routes.search.get_vector_store")
    def test_basic_search(self, mock_get_vs, client):
        meta = {
            "categoryID": 1,
            "subcategoryID": 1,
            "slug": "visa",
            "question": "How to apply?",
            "answer": "Apply online.",
            "google_index": "QnAs2",
            "parent": None,
        }
        doc = _make_doc(meta)
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(doc, 0.92)],
            all_docs_metadata=[meta],
        )

        resp = client.post(
            "/search",
            json={
                "query": "visa application",
                "googleSheetId": "sheet123",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert len(body["results"]) == 1
        assert body["results"][0]["question"] == "How to apply?"

    @patch("routes.search.get_vector_store")
    def test_invalid_source(self, mock_get_vs, client):
        resp = client.post(
            "/search",
            json={
                "query": "test",
                "googleSheetId": "sheet123",
                "source": "InvalidSheet",
            },
        )

        assert resp.status_code == 400

    @patch("routes.search.translate")
    @patch("routes.search.get_vector_store")
    def test_search_with_translation(self, mock_get_vs, mock_translate, client):
        meta = {
            "categoryID": 1,
            "subcategoryID": 1,
            "slug": "",
            "question": "What is this?",
            "answer": "An answer.",
            "google_index": "QnAs3",
            "parent": None,
        }
        doc = _make_doc(meta)
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(doc, 0.85)],
            all_docs_metadata=[meta],
        )
        mock_translate.side_effect = (
            lambda from_lang, to_lang, text: f"[{to_lang}]{text}"
        )

        resp = client.post(
            "/search",
            json={
                "query": "Qu'est-ce que c'est?",
                "googleSheetId": "sheet123",
                "lang": "fr",
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        # question & answer should be translated to fr
        assert body["results"][0]["question"].startswith("[fr]")
        assert body["results"][0]["answer"].startswith("[fr]")

    def test_search_missing_required_field(self, client):
        """Omitting googleSheetId should return 422."""
        resp = client.post("/search", json={"query": "hello"})
        assert resp.status_code == 422

    @patch("routes.search.get_vector_store")
    def test_search_deduplicates_by_google_index(self, mock_get_vs, client):
        meta = {
            "categoryID": 1,
            "subcategoryID": 1,
            "slug": "",
            "question": "Q",
            "answer": "A",
            "google_index": "QnAs5",
            "parent": None,
        }
        doc1 = _make_doc(meta)
        doc2 = _make_doc(meta)  # same google_index → duplicate
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(doc1, 0.9), (doc2, 0.88)],
            all_docs_metadata=[meta],
        )

        resp = client.post(
            "/search",
            json={
                "query": "test",
                "googleSheetId": "sheet123",
            },
        )

        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1
