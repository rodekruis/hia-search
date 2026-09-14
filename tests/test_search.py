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

    @patch("routes.search.get_vector_store")
    def test_parent_result_includes_children_with_their_scores(self, mock_get_vs, client):
        parent = {
            "categoryID": 1, "subcategoryID": 1, "slug": "housing", "parent": None,
            "question": "Housing?", "answer": "Overview.", "google_index": "QnAs1",
        }
        child_hit = {**parent, "slug": "", "parent": "housing", "question": "Rent?",
                     "answer": "Rent info.", "google_index": "QnAs2"}
        child_miss = {**child_hit, "question": "Buy?", "answer": "Buy info.", "google_index": "QnAs3"}
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(_make_doc(parent), 0.9), (_make_doc(child_hit), 0.7)],
            all_docs_metadata=[parent, child_hit, child_miss],
        )

        resp = client.post("/search", json={"query": "q", "googleSheetId": "s"})

        results = resp.json()["results"]
        # parent and its child hit collapse into one parent entry
        assert len(results) == 1
        assert results[0]["question"] == "Housing?"
        assert "google_index" not in results[0]
        children = {c["question"]: c["score"] for c in results[0]["children"]}
        assert children == {"Rent?": 0.7, "Buy?": 0.0}

    @patch("routes.search.get_vector_store")
    def test_child_hit_is_promoted_to_its_parent(self, mock_get_vs, client):
        parent = {
            "categoryID": 1, "subcategoryID": 1, "slug": "work", "parent": None,
            "question": "Work?", "answer": "Overview.", "google_index": "QnAs1",
        }
        child = {**parent, "slug": "", "parent": "work", "question": "Permit?",
                 "answer": "Permit info.", "google_index": "QnAs2"}
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(_make_doc(child), 0.8)],
            all_docs_metadata=[parent, child],
        )

        resp = client.post("/search", json={"query": "q", "googleSheetId": "s"})

        result = resp.json()["results"][0]
        assert result["question"] == "Work?"
        assert result["slug"] == "work"
        # parent carries the matched child's score
        assert result["score"] == 0.8
        assert [c["question"] for c in result["children"]] == ["Permit?"]

    @patch("routes.search.get_vector_store")
    def test_child_with_unknown_parent_is_returned_as_is(self, mock_get_vs, client):
        orphan = {
            "categoryID": 1, "subcategoryID": 1, "slug": "", "parent": "gone",
            "question": "Orphan?", "answer": "A.", "google_index": "QnAs9",
        }
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(_make_doc(orphan), 0.5)], all_docs_metadata=[orphan]
        )

        resp = client.post("/search", json={"query": "q", "googleSheetId": "s"})

        result = resp.json()["results"][0]
        assert result["question"] == "Orphan?" and result["children"] is None

    @patch("routes.search.translate")
    @patch("routes.search.get_vector_store")
    def test_translation_covers_query_and_children(self, mock_get_vs, mock_translate, client):
        parent = {
            "categoryID": 1, "subcategoryID": 1, "slug": "p", "parent": None,
            "question": "P?", "answer": "PA.", "google_index": "QnAs1",
        }
        child = {**parent, "slug": "", "parent": "p", "question": "C?", "answer": "CA.",
                 "google_index": "QnAs2"}
        vs = _make_vector_store(
            docs_and_scores=[(_make_doc(parent), 0.9)], all_docs_metadata=[parent, child]
        )
        mock_get_vs.return_value = vs
        mock_translate.side_effect = lambda from_lang, to_lang, text: f"[{to_lang}]{text}"

        resp = client.post(
            "/search", json={"query": "vraag", "googleSheetId": "s", "lang": "nl", "k": 3}
        )

        # query translated to English before retrieval, with the requested k
        vs.similarity_search_with_score.assert_called_once_with(query="[en]vraag", k=3)
        result = resp.json()["results"][0]
        assert result["question"] == "[nl]P?" and result["answer"] == "[nl]PA."
        assert result["children"][0]["question"] == "[nl]C?"
        assert result["children"][0]["answer"] == "[nl]CA."

    @patch("routes.search.get_vector_store")
    def test_english_query_is_not_translated(self, mock_get_vs, client):
        meta = {
            "categoryID": 1, "subcategoryID": 1, "slug": "", "parent": None,
            "question": "Q", "answer": "A", "google_index": "QnAs1",
        }
        vs = _make_vector_store(docs_and_scores=[(_make_doc(meta), 0.9)], all_docs_metadata=[meta])
        mock_get_vs.return_value = vs

        with patch("routes.search.translate") as mock_translate:
            client.post("/search", json={"query": "hello", "googleSheetId": "s"})

        mock_translate.assert_not_called()
        vs.similarity_search_with_score.assert_called_once_with(query="hello", k=5)
        mock_get_vs.assert_called_once_with("s", check_if_exists=True)

    @patch("routes.search.get_vector_store")
    def test_query_text_is_not_logged(self, mock_get_vs, client, caplog):
        import logging

        caplog.set_level(logging.INFO)
        meta = {
            "categoryID": 1, "subcategoryID": 1, "slug": "", "parent": None,
            "question": "Q", "answer": "A", "google_index": "QnAs1",
        }
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(_make_doc(meta), 0.9)], all_docs_metadata=[meta]
        )

        client.post("/search", json={"query": "hiv test anonymous", "googleSheetId": "s", "k": 3})

        logged = " ".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
        assert "hiv test anonymous" not in logged
        entry = next(r for r in caplog.records if r.getMessage() == "search").__dict__
        assert {k: entry[k] for k in ("googleSheetId", "lang", "k", "query_chars", "n_results")} == {
            "googleSheetId": "s", "lang": "en", "k": 3, "query_chars": 18, "n_results": 1
        }


class TestSearchTracing:
    """The search root span carries everything an observation-level evaluator needs."""

    @pytest.fixture()
    def span(self, monkeypatch):
        import utils.tracing as tracing

        client = MagicMock()
        span = MagicMock(name="span")
        client.start_as_current_observation.return_value.__enter__.return_value = span
        monkeypatch.setattr(tracing, "_clients", {"search": client})
        monkeypatch.setattr(tracing, "propagate_attributes", MagicMock())
        return span, client, tracing.propagate_attributes

    @patch("routes.search.translate")
    @patch("routes.search.get_vector_store")
    def test_span_has_evaluator_fields_and_scores(self, mock_get_vs, mock_translate, client, span):
        span, lf_client, propagate = span
        mock_translate.side_effect = lambda from_lang, to_lang, text: f"[{to_lang}]{text}"
        m1 = {"categoryID": 1, "subcategoryID": 1, "slug": "", "parent": None,
              "question": "Where is the GP?", "answer": "Main street.", "google_index": "QnAs1"}
        m2 = {**m1, "question": "Emergency?", "answer": "Call 112.", "google_index": "QnAs2"}
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(_make_doc(m1), 0.81), (_make_doc(m2), 0.42)],
            all_docs_metadata=[m1, m2],
        )

        resp = client.post(
            "/search", json={"query": "dokter", "googleSheetId": "sheetX", "lang": "nl", "k": 2}
        )
        assert resp.status_code == 200

        propagate.assert_called_once_with(
            session_id=None, user_id=None, tags=["sheet:sheetX", "channel:search", "lang:nl"]
        )
        # input is the query as used for retrieval (English)
        lf_client.start_as_current_observation.assert_called_once_with(
            as_type="span", name="search", input="[en]dokter"
        )
        update = span.update.call_args.kwargs
        context = "[1] Q: Where is the GP?\nA: Main street.\n\n[2] Q: Emergency?\nA: Call 112."
        assert update["output"] == context
        assert update["metadata"] == {
            "search_query": "[en]dokter",
            "retrieved_context": context,
            "original_query": "dokter",
            "lang": "nl",
            "k": 2,
            "n_results": 2,
            "top_score": 0.81,
        }
        scores = {c.kwargs["name"]: c.kwargs for c in span.score.call_args_list}
        assert scores["top_score"]["value"] == 0.81
        assert scores["n_results"]["value"] == 2.0
        assert scores["zero_results"]["value"] is False
        assert scores["zero_results"]["data_type"] == "BOOLEAN"

    @patch("routes.search.get_vector_store")
    def test_zero_results_scored(self, mock_get_vs, client, span):
        span, _, _ = span
        mock_get_vs.return_value = _make_vector_store(docs_and_scores=[], all_docs_metadata=[])

        resp = client.post("/search", json={"query": "nothing", "googleSheetId": "s"})

        assert resp.status_code == 200 and resp.json()["results"] == []
        update = span.update.call_args.kwargs
        assert update["output"] == "" and update["metadata"]["top_score"] == 0.0
        scores = {c.kwargs["name"]: c.kwargs["value"] for c in span.score.call_args_list}
        assert scores == {"top_score": 0.0, "n_results": 0.0, "zero_results": True}

    @patch("routes.search.get_vector_store")
    def test_search_works_without_tracing(self, mock_get_vs, client):
        meta = {"categoryID": 1, "subcategoryID": 1, "slug": "", "parent": None,
                "question": "Q", "answer": "A", "google_index": "QnAs1"}
        mock_get_vs.return_value = _make_vector_store(
            docs_and_scores=[(_make_doc(meta), 0.9)], all_docs_metadata=[meta]
        )
        resp = client.post("/search", json={"query": "q", "googleSheetId": "s"})
        assert resp.status_code == 200 and len(resp.json()["results"]) == 1
