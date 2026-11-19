"""Incremental refresh with stateful Azure fakes; no external calls."""

from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents.models import IndexingResult
from fastapi import HTTPException
from langchain_core.documents import Document

import utils.vector_store as vs
from utils.document_loader import DocumentLoader, SourceSnapshot
from utils.translator import TranslationError


class MemorySearch:
    def __init__(self):
        self.documents = {}
        self.events = []
        self.responses = {}
        self.hidden = set()
        self.hide_uploads = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_document(self, key):
        if key not in self.documents:
            raise ResourceNotFoundError()
        return copy.deepcopy(self.documents[key])

    def search(self, **kwargs):
        self.events.append(("search", kwargs))
        documents = [
            copy.deepcopy(doc) for key, doc in self.documents.items()
            if key != vs.INDEX_METADATA_ID and key not in self.hidden
        ]
        excluded = re.search(r"not search\.in\(id, '([^']+)'\)", kwargs.get("filter", ""))
        if excluded:
            seen = set(excluded.group(1).split(","))
            documents = [doc for doc in documents if doc["id"] not in seen]
        return documents[:kwargs.get("top", 1000)]

    def _write(self, action, documents):
        self.events.append((action, copy.deepcopy(documents)))
        responses = self.responses.get(action, [])
        codes = responses.pop(0) if responses else [200] * len(documents)
        results = []
        for document, code in zip(documents, codes, strict=True):
            key = document["id"]
            if code < 400:
                if action == "delete":
                    self.documents.pop(key, None)
                elif action == "upload":
                    self.documents[key] = copy.deepcopy(document)
                    if self.hide_uploads:
                        self.hidden.add(key)
                else:
                    self.documents.setdefault(key, {}).update(copy.deepcopy(document))
            results.append(IndexingResult(key=key, succeeded=code < 400, status_code=code))
        return results

    def upload_documents(self, documents):
        return self._write("upload", documents)

    def merge_documents(self, documents):
        return self._write("merge", documents)

    def merge_or_upload_documents(self, documents):
        return self._write("meta", documents)

    def delete_documents(self, documents):
        return self._write("delete", documents)


@pytest.fixture
def refresh(monkeypatch):
    search = MemorySearch()
    indexes = MagicMock()
    indexes.__enter__.return_value = indexes
    indexes.get_index.side_effect = lambda name: (
        indexes.definition if indexes.definition is not None else _missing()
    )
    indexes.definition = None
    indexes.create_index.side_effect = lambda definition: setattr(indexes, "definition", definition)
    loader = MagicMock()
    loader.translate_document.side_effect = lambda doc: doc.model_copy(
        update={"page_content": f"English: {doc.page_content}"}
    )
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0, 1.0, 2.0]
    embedder.embed_documents.side_effect = lambda texts: [
        [float(len(text)), 1.0, 2.0] for text in texts
    ]
    monkeypatch.setattr(vs, "SearchIndexClient", lambda *args, **kwargs: indexes)
    monkeypatch.setattr(vs, "SearchClient", lambda *args, **kwargs: search)
    monkeypatch.setattr(vs, "DocumentLoader", MagicMock(return_value=loader))
    monkeypatch.setattr(vs, "_openai_embedder", MagicMock(return_value=embedder))
    monkeypatch.setattr(vs.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("ENVIRONMENT", "test")

    def run(documents, digest="first", force=False, data=None):
        loader.load_source.return_value = SourceSnapshot(digest, documents)
        return vs.ensure_index_fresh(
            "index", force=force, source_sheet_id="ShEet_ABC", document_data=data,
        )

    return SimpleNamespace(
        run=run, search=search, indexes=indexes, loader=loader, embedder=embedder,
    )


def _missing():
    raise ResourceNotFoundError()


def entry(text="Question and answer", row="QnAs0", category=1):
    return Document(
        page_content=text,
        metadata={"source": "QnAs", "google_index": row, "categoryID": category},
    )


def content(refresh):
    return {
        key: doc for key, doc in refresh.search.documents.items() if key != vs.INDEX_METADATA_ID
    }


def metadata(refresh):
    return refresh.search.documents[vs.INDEX_METADATA_ID]


def test_new_index_and_unchanged_refresh_do_not_reembed(refresh):
    result = refresh.run([entry()])
    assert result == {
        "status": "updated", "index_name": "index", "document_count": 1,
        "added": 1, "removed": 0,
    }
    doc = next(iter(content(refresh).values()))
    assert re.fullmatch("[0-9a-f]{64}", doc["id"])
    assert doc["content"] == "English: Question and answer"
    assert metadata(refresh)["source_sheet_id"] == "ShEet_ABC"
    assert metadata(refresh)["sheet_hash"] == "first"
    assert not {"content", "content_vector"} & metadata(refresh).keys()
    refresh.indexes.delete_index.assert_not_called()
    refresh.embedder.embed_query.assert_not_called()
    refresh.loader.translate_document.reset_mock()
    refresh.embedder.reset_mock()
    refresh.search.events.clear()

    assert refresh.run([entry()])["status"] == "unchanged"
    refresh.loader.translate_document.assert_not_called()
    refresh.embedder.embed_documents.assert_not_called()
    assert [action for action, _ in refresh.search.events] == ["search"]


def test_edit_uploads_only_new_entry_before_deleting_obsolete(refresh):
    refresh.run([entry("Original answer"), entry("Keep this answer", "QnAs1")])
    preserved = next(
        copy.deepcopy(doc) for doc in content(refresh).values() if "Keep this" in doc["content"]
    )
    refresh.embedder.embed_documents.reset_mock()
    refresh.search.events.clear()

    result = refresh.run([entry("Edited answer"), entry("Keep this answer", "QnAs1")], "second")

    assert (result["added"], result["removed"]) == (1, 1)
    refresh.embedder.embed_documents.assert_called_once_with(["English: Edited answer"])
    assert content(refresh)[preserved["id"]] == preserved
    actions = [action for action, _ in refresh.search.events]
    assert actions.index("upload") < actions.index("delete") < len(actions) - 1
    assert "search" in actions[actions.index("upload") + 1:actions.index("delete")]
    assert actions[-1] == "meta"


def test_row_moves_merge_metadata_without_embedding(refresh):
    refresh.run([entry("First answer"), entry("Second answer", "QnAs1")])
    ids = set(content(refresh))
    refresh.embedder.reset_mock()
    result = refresh.run(
        [entry("Second answer", "QnAs0"), entry("First answer", "QnAs1")], "reordered",
    )
    assert (result["added"], result["removed"]) == (0, 0)
    assert set(content(refresh)) == ids
    refresh.embedder.embed_documents.assert_not_called()
    for doc in content(refresh).values():
        expected = "QnAs1" if "First" in doc["content"] else "QnAs0"
        assert json.loads(doc["metadata"])["google_index"] == expected


def test_distinct_metadata_and_identical_duplicates_are_preserved(refresh):
    result = refresh.run([
        entry(), entry(row="QnAs1"), entry(row="QnAs2", category=2),
    ])
    assert result["document_count"] == 3
    assert len(content(refresh)) == 3
    assert len({doc["id"] for doc in content(refresh).values()}) == 3


def test_force_reconciles_without_reembedding_unchanged_entries(refresh):
    refresh.run([entry()])
    refresh.embedder.reset_mock()
    assert refresh.run([entry()], force=True)["status"] == "updated"
    refresh.embedder.embed_documents.assert_not_called()


def test_same_hash_with_interrupted_changes_is_reconciled(refresh):
    refresh.run([entry()])
    refresh.search.documents["abandoned"] = {"id": "abandoned", "metadata": "{}"}
    refresh.embedder.reset_mock()
    result = refresh.run([entry()])
    assert result["removed"] == 1
    assert "abandoned" not in refresh.search.documents
    refresh.embedder.embed_documents.assert_not_called()


@pytest.mark.parametrize("failure", [
    HTTPException(502, "Could not read sheet"), TranslationError("Translation failed"),
])
def test_read_or_translation_failure_preserves_existing_content(refresh, failure):
    refresh.run([entry()])
    previous = copy.deepcopy(content(refresh))
    if isinstance(failure, HTTPException):
        refresh.loader.load_source.side_effect = failure
    else:
        refresh.loader.translate_document.side_effect = failure
    with pytest.raises(type(failure)):
        refresh.run([entry("Edited answer")], "second")
    assert content(refresh) == previous
    assert metadata(refresh)["sheet_hash"] == "first"
    assert metadata(refresh)["last_error"]


def test_partial_upload_failure_is_retryable_without_reembedding_successes(refresh):
    refresh.run([entry("Old answer")])
    previous_ids = set(content(refresh))
    refresh.search.responses["upload"] = [[200, 400]]
    with pytest.raises(HttpResponseError):
        refresh.run([entry("New answer one"), entry("New answer two", "QnAs1")], "second")
    assert previous_ids <= content(refresh).keys()
    assert metadata(refresh)["sheet_hash"] == "first"
    refresh.embedder.embed_documents.reset_mock()

    result = refresh.run(
        [entry("New answer one"), entry("New answer two", "QnAs1")], "second",
    )
    assert result["added"] == 1
    assert len(refresh.embedder.embed_documents.call_args.args[0]) == 1
    assert metadata(refresh)["sheet_hash"] == "second"
    assert len(content(refresh)) == 2


def test_transient_document_failure_retries_only_failed_documents(refresh):
    refresh.search.responses["upload"] = [[200, 503], [200]]
    refresh.run([entry(), entry("Another answer", "QnAs1")])
    uploads = [docs for action, docs in refresh.search.events if action == "upload"]
    assert [len(docs) for docs in uploads] == [2, 1]
    refresh.embedder.embed_documents.assert_called_once()


def test_visibility_timeout_prevents_deletion_and_hash_commit(refresh, monkeypatch):
    refresh.run([entry()])
    previous_ids = set(content(refresh))
    refresh.search.hide_uploads = True
    ticks = iter([0, 31])
    monkeypatch.setattr(vs.time, "monotonic", lambda: next(ticks))
    with pytest.raises(HttpResponseError, match="searchable"):
        refresh.run([entry("New answer")], "second")
    assert previous_ids <= content(refresh).keys()
    assert metadata(refresh)["sheet_hash"] == "first"


@pytest.mark.parametrize("action", ["delete", "meta"])
def test_failed_delete_or_final_commit_leaves_refresh_retryable(refresh, action):
    refresh.run([entry()])
    refresh.search.responses[action] = [[400]]
    with pytest.raises(HttpResponseError):
        refresh.run([entry("New answer")], "second")
    assert metadata(refresh)["sheet_hash"] == "first"
    refresh.embedder.reset_mock()
    refresh.run([entry("New answer")], "second")
    refresh.embedder.embed_documents.assert_not_called()
    assert metadata(refresh)["sheet_hash"] == "second"


def test_valid_empty_source_clears_only_content_and_preserves_usage(refresh):
    refresh.run([entry()])
    metadata(refresh)["last_searched_at"] = "2026-11-18T00:00:00Z"
    created = metadata(refresh)["created_at"]
    refresh.embedder.reset_mock()
    result = refresh.run([], "empty")
    assert result["document_count"] == 0
    assert content(refresh) == {}
    assert metadata(refresh)["created_at"] == created
    assert metadata(refresh)["last_searched_at"] == "2026-11-18T00:00:00Z"
    assert refresh.run([], "empty")["status"] == "unchanged"
    refresh.embedder.embed_documents.assert_not_called()


def test_json_index_requires_explicit_payload(refresh):
    refresh.run([entry()], data={"values": []})
    refresh.loader.load_source.reset_mock()
    with pytest.raises(HTTPException, match="new JSON payload"):
        vs.ensure_index_fresh("index")
    refresh.loader.load_source.assert_not_called()
    assert refresh.run([entry("New answer")], "second", data={"values": []})["added"] == 1


@pytest.mark.parametrize("field,value", [
    ("embedding_model", "other-model"), ("source_sheet_id", "different-sheet"),
    ("index_environment", "another-environment"),
])
def test_incompatible_source_model_or_owner_is_rejected(refresh, field, value):
    refresh.run([entry()])
    metadata(refresh)[field] = value
    previous = copy.deepcopy(content(refresh))
    with pytest.raises(HTTPException):
        refresh.run([entry("New answer")], "second", force=True)
    assert content(refresh) == previous


def test_vector_dimension_change_is_rejected_even_with_force(refresh):
    refresh.run([entry()])
    refresh.embedder.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    previous = copy.deepcopy(content(refresh))
    with pytest.raises(HTTPException, match="dimensions"):
        refresh.run([entry("New answer")], "second", force=True)
    assert content(refresh) == previous


def test_all_existing_ids_are_paged_before_deletion(refresh):
    documents = [entry(f"Answer number {i}", f"QnAs{i}") for i in range(1001)]
    refresh.run(documents)
    refresh.embedder.reset_mock()
    result = refresh.run([], "empty")
    assert result["removed"] == 1001
    assert content(refresh) == {}
    queries = [kwargs for action, kwargs in refresh.search.events if action == "search"]
    assert any("search.in" in query["filter"] for query in queries)
    refresh.embedder.embed_documents.assert_not_called()


def test_real_json_loader_is_canonical(refresh, monkeypatch):
    monkeypatch.setattr(vs, "DocumentLoader", DocumentLoader)
    monkeypatch.setattr("utils.document_loader.detect_language", lambda text, **kwargs: "en")
    headers = [
        "#CATEGORY", "#SUBCATEGORY", "#SLUG", "#VISIBLE", "#PARENT", "#QUESTION", "#ANSWER",
    ]
    payload = {"values": [headers, [1, 2, "", "Show", "", "Hello", "World"]]}
    result = vs.ensure_index_fresh("index", source_sheet_id="ShEet_ABC", document_data=payload)
    assert result["added"] == 1
    assert next(iter(content(refresh).values()))["content"] == "Hello World"
    refresh.embedder.reset_mock()
    payload["values"].append(["", " "])
    assert vs.ensure_index_fresh("index", document_data=payload)["status"] == "unchanged"
    refresh.embedder.embed_documents.assert_not_called()
    refresh.embedder.embed_query.assert_not_called()


@pytest.fixture
def csv_refresh(refresh, monkeypatch):
    monkeypatch.setattr(vs, "DocumentLoader", DocumentLoader)
    monkeypatch.setattr("utils.document_loader.detect_language", lambda text, **kwargs: "en")
    monkeypatch.delenv("GOOGLE_SHEETS_API_KEY", raising=False)
    responses = {
        "Q&As": (
            b"#CATEGORY,#SUBCATEGORY,#SLUG,#VISIBLE,#PARENT,#QUESTION,#ANSWER\n"
            b"1,2,,Show,,Question one,Answer one\n"
            b"1,2,,Show,,Question two,Answer two\n"
        ),
        "Offers": (
            b"#CATEGORY,#SUBCATEGORY,#SLUG,#VISIBLE,#NAME,#DESCRIPTION,"
            b"#PHONENUMBERS,#EMAILS,#WEBURLS,#ADDRESS,#OPENWEEK,#OPENWEEKEND,"
            b"#NEEDTOKNOW,#MOREINFO\n"
        ),
    }
    session = MagicMock()
    session.__enter__.return_value = session

    def get_csv(url, *, params, **kwargs):
        value = responses[params["sheet"]]
        if isinstance(value, Exception):
            raise value
        response = MagicMock()
        response.content = value
        response.headers = {"Content-Type": "text/csv; charset=utf-8"}
        return response

    session.get.side_effect = get_csv
    monkeypatch.setattr("utils.document_loader.requests.Session", lambda: session)
    return refresh, responses


def test_real_csv_refresh_reuses_unchanged_embeddings(csv_refresh):
    refresh, responses = csv_refresh
    assert vs.ensure_index_fresh("index", source_sheet_id="ShEet_ABC")["added"] == 2
    previous = copy.deepcopy(content(refresh))
    refresh.embedder.reset_mock()
    assert vs.ensure_index_fresh("index")["status"] == "unchanged"
    refresh.embedder.embed_documents.assert_not_called()

    responses["Q&As"] = responses["Q&As"].replace(b"Answer one", b"Changed answer")
    result = vs.ensure_index_fresh("index")
    assert (result["added"], result["removed"]) == (1, 1)
    refresh.embedder.embed_documents.assert_called_once_with(["Question one Changed answer"])
    unchanged = next(
        doc for doc in previous.values() if doc["content"] == "Question two Answer two"
    )
    assert content(refresh)[unchanged["id"]] == unchanged


def test_real_header_only_csv_clears_content(csv_refresh):
    refresh, responses = csv_refresh
    vs.ensure_index_fresh("index", source_sheet_id="ShEet_ABC")
    responses["Q&As"] = responses["Q&As"].splitlines(keepends=True)[0]
    refresh.embedder.reset_mock()
    result = vs.ensure_index_fresh("index")
    assert result["removed"] == 2
    assert content(refresh) == {}
    assert metadata(refresh)["sheet_hash"]
    refresh.embedder.embed_documents.assert_not_called()


@pytest.mark.parametrize("failure", ["incomplete", "unavailable"])
def test_invalid_csv_refresh_preserves_existing_index(csv_refresh, failure):
    refresh, responses = csv_refresh
    vs.ensure_index_fresh("index", source_sheet_id="ShEet_ABC")
    previous = copy.deepcopy(content(refresh))
    previous_hash = metadata(refresh)["sheet_hash"]
    refresh.search.events.clear()
    refresh.embedder.reset_mock()
    if failure == "incomplete":
        responses["Q&As"] = responses["Q&As"].replace(b"Answer one", b"")
    else:
        responses["Offers"] = requests.Timeout()
    with pytest.raises(HTTPException):
        vs.ensure_index_fresh("index")
    assert content(refresh) == previous
    assert metadata(refresh)["sheet_hash"] == previous_hash
    assert metadata(refresh)["last_error"]
    assert not any(action in ("upload", "delete") for action, _ in refresh.search.events)
    refresh.embedder.embed_documents.assert_not_called()


def test_new_empty_index_is_initialized_once(refresh):
    assert refresh.run([])["document_count"] == 0
    assert set(refresh.search.documents) == {vs.INDEX_METADATA_ID}
    assert refresh.run([])["status"] == "unchanged"
    refresh.embedder.embed_query.assert_called_once()


def test_recovered_unchanged_source_clears_error_without_embedding(refresh):
    refresh.run([entry()])
    refresh.loader.load_source.side_effect = HTTPException(502, "Unavailable")
    with pytest.raises(HTTPException):
        refresh.run([entry()])
    refresh.loader.load_source.side_effect = None
    refresh.embedder.reset_mock()
    assert refresh.run([entry()])["status"] == "unchanged"
    assert metadata(refresh)["last_error"] is None
    refresh.embedder.embed_documents.assert_not_called()


def test_failure_start_is_kept_across_repeated_failures_and_reset_after_recovery(refresh):
    refresh.run([entry()])
    refresh.loader.load_source.side_effect = HTTPException(502, "Unavailable")
    with pytest.raises(HTTPException):
        refresh.run([entry()])
    first = metadata(refresh)["last_error"]
    since = vs.failing_since(first)
    assert since is not None and first.endswith("502: Unavailable")

    refresh.loader.load_source.side_effect = HTTPException(404, "Gone")
    with pytest.raises(HTTPException):
        refresh.run([entry()])
    assert vs.failing_since(metadata(refresh)["last_error"]) == since
    assert metadata(refresh)["last_error"].endswith("404: Gone")

    refresh.loader.load_source.side_effect = None
    refresh.run([entry()])
    assert vs.failing_since(metadata(refresh)["last_error"]) is None


@pytest.mark.parametrize("value", [None, "", "502: legacy error", "not-a-date | x", "2024-01-01 | x"])
def test_failing_since_ignores_missing_or_unparseable_prefixes(value):
    assert vs.failing_since(value) is None


def test_incomplete_registration_response_does_not_create_an_error_only_metadata_record(refresh):
    refresh.search.merge_or_upload_documents = MagicMock(return_value=[])
    with pytest.raises(HttpResponseError, match="Incomplete indexing"):
        refresh.run([entry()])
    assert refresh.search.documents == {}


def test_invalid_vectors_leave_existing_content_untouched(refresh):
    refresh.run([entry()])
    previous = copy.deepcopy(content(refresh))
    refresh.embedder.embed_documents.side_effect = lambda texts: [
        [float("nan"), 1.0, 2.0] for _ in texts
    ]
    with pytest.raises(HttpResponseError, match="invalid vectors"):
        refresh.run([entry("New answer")], "second")
    assert content(refresh) == previous
    assert metadata(refresh)["sheet_hash"] == "first"


def test_index_created_by_a_concurrent_request_is_reused(refresh):
    def created_concurrently(definition):
        refresh.indexes.definition = definition
        raise HttpResponseError("Cannot create index because it already exists.")

    refresh.indexes.create_index.side_effect = created_concurrently
    result = refresh.run([entry()])

    assert result["status"] == "updated"
    assert len(content(refresh)) == 1
    assert metadata(refresh)["sheet_hash"] == "first"


def test_concurrently_created_index_is_awaited_until_visible(refresh):
    pending = {}

    def created_concurrently(definition):
        pending["definition"] = definition
        raise HttpResponseError("Cannot create index because it already exists.")

    lookups = iter([None, None, None, None])
    original_get_index = refresh.indexes.get_index.side_effect

    def eventually_visible(name):
        if refresh.indexes.definition is None and next(lookups, "visible") == "visible":
            refresh.indexes.definition = pending["definition"]
        return original_get_index(name)

    refresh.indexes.create_index.side_effect = created_concurrently
    refresh.indexes.get_index.side_effect = eventually_visible

    assert refresh.run([entry()])["status"] == "updated"
    assert len(content(refresh)) == 1


def test_index_creation_failure_is_raised_when_no_index_exists(refresh):
    refresh.indexes.create_index.side_effect = HttpResponseError("quota exceeded")
    with pytest.raises(HttpResponseError, match="quota"):
        refresh.run([entry()])
    assert content(refresh) == {}
