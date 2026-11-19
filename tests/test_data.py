"""Tests for the /create-vector-store and /delete-vector-store endpoints."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import inspect
import pytest

import routes.data as data_routes
from utils.vector_store import googleid_to_vectorstoreid

WRITE = {"Authorization": "test-write-key"}


class TestDataEndpointsShape:
    def test_handlers_are_sync_so_rebuilds_do_not_block_the_event_loop(self):
        # an `async def` handler would run minutes of blocking I/O on the event loop
        assert not inspect.iscoroutinefunction(data_routes.create_vector_store)
        assert not inspect.iscoroutinefunction(data_routes.delete_vector_store)


class TestCreateVectorStore:
    """Tests for POST /create-vector-store."""

    def test_openapi_example_shows_empty_data(self, client):
        schema = client.get("/openapi.json").json()
        data_schema = schema["components"]["schemas"]["VectorStorePayload"]["properties"]["data"]
        assert data_schema["examples"] == [{}]

    @pytest.mark.parametrize("data", [None, {}], ids=["omitted", "empty"])
    @patch("routes.data.ensure_index_fresh")
    def test_create_from_google_sheet(self, mock_create, client, data):
        mock_create.return_value = {"status": "updated", "document_count": 326}

        payload: dict[str, object] = {"googleSheetId": "sheet123"}
        if data is not None:
            payload["data"] = data
        resp = client.post(
            "/create-vector-store",
            json=payload,
            headers=WRITE,
        )

        assert resp.status_code == 200
        index_name = googleid_to_vectorstoreid("sheet123")
        assert resp.json() == f"Created vector store index {index_name} with 326 documents."
        mock_create.assert_called_once_with(
            index_name, force=False, source_sheet_id="sheet123", document_data=None,
        )

    @patch("routes.data.ensure_index_fresh")
    def test_create_from_json_with_auth(self, mock_create, client):
        mock_create.return_value = {"status": "updated", "document_count": 10}

        resp = client.post(
            "/create-vector-store",
            json={
                "googleSheetId": "sheet123",
                "data": {"values": [["col1"], ["val1"]]},
            },
            headers={"Authorization": "test-write-key"},
        )

        assert resp.status_code == 200
        mock_create.assert_called_once_with(
            googleid_to_vectorstoreid("sheet123"), force=False, source_sheet_id="sheet123",
            document_data={"values": [["col1"], ["val1"]]},
        )

    @patch("routes.data.ensure_index_fresh")
    def test_create_from_json_unauthorized(self, mock_create, client):
        resp = client.post(
            "/create-vector-store",
            json={
                "googleSheetId": "sheet123",
                "data": {"values": [["col1"], ["val1"]]},
            },
            headers={"Authorization": "wrong-key"},
        )

        assert resp.status_code == 401

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "test-api-key"}])
    @patch("routes.data.ensure_index_fresh")
    def test_create_from_google_sheet_requires_write_key(self, mock_create, client, headers):
        """Rebuilding costs translation + embeddings: the read key or no key is not enough."""
        resp = client.post("/create-vector-store", json={"googleSheetId": "sheet123"}, headers=headers)

        assert resp.status_code == 401
        mock_create.assert_not_called()

    @patch("routes.data.ensure_index_fresh")
    def test_unchanged_response_and_force(self, mock_refresh, client):
        mock_refresh.return_value = {"status": "unchanged", "document_count": 12}
        response = client.post(
            "/create-vector-store", json={"googleSheetId": "sheet123", "force": True},
            headers=WRITE,
        )
        assert response.status_code == 200
        index_name = googleid_to_vectorstoreid("sheet123")
        assert response.json() == f"Vector store index {index_name} is unchanged with 12 documents."
        assert mock_refresh.call_args.kwargs["force"] is True


class TestDeleteVectorStore:
    """Tests for DELETE /delete-vector-store."""

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "test-api-key"}])
    @patch("routes.data.SearchIndexClient")
    def test_delete_requires_write_key(self, mock_index_client_cls, client, headers):
        resp = client.request(
            "DELETE", "/delete-vector-store", json={"googleSheetId": "sheet123"}, headers=headers
        )

        assert resp.status_code == 401
        mock_index_client_cls.return_value.delete_index.assert_not_called()

    @patch("routes.data.SearchIndexClient")
    def test_delete_success(self, mock_index_client_cls, client):
        mock_client = MagicMock()
        mock_index_client_cls.return_value = mock_client

        resp = client.request(
            "DELETE",
            "/delete-vector-store",
            json={"googleSheetId": "sheet123"},
            headers=WRITE,
        )

        assert resp.status_code == 200
        mock_client.delete_index.assert_called_once_with(googleid_to_vectorstoreid("sheet123"))

    @patch("routes.data.SearchIndexClient")
    def test_delete_not_found(self, mock_index_client_cls, client):
        mock_client = MagicMock()
        mock_client.delete_index.side_effect = Exception("Index not found")
        mock_index_client_cls.return_value = mock_client

        resp = client.request(
            "DELETE",
            "/delete-vector-store",
            json={"googleSheetId": "nonexistent"},
            headers=WRITE,
        )

        assert resp.status_code == 400

def _metadata(sheet_id, **overrides):
    return {
        "source_sheet_id": sheet_id, "source_type": "googlesheet", "index_owner": "hia-search",
        "index_environment": "prod", "created_at": "2025-01-01T00:00:00Z",
        "last_refreshed_at": None, "last_searched_at": None,
        "last_error": "2025-02-01T10:00:00+00:00 | 502: Could not read sheet",
        **overrides,
    }


class TestListVectorStores:
    """Tests for GET /vector-stores."""

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "test-api-key"}])
    @patch("routes.data.list_managed_indexes")
    def test_requires_write_key(self, mock_list, client, headers):
        assert client.get("/vector-stores", headers=headers).status_code == 401
        mock_list.assert_not_called()

    @patch("utils.vector_store.SearchClient")
    @patch("utils.vector_store.SearchIndexClient")
    def test_lists_only_recognized_indexes_of_this_environment(
        self, mock_index_client_cls, mock_search_client_cls, client, monkeypatch,
    ):
        monkeypatch.setenv("ENVIRONMENT", "prod")
        from azure.core.exceptions import ResourceNotFoundError

        good = googleid_to_vectorstoreid("good")
        documents = {
            good: _metadata("good"),
            googleid_to_vectorstoreid("other-owner"): _metadata("other-owner", index_owner="x"),
            googleid_to_vectorstoreid("other-env"): _metadata("other-env", index_environment="dev"),
            googleid_to_vectorstoreid("renamed"): _metadata("not-renamed"),
            googleid_to_vectorstoreid("no-metadata"): None,
        }
        mock_index_client_cls.return_value.__enter__.return_value.list_index_names.return_value = [
            *documents, "hia-test-abc", "unrelated-index",
        ]

        def search_client(endpoint, index_name, **kwargs):
            search = MagicMock()
            search.__enter__.return_value = search
            if documents.get(index_name) is None:
                search.get_document.side_effect = ResourceNotFoundError("missing")
            else:
                search.get_document.return_value = documents[index_name]
            return search

        mock_search_client_cls.side_effect = search_client

        resp = client.get("/vector-stores", headers=WRITE)

        assert resp.status_code == 200
        assert resp.json() == [{
            "index_name": good, "source_sheet_id": "good", "source_type": "googlesheet",
            "created_at": "2025-01-01T00:00:00Z", "last_refreshed_at": None,
            "last_searched_at": None,
            "last_error": "2025-02-01T10:00:00+00:00 | 502: Could not read sheet",
            "failing_since": "2025-02-01T10:00:00Z",
        }]
        opened = {call.kwargs["index_name"] for call in mock_search_client_cls.call_args_list}
        assert "hia-test-abc" not in opened and "unrelated-index" not in opened
