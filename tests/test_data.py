"""Tests for the /create-vector-store and /delete-vector-store endpoints."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import inspect
import pytest

import routes.data as data_routes

WRITE = {"Authorization": "test-write-key"}


class TestDataEndpointsShape:
    def test_handlers_are_sync_so_rebuilds_do_not_block_the_event_loop(self):
        # an `async def` handler would run minutes of blocking I/O on the event loop
        assert not inspect.iscoroutinefunction(data_routes.create_vector_store)
        assert not inspect.iscoroutinefunction(data_routes.delete_vector_store)


class TestCreateVectorStore:
    """Tests for POST /create-vector-store."""

    @patch("routes.data.create_vector_store_index")
    def test_create_from_google_sheet(self, mock_create, client):
        mock_vs = MagicMock()
        mock_vs.n_documents_added = 326
        # Azure AI Search's count lags behind indexing: right after a rebuild it reads 0
        mock_vs.client.get_document_count.return_value = 0
        mock_create.return_value = mock_vs

        resp = client.post(
            "/create-vector-store",
            json={
                "googleSheetId": "sheet123",
            },
            headers=WRITE,
        )

        assert resp.status_code == 200
        assert resp.json() == "Created vector store index sheet123 with 326 documents."
        mock_create.assert_called_once_with(
            document_type="googlesheet",
            document_id="sheet123",
            document_data={},
        )

    @patch("routes.data.create_vector_store_index")
    def test_create_from_json_with_auth(self, mock_create, client):
        mock_vs = MagicMock()
        mock_vs.n_documents_added = 10
        mock_create.return_value = mock_vs

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
            document_type="json",
            document_id="sheet123",
            document_data={"values": [["col1"], ["val1"]]},
        )

    @patch("routes.data.create_vector_store_index")
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
    @patch("routes.data.create_vector_store_index")
    def test_create_from_google_sheet_requires_write_key(self, mock_create, client, headers):
        """Rebuilding costs translation + embeddings: the read key or no key is not enough."""
        resp = client.post("/create-vector-store", json={"googleSheetId": "sheet123"}, headers=headers)

        assert resp.status_code == 401
        mock_create.assert_not_called()


class TestDeleteVectorStore:
    """Tests for DELETE /delete-vector-store."""

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "test-api-key"}])
    @patch("routes.data.invalidate_vector_store")
    @patch("routes.data.SearchIndexClient")
    def test_delete_requires_write_key(self, mock_index_client_cls, mock_invalidate, client, headers):
        resp = client.request(
            "DELETE", "/delete-vector-store", json={"googleSheetId": "sheet123"}, headers=headers
        )

        assert resp.status_code == 401
        mock_index_client_cls.return_value.delete_index.assert_not_called()
        mock_invalidate.assert_not_called()

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
        mock_client.delete_index.assert_called_once()

    @patch("routes.data.invalidate_vector_store")
    @patch("routes.data.SearchIndexClient")
    def test_delete_evicts_cached_store(self, mock_index_client_cls, mock_invalidate, client):
        resp = client.request(
            "DELETE", "/delete-vector-store", json={"googleSheetId": "sheet123"}, headers=WRITE
        )
        assert resp.status_code == 200
        mock_invalidate.assert_called_once_with("sheet123")

    @patch("routes.data.invalidate_vector_store")
    @patch("routes.data.SearchIndexClient")
    def test_failed_delete_keeps_cache(self, mock_index_client_cls, mock_invalidate, client):
        mock_index_client_cls.return_value.delete_index.side_effect = Exception("nope")
        client.request("DELETE", "/delete-vector-store", json={"googleSheetId": "sheet123"}, headers=WRITE)
        mock_invalidate.assert_not_called()

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
