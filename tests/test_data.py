"""Tests for the /create-vector-store and /delete-vector-store endpoints."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest


class TestCreateVectorStore:
    """Tests for POST /create-vector-store."""

    @patch("routes.data.create_vector_store_index")
    def test_create_from_google_sheet(self, mock_create, client):
        mock_vs = MagicMock()
        mock_vs.client.get_document_count.return_value = 42
        mock_create.return_value = mock_vs

        resp = client.post(
            "/create-vector-store",
            json={
                "googleSheetId": "sheet123",
            },
        )

        assert resp.status_code == 200
        mock_create.assert_called_once_with(
            document_type="googlesheet",
            document_id="sheet123",
            document_data={},
        )

    @patch("routes.data.create_vector_store_index")
    def test_create_from_json_with_auth(self, mock_create, client):
        mock_vs = MagicMock()
        mock_vs.client.get_document_count.return_value = 10
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


class TestDeleteVectorStore:
    """Tests for DELETE /delete-vector-store."""

    @patch("routes.data.SearchIndexClient")
    def test_delete_success(self, mock_index_client_cls, client):
        mock_client = MagicMock()
        mock_index_client_cls.return_value = mock_client

        resp = client.request(
            "DELETE",
            "/delete-vector-store",
            json={"googleSheetId": "sheet123"},
        )

        assert resp.status_code == 200
        mock_client.delete_index.assert_called_once()

    @patch("routes.data.SearchIndexClient")
    def test_delete_not_found(self, mock_index_client_cls, client):
        mock_client = MagicMock()
        mock_client.delete_index.side_effect = Exception("Index not found")
        mock_index_client_cls.return_value = mock_client

        resp = client.request(
            "DELETE",
            "/delete-vector-store",
            json={"googleSheetId": "nonexistent"},
        )

        assert resp.status_code == 400
