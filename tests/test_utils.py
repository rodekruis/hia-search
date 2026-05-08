"""Tests for utility modules: translator, prompt_loader, constants, vector_store helpers."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

# ---------------------------------------------------------------------------
# translator.py
# ---------------------------------------------------------------------------


class TestTranslate:

    @patch("utils.translator.requests.post")
    def test_translate_normal(self, mock_post):
        mock_post.return_value.json.return_value = [
            {"translations": [{"text": "Hello"}]}
        ]

        from utils.translator import translate

        result = translate(from_lang="nl", to_lang="en", text="Hallo")

        assert result == "Hello"
        mock_post.assert_called_once()

    def test_translate_empty_string(self):
        from utils.translator import translate

        assert translate(from_lang="en", to_lang="nl", text="") == ""

    def test_translate_nan(self):
        import math
        from utils.translator import translate

        assert translate(from_lang="en", to_lang="nl", text=float("nan")) == ""

    @patch("utils.translator.requests.post")
    def test_translate_whitespace_only(self, mock_post):
        from utils.translator import translate

        result = translate(from_lang="en", to_lang="nl", text="   ")
        assert result == ""
        mock_post.assert_not_called()


class TestDetectLanguage:

    @patch("utils.translator.requests.post")
    def test_detect_english(self, mock_post):
        mock_post.return_value.json.return_value = [{"language": "en"}]

        from utils.translator import detect_language

        result = detect_language("Hello world")

        assert result == "en"

    @patch("utils.translator.requests.post")
    def test_detect_italian(self, mock_post):
        mock_post.return_value.json.return_value = [{"language": "it"}]

        from utils.translator import detect_language

        result = detect_language("Ciao mondo")

        assert result == "it"

    def test_detect_empty(self):
        from utils.translator import detect_language

        assert detect_language("") == "en"

    def test_detect_whitespace(self):
        from utils.translator import detect_language

        assert detect_language("   ") == "en"


# ---------------------------------------------------------------------------
# constants.py – DocumentMetadata
# ---------------------------------------------------------------------------


class TestDocumentMetadata:

    def test_metadata_fields_exist(self):
        from utils.constants import DocumentMetadata

        dm = DocumentMetadata()
        assert dm.CATEGORY == "categoryID"
        assert dm.SUBCATEGORY == "subcategoryID"
        assert dm.SLUG == "slug"
        assert dm.QUESTION == "question"
        assert dm.ANSWER == "answer"
        assert dm.GOOGLE_INDEX == "google_index"
        assert dm.PARENT == "parent"
        assert dm.SCORE == "score"
        assert dm.CHILDREN == "children"


# ---------------------------------------------------------------------------
# vector_store.py – googleid_to_vectorstoreid
# ---------------------------------------------------------------------------


class TestGoogleIdConversion:

    def test_basic_conversion(self):
        from utils.vector_store import googleid_to_vectorstoreid

        result = googleid_to_vectorstoreid("ABC-123")
        assert result == "abc-123"

    def test_strips_special_chars(self):
        from utils.vector_store import googleid_to_vectorstoreid

        result = googleid_to_vectorstoreid("My_Sheet!@#ID")
        assert all(c.isalnum() or c == "-" for c in result)

    def test_truncates_to_128(self):
        from utils.vector_store import googleid_to_vectorstoreid

        long_id = "a" * 200
        result = googleid_to_vectorstoreid(long_id)
        assert len(result) <= 128


# ---------------------------------------------------------------------------
# prompt_loader.py
# ---------------------------------------------------------------------------


class TestPromptLoader:

    @patch("utils.prompt_loader.pd.read_csv")
    def test_loads_prompt_from_google_sheet(self, mock_read_csv):
        import pandas as pd
        from utils.prompt_loader import PromptLoader

        mock_read_csv.return_value = pd.DataFrame(
            {
                "#KEY": ["#system-prompt"],
                "#VALUE": ["  You are a helpful assistant.  "],
            }
        )

        loader = PromptLoader(document_type="googlesheet", document_id="sheet123")
        prompt = loader.get_prompt()

        assert prompt == "You are a helpful assistant."

    @patch("utils.prompt_loader.pd.read_csv")
    def test_returns_empty_when_no_prompt(self, mock_read_csv):
        import pandas as pd
        from utils.prompt_loader import PromptLoader

        mock_read_csv.return_value = pd.DataFrame(
            {
                "#KEY": ["#other-key"],
                "#VALUE": ["some value"],
            }
        )

        loader = PromptLoader(document_type="googlesheet", document_id="sheet123")
        prompt = loader.get_prompt()

        assert prompt == ""

    def test_json_loader_no_data_raises(self):
        from utils.prompt_loader import PromptLoader
        from fastapi import HTTPException

        loader = PromptLoader(document_type="json", document_id="x")
        with pytest.raises(HTTPException):
            loader.get_prompt()

    def test_invalid_type_raises(self):
        from utils.prompt_loader import PromptLoader
        from fastapi import HTTPException

        loader = PromptLoader(document_type="unknown", document_id="x")
        with pytest.raises(HTTPException):
            loader.get_prompt()


# ---------------------------------------------------------------------------
# groundedness.py
# ---------------------------------------------------------------------------


class TestGroundedness:

    @patch("utils.groundedness.requests.post")
    def test_no_ungrounded_content(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "ungroundedDetected": False,
            "ungroundedPercentage": 0.0,
            "ungroundedDetails": [],
        }

        from utils.groundedness import detect_groundness

        result = detect_groundness(
            content_text="All facts are grounded.",
            grounding_sources=["source1"],
            query="test query",
        )

        assert result == "All facts are grounded."

    @patch("utils.groundedness.requests.post")
    def test_api_error_returns_original(self, mock_post):
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        from utils.groundedness import detect_groundness

        result = detect_groundness(
            content_text="Some text.",
            grounding_sources=["src"],
            query="q",
        )

        assert result == "Some text."


# ---------------------------------------------------------------------------
# search helpers
# ---------------------------------------------------------------------------


class TestSearchHelpers:

    def test_get_score_google_index_found(self):
        from routes.search import get_score_google_index

        doc = MagicMock()
        doc.metadata = {"google_index": "QnAs5"}
        docs_and_scores = [(doc, 0.95)]

        score = get_score_google_index(docs_and_scores, "QnAs5")
        assert score == 0.95

    def test_get_score_google_index_not_found(self):
        from routes.search import get_score_google_index

        score = get_score_google_index([], "QnAs99")
        assert score == 0.0
