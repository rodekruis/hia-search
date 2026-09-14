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

    @patch("utils.translator.requests.post")
    def test_translate_returns_original_on_request_error(self, mock_post):
        import requests
        from utils.translator import translate

        mock_post.side_effect = requests.ConnectionError("down")
        assert translate(from_lang="nl", to_lang="en", text="Hallo") == "Hallo"

    @patch("utils.translator.requests.post")
    def test_translate_returns_original_on_http_error(self, mock_post):
        import requests
        from utils.translator import translate

        mock_post.return_value.raise_for_status.side_effect = requests.HTTPError("401")
        assert translate(from_lang="nl", to_lang="en", text="Hallo") == "Hallo"

    @patch("utils.translator.requests.post")
    def test_translate_returns_original_on_malformed_payload(self, mock_post):
        from utils.translator import translate

        mock_post.return_value.json.return_value = [{}]
        assert translate(from_lang="nl", to_lang="en", text="Hallo") == "Hallo"

    @patch("utils.translator.requests.post")
    def test_translate_sends_timeout_and_languages(self, mock_post):
        from utils.translator import translate

        mock_post.return_value.json.return_value = [{"translations": [{"text": "x"}]}]
        translate(from_lang="nl", to_lang="en", text="Hallo")
        kwargs = mock_post.call_args.kwargs
        assert kwargs["timeout"] == 10
        assert kwargs["params"]["from"] == ["nl"] and kwargs["params"]["to"] == ["en"]
        assert kwargs["json"] == [{"text": "Hallo"}]


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

    @patch("utils.translator.requests.post")
    def test_detect_falls_back_to_english_on_error(self, mock_post):
        import requests
        from utils.translator import detect_language

        mock_post.side_effect = requests.Timeout()
        assert detect_language("Ciao mondo") == "en"

    @patch("utils.translator.requests.post")
    def test_detect_falls_back_to_english_on_malformed_payload(self, mock_post):
        from utils.translator import detect_language

        mock_post.return_value.json.return_value = []
        assert detect_language("Ciao mondo") == "en"


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

    @patch("utils.prompt_loader.pd.read_csv")
    def test_returns_empty_when_sheet_unreachable(self, mock_read_csv):
        import urllib.error
        from utils.prompt_loader import PromptLoader

        mock_read_csv.side_effect = urllib.error.HTTPError(
            url="u", code=404, msg="nf", hdrs=None, fp=None
        )
        assert PromptLoader(document_type="googlesheet", document_id="x").get_prompt() == ""

    @patch("utils.prompt_loader.pd.read_csv")
    def test_returns_empty_when_key_column_missing(self, mock_read_csv):
        import pandas as pd
        from utils.prompt_loader import PromptLoader

        mock_read_csv.return_value = pd.DataFrame({"foo": ["bar"]})
        assert PromptLoader(document_type="googlesheet", document_id="x").get_prompt() == ""

    @patch("utils.prompt_loader.pd.read_csv")
    def test_ignores_rows_without_key(self, mock_read_csv):
        import pandas as pd
        from utils.prompt_loader import PromptLoader

        mock_read_csv.return_value = pd.DataFrame(
            {"#KEY": [None, "#system-prompt"], "#VALUE": ["junk", "Be kind."]}
        )
        assert PromptLoader(document_type="googlesheet", document_id="x").get_prompt() == "Be kind."

    def test_json_loader_reads_header_row(self):
        from utils.prompt_loader import PromptLoader

        loader = PromptLoader(
            document_type="json",
            document_data={
                "values": [["#KEY", "#VALUE"], ["#other", "x"], ["#system-prompt", " Hi "]]
            },
        )
        assert loader.get_prompt() == "Hi"


class TestGetSystemPrompt:
    @pytest.fixture(autouse=True)
    def _clear(self):
        from utils.prompt_loader import clear_prompt_cache

        clear_prompt_cache()
        yield
        clear_prompt_cache()

    @patch("utils.prompt_loader.PromptLoader")
    def test_fetches_once_within_ttl(self, mock_loader):
        from utils.prompt_loader import get_system_prompt

        mock_loader.return_value.get_prompt.return_value = "Be kind."

        assert get_system_prompt("sheetA") == "Be kind."
        assert get_system_prompt("sheetA") == "Be kind."

        mock_loader.assert_called_once_with(document_type="googlesheet", document_id="sheetA")

    @patch("utils.prompt_loader.PromptLoader")
    def test_cache_is_per_sheet(self, mock_loader):
        from utils.prompt_loader import get_system_prompt

        mock_loader.return_value.get_prompt.side_effect = ["A", "B"]

        assert get_system_prompt("sheetA") == "A"
        assert get_system_prompt("sheetB") == "B"
        assert get_system_prompt("sheetA") == "A"
        assert mock_loader.call_count == 2

    @patch("utils.prompt_loader.time.monotonic")
    @patch("utils.prompt_loader.PromptLoader")
    def test_refetches_after_ttl(self, mock_loader, mock_clock):
        import utils.prompt_loader as pl

        mock_loader.return_value.get_prompt.side_effect = ["old", "new"]
        mock_clock.side_effect = [0.0, pl.PROMPT_CACHE_TTL_S - 1, pl.PROMPT_CACHE_TTL_S + 1]

        assert pl.get_system_prompt("s") == "old"
        assert pl.get_system_prompt("s") == "old"
        assert pl.get_system_prompt("s") == "new"

    @patch("utils.prompt_loader.PromptLoader")
    def test_falls_back_to_default_file_and_caches_it(self, mock_loader):
        import utils.prompt_loader as pl

        mock_loader.return_value.get_prompt.return_value = ""
        expected = pl.DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")

        assert pl.get_system_prompt("s") == expected
        assert expected.strip() != ""
        pl.get_system_prompt("s")
        # a sheet without a prompt must not be re-fetched every turn either
        mock_loader.assert_called_once()


# ---------------------------------------------------------------------------
# auth.py
# ---------------------------------------------------------------------------


class TestAuthHelpers:
    def test_require_write_key_accepts_and_rejects(self, monkeypatch):
        from fastapi import HTTPException
        from utils.auth import require_write_key

        monkeypatch.setenv("API_KEY_WRITE", "w-key")
        require_write_key("w-key")
        with pytest.raises(HTTPException) as exc:
            require_write_key("nope")
        assert exc.value.status_code == 401

    def test_missing_configured_key_rejects_everything(self, monkeypatch):
        """An unset API key must not turn into 'anything goes'."""
        from fastapi import HTTPException
        from utils.auth import require_read_key

        monkeypatch.setenv("API_KEY", "")
        with pytest.raises(HTTPException):
            require_read_key("")
        with pytest.raises(HTTPException):
            require_read_key(None)

    def test_twilio_tokens_must_be_object(self, monkeypatch):
        from utils.auth import _twilio_token_for

        monkeypatch.setenv("TWILIO_AUTH_TOKENS", '["tok"]')
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "fallback")
        assert _twilio_token_for("sheet") is None

    def test_twilio_tokens_map_ignores_global_fallback(self, monkeypatch):
        from utils.auth import _twilio_token_for

        monkeypatch.setenv("TWILIO_AUTH_TOKENS", '{"a": "tok-a"}')
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "fallback")
        assert _twilio_token_for("a") == "tok-a"
        assert _twilio_token_for("b") is None
        assert _twilio_token_for(None) is None

    def test_twilio_single_token_used_without_map(self, monkeypatch):
        from utils.auth import _twilio_token_for

        monkeypatch.delenv("TWILIO_AUTH_TOKENS", raising=False)
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "fallback")
        assert _twilio_token_for("anything") == "fallback"


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
