"""Tests for utils/document_loader.py: HIA sheet cleaning, JSON/Google Sheet loading, validation."""

from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import HTTPException

from utils.document_loader import DocumentLoader, uuid_hash


def _qna_sheet(rows: list[dict]) -> pd.DataFrame:
    """Build a raw Q&As sheet as exported from Google Sheets (HIA column headers)."""
    defaults = {
        "#CATEGORY": 1,
        "#SUBCATEGORY": 10,
        "#SLUG": "",
        "#VISIBLE": "Show",
        "#PARENT": "",
        "#QUESTION": "Q",
        "#ANSWER": "A",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _offers_sheet(rows: list[dict]) -> pd.DataFrame:
    # Empty cells arrive as NaN from read_csv; combine_fields relies on that.
    nan = float("nan")
    defaults = {
        "#CATEGORY": 2,
        "#SUBCATEGORY": 20,
        "#SLUG": "",
        "#VISIBLE": "Show",
        "#NAME": "Shelter",
        "#DESCRIPTION": nan,
        "#PHONENUMBERS": nan,
        "#EMAILS": nan,
        "#WEBURLS": nan,
        "#ADDRESS": nan,
        "#OPENWEEK": nan,
        "#OPENWEEKEND": nan,
        "#NEEDTOKNOW": nan,
        "#MOREINFO": nan,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_uuid_hash_is_deterministic():
    assert uuid_hash("abc") == uuid_hash("abc")
    assert uuid_hash("abc") != uuid_hash("abd")


class TestCleanQnAsSheet:
    def setup_method(self):
        self.loader = DocumentLoader(document_type="googlesheet", document_id="s")

    def test_renames_columns_and_builds_text(self):
        df = self.loader._clean_QnAs_sheet(
            _qna_sheet([{"#QUESTION": "How?", "#ANSWER": "Like this.", "#SLUG": "how"}])
        )
        assert list(df.columns) == [
            "google_index", "categoryID", "subcategoryID", "slug", "parent",
            "question", "answer", "text", "source",
        ]
        row = df.iloc[0]
        assert row["google_index"] == "QnAs0"
        assert row["text"] == "How? Like this."
        assert row["source"] == "QnAs"
        assert row["categoryID"] == 1 and row["subcategoryID"] == 10

    @pytest.mark.parametrize("visible", ["hide", "Hidden", "no", "0", "-"])
    def test_drops_hidden_rows(self, visible):
        df = self.loader._clean_QnAs_sheet(
            _qna_sheet([{"#VISIBLE": visible}, {"#VISIBLE": "Show", "#QUESTION": "kept"}])
        )
        assert df["question"].tolist() == ["kept"]

    def test_drops_rows_without_category_or_subcategory(self):
        df = self.loader._clean_QnAs_sheet(
            _qna_sheet(
                [
                    {"#CATEGORY": None, "#QUESTION": "no cat"},
                    {"#SUBCATEGORY": "", "#QUESTION": "no subcat"},
                    {"#QUESTION": "kept"},
                ]
            )
        )
        assert df["question"].tolist() == ["kept"]

    def test_drops_rows_with_empty_question_or_answer(self):
        df = self.loader._clean_QnAs_sheet(
            _qna_sheet(
                [
                    {"#QUESTION": "", "#ANSWER": "a"},
                    {"#QUESTION": "q", "#ANSWER": "  "},
                    {"#QUESTION": "kept", "#ANSWER": "a"},
                ]
            )
        )
        assert df["question"].tolist() == ["kept"]

    def test_strips_html_newlines_and_bold_markers(self):
        df = self.loader._clean_QnAs_sheet(
            _qna_sheet([{"#QUESTION": "<b>Q</b>", "#ANSWER": "line1\nline2 **bold**"}])
        )
        assert df.iloc[0]["text"] == "Q line1 line2 bold"

    def test_google_index_preserves_original_row_number(self):
        """The index must point at the sheet row, even after rows are dropped."""
        df = self.loader._clean_QnAs_sheet(
            _qna_sheet([{"#VISIBLE": "hide"}, {"#QUESTION": "second"}])
        )
        assert df.iloc[0]["google_index"] == "QnAs1"


class TestCleanOffersSheet:
    def setup_method(self):
        self.loader = DocumentLoader(document_type="googlesheet", document_id="s")

    def test_combines_present_fields_only(self):
        df = self.loader._clean_Offers_sheet(
            _offers_sheet(
                [
                    {
                        "#NAME": "Shelter",
                        "#DESCRIPTION": "Warm beds",
                        "#PHONENUMBERS": "0800",
                        "#ADDRESS": "Main st 1",
                    }
                ]
            )
        )
        text = df.iloc[0]["text"]
        assert text.startswith("Shelter. ")
        assert "Description: Warm beds" in text
        assert "Phone Number: 0800." in text
        assert "Address: Main st 1." in text
        assert "Email" not in text and "Website" not in text
        assert df.iloc[0]["google_index"] == "Offers0"
        assert df.iloc[0]["source"] == "Offers"

    def test_all_optional_fields_rendered(self):
        df = self.loader._clean_Offers_sheet(
            _offers_sheet(
                [
                    {
                        "#EMAILS": "a@b.c",
                        "#WEBURLS": "https://x.y",
                        "#OPENWEEK": "9-17",
                        "#OPENWEEKEND": "closed",
                        "#NEEDTOKNOW": "Bring ID",
                        "#MOREINFO": "See site",
                    }
                ]
            )
        )
        text = df.iloc[0]["text"]
        for fragment in [
            "Email: a@b.c.",
            "Link to Website: https://x.y",
            "Opening Hours: 9-17.",
            "Opening Hours on Weekends: closed.",
            "What you need to know: Bring ID",
            "Further information: See site",
        ]:
            assert fragment in text


class TestToDataframe:
    def test_json_uses_first_row_as_header(self):
        loader = DocumentLoader(
            document_type="json",
            document_data={
                "values": [
                    ["#CATEGORY", "#SUBCATEGORY", "#SLUG", "#VISIBLE", "#PARENT", "#QUESTION", "#ANSWER"],
                    ["1", "2", "", "Show", "", "Q1", "A1"],
                    ["1", "2", "", "hide", "", "Q2", "A2"],
                ]
            },
        )
        df = loader._to_dataframe()
        assert df["question"].tolist() == ["Q1"]
        assert df.iloc[0]["categoryID"] == 1

    def test_json_without_data_raises_400(self):
        loader = DocumentLoader(document_type="json", document_data={})
        with pytest.raises(HTTPException) as exc:
            loader._to_dataframe()
        assert exc.value.status_code == 400

    def test_unknown_type_raises_500(self):
        loader = DocumentLoader(document_type="xml", document_id="s")
        with pytest.raises(HTTPException) as exc:
            loader._to_dataframe()
        assert exc.value.status_code == 500

    @patch("utils.document_loader.pd.read_csv")
    def test_googlesheet_concatenates_qnas_and_offers(self, mock_read_csv):
        mock_read_csv.side_effect = [
            _qna_sheet([{"#QUESTION": "Q1"}]),
            _offers_sheet([{"#NAME": "Offer1"}]),
        ]
        loader = DocumentLoader(document_type="googlesheet", document_id="sheet-1")
        df = loader._to_dataframe()

        urls = [call.args[0] for call in mock_read_csv.call_args_list]
        assert "sheet=Q%26As" in urls[0] and "sheet-1" in urls[0]
        assert "sheet=Offers" in urls[1]
        assert df["source"].tolist() == ["QnAs", "Offers"]

    @patch("utils.document_loader.pd.read_csv")
    def test_googlesheet_http_error_propagates_status(self, mock_read_csv):
        mock_read_csv.side_effect = urllib.error.HTTPError(
            url="u", code=404, msg="not found", hdrs=None, fp=None
        )
        loader = DocumentLoader(document_type="googlesheet", document_id="missing")
        with pytest.raises(HTTPException) as exc:
            loader._to_dataframe()
        assert exc.value.status_code == 404
        assert "missing" in exc.value.detail


class TestValidation:
    def setup_method(self):
        self.loader = DocumentLoader(document_type="json")

    @pytest.mark.parametrize(
        "content,empty",
        [("abc", False), ("a b c", True), ("12 3 -- !!", True), ("x1yzw2", False), ("xy1", True), ("", True)],
    )
    def test_check_emptiness_requires_three_consecutive_letters(self, content, empty):
        assert self.loader._check_emptiness(content) is empty

    def test_validate_loading_drops_empty_documents(self):
        from langchain_core.documents import Document

        docs = [
            Document(page_content="real text", metadata={"i": 1}),
            Document(page_content="1 2 3", metadata={"i": 2}),
        ]
        valid = self.loader._validate_loading(docs)
        assert [d.metadata["i"] for d in valid] == [1]


class TestLoad:
    @patch("utils.document_loader.translate")
    @patch("utils.document_loader.detect_language")
    def test_load_translates_non_english_rows_and_keeps_metadata(
        self, mock_detect, mock_translate
    ):
        mock_detect.side_effect = lambda text: "nl" if "Hallo" in text else "en"
        mock_translate.side_effect = lambda from_lang, to_lang, text: f"[en]{text}"
        loader = DocumentLoader(
            document_type="json",
            document_data={
                "values": [
                    ["#CATEGORY", "#SUBCATEGORY", "#SLUG", "#VISIBLE", "#PARENT", "#QUESTION", "#ANSWER"],
                    ["1", "2", "hallo", "Show", "", "Hallo", "Wereld"],
                    ["1", "2", "", "Show", "hallo", "Hello", "World"],
                ]
            },
        )

        docs = loader.load()

        assert [d.page_content for d in docs] == ["[en]Hallo Wereld", "Hello World"]
        assert docs[0].metadata["google_index"] == "QnAs1"
        assert docs[0].metadata["slug"] == "hallo"
        assert docs[1].metadata["parent"] == "hallo"
        mock_translate.assert_called_once()
