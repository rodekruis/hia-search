from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List
from urllib.parse import quote

import pandas as pd
import requests
from fastapi import HTTPException
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.constants import DocumentMetadata
from utils.translator import detect_language, translate

logger = logging.getLogger(__name__)

dm = DocumentMetadata()

Cell = str | int | float | bool | None
SheetValues = dict[str, list[list[str]]]
_COMMON_HEADERS = ("#CATEGORY", "#SUBCATEGORY", "#SLUG", "#VISIBLE")
_HIDDEN_VALUES = ("hide", "hidden", "no", "0", "-")
_HEADERS = {
    "Q&As": (*_COMMON_HEADERS, "#PARENT", "#QUESTION", "#ANSWER"),
    "Offers": (
        *_COMMON_HEADERS, "#NAME", "#DESCRIPTION", "#PHONENUMBERS", "#EMAILS",
        "#WEBURLS", "#ADDRESS", "#OPENWEEK", "#OPENWEEKEND", "#NEEDTOKNOW", "#MOREINFO",
    ),
}


class _JsonValues(BaseModel):
    model_config = ConfigDict(strict=True)

    values: list[list[Cell]]


@dataclass(frozen=True)
class SourceSnapshot:
    sheet_hash: str
    documents: list[Document]


def _normalize_values(rows: Sequence[Sequence[Cell]]) -> list[list[str]]:
    normalized = []
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, float) and not math.isfinite(value):
                raise HTTPException(status_code=422, detail="Source contains an invalid number.")
            if value is None:
                text = ""
            elif isinstance(value, bool):
                text = "TRUE" if value else "FALSE"
            elif isinstance(value, float) and value.is_integer():
                text = str(int(value))
            else:
                text = str(value).strip()
            cells.append(text)
        while cells and cells[-1] == "":
            cells.pop()
        normalized.append(cells)
    while normalized and not normalized[-1]:
        normalized.pop()
    return normalized


def _column(df: pd.DataFrame, tag: str) -> str:
    matches = [
        column for column in df.columns
        if re.search(re.escape(tag) + r"(?![A-Za-z0-9_])", str(column))
    ]
    if len(matches) != 1:
        raise HTTPException(status_code=422, detail=f"Source requires one {tag} column.")
    return matches[0]


def _validate_visible_rows(df: pd.DataFrame, name: str, columns: dict[str, str]) -> None:
    if name == "Q&As":
        content_tags = ("#QUESTION", "#ANSWER")
        required_tags = ("#CATEGORY", "#SUBCATEGORY", *content_tags)
    else:
        content_tags = _HEADERS["Offers"][4:]
        required_tags = ("#CATEGORY", "#SUBCATEGORY", "#NAME")
    visible = ~df[columns["#VISIBLE"]].str.lower().isin(_HIDDEN_VALUES)
    has_content = df[[columns[tag] for tag in content_tags]].ne("").any(axis=1)
    missing_required = df[[columns[tag] for tag in required_tags]].eq("").any(axis=1)
    if (visible & has_content & missing_required).any():
        raise HTTPException(
            status_code=422, detail=f"Source contains incomplete visible {name} entries."
        )


def uuid_hash(content: str) -> str:
    """Create a unique hash from the page content of a document"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))


class DocumentLoader:
    """
    Document loading class that:
        1. Loads documents from source
        2. Returns list of Langchain Documents
    """

    output: List[Document]

    def __init__(
        self,
        document_type: str,
        document_id: str = None,
        document_data: dict = None,
        **kwargs: dict,
    ):
        self.document_type = document_type
        self.document_id = document_id
        self.document_data = document_data
        self.__dict__.update(kwargs)

    def _clean_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning on sheet
        """
        # rename columns to standard format
        df = df.rename(
            columns={
                _column(df, "#CATEGORY"): dm.CATEGORY,
                _column(df, "#SUBCATEGORY"): dm.SUBCATEGORY,
                _column(df, "#SLUG"): dm.SLUG,
                _column(df, "#VISIBLE"): "visible",
            }
        )
        # filter out what's hidden
        df = df[
            ~df["visible"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(_HIDDEN_VALUES)
        ].drop(columns=["visible"])
        # filter out rows with empty CATEGORY or SUBCATEGORY
        df = df[df[dm.CATEGORY].astype(str).str.strip() != ""]
        df = df[df[dm.SUBCATEGORY].astype(str).str.strip() != ""]
        df = df.dropna(subset=[dm.CATEGORY, dm.SUBCATEGORY])
        for column in (dm.CATEGORY, dm.SUBCATEGORY):
            numeric = pd.to_numeric(df[column], errors="raise")
            if (
                numeric.isna().any() or (numeric % 1 != 0).any()
                or (numeric < -(2**63)).any() or (numeric >= 2**63).any()
            ):
                raise ValueError("Category identifiers must be integers.")
            df[column] = numeric.astype("int64")
        # add text column
        df["text"] = ""
        return df

    def _clean_QnAs_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the Q&As sheet by removing empty rows and columns, and standardizing the column names.
        """
        df = self._clean_sheet(df)
        # add google index (row number) as a column
        df[dm.GOOGLE_INDEX] = "QnAs" + df.index.astype(str)
        # rename columns to standard format
        df = df.rename(
            columns={
                _column(df, "#PARENT"): dm.PARENT,
                _column(df, "#QUESTION"): dm.QUESTION,
                _column(df, "#ANSWER"): dm.ANSWER,
            }
        )
        # Filter out rows with empty Q&As
        df = df[df[dm.QUESTION].astype(str).str.strip() != ""]
        df = df[df[dm.ANSWER].astype(str).str.strip() != ""]
        # keep only relevant columns
        df = df[
            [
                dm.GOOGLE_INDEX,
                dm.CATEGORY,
                dm.SUBCATEGORY,
                dm.SLUG,
                dm.PARENT,
                dm.QUESTION,
                dm.ANSWER,
                "text",
            ]
        ]
        # clean text
        df["text"] = df[dm.QUESTION] + " " + df[dm.ANSWER]
        df["text"] = df["text"].str.replace(r"<[^<]+?>", "", regex=True)
        df["text"] = df["text"].str.replace(r"\n", " ", regex=True)
        df["text"] = df["text"].str.replace("**", "")
        df["text"] = df["text"].astype(str)
        df = df.dropna(subset=["text", dm.GOOGLE_INDEX])
        # add google sheet name
        df["source"] = "QnAs"
        return df

    def _clean_Offers_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the Offers sheet by removing empty rows and columns, and standardizing the column names.
        """
        df = self._clean_sheet(df)
        # add google index (row number) as a column
        df[dm.GOOGLE_INDEX] = "Offers" + df.index.astype(str)
        # rename columns to standard format
        df = df.rename(
            columns={
                _column(df, "#NAME"): dm.NAME,
                _column(df, "#DESCRIPTION"): dm.DESCRIPTION,
                _column(df, "#PHONENUMBERS"): dm.PHONENUMBERS,
                _column(df, "#EMAILS"): dm.EMAILS,
                _column(df, "#WEBURLS"): dm.WEBURLS,
                _column(df, "#ADDRESS"): dm.ADDRESS,
                _column(df, "#OPENWEEK"): dm.OPENWEEK,
                _column(df, "#OPENWEEKEND"): dm.OPENWEEKEND,
                _column(df, "#NEEDTOKNOW"): dm.NEEDTOKNOW,
                _column(df, "#MOREINFO"): dm.MOREINFO,
            }
        )
        # keep only relevant columns
        df = df[
            [
                dm.GOOGLE_INDEX,
                dm.CATEGORY,
                dm.SUBCATEGORY,
                dm.SLUG,
                dm.NAME,
                dm.DESCRIPTION,
                dm.PHONENUMBERS,
                dm.EMAILS,
                dm.WEBURLS,
                dm.ADDRESS,
                dm.OPENWEEK,
                dm.OPENWEEKEND,
                dm.NEEDTOKNOW,
                dm.MOREINFO,
                "text",
            ]
        ]
        # filter out offers without a name (e.g. placeholder rows with only IDs)
        df = df[df[dm.NAME].notna() & (df[dm.NAME].astype(str).str.strip() != "")]
        if df.empty:
            return df.assign(source="Offers")

        # combine fields into column text
        def combine_fields(row):
            text = row[dm.NAME] + ". "
            if not pd.isna(row[dm.DESCRIPTION]) and str(row[dm.DESCRIPTION]).strip() != "" and str(row[dm.DESCRIPTION]) != "nan":
                text = text + "Description: " + str(row[dm.DESCRIPTION]) + " "
            if not pd.isna(row[dm.PHONENUMBERS]) and str(row[dm.PHONENUMBERS]).strip() != "" and str(row[dm.PHONENUMBERS]) != "nan":
                text = text + "Phone Number: " + str(row[dm.PHONENUMBERS]) + ". "
            if not pd.isna(row[dm.EMAILS]) and str(row[dm.EMAILS]).strip() != "" and str(row[dm.EMAILS]) != "nan":
                text = text + "Email: " + str(row[dm.EMAILS]) + ". "
            if not pd.isna(row[dm.WEBURLS]) and str(row[dm.WEBURLS]).strip() != "" and str(row[dm.WEBURLS]) != "nan":
                text = text + "Link to Website: " + str(row[dm.WEBURLS]) + " "
            if not pd.isna(row[dm.ADDRESS]) and str(row[dm.ADDRESS]).strip() != "" and str(row[dm.ADDRESS]) != "nan":
                text = text + "Address: " + str(row[dm.ADDRESS]) + ". "
            if not pd.isna(row[dm.OPENWEEK]) and str(row[dm.OPENWEEK]).strip() != "" and str(row[dm.OPENWEEK]) != "nan":
                text = text + "Opening Hours: " + str(row[dm.OPENWEEK]) + ". "
            if not pd.isna(row[dm.OPENWEEKEND]) and str(row[dm.OPENWEEKEND]).strip() != "" and str(row[dm.OPENWEEKEND]) != "nan":
                text = text + "Opening Hours on Weekends: " + str(row[dm.OPENWEEKEND]) + ". "
            if not pd.isna(row[dm.NEEDTOKNOW]) and str(row[dm.NEEDTOKNOW]).strip() != "" and str(row[dm.NEEDTOKNOW]) != "nan":
                text = text + "What you need to know: " + str(row[dm.NEEDTOKNOW]) + " "
            if not pd.isna(row[dm.MOREINFO]) and str(row[dm.MOREINFO]).strip() != "" and str(row[dm.MOREINFO]) != "nan":
                text = text + "Further information: " + str(row[dm.MOREINFO])
            return text

        df["text"] = df.apply(combine_fields, axis=1)
        # clean text
        df["text"] = df["text"].str.replace(r"<[^<]+?>", "", regex=True)
        df["text"] = df["text"].str.replace(r"\n", " ", regex=True)
        df["text"] = df["text"].str.replace("**", "")
        df["text"] = df["text"].astype(str)
        df = df.dropna(subset=["text", dm.GOOGLE_INDEX])
        # add google sheet name
        df["source"] = "Offers"
        return df

    def _read_values(self) -> SheetValues:
        if self.document_type.lower() == "json":
            try:
                payload = _JsonValues.model_validate(self.document_data)
            except ValidationError:
                raise HTTPException(status_code=400, detail="Invalid JSON sheet values.") from None
            return {"Q&As": _normalize_values(payload.values)}
        if self.document_type.lower() != "googlesheet":
            raise HTTPException(status_code=500, detail="Unsupported document type.")

        retry = Retry(
            total=2, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods={"GET"}, respect_retry_after_header=False,
        )
        values: SheetValues = {}
        url = (
            "https://docs.google.com/spreadsheets/d/"
            f"{quote(self.document_id or '', safe='')}/gviz/tq"
        )
        try:
            with requests.Session() as session:
                session.mount("https://", HTTPAdapter(max_retries=retry))
                for name in ("Q&As", "Offers"):
                    response = session.get(
                        url,
                        params={"tqx": "out:csv", "headers": 1, "sheet": name},
                        timeout=(5, 30),
                    )
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
                    if content_type.strip().lower() != "text/csv":
                        raise HTTPException(status_code=502, detail="Invalid CSV response.")
                    text = response.content.decode("utf-8-sig")
                    rows = csv.reader(io.StringIO(text, newline=""), strict=True)
                    values[name] = _normalize_values(list(rows))
        except requests.RequestException:
            logger.warning("Google Sheet CSV request failed")
            raise HTTPException(
                status_code=502, detail="Could not read the Google Sheet."
            ) from None
        except (csv.Error, UnicodeDecodeError):
            logger.warning("Google Sheet CSV response is invalid")
            raise HTTPException(status_code=502, detail="Invalid CSV response.") from None
        return values

    def _to_dataframe(self, values: SheetValues | None = None) -> pd.DataFrame:
        if values is None:
            values = self._read_values()
        frames = []
        for name, rows in values.items():
            if not rows or not rows[0]:
                raise HTTPException(status_code=422, detail=f"Missing {name} headers.")
            # Cells under unlabelled columns (e.g. notes) are kept but never indexed.
            width = max(len(row) for row in rows)
            headers = rows[0] + [""] * (width - len(rows[0]))
            df = pd.DataFrame(
                [row + [""] * (width - len(row)) for row in rows[1:]],
                columns=headers,
            )
            columns = {tag: _column(df, tag) for tag in _HEADERS[name]}
            _validate_visible_rows(df, name, columns)
            if self.document_type.lower() == "json":
                df.index = range(1, len(df) + 1)
            try:
                frames.append(
                    self._clean_QnAs_sheet(df) if name == "Q&As"
                    else self._clean_Offers_sheet(df)
                )
            except (ValueError, TypeError, OverflowError):
                raise HTTPException(
                    status_code=422, detail=f"Invalid {name} category values."
                ) from None
        df = pd.concat(frames, ignore_index=True)
        return df.astype(object).where(pd.notna(df), None)

    def load_source(self) -> SourceSnapshot:
        values = self._read_values()
        df = self._to_dataframe(values)
        documents = DataFrameLoader(df, page_content_column="text").load()
        valid_documents = self._validate_loading(documents)
        if len(valid_documents) != len(documents):
            raise HTTPException(status_code=422, detail="Source contains unindexable entries.")
        digest = hashlib.sha256(
            json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        return SourceSnapshot(sheet_hash=digest, documents=valid_documents)

    @staticmethod
    def translate_document(document: Document) -> Document:
        language = detect_language(document.page_content, strict=True)
        text = document.page_content
        if language != "en":
            text = translate(from_lang=language, to_lang="en", text=text, strict=True)
        return document.model_copy(update={"page_content": text})

    def _load(self):
        return [self.translate_document(doc) for doc in self.load_source().documents]

    def _check_emptiness(self, page_content: str) -> bool:
        """
        Check if document content is empty of text, by evaluating whether there are at least 3 consecutive alphabet characters in there
        """
        min_consecutive_chars = 3
        for letter in page_content:
            if letter.isalpha():
                min_consecutive_chars -= 1
            else:
                min_consecutive_chars = 3
            if min_consecutive_chars == 0:
                return False
        return True

    def _validate_loading(self, documents: List[Document]):
        """
        Validates if the text from documents is loaded
        Logs all documents from which no text has been loaded
        Raises info if one or more documents is not loaded
        Return a new list without documents which didn't pass emptiness check
        """
        sources_not_loaded = []
        valid_documents = []
        for doc in documents:
            if self._check_emptiness(doc.page_content):
                logger.warning("Source entry contains no indexable text")
                sources_not_loaded.append(self.document_type.lower())
            else:
                valid_documents.append(doc)
        if sources_not_loaded:
            logger.info(
                f"Empty documents rendered from the following sources: {sources_not_loaded}"
            )
        return valid_documents

    def load(self) -> List[Document]:
        """
        Loads the documents
        Validates whether they are properly loaded
        """
        documents = self._load()
        valid_documents = self._validate_loading(documents)
        logger.info(f"Loaded {len(valid_documents)} documents")
        return valid_documents
