from typing import List

import urllib
from langchain_core.documents import Document
import pandas as pd
from utils.constants import DocumentMetadata
from utils.logger import logger
from langchain_community.document_loaders import DataFrameLoader
import uuid
from fastapi import HTTPException
from cleantext import clean
from utils.translator import translate, detect_language

dm = DocumentMetadata()


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
                next(c for c in df.columns if "#CATEGORY" in c): dm.CATEGORY,
                next(c for c in df.columns if "#SUBCATEGORY" in c): dm.SUBCATEGORY,
                next(c for c in df.columns if "#SLUG" in c): dm.SLUG,
                next(c for c in df.columns if "#VISIBLE" in c): "visible",
            }
        )
        # filter out rows with empty CATEGORY or SUBCATEGORY
        df = df[df[dm.CATEGORY].astype(str).str.strip() != ""]
        df = df[df[dm.SUBCATEGORY].astype(str).str.strip() != ""]
        df = df.dropna(subset=[dm.CATEGORY, dm.SUBCATEGORY])
        df[dm.CATEGORY] = df[dm.CATEGORY].astype(int)
        df[dm.SUBCATEGORY] = df[dm.SUBCATEGORY].astype(int)
        # filter out what's hidden
        df = df[
            ~df["visible"]
            .astype(str)
            .str.lower()
            .isin(["hide", "hidden", "no", "0", "-"])
        ].drop(columns=["visible"])
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
                next(c for c in df.columns if "#PARENT" in c): dm.PARENT,
                next(c for c in df.columns if "#QUESTION" in c): dm.QUESTION,
                next(c for c in df.columns if "#ANSWER" in c): dm.ANSWER,
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
                next(c for c in df.columns if "#NAME" in c): dm.NAME,
                next(c for c in df.columns if "#DESCRIPTION" in c): dm.DESCRIPTION,
                next(c for c in df.columns if "#PHONENUMBERS" in c): dm.PHONENUMBERS,
                next(c for c in df.columns if "#EMAILS" in c): dm.EMAILS,
                next(c for c in df.columns if "#WEBURLS" in c): dm.WEBURLS,
                next(c for c in df.columns if "#ADDRESS" in c): dm.ADDRESS,
                next(c for c in df.columns if "#OPENWEEK" in c): dm.OPENWEEK,
                next(c for c in df.columns if "#OPENWEEKEND" in c): dm.OPENWEEKEND,
                next(c for c in df.columns if "#NEEDTOKNOW" in c): dm.NEEDTOKNOW,
                next(c for c in df.columns if "#MOREINFO" in c): dm.MOREINFO,
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

        # combine fields into column text
        def combine_fields(row):
            text = row[dm.NAME] + ". "
            if ~pd.isna(row[dm.DESCRIPTION]) and str(row[dm.DESCRIPTION]).strip() != "" and str(row[dm.DESCRIPTION]) != "nan":
                text = text + "Description: " + str(row[dm.DESCRIPTION]) + " "
            if ~pd.isna(row[dm.PHONENUMBERS]) and str(row[dm.PHONENUMBERS]).strip() != "" and str(row[dm.PHONENUMBERS]) != "nan":
                text = text + "Phone Number: " + str(row[dm.PHONENUMBERS]) + ". "
            if ~pd.isna(row[dm.EMAILS]) and str(row[dm.EMAILS]).strip() != "" and str(row[dm.EMAILS]) != "nan":
                text = text + "Email: " + str(row[dm.EMAILS]) + ". "
            if ~pd.isna(row[dm.WEBURLS]) and str(row[dm.WEBURLS]).strip() != "" and str(row[dm.WEBURLS]) != "nan":
                text = text + "Link to Website: " + str(row[dm.WEBURLS]) + " "
            if ~pd.isna(row[dm.ADDRESS]) and str(row[dm.ADDRESS]).strip() != "" and str(row[dm.ADDRESS]) != "nan":
                text = text + "Address: " + str(row[dm.ADDRESS]) + ". "
            if ~pd.isna(row[dm.OPENWEEK]) and str(row[dm.OPENWEEK]).strip() != "" and str(row[dm.OPENWEEK]) != "nan":
                text = text + "Opening Hours: " + str(row[dm.OPENWEEK]) + ". "
            if ~pd.isna(row[dm.OPENWEEKEND]) and str(row[dm.OPENWEEKEND]).strip() != "" and str(row[dm.OPENWEEKEND]) != "nan":
                text = text + "Opening Hours on Weekends: " + str(row[dm.OPENWEEKEND]) + ". "
            if ~pd.isna(row[dm.NEEDTOKNOW]) and str(row[dm.NEEDTOKNOW]).strip() != "" and str(row[dm.NEEDTOKNOW]) != "nan":
                text = text + "What you need to know: " + str(row[dm.NEEDTOKNOW]) + " "
            if ~pd.isna(row[dm.MOREINFO]) and str(row[dm.MOREINFO]).strip() != "" and str(row[dm.MOREINFO]) != "nan":
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

    def _to_dataframe(self):
        """
        Loads a pandas DataFrame based on the document type. Google Sheet and JSON are currently supported.
        """
        if self.document_type.lower() == "googlesheet":
            logger.info(f"Loading {self.document_id} from Google Sheet.")
            # first load Q&A sheet
            url = f"https://docs.google.com/spreadsheets/d/{self.document_id}/gviz/tq?tqx=out:csv&sheet=Q%26As"
            try:
                df_QnAs = pd.read_csv(url)
            except urllib.error.HTTPError as e:
                raise HTTPException(
                    status_code=e.code,
                    detail=f"Could not load Q&A from Google Sheet with ID {self.document_id}: {e}",
                )
            df_QnAs = self._clean_QnAs_sheet(df_QnAs)
            # then load Offers sheet and append to Q&A
            url = f"https://docs.google.com/spreadsheets/d/{self.document_id}/gviz/tq?tqx=out:csv&sheet=Offers"
            try:
                df_Offers = pd.read_csv(url)
            except urllib.error.HTTPError as e:
                raise HTTPException(
                    status_code=e.code,
                    detail=f"Could not load Offers from Google Sheet with ID {self.document_id}: {e}",
                )
            df_Offers = self._clean_Offers_sheet(df_Offers)
            df = pd.concat([df_QnAs, df_Offers], ignore_index=True)

        elif self.document_type.lower() == "json":
            logger.info("Loading from JSON.")
            # N.B. only Q&A content is supported through json
            if not self.document_data:
                raise HTTPException(
                    status_code=400, detail="No document data provided."
                )
            df = pd.DataFrame(self.document_data["values"])
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
            df = self._clean_QnAs_sheet(df)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Loader of document type {self.document_type} not available.",
            )
        return df

    def _load(self):
        df = self._to_dataframe()

        # Translate content to English
        def translate_row(row):
            detected_lang = detect_language(row["text"])
            if detected_lang != "en":
                translated_text = translate(
                    from_lang=detected_lang, to_lang="en", text=row["text"]
                )
                return translated_text
            else:
                return row["text"]

        df["text"] = df.apply(translate_row, axis=1)
        # map to langchain doc
        documents = DataFrameLoader(df, page_content_column="text").load()
        return documents

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
                logger.warning(
                    f"Was not able to extract text from document {doc.metadata}; removed from the outputs."
                )
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
