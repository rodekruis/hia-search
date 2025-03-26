from typing import List
from langchain.schema import Document
import pandas as pd
from utils.constants import DocumentMetadata
from utils.logger import logger
from langchain_community.document_loaders import DataFrameLoader
import uuid
from fastapi import HTTPException
from cleantext import clean

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

    def _to_dataframe(self):
        """
        Loads a pandas DataFrame based on the document type. Google Sheet and JSON are currently supported.
        """
        if self.document_type.lower() == "googlesheet":
            logger.info(f"Loading {self.document_id} from Google Sheet.")
            sheet_name = "Q%26As"
            url = f"https://docs.google.com/spreadsheets/d/{self.document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            df = pd.read_csv(url)
        elif self.document_type.lower() == "json":
            logger.info("Loading from JSON.")
            if not self.document_data:
                raise HTTPException(
                    status_code=400, detail="No document data provided."
                )
            df = pd.DataFrame(self.document_data["values"])
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Loader of document type {self.document_type} not available.",
            )
        df[dm.GOOGLE_INDEX] = df.index  # add google index (row number) as a column
        # rename columns to standard format
        df = df.rename(
            columns={
                next(c for c in df.columns if "#CATEGORY" in c): dm.CATEGORY,
                next(c for c in df.columns if "#SUBCATEGORY" in c): dm.SUBCATEGORY,
                next(c for c in df.columns if "#SLUG" in c): dm.SLUG,
                next(c for c in df.columns if "#PARENT" in c): dm.PARENT,
                next(c for c in df.columns if "#QUESTION" in c): dm.QUESTION,
                next(c for c in df.columns if "#ANSWER" in c): dm.ANSWER,
                next(c for c in df.columns if "#VISIBLE" in c): "visible",
            }
        )
        df = df[df["visible"] != "Hide"]  # keep what's not hidden
        return df

    def _load(self):
        df = self._to_dataframe()

        # clean text
        df["text"] = df[dm.QUESTION] + " " + df[dm.ANSWER]
        df["text"] = df["text"].str.replace(r"<[^<]+?>", "", regex=True)
        df["text"] = df["text"].str.replace(r"\n", " ", regex=True)
        df["text"] = df["text"].apply(
            lambda x: clean(
                x,
                lower=False,
                no_line_breaks=True,
                no_urls=True,
                no_emoji=True,
            )
        )
        df["text"] = df["text"].str.replace("[<URL>]", "")
        df["text"] = df["text"].str.replace("(<URL>)", "")
        df["text"] = df["text"].str.replace("<URL>", "")
        df["text"] = df["text"].str.replace("**", "")
        df["text"] = df["text"].astype(str)

        # extra filters
        df = df.dropna(subset=["text", dm.GOOGLE_INDEX, dm.CATEGORY, dm.SUBCATEGORY])
        df[dm.CATEGORY] = df[dm.CATEGORY].astype(int)
        df[dm.SUBCATEGORY] = df[dm.SUBCATEGORY].astype(int)
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
