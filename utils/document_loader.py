import copy
from collections import OrderedDict
from typing import List
from langchain.schema import Document
import pandas as pd
from utils.constants import DocumentMetadata
from utils.logger import logger
from langchain_community.document_loaders import DataFrameLoader
import uuid
import re
from cleantext import clean

dm = DocumentMetadata()


def uuid_hash(content: str) -> str:
    """Create a unique hash from the page content of a document"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))


class DocumentLoader:
    """
    Document loading class that:
        1. Loads documents from path
        2. Uses the hash of the page content to create a unique documentID
        3. Returns list of Langchain Document

    Input
    --------
    document_path:
        str
        Location of the raw document(s)
    document_id:
        str
        Unique identifier of the document
    document_type:
        str
        Data type of document(s)
    kwargs:
        dict
        Input parameters that are conditional to the document type

    Output
    --------
    The output is a list in Document type. Each element has two attributes:
    page_content:
        str
        Content of documents
    metadata:
        dictionary
        Information of each page_content
    """

    output: List[Document]

    def __init__(
        self,
        document_type: str,
        document_path: str = None,
        document_id: str = None,
        **kwargs: dict,
    ):
        self.document_path = document_path
        self.document_id = document_id
        self.document_type = document_type
        self.__dict__.update(kwargs)
        self.loader = self._set_loader()

    def _set_loader(self):
        """
        Instantiates a document loader based on the document type. PDF and Excel are allowed to be loaded
        PDF document is loaded from the BaseLoader langchain class
        Excel document is loaded from a custom loder inherented from the langchain class
        """
        if self.document_type.lower() == "googlesheet":
            logger.info("Loading from Google Sheet with pandas.")
            sheet_name = "Q%26As"
            url = f"https://docs.google.com/spreadsheets/d/{self.document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            return url
        else:
            raise NotImplementedError(
                f"Loader of document type {self.document_type} not available. Only loading of 'googlesheet' is currently implemented."
            )

    def _to_dataframe(self):
        if self.document_type.lower() == "googlesheet":
            df = pd.read_csv(self.loader)
            df[dm.GOOGLE_INDEX] = df.index
            df = df[df["Visible?\n#VISIBLE"] == "Show"]  # only visible entries
        else:
            raise NotImplementedError(
                f"Loader of document type {self.document_type} not available. Only loading of 'googlesheet' is currently implemented."
            )
        return df

    def _load(self):
        df = self._to_dataframe()

        # clean text
        df = df.rename(
            columns={
                "The Answer (can be multi-line)\n#ANSWER": dm.ANSWER,
                "The Question (should be 1 line)\n#QUESTION": dm.QUESTION,
            }
        )
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
        df = df.rename(
            columns={
                "Category ID\n#CATEGORY": dm.CATEGORY,
                "Sub-Category ID\n#SUBCATEGORY": dm.SUBCATEGORY,
                "Unique name/part of URL\n#SLUG": dm.SLUG,
                "Name/Slug of parent Question\n#PARENT": dm.PARENT,
            }
        )
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
        logger.info(f"Started loading")
        documents = self._load()
        valid_documents = self._validate_loading(documents)
        logger.info(f"Loaded {len(valid_documents)} documents")
        return valid_documents
