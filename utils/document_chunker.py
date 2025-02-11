import copy
from pathlib import Path
from typing import List
import re

from langchain.schema import Document
from langchain.text_splitter import SpacyTextSplitter
from utils.logger import logger
from utils.constants import DocumentMetadata


dm = DocumentMetadata()


class DocumentChunker:
    """
    Document chunking class that:
        1. Chunk the documents into smaller pieces, so that it's ready for embedding.
        2. Add metadata on chunk level: nth number of chunk in document, chunk ID
        3. Returns chunked documents with metadata

    Input
    --------
    documents:
        List[Document]
        list of loaded documents
    chunking_strategy:
        str
        chunking strategy, f.e. tokenized sentence chunking
    kwargs:
        TypedDict
        input parameters that are conditional to the chunking strategy
    """

    output: List[Document]

    def __init__(self, chunking_strategy: str, **kwargs: dict):
        self.chunking_strategy = chunking_strategy
        self.__dict__.update(kwargs)
        self.chunker = self._set_chunker()

    def _set_chunker(self):
        """instantiates a document chunker based on the document strategy.
        Custom chunking strategies can be added by creating one that inherets from the ** langchain class.
        and adding them here.
        """
        if self.chunking_strategy.lower() == "sentencesplitting":
            """Splitting text using Spacy package."""
            logger.info("Using Langchain SpacyTextSplitter for chunking the documents")
            return SpacyTextSplitter(
                chunk_size=self.kwargs["chunk_size"],
                chunk_overlap=self.kwargs["chunk_overlap"],
                separator=self.kwargs.get("separator", "\n\n"),
                pipeline=self.kwargs.get("pipeline", "en_core_web_sm"),
            )
        elif self.chunking_strategy.lower() == "tokenizedsentencesplitting":
            """Splitting text using Spacy package."""
            logger.info("Using Langchain SpacyTextSplitter for chunking the documents")
            return SpacyTextSplitter.from_tiktoken_encoder(
                chunk_size=self.kwargs["chunk_size"],
                chunk_overlap=self.kwargs["chunk_overlap"],
                separator=self.kwargs.get("separator", "\n\n"),
                pipeline=self.kwargs.get("pipeline", "en_core_web_sm"),
                encoding_name=self.kwargs.get("encoding_name", "cl100k_base"),
            )
        else:
            raise NotImplementedError(
                f"Chunking strategy {self.chunking_strategy} not available. Only 'SentenceSplitting' is currently implemented."
            )

    def _get_urls_from_page_content(self, document: Document) -> str:
        """
        Filters out URLs in page_content, adds URLs to document metadata if exists. Returns string, because metadata from Langchain Document does not accept list format.
        """
        # regular expresson to match URLs
        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # merge all urls into a string, stored as "urls" in metadata
        urls = [
            url.split("\\", 1)[0] for url in url_pattern.findall(document.page_content)
        ]
        if urls:
            return "\n ".join(urls)
        else:
            return ""

    def _add_chunk_metadata(self, chunked_documents: List[Document]) -> List[Document]:
        """
        Adds information to metadata about the ordering index of the chunk within a document ('nth chunk')
        Logs number of tokens
        """
        chunked_docs_with_metadata: List[Document] = []
        google_index = None
        nth_chunk = 0
        for chunked_doc in chunked_documents:

            # if the previous chunk was in the same document (based on the document id of the chunk), this is the
            # n+1th chunk in the document
            if google_index == chunked_doc.metadata[dm.GOOGLE_INDEX]:
                nth_chunk += 1
            # if the previous chunk was in a different document, this is the first chunk of the document
            else:
                nth_chunk = 0

            # creating a deep copy to prevent the original dict from being updated
            new_metadata = copy.deepcopy(chunked_doc.metadata)
            new_metadata[dm.NTH_CHUNK] = nth_chunk
            # new_metadata[dm.URLS] = self._get_urls_from_page_content(chunked_doc)
            doc_with_new_metadata = Document(
                page_content=chunked_doc.page_content, metadata=new_metadata
            )
            chunked_docs_with_metadata.append(doc_with_new_metadata)
            google_index = chunked_doc.metadata[dm.GOOGLE_INDEX]

        return chunked_docs_with_metadata

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits the documents into chunks according to chunking strategy
        Adds metadata
        """
        chunked_documents = self.chunker.split_documents(documents=documents)
        return self._add_chunk_metadata(chunked_documents)
