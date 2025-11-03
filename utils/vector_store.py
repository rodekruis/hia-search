from __future__ import annotations
import re
import copy
from pathlib import Path
from typing import List
from fastapi import HTTPException
from chromadb import PersistentClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswParameters,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from utils.logger import logger
from utils.constants import DocumentMetadata
import os

DEFAULT_HUGGING_FACE_MODEL = "sentence-transformers/all-mpnet-base-v2"


dm = DocumentMetadata()


def googleid_to_vectorstoreid(googleid: str) -> str:
    """
    Convert a Google Sheet ID to a valid vector store ID:
    * 2-128 characters, lowercase
    * only letters, numbers and dashes ("-")
    * first character must be a letter or number
    * no consecutive dashes
    * example: hia-faq-ukraine-en
    """
    googleid = re.sub(r"[^a-z0-9-]", "", googleid.lower())
    googleid = googleid[:128]
    return googleid


class VectorStore:
    """
    Vector storage for chunked documents and embeddings
        1. Embeds chunked documents
        2. Save to vector store
    """

    output: List[Document]

    def __init__(
        self,
        store_path: str,
        store_service: str,
        store_password: str = None,
        embedding_source: str = None,
        embedding_model: str = None,
        store_id: str = "chunked_document_embeddings",
    ):
        self.store_id = store_id
        self.embedding_source = embedding_source
        self.embedding_model = embedding_model
        self.store_service = store_service
        self.store_password = store_password
        self.store_path = store_path
        self.embedder = self._set_embedder()
        self.client = self._set_client()
        self.langchain_client = self._set_langchain_client()

    def _set_embedder(self):
        """Sets the document embedder based on the input embedding model or embedding source.
        If no embedding model is given, a default model is used.
        """
        if self.embedding_source.lower() == "openai":
            return AzureOpenAIEmbeddings(
                deployment=self.embedding_model,
                chunk_size=1,
            )

        elif self.embedding_source.lower() == "huggingface":
            if self.embedding_model is None:
                self.embedding_model = DEFAULT_HUGGING_FACE_MODEL
            return HuggingFaceEmbeddings(model_name=self.embedding_model)

        else:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding source {self.embedding_source} not available. Only embedding models from 'HuggingFace' or 'OpenAI' are currently available.",
            )

    def _create_azuresearch_index(self):
        """Create a new index in Azure Search"""
        client = SearchIndexClient(
            self.store_path, AzureKeyCredential(self.store_password)
        )
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(self.embedder.embed_query("Text")),
                vector_search_profile_name="HnswProfile",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="Hnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="HnswProfile",
                    algorithm_configuration_name="Hnsw",
                )
            ],
        )
        client.create_index(
            SearchIndex(name=self.store_id, fields=fields, vector_search=vector_search)
        )

    def _set_client(self):
        """Sets the vector store client"""
        if self.store_service.lower() == "chroma":
            return PersistentClient(db_path=Path(self.store_path))
        elif self.store_service.lower() == "azuresearch":
            return SearchClient(
                self.store_path,
                index_name=self.store_id,
                credential=AzureKeyCredential(self.store_password),
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Vectore store {self.store_service} not available. Only 'chroma' or 'azuresearch' are currently available.",
            )

    def _add_embedding_model_to_metadata(self, metadatas: list) -> List[dict]:
        """
        Add the embedding model used to embed the documents to the metadata
        """
        new_metadatas = []
        for metadata in metadatas:
            new_metadata = copy.deepcopy(metadata)
            new_metadata[dm.EMBEDDING_MODEL] = self.embedding_model
            new_metadatas.append(new_metadata)
        return new_metadatas

    def _set_langchain_client(self):
        """Set the vector store langchain client"""
        if self.store_service.lower() == "chroma":
            logger.info("Initializing ChromaDB")
            _ = self.client.get_or_create_collection(self.store_id)
            return Chroma(
                store_id=self.store_id,
                embedding_function=self.embedder,
                client=self.client,
            )
        elif self.store_service.lower() == "azuresearch":
            return AzureSearch(
                azure_search_endpoint=self.store_path,
                azure_search_key=self.store_password,
                index_name=self.store_id,
                embedding_function=self.embedder.embed_query,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Vectore store {self.store_service} not available. Only 'chroma' or 'azuresearch' are currently available.",
            )

    def add_documents(self, chunked_documents: List[Document]) -> int:
        """
        Add new incoming chunked documents to the vector store
        If the collection/index specified by store_id is not empty, replace all content
        Add metadata regarding the embedding model
        """
        n_docs_added = 0
        if len(chunked_documents) > 0:
            n_docs_in_collection = 0
            if self.store_service.lower() == "chroma":
                n_docs_in_collection = self.client.get_or_create_collection(
                    self.store_id
                ).count()
            elif self.store_service.lower() == "azuresearch":
                n_docs_in_collection = self.client.get_document_count()

            if n_docs_in_collection > 0:
                logger.info(
                    f"Vector store already contains {n_docs_in_collection} documents. Replacing everything."
                )
                if self.store_service.lower() == "chroma":
                    self.client.delete_collection(self.store_id)
                    _ = self.client.get_or_create_collection(self.store_id)
                elif self.store_service.lower() == "azuresearch":
                    index_client = SearchIndexClient(
                        self.store_path, AzureKeyCredential(self.store_password)
                    )
                    index_client.delete_index(self.store_id)
                    self._create_azuresearch_index()

            n_docs_added = len(chunked_documents)
            logger.info(f"Adding {n_docs_added} new incoming chunked documents")
            documents = []
            metadatas = []
            ids = []
            for doc in chunked_documents:
                documents.append(doc.page_content)
                metadatas.append(doc.metadata)
                ids.append(
                    f"{doc.metadata[dm.GOOGLE_INDEX]}_{doc.metadata[dm.NTH_CHUNK]}"
                )
            metadatas = self._add_embedding_model_to_metadata(metadatas)
            self.langchain_client.add_texts(
                texts=documents, metadatas=metadatas, ids=ids
            )

            return n_docs_added

    def count_documents(self) -> int:
        """Count the number of documents in the vector store"""
        n_docs_in_collection = None
        if self.store_service.lower() == "chroma":
            n_docs_in_collection = self.client.get_or_create_collection(
                self.store_id
            ).count()
        elif self.store_service.lower() == "azuresearch":
            n_docs_in_collection = self.client.get_document_count()
        return n_docs_in_collection

    def get_documents(self) -> List[Document]:
        """Get all documents from the vector store"""
        if self.store_service.lower() == "chroma":
            docs = self.client.get_all_documents()
        elif self.store_service.lower() == "azuresearch":
            docs = [d for d in self.client.search(search_text="*")]
        return docs

    def similarity_search(self, query: str, k: int) -> List[Document]:
        """Search for similar documents in the vector store"""
        return self.langchain_client.similarity_search(query=query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int
    ) -> List[(Document, float)]:
        """Search for similar documents in the vector store and return with scores"""
        return self.langchain_client.similarity_search_with_score(query=query, k=k)


def get_vector_store(vector_store_id: str) -> VectorStore:
    """Get vector store from Azure Search."""
    try:
        vector_store = VectorStore(
            store_path=os.environ["VECTOR_STORE_ADDRESS"],
            store_service="azuresearch",
            store_password=os.environ["VECTOR_STORE_PASSWORD"],
            embedding_source="OpenAI",
            embedding_model=os.environ["MODEL_EMBEDDINGS"],
            store_id=vector_store_id,
        )
    except Exception as ex:
        raise HTTPException(
            status_code=400,
            detail=f"Vector store {vector_store_id} not found, did you create it?",
        )
    return vector_store
