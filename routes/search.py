from __future__ import annotations
from fastapi import (
    Depends,
    APIRouter,
    HTTPException,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from utils.vector_store import VectorStore, googleid_to_vectorstoreid
from utils.constants import DocumentMetadata
from utils.document_loader import DocumentLoader
from utils.logger import logger
import orjson
from typing import Any
from time import perf_counter
from utils.translator import translate
import os

dm = DocumentMetadata()

router = APIRouter()

key_query_scheme = APIKeyHeader(name="Authorization")

azure_search_index_client = SearchIndexClient(
    os.environ["VECTOR_STORE_ADDRESS"],
    AzureKeyCredential(os.environ["VECTOR_STORE_PASSWORD"]),
)

# initialize vector stores
vector_dbs = {}
for index in azure_search_index_client.list_indexes():
    vector_dbs[index.name] = VectorStore(
        store_path=os.environ["VECTOR_STORE_ADDRESS"],
        store_service="azuresearch",
        store_password=os.environ["VECTOR_STORE_PASSWORD"],
        embedding_source="OpenAI",
        embedding_model=os.environ["MODEL_EMBEDDINGS"],
        store_id=index.name,
    )


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


class SearchPayload(BaseModel):
    query: str = Field(
        ...,
        description="""Text of the question""",
    )
    googleSheetId: str = Field(
        ...,
        description="""HIA Google sheet ID""",
    )
    k: int = Field(
        5,
        description="""Number of results to return""",
    )
    lang: str = Field(
        "en",
        description="""Language of the query; results will be translated to this language""",
    )


def get_score_google_index(docs_and_scores, google_index: str):
    scores = [0.0]
    for doc_and_score in docs_and_scores:
        doc = doc_and_score[0]
        score = doc_and_score[1]
        if doc.metadata["google_index"] == google_index:
            scores.append(score)
    return max(scores)


@router.post("/search")
async def search(payload: SearchPayload, api_key: str = Depends(key_query_scheme)):
    """Search in HIA."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # load document from Google Sheet
    t2_start = perf_counter()
    doc_loader = DocumentLoader(
        document_type="googlesheet", document_id=payload.googleSheetId
    )
    df = doc_loader._to_dataframe()
    t2_stop = perf_counter()
    logger.info(f"Elapsed time loading dataframe: {float(t2_stop - t2_start)} seconds")

    # load vector store
    vector_store_id = googleid_to_vectorstoreid(payload.googleSheetId)
    if vector_store_id not in vector_dbs:
        # try again in case the vector store was just added
        vector_store_ids = [i.name for i in azure_search_index_client.list_indexes()]
        if vector_store_id in vector_store_ids:
            vector_dbs[vector_store_id] = VectorStore(
                store_path=os.environ["VECTOR_STORE_ADDRESS"],
                store_service="azuresearch",
                store_password=os.environ["VECTOR_STORE_PASSWORD"],
                embedding_source="OpenAI",
                embedding_model=os.environ["MODEL_EMBEDDINGS"],
                store_id=vector_store_id,
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Vector store {vector_store_id} not found."
            )

    # translate if necessary
    if payload.lang != "en":
        logger.info(f"Translating query from {payload.lang} to en")
        payload.query = translate(
            from_lang=payload.lang, to_lang="en", text=payload.query
        )

    # retrieve documents
    t2_start = perf_counter()
    logger.info(f"Searching for {payload.k} results with query {payload.query}")
    docs_and_scores = vector_dbs[vector_store_id].similarity_search_with_score(
        query=payload.query, k=payload.k
    )

    t2_stop = perf_counter()
    logger.info(
        f"Elapsed time retrieving documents: {float(t2_stop - t2_start)} seconds"
    )
    logger.info(len(docs_and_scores))

    # build results they way HIA likes them
    t2_start = perf_counter()
    results = []
    for doc_and_score in docs_and_scores:
        doc = doc_and_score[0]
        score = doc_and_score[1]

        result = {
            "category": doc.metadata[dm.CATEGORY],
            "subcategory": doc.metadata[dm.SUBCATEGORY],
            "slug": doc.metadata[dm.SLUG],
            "question": doc.metadata[dm.QUESTION],
            "answer": doc.metadata[dm.ANSWER],
            "score": score,
            "children": None,
            "google_index": doc.metadata[dm.GOOGLE_INDEX],
        }

        # if doc is a parent question, add children
        if result[dm.SLUG]:
            children = []
            df_child = df[
                df["Name/Slug of parent Question\n#PARENT"] == result[dm.SLUG]
            ]
            for ix, row in df_child.iterrows():
                children.append(
                    {
                        "question": row["The Question (should be 1 line)\n#QUESTION"],
                        "answer": row["The Answer (can be multi-line)\n#ANSWER"],
                        "score": get_score_google_index(
                            docs_and_scores, row[dm.GOOGLE_INDEX]
                        ),
                    }
                )
            if len(children) > 0:
                result["children"] = children

        # if doc is a child question, add parent and siblings
        if doc.metadata[dm.PARENT]:
            parent = df[
                df["Unique name/part of URL\n#SLUG"] == doc.metadata[dm.PARENT]
            ].to_dict(orient="records")
            if len(parent) > 0:
                parent = parent[0]
                children = []
                df_child = df[
                    df["Name/Slug of parent Question\n#PARENT"]
                    == parent["Unique name/part of URL\n#SLUG"]
                ]
                for ix, row in df_child.iterrows():
                    children.append(
                        {
                            "question": row[
                                "The Question (should be 1 line)\n#QUESTION"
                            ],
                            "answer": row["The Answer (can be multi-line)\n#ANSWER"],
                            "score": get_score_google_index(
                                docs_and_scores, row[dm.GOOGLE_INDEX]
                            ),
                        }
                    )
                result = {
                    "category": parent["Category ID\n#CATEGORY"],
                    "subcategory": parent["Sub-Category ID\n#SUBCATEGORY"],
                    "slug": parent["Unique name/part of URL\n#SLUG"],
                    "question": parent["The Question (should be 1 line)\n#QUESTION"],
                    "answer": parent["The Answer (can be multi-line)\n#ANSWER"],
                    "score": get_score_google_index(
                        docs_and_scores, doc.metadata[dm.GOOGLE_INDEX]
                    ),
                    "children": children,
                    "google_index": parent[dm.GOOGLE_INDEX],
                }

        results.append(result)

    logger.info(f"results: {results}")

    # keep only unique results
    results = list({v["google_index"]: v for v in results}.values())
    # remove google_index from results
    for result in results:
        result.pop("google_index")

    # translate results if necessary
    if payload.lang != "en":
        logger.info(f"Translating result from en to {payload.lang}")
        for result in results:
            result["question"] = translate(
                from_lang="en", to_lang=payload.lang, text=result["question"]
            )
            result["answer"] = translate(
                from_lang="en", to_lang=payload.lang, text=result["answer"]
            )
            if result["children"]:
                for child in result["children"]:
                    child["question"] = translate(
                        from_lang="en", to_lang=payload.lang, text=child["question"]
                    )
                    child["answer"] = translate(
                        from_lang="en", to_lang=payload.lang, text=child["answer"]
                    )

    t2_stop = perf_counter()
    logger.info(f"Elapsed time preparing results: {float(t2_stop - t2_start)} seconds")
    logger.info(f"{len(results)} results found")
    logger.info(f"results: {results}")

    return ORJSONResponse(
        status_code=200,
        content={"results": results},
    )
