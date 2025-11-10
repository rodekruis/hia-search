from __future__ import annotations

import pandas as pd
from fastapi import (
    Depends,
    APIRouter,
    HTTPException,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from utils.vector_store import VectorStore, googleid_to_vectorstoreid, get_vector_store
from utils.constants import DocumentMetadata
import json
from utils.logger import logger
import orjson
from typing import Any
from utils.translator import translate
import os

dm = DocumentMetadata()

router = APIRouter()

key_query_scheme = APIKeyHeader(name="Authorization")


def get_score_google_index(docs_and_scores, google_index: str) -> float:
    """Get the maximum score for a given google_index."""
    scores = [0.0]
    for doc_and_score in docs_and_scores:
        doc = doc_and_score[0]
        score = doc_and_score[1]
        if doc.metadata["google_index"] == google_index:
            scores.append(score)
    return max(scores)


class ORJSONResponse(JSONResponse):
    """Custom JSONResponse class that uses orjson for serialization."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


class SearchPayload(BaseModel):
    """Search payload."""

    query: str = Field(
        ...,
        description="""Text of the search query""",
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
        description="""Language of the search query; results will be translated to this language""",
    )


@router.post("/search")
async def search(payload: SearchPayload, api_key: str = Depends(key_query_scheme)):
    """Search HIA."""

    if api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # load vector store
    vector_store = get_vector_store(payload.googleSheetId)

    logger.info(f"Search query: '{payload.query}'")

    # translate if necessary
    if payload.lang != "en":
        payload.query = translate(
            from_lang=payload.lang, to_lang="en", text=payload.query
        )
        logger.info(
            f"Search query translated from {payload.lang} to en: '{payload.query}'"
        )

    # retrieve documents
    docs_and_scores = vector_store.similarity_search_with_score(
        query=payload.query, k=payload.k
    )

    # build results they way HIA likes them
    df = pd.DataFrame.from_records(
        [
            json.loads(doc["metadata"], strict=False)
            for doc in vector_store.get_documents()
        ]
    )  # load all documents from vector store
    results = []
    for doc_and_score in docs_and_scores:
        doc = doc_and_score[0]
        score = doc_and_score[1]

        result = {
            dm.CATEGORY: doc.metadata[dm.CATEGORY],
            dm.SUBCATEGORY: doc.metadata[dm.SUBCATEGORY],
            dm.SLUG: doc.metadata[dm.SLUG],
            dm.QUESTION: doc.metadata[dm.QUESTION],
            dm.ANSWER: doc.metadata[dm.ANSWER],
            dm.SCORE: score,
            dm.CHILDREN: None,
            dm.GOOGLE_INDEX: doc.metadata[dm.GOOGLE_INDEX],
        }

        # if result is a parent question, add children
        if result[dm.SLUG]:
            children = []
            df_children = df[df[dm.PARENT] == result[dm.SLUG]]
            for ix, row in df_children.iterrows():
                children.append(
                    {
                        dm.CATEGORY: row[dm.CATEGORY],
                        dm.SUBCATEGORY: row[dm.SUBCATEGORY],
                        dm.QUESTION: row[dm.QUESTION],
                        dm.ANSWER: row[dm.ANSWER],
                        dm.SCORE: get_score_google_index(
                            docs_and_scores, row[dm.GOOGLE_INDEX]
                        ),
                    }
                )
            if len(children) > 0:
                result[dm.CHILDREN] = children

        # if result is a child question, add parent and siblings
        if doc.metadata[dm.PARENT]:
            parent = df[df[dm.SLUG] == doc.metadata[dm.PARENT]].to_dict(
                orient="records"
            )
            if len(parent) > 0:
                parent = parent[0]
                children = []
                df_children = df[df[dm.PARENT] == parent[dm.SLUG]]
                for ix, row in df_children.iterrows():
                    children.append(
                        {
                            dm.CATEGORY: row[dm.CATEGORY],
                            dm.SUBCATEGORY: row[dm.SUBCATEGORY],
                            dm.QUESTION: row[dm.QUESTION],
                            dm.ANSWER: row[dm.ANSWER],
                            dm.SCORE: get_score_google_index(
                                docs_and_scores, row[dm.GOOGLE_INDEX]
                            ),
                        }
                    )
                result = {
                    dm.CATEGORY: parent[dm.CATEGORY],
                    dm.SUBCATEGORY: parent[dm.SUBCATEGORY],
                    dm.SLUG: parent[dm.SLUG],
                    dm.QUESTION: parent[dm.QUESTION],
                    dm.ANSWER: parent[dm.ANSWER],
                    dm.SCORE: get_score_google_index(
                        docs_and_scores, doc.metadata[dm.GOOGLE_INDEX]
                    ),
                    dm.CHILDREN: children,
                    dm.GOOGLE_INDEX: parent[dm.GOOGLE_INDEX],
                }

        results.append(result)

    # keep only unique results
    results = list({v[dm.GOOGLE_INDEX]: v for v in results}.values())
    # remove google_index from results
    for result in results:
        result.pop(dm.GOOGLE_INDEX)

    # translate results if necessary
    if payload.lang != "en":
        for result in results:
            result[dm.QUESTION] = translate(
                from_lang="en", to_lang=payload.lang, text=result[dm.QUESTION]
            )
            result[dm.ANSWER] = translate(
                from_lang="en", to_lang=payload.lang, text=result[dm.ANSWER]
            )
            if result["children"]:
                for child in result["children"]:
                    child[dm.QUESTION] = translate(
                        from_lang="en", to_lang=payload.lang, text=child[dm.QUESTION]
                    )
                    child[dm.ANSWER] = translate(
                        from_lang="en", to_lang=payload.lang, text=child[dm.ANSWER]
                    )

    return ORJSONResponse(
        status_code=200,
        content={"results": results},
    )
