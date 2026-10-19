from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from utils.vector_store import get_vector_store
from utils.constants import DocumentMetadata
import json
import logging
import orjson
from typing import Any
from utils.translator import translate
from utils.tracing import observe
import os

logger = logging.getLogger(__name__)

dm = DocumentMetadata()

router = APIRouter()


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
        "14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04",
        description="""HIA Google Spreadsheet ID""",
    )
    k: int = Field(
        5,
        description="""Number of results to return""",
    )
    lang: str = Field(
        "en",
        description="""Language of the search query; results will be translated to this language""",
    )
    source: str = Field(
        "Q&As",
        description="""Source sheet of the search; can only be 'Q&As'""",
    )


def _build_results(docs_and_scores, vector_store) -> list[dict]:
    """Build results the way HIA likes them: parents with children, children promoted to their parent."""
    df = pd.DataFrame.from_records(
        [
            json.loads(doc["metadata"], strict=False)
            for doc in vector_store.get_documents()
        ]
    )  # load all documents from vector store, needed to find parent and child questions for results
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
    return results


def _retrieved_context(docs_and_scores) -> str:
    """Numbered Q&A block, same shape as the chat-turn span's `retrieved_context`."""
    return "\n\n".join(
        f"[{index}] Q: {doc.metadata[dm.QUESTION]}\nA: {doc.metadata[dm.ANSWER]}"
        for index, (doc, _score) in enumerate(docs_and_scores, start=1)
    )


def _record_search(span, payload: SearchPayload, original_query: str, docs_and_scores, results) -> None:
    """Write everything an observation-level evaluator needs onto the root span."""
    context = _retrieved_context(docs_and_scores)
    top_score = max((score for _doc, score in docs_and_scores), default=0.0)
    span.update(
        output=context,
        metadata={
            # same keys as chat-turn so one evaluator mapping serves both channels
            "search_query": payload.query,
            "retrieved_context": context,
            "original_query": original_query,
            "lang": payload.lang,
            "k": payload.k,
            "n_results": len(results),
            "top_score": top_score,
        },
    )
    # cheap deterministic scores: pre-filter for LLM judges and trendable on their own
    span.score(name="top_score", value=float(top_score), data_type="NUMERIC")
    span.score(name="n_results", value=float(len(results)), data_type="NUMERIC")
    span.score(name="zero_results", value=len(results) == 0, data_type="BOOLEAN")


# Plain `def`: embedding, search and translation are blocking HTTP calls and must
# run in the threadpool, not on the event loop.
@router.post("/search", tags=["search"])
def search(payload: SearchPayload):
    """Search HIA."""

    if payload.source not in ["Q&As"]:
        raise HTTPException(status_code=400, detail="Invalid source; must be 'Q&As'")

    # load vector store
    vector_store = get_vector_store(payload.googleSheetId, check_if_exists=True)

    # translate if necessary
    original_query = payload.query
    if payload.lang != "en":
        payload.query = translate(
            from_lang=payload.lang, to_lang="en", text=payload.query
        )

    tags = [f"sheet:{payload.googleSheetId}", "channel:search", f"lang:{payload.lang}"]
    with observe("search", project="search", input=payload.query, tags=tags) as span:
        docs_and_scores = vector_store.similarity_search_with_score(
            query=payload.query, k=payload.k
        )
        results = _build_results(docs_and_scores, vector_store)
        if span is not None:
            _record_search(span, payload, original_query, docs_and_scores, results)

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

    # Metadata only; the query text itself is not logged.
    logger.info(
        "search",
        extra={
            "googleSheetId": payload.googleSheetId,
            "lang": payload.lang,
            "k": payload.k,
            "query_chars": len(payload.query),
            "n_results": len(results),
        },
    )

    return ORJSONResponse(
        status_code=200,
        content={"results": results},
    )
