"""Similarity search over the book catalogue.

Pinecone scores a cosine match between -1 and 1, higher being closer, and that
is the number MIN_SIMILARITY is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from . import config, store


@dataclass
class Hit:
    rank: int
    similarity: float
    text: str
    title: str
    authors: str
    year: str
    rating: float
    description: str


def as_hit(document: Document, score: float, rank: int) -> Hit:
    metadata = document.metadata
    return Hit(
        rank=rank,
        similarity=score,
        text=document.page_content,
        title=metadata.get("title", "Untitled"),
        authors=metadata.get("authors", ""),
        year=metadata.get("year", ""),
        rating=float(metadata.get("rating") or 0.0),
        description=metadata.get("description", ""),
    )


def retriever(top_k: int | None = None,
              min_similarity: float | None = None) -> VectorStoreRetriever:
    """The same search as `search()`, as a Runnable for use inside a chain.

    Yields bare Documents; `search()` is the one that carries scores.
    """
    return store.vectorstore().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k or config.TOP_K,
            "score_threshold": (config.MIN_SIMILARITY if min_similarity is None
                                else min_similarity),
        },
    )


def search(question: str, top_k: int | None = None,
           min_similarity: float | None = None) -> list[Hit]:
    question = (question or "").strip()
    if not question:
        return []

    top_k = top_k or config.TOP_K
    floor = config.MIN_SIMILARITY if min_similarity is None else min_similarity

    # No size check first: an empty index simply returns no matches, and
    # every extra call here is a network round trip.
    scored = store.vectorstore().similarity_search_with_relevance_scores(
        question, k=top_k
    )

    hits = []
    for document, score in scored:
        if score < floor:
            continue
        hits.append(as_hit(document, float(score), len(hits) + 1))

    return hits
