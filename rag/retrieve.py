"""Similarity search over the book collection.

LangChain's `similarity_search_with_relevance_scores` reads `hnsw:space` off
the collection and picks `1 - distance` as its relevance score, which is the
same number this project has always thresholded on — so MIN_SIMILARITY keeps
its meaning.
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

    Returns bare Documents, so the UI's relevance bars use `search()` instead.
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

    total = store.count()
    if total == 0:
        return []

    scored = store.vectorstore().similarity_search_with_relevance_scores(
        question, k=min(top_k, total)
    )

    hits = []
    for document, score in scored:
        if score < floor:
            continue
        hits.append(as_hit(document, float(score), len(hits) + 1))

    return hits
