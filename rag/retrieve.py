"""Similarity search over the book collection."""
from __future__ import annotations

from dataclasses import dataclass

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


def search(question: str, top_k: int | None = None,
           min_similarity: float | None = None) -> list[Hit]:
    question = (question or "").strip()
    if not question:
        return []

    top_k = top_k or config.TOP_K
    floor = config.MIN_SIMILARITY if min_similarity is None else min_similarity

    coll = store.collection()
    if coll.count() == 0:
        return []

    # Query-side embedding is computed here so BGE's instruction prefix is
    # applied; passing query_texts would embed it as if it were a document.
    result = coll.query(
        query_embeddings=[store.embed_query(question)],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = 1.0 - float(distance)
        if similarity < floor:
            continue

        hits.append(Hit(
            rank=len(hits) + 1,
            similarity=similarity,
            text=text,
            title=metadata.get("title", "Untitled"),
            authors=metadata.get("authors", ""),
            year=metadata.get("year", ""),
            rating=float(metadata.get("rating") or 0.0),
            description=metadata.get("description", ""),
        ))

    return hits
