"""CSV -> one record per book -> embedded into Chroma.

Two decisions worth stating:

1. **One document per book, not fixed-size chunks.** A book row is already a
   self-contained record. Splitting on a token window would cut one blurb in
   half and glue it to an unrelated book, so "a novel about a desert planet"
   would retrieve fragments instead of books.

2. **Resumable by construction.** Each book gets a deterministic id and is
   written with `upsert`, so re-running skips what is already stored instead
   of starting over. An interrupted run costs you nothing.
"""
from __future__ import annotations

import math
import sys

import pandas as pd

from . import config, store



def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _year(value) -> str:
    """1997.0 -> 1997; pandas reads an int column with blanks as float."""
    text = _clean(value)
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def load_books(path: str, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    if limit:
        frame = frame.head(limit)
    return frame


def build_record(row: pd.Series, columns: list[str]) -> tuple[str, dict]:
    """Return the text to embed plus the metadata kept alongside it.

    Only meaning-bearing prose is embedded. Publication year, rating, ids and
    language codes are stored as metadata but deliberately kept OUT of the
    vector: they are identical boilerplate on every record, so they pull all
    the vectors toward a common centroid and wash out the signal that actually
    distinguishes one book from another.

    It matters more than it looks, because all-MiniLM-L6-v2 truncates at 256
    tokens. Thirty tokens of "Published: / Average rating: / Book id:" at the
    front push that much of the blurb straight out of the window.
    """
    title = _clean(row.get("title"))
    authors = _clean(row.get("authors"))
    year = _year(row.get("original_publication_year"))
    rating = _clean(row.get("average_rating"))
    description = _clean(row.get("description"))

    # Title and author first as light context, then straight into the blurb,
    # which is what makes "a book about grief" work at all.
    lead = " by ".join(p for p in [title, authors] if p)
    embed_text = ". ".join(p for p in [lead, description] if p)

    metadata = {
        "title": title or "Untitled",
        "authors": authors,
        "year": year,
        "rating": float(rating) if rating.replace(".", "", 1).isdigit() else 0.0,
        # Shown in the UI and given to the model, but not embedded.
        "description": description[:1500],
    }
    return embed_text, metadata


def book_id(row: pd.Series, position: int) -> str:
    raw = _clean(row.get("book_id")) or _clean(row.get("goodreads_book_id"))
    return f"book-{raw or position}"


def ingest(path: str | None = None, limit: int | None = None,
           reset: bool = False, progress=None) -> dict:
    path = path or config.BOOKS_CSV
    frame = load_books(path, limit)
    columns = list(frame.columns)
    coll = store.collection(reset=reset)

    existing = set()
    if not reset:
        # Resume: ask Chroma what it already has rather than re-embedding it.
        try:
            existing = set(coll.get(include=[])["ids"])
        except Exception:
            existing = set()

    ids, documents, metadatas = [], [], []
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        identifier = book_id(row, position)
        if identifier in existing:
            continue

        text, metadata = build_record(row, columns)
        if not text.strip():
            continue

        ids.append(identifier)
        documents.append(text)
        metadatas.append(metadata)

    total = len(ids)
    if total == 0:
        return {"added": 0, "skipped": len(frame), "total": coll.count()}

    for start in range(0, total, config.EMBED_BATCH):
        stop = min(start + config.EMBED_BATCH, total)
        coll.upsert(
            ids=ids[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
        )
        if progress:
            progress(stop, total)

    return {"added": total, "skipped": len(frame) - total, "total": coll.count()}


if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    limit = int(positional[0]) if positional else None
    reset = "--reset" in sys.argv

    def show(done, total):
        print(f"  embedded {done}/{total}", flush=True)

    print(f"Ingesting {config.BOOKS_CSV} (limit={limit or 'all'}, reset={reset})")
    print(ingest(limit=limit, reset=reset, progress=show))
