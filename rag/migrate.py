"""One-off copy of a local Chroma collection into Pinecone.

The vectors already exist on disk, so nothing is re-embedded here: each record
is read out of Chroma with its embedding and written straight to Pinecone under
the same id. A 4,766-book catalogue moves in a couple of minutes without the
embedding model ever being loaded.

    python -m rag.migrate            # copy everything
    python -m rag.migrate --reset    # clear the Pinecone namespace first

Records are read in pages and written in batches: the whole catalogue is about
15MB of float32, and Pinecone caps a single upsert at 2MB.
"""
from __future__ import annotations

import sys

import chromadb
from chromadb.config import Settings

from . import config, store

READ_PAGE = 500
UPSERT_BATCH = 100


def source_collection():
    client = chromadb.PersistentClient(
        path=config.CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(config.COLLECTION)


def records(collection):
    """Yield (id, embedding, metadata, document) a page at a time."""
    total = collection.count()
    for offset in range(0, total, READ_PAGE):
        page = collection.get(
            include=["embeddings", "metadatas", "documents"],
            limit=READ_PAGE,
            offset=offset,
        )
        for identifier, embedding, metadata, document in zip(
            page["ids"], page["embeddings"], page["metadatas"], page["documents"]
        ):
            yield identifier, embedding, metadata, document


def migrate(reset: bool = False, progress=None) -> dict:
    collection = source_collection()
    total = collection.count()
    if total == 0:
        return {"copied": 0, "source": 0}

    index = store.index()
    if reset:
        try:
            index.delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
        except Exception:
            pass  # nothing to delete in an empty namespace

    copied, batch = 0, []
    for identifier, embedding, metadata, document in records(collection):
        batch.append({
            "id": identifier,
            "values": [float(value) for value in embedding],
            # Pinecone stores vectors and metadata, not documents, so the
            # embedded text rides along under the store's text key.
            "metadata": {**metadata, store.TEXT_KEY: document},
        })

        if len(batch) >= UPSERT_BATCH:
            index.upsert(vectors=batch, namespace=config.PINECONE_NAMESPACE)
            copied += len(batch)
            batch = []
            if progress:
                progress(copied, total)

    if batch:
        index.upsert(vectors=batch, namespace=config.PINECONE_NAMESPACE)
        copied += len(batch)
        if progress:
            progress(copied, total)

    return {"copied": copied, "source": total}


if __name__ == "__main__":
    reset = "--reset" in sys.argv

    def show(done, total):
        print(f"  copied {done}/{total}", flush=True)

    print(f"Copying {config.CHROMA_DIR}:{config.COLLECTION} "
          f"-> Pinecone index {config.PINECONE_INDEX} (reset={reset})")
    print(migrate(reset=reset, progress=show))
