"""ChromaDB wrapper with local, quota-free embeddings.

Embeddings run in-process via fastembed (ONNX, no PyTorch — which matters on a
machine with little free RAM). A hosted embedding API meters every record, and
a 4,766-book catalogue exhausts a free tier in a single run; local embedding
has no cap, no key and no per-record cost.

Model: BAAI/bge-base-en-v1.5 (768-dim), set via EMBED_MODEL. Its 384-dim
sibling bge-small-en-v1.5 is the cheaper option — a third of the index size and
several times faster to embed. Either way it must be BGE: all-MiniLM-L6-v2 was
measured on this catalogue and failed badly at thematic search — for "a
dystopian novel about surveillance and government control" it put spy thrillers
on top and left 1984, Brave New World and The Handmaid's Tale outside the top
100. BGE puts them in the top three.

A collection is fixed-width, so changing EMBED_MODEL requires `--reset`.

BGE is an *asymmetric* model: documents are embedded plain, queries get an
instruction prefix. fastembed's `query_embed` applies it, so queries go through
`embed_query` here rather than Chroma's `query_texts`.
"""
from __future__ import annotations

import logging

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from fastembed import TextEmbedding

from . import config

_model: TextEmbedding | None = None


def model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(config.EMBED_MODEL)
    return _model


class LocalEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # batch_size is pinned rather than left to fastembed's default of 256.
        # Without it, memory depends on how many documents the *caller* hands
        # over, so a large upsert silently becomes a 7GB forward pass and the
        # process is OOM-killed. Bounding it here means no caller can do that.
        return [
            vector.tolist()
            for vector in model().embed(list(input), batch_size=config.EMBED_BATCH)
        ]

    def name(self) -> str:
        return f"fastembed-{config.EMBED_MODEL.split('/')[-1]}"


def embed_query(text: str) -> list[float]:
    """Query-side embedding, with BGE's retrieval instruction prefix applied."""
    return next(iter(model().query_embed([text]))).tolist()


_client = None
_collection = None

# Chroma calls posthog's old `capture(user_id, event, properties)`; posthog >= 6
# takes one positional argument, so every event raises a TypeError that Chroma
# logs. Telemetry is off below, but Chroma tries the call regardless, so the
# only way to keep the console readable on an env with a newer posthog is to
# silence the logger that reports it. requirements.txt pins the working version.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def client():
    """One client per process. Streamlit reruns the script on every
    interaction, so building a fresh client each time is pure overhead."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def collection(reset: bool = False):
    global _collection
    if _collection is not None and not reset:
        return _collection

    chroma = client()

    if reset:
        try:
            chroma.delete_collection(config.COLLECTION)
        except Exception:
            pass  # nothing to delete on a first run

    _collection = chroma.get_or_create_collection(
        name=config.COLLECTION,
        embedding_function=LocalEmbeddingFunction(),
        # BGE vectors are meant to be compared by cosine; Chroma defaults to L2,
        # which ranks differently.
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def count() -> int:
    try:
        return collection().count()
    except Exception:
        return 0
