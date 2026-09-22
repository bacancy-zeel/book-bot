"""Vector store: LangChain's Chroma wrapper over a local, persistent DB.

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
instruction prefix. `FastEmbedEmbeddings` honours that split — `embed_query`
calls fastembed's `query_embed`, `embed_documents` its plain `embed` — so the
LangChain interface preserves the behaviour the retrieval quality depends on.
"""
from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from . import config

# Chroma calls posthog's old `capture(user_id, event, properties)`; posthog >= 6
# takes one positional argument, so every event raises a TypeError that Chroma
# logs. Telemetry is off below, but Chroma tries the call regardless, so the
# only way to keep the console readable on an env with a newer posthog is to
# silence the logger that reports it. requirements.txt pins the working version.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

_embeddings: FastEmbedEmbeddings | None = None
_client = None
_vectorstore: Chroma | None = None


def embeddings() -> FastEmbedEmbeddings:
    """The ONNX model, loaded once — it costs a second or two and ~400MB."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name=config.EMBED_MODEL,
            batch_size=config.EMBED_BATCH,
        )
    return _embeddings


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


def vectorstore(reset: bool = False) -> Chroma:
    global _vectorstore
    if _vectorstore is not None and not reset:
        return _vectorstore

    _vectorstore = Chroma(
        client=client(),
        collection_name=config.COLLECTION,
        embedding_function=embeddings(),
        # BGE vectors are meant to be compared by cosine; Chroma defaults to L2,
        # which ranks differently. LangChain reads this back off the collection
        # to pick `1 - distance` as its relevance score.
        collection_metadata={"hnsw:space": "cosine"},
    )
    if reset:
        _vectorstore.reset_collection()
    return _vectorstore


def count() -> int:
    try:
        return vectorstore()._collection.count()
    except Exception:
        return 0
