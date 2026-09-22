"""Vector store: a Pinecone serverless index reached through LangChain.

Embeddings run in-process via fastembed (ONNX, no PyTorch — which matters on a
machine with little free RAM). A hosted embedding API meters every record, and
a 4,766-book catalogue exhausts a free tier in a single run; local embedding
has no cap, no key and no per-record cost. Only the vectors are hosted.

Model: BAAI/bge-base-en-v1.5 (768-dim), set via EMBED_MODEL. Its 384-dim
sibling bge-small-en-v1.5 is the cheaper option — a third of the index size and
several times faster to embed. Either way it must be BGE: all-MiniLM-L6-v2 was
measured on this catalogue and failed badly at thematic search — for "a
dystopian novel about surveillance and government control" it put spy thrillers
on top and left 1984, Brave New World and The Handmaid's Tale outside the top
100. BGE puts them in the top three.

An index is fixed-width, so changing EMBED_MODEL means a new index.

BGE is an *asymmetric* model: documents are embedded plain, queries get an
instruction prefix. `FastEmbedEmbeddings` applies that split — `embed_query`
goes through fastembed's `query_embed`, `embed_documents` through its plain
`embed`.
"""
from __future__ import annotations

import logging
from typing import Callable

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from . import config

logger = logging.getLogger(__name__)

DIMENSION = 768
# Where a document's own text is kept in Pinecone's metadata. Pinecone stores
# vectors and metadata, not documents, so the text has to ride along.
TEXT_KEY = "text"

_embeddings: FastEmbedEmbeddings | None = None
_client: Pinecone | None = None
_index = None
_vectorstore: PineconeVectorStore | None = None


class BookVectorStore(PineconeVectorStore):
    """Pinecone scores a cosine match as a *similarity*, which is already the
    number MIN_SIMILARITY is measured against. The base class assumes a
    distance and returns `1 - score`, which would invert the ranking."""

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda score: score


def embeddings() -> FastEmbedEmbeddings:
    """The ONNX model, loaded once — it costs a second or two and ~400MB."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name=config.EMBED_MODEL,
            batch_size=config.EMBED_BATCH,
        )
    return _embeddings


def client() -> Pinecone:
    global _client
    if _client is None:
        if not config.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        _client = Pinecone(api_key=config.PINECONE_API_KEY)
    return _client


def index(create: bool = True):
    """The Pinecone index, created on first use if it is missing."""
    global _index
    if _index is not None:
        return _index

    pinecone = client()
    if create and not pinecone.has_index(config.PINECONE_INDEX):
        pinecone.create_index(
            name=config.PINECONE_INDEX,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD,
                                region=config.PINECONE_REGION),
        )

    _index = pinecone.Index(config.PINECONE_INDEX)
    return _index


def vectorstore(reset: bool = False) -> PineconeVectorStore:
    global _vectorstore
    if _vectorstore is not None and not reset:
        return _vectorstore

    if reset:
        try:
            index().delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
        except Exception:
            pass  # nothing to delete in an empty namespace

    _vectorstore = BookVectorStore(
        index=index(),
        embedding=embeddings(),
        text_key=TEXT_KEY,
        namespace=config.PINECONE_NAMESPACE,
    )
    return _vectorstore


def count() -> int:
    try:
        stats = index(create=False).describe_index_stats()
        if config.PINECONE_NAMESPACE:
            namespace = stats.get("namespaces", {}).get(
                config.PINECONE_NAMESPACE, {})
            return int(namespace.get("vector_count", 0))
        return int(stats.get("total_vector_count", 0))
    except Exception as error:
        # The page reads this to decide whether the catalogue is empty, so a
        # missing key or an unreachable index must not take the app down — but
        # it should say so rather than look like an empty catalogue.
        logger.warning("Could not read the Pinecone index: %s", error)
        return 0
