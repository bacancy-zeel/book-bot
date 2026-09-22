"""Settings, all env-driven so the same code runs on a laptop or a server."""
import os
from dotenv import load_dotenv

load_dotenv()


def _int(name, default):
    return int(os.getenv(name, default))


def _float(name, default):
    return float(os.getenv(name, default))


BOOKS_CSV = os.getenv("BOOKS_CSV", "./data/book.csv")
# 768-dim BGE. Changing this changes the vector width, and a Pinecone index is
# fixed-width — switching models means a new index and a `--reset` rebuild.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "book-bot")
# Where a missing index gets created. Both are free-tier serverless defaults.
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# Namespaces partition one index; "" is Pinecone's default namespace.
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

# Only `rag.migrate` reads these — the source Chroma store it copies from.
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
COLLECTION = os.getenv("COLLECTION", "books")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
# Free-tier request quota is counted per model per day, so a sibling model
# is a fresh bucket. Tried in order when the primary returns 429.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite",
    ).split(",") if m.strip()
]

TOP_K = _int("TOP_K", 5)
# Chroma returns cosine *distance*; similarity is 1 - distance. Chunks below
# this are dropped so the model is not handed unrelated books. It is a coarse
# junk filter — the prompt is what actually makes the bot say "I don't know".
MIN_SIMILARITY = _float("MIN_SIMILARITY", 0.15)

# Where the ONNX model is cached. It must be writable, which on a host with a
# read-only filesystem means somewhere under /tmp.
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "/tmp/fastembed_cache")

# Rows embedded per forward pass. Local embedding has no quota, so this is
# purely a memory/throughput knob — but it is a steep one. A batch is padded to
# its longest member, and most blurbs here reach BGE's 512-token limit, so the
# transformer's activations cost roughly 20MB per row: 32 rows peak at ~1.7GB,
# 256 rows at ~7.5GB, which is what the Linux OOM killer ends. Raise it only if
# you have measured the headroom.
EMBED_BATCH = _int("EMBED_BATCH", 32)
