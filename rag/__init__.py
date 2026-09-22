"""Retrieval pipeline: ingest, store, search, generate.

The cache locations below are set here because this package is imported before
anything reaches fastembed or huggingface_hub, and those read the environment
once at import and never look again. Both default to a directory under $HOME,
which is read-only on a host that gives you exactly one writable path — so
fetching the embedding model fails there with "Read-only file system" even
though fastembed's own cache_dir points somewhere fine. setdefault, so a
machine with a real home directory can still point them wherever it likes.
"""
import os

os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_XET_CACHE", "/tmp/huggingface/xet")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
