"""Vercel entry point.

The platform looks for an ASGI app in this module and routes every request to
it; `vercel.json` sends the whole path space here, so FastAPI's own router
decides what each URL means. Nothing else lives in this file — running locally
should exercise the same application object.
"""
import sys
from pathlib import Path

# On Vercel this file is imported from inside api/, so the project root has to
# be on the path before `main` can be found.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402

__all__ = ["app"]
