"""Book Bot — a RAG chatbot over a book catalogue, served by FastAPI.

Pipeline: CSV -> one Document per book -> local ONNX embeddings -> Pinecone
cosine search -> Gemini writes a grounded answer citing the books it used.

Nothing is stored server-side. A conversation lives in the page that is having
it: the browser keeps the turns so far and sends them with the next question,
which is what lets a follow-up be understood, and closing the tab ends it.

`GET /` serves the chat page; the JSON endpoints under `/api` are the same
surface the page uses, so anything else can drive the bot too — see the
generated docs at `/docs`.

Endpoints are plain `def`, not `async def`: embedding a query and calling
Gemini both block, and FastAPI runs a sync endpoint in a worker thread instead
of stalling the event loop.
"""
from __future__ import annotations

import html
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import markdown
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from rag import store
from rag.generate import GenerationError, answer
from rag.retrieve import search

EXAMPLES = [
    "Recommend a dystopian novel about surveillance",
    "What are the highest rated fantasy books?",
    "A book about grief that isn't depressing",
    "Who wrote The Hunger Games?",
]

app = FastAPI(
    title="Book Bot",
    description="Ask questions about a book catalogue; answers cite the books "
                "they were built from.",
    version="1.0.0",
)

# Anchored to this file rather than the working directory, so the app behaves
# the same whatever directory it is launched from.
HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=HERE / "templates")

# `public/` is what a static host serves straight off its CDN, and such a host
# has no reason to ship it to the application as well — so it may simply not be
# there. Mounting it unconditionally would end the process at import, since
# StaticFiles checks its directory up front. Serve it when it is present, which
# is what makes a local run self-contained.
STATIC = HERE / "public" / "static"
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


# --------------------------------------------------------------------------
# request/response models — these are what /docs documents
# --------------------------------------------------------------------------
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Ask(BaseModel):
    question: str = Field(min_length=1, description="The user's question.")
    history: list[Turn] = Field(
        default_factory=list,
        description="Earlier turns of this conversation, oldest first. The "
                    "server keeps none of its own, so a follow-up question "
                    "only makes sense if they are sent back.",
    )


class Source(BaseModel):
    rank: int
    similarity: float
    title: str
    authors: str
    year: str
    rating: float
    description: str


class Reply(BaseModel):
    reply: str
    reply_html: str
    sources: list[Source]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def render(text: str) -> str:
    """Markdown -> HTML, with the model's own output escaped first.

    Escaping before conversion means a stray `<script>` in an answer stays
    visible text; only the tags Markdown itself emits reach the page.
    """
    return markdown.markdown(html.escape(text), extensions=["sane_lists"])


def as_messages(history: list[Turn]) -> list[BaseMessage]:
    return [
        HumanMessage(content=turn.content) if turn.role == "user"
        else AIMessage(content=turn.content)
        for turn in history
    ]


def as_sources(hits) -> list[dict]:
    out = []
    for hit in hits or []:
        hit = hit if isinstance(hit, dict) else asdict(hit)
        out.append({
            "rank": hit.get("rank", 0),
            "similarity": float(hit.get("similarity") or 0.0),
            "title": hit.get("title", ""),
            "authors": hit.get("authors", ""),
            "year": hit.get("year", ""),
            "rating": float(hit.get("rating") or 0.0),
            "description": hit.get("description", ""),
        })
    return out


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "examples": EXAMPLES,
        "books": store.count(),
    })


# --------------------------------------------------------------------------
# api
# --------------------------------------------------------------------------
@app.get("/api/status", summary="Catalogue size")
def status() -> dict:
    return {"books": store.count()}


@app.post("/api/chat", response_model=Reply, summary="Ask a question")
def chat(ask: Ask) -> dict:
    question = ask.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Ask something.")

    hits = search(question)
    try:
        reply = answer(question, hits, as_messages(ask.history))
    except GenerationError as error:
        reply = (
            f"**Could not generate an answer.** {error}\n\n"
            "Retrieval still worked — the matching books are listed below."
        )

    return {
        "reply": reply,
        "reply_html": render(reply),
        "sources": as_sources(hits),
    }

# --- temporary routing probe; remove once the host's path handling is known ---
@app.api_route("/{probe_path:path}", methods=["GET"], include_in_schema=False)
def _probe(request: Request, probe_path: str):
    scope = request.scope
    return {
        "path": scope.get("path"),
        "raw_path": str(scope.get("raw_path")),
        "root_path": scope.get("root_path"),
        "query": str(scope.get("query_string")),
        "vercel_headers": {k: v for k, v in request.headers.items()
                           if "vercel" in k.lower() or k.lower() in
                           ("x-forwarded-path", "x-original-uri", "host")},
    }
