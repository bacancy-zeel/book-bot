"""Book Bot — a RAG chatbot over a book catalogue, served by FastAPI.

Pipeline: CSV -> one Document per book -> local ONNX embeddings -> ChromaDB
cosine search -> Gemini writes a grounded answer citing the books it used.

Only generation touches a hosted API. Embedding runs locally, so indexing the
whole catalogue costs nothing and is not rate limited.

The HTTP layer is deliberately thin: it owns request shapes and Markdown
rendering, and everything else lives in `rag/` and `history.py`, which know
nothing about the web. `GET /` serves the chat page; the JSON endpoints under
`/api` are the same surface the page uses, so anything else can drive the bot
too — see the generated docs at `/docs`.

Endpoints are plain `def`, not `async def`: embedding a query and calling
Gemini both block, and FastAPI runs a sync endpoint in a worker thread instead
of stalling the event loop. History opens a SQLite connection per call, which
is what makes that safe.
"""
from __future__ import annotations

import html
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import markdown
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import history
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
# Anchored to this file rather than the working directory, so `uvicorn
# main:app` behaves the same whatever directory it is launched from.
HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

history.init()


# --------------------------------------------------------------------------
# request/response models — these are what /docs documents
# --------------------------------------------------------------------------
class Ask(BaseModel):
    question: str = Field(min_length=1, description="The user's question.")
    conversation_id: str | None = Field(
        default=None,
        description="Continues that conversation. Omit to start a new one.",
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
    conversation_id: str
    title: str
    reply: str
    reply_html: str
    sources: list[Source]


class Message(BaseModel):
    role: str
    content: str
    content_html: str
    sources: list[Source]


class Conversation(BaseModel):
    id: str
    title: str
    updated_at: float
    group: str


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def render(text: str) -> str:
    """Markdown -> HTML, with the model's own output escaped first.

    Escaping before conversion means a stray `<script>` in an answer stays
    visible text; only the tags Markdown itself emits reach the page.
    """
    return markdown.markdown(html.escape(text), extensions=["sane_lists"])


def group_of(timestamp: float) -> str:
    day = datetime.fromtimestamp(timestamp).date()
    days = (datetime.now().date() - day).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return "Previous 7 days"
    if days < 30:
        return "Previous 30 days"
    return "Older"


def as_sources(hits) -> list[dict]:
    """Hits arrive as dataclasses live, and as plain dicts out of history."""
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


def require(conversation_id: str) -> None:
    if not history.exists(conversation_id):
        raise HTTPException(status_code=404, detail="No such conversation.")


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


@app.get("/api/conversations", response_model=list[Conversation],
         summary="Every conversation, most recently used first")
def conversations() -> list[dict]:
    return [
        {"id": c["id"], "title": c["title"], "updated_at": c["updated_at"],
         "group": group_of(c["updated_at"])}
        for c in history.conversations()
    ]


@app.get("/api/conversations/{conversation_id}", response_model=list[Message],
         summary="The turns of one conversation")
def messages(conversation_id: str) -> list[dict]:
    require(conversation_id)
    out = []
    for message in history.ConversationHistory(conversation_id).messages:
        text = str(message.content)
        out.append({
            "role": "user" if message.type == "human" else "assistant",
            "content": text,
            "content_html": render(text) if message.type != "human" else "",
            "sources": as_sources(message.additional_kwargs.get("hits")),
        })
    return out


@app.post("/api/chat", response_model=Reply, summary="Ask a question")
def chat(ask: Ask) -> dict:
    question = ask.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Ask something.")

    conversation_id = ask.conversation_id
    if conversation_id:
        require(conversation_id)
    else:
        conversation_id = history.create(history.title_from(question))

    chat_history = history.ConversationHistory(conversation_id)
    # Read the thread before this turn is added, so the model sees the
    # conversation as it stood when the question was asked.
    thread = chat_history.messages
    chat_history.add_user_message(question)

    hits = search(question)
    try:
        reply = answer(question, hits, thread)
    except GenerationError as error:
        reply = (
            f"**Could not generate an answer.** {error}\n\n"
            "Retrieval still worked — the matching books are listed below."
        )

    sources = as_sources(hits)
    chat_history.add_messages([history.answer_message(reply, sources)])

    return {
        "conversation_id": conversation_id,
        "title": history.title_from(question),
        "reply": reply,
        "reply_html": render(reply),
        "sources": sources,
    }


@app.delete("/api/conversations/{conversation_id}", status_code=204,
            summary="Delete one conversation")
def delete(conversation_id: str) -> None:
    require(conversation_id)
    history.delete(conversation_id)


@app.delete("/api/conversations", status_code=204,
            summary="Delete every conversation")
def delete_all() -> None:
    history.delete_all()
