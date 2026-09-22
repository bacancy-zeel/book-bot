"""Persistent chat history for the Book Bot UI.

Conversations live in a small SQLite file of their own — `st.session_state` is
wiped by a browser refresh or a server restart, so a sidebar built on it would
lose every past chat. SQLite keeps the store single-file and stdlib-only.

The per-conversation view is a LangChain `BaseChatMessageHistory`, so the
stored turns drop straight into a prompt's MessagesPlaceholder. The retrieved
books that backed an answer ride along in the message's `additional_kwargs`,
where the UI reads them back to redraw its source cards.

A connection is opened per call rather than cached: Streamlit reruns the script
on its own threads, and a shared sqlite3 connection is not thread-safe.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from typing import Sequence

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# This module is imported before rag.config, which is what normally reads the
# .env — without this, HISTORY_DB set there would be ignored.
load_dotenv()

DB_PATH = os.getenv("HISTORY_DB", "./chat_history.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    hits            TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation
    ON messages (conversation_id, id);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")  # so deleting a chat drops its turns
    return con


def init() -> None:
    with _connect() as con:
        con.executescript(SCHEMA)


def title_from(question: str, limit: int = 44) -> str:
    """A sidebar label built from the first thing the user asked."""
    text = re.sub(r"\s+", " ", (question or "").strip()) or "New chat"
    return text if len(text) <= limit else text[:limit].rstrip(" ,.;:") + "…"


def create(title: str) -> str:
    conversation_id = uuid.uuid4().hex
    now = time.time()
    with _connect() as con:
        con.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (conversation_id, title, now, now),
        )
    return conversation_id


def conversations() -> list[dict]:
    """Every chat, most recently used first."""
    with _connect() as con:
        rows = con.execute(
            "SELECT id, title, created_at, updated_at FROM conversations"
            " ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def exists(conversation_id: str | None) -> bool:
    if not conversation_id:
        return False
    with _connect() as con:
        row = con.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


class ConversationHistory(BaseChatMessageHistory):
    """The turns of one conversation, as LangChain messages."""

    def __init__(self, conversation_id: str | None) -> None:
        self.conversation_id = conversation_id

    @property
    def messages(self) -> list[BaseMessage]:
        if not self.conversation_id:
            return []  # a chat that has not been started yet
        with _connect() as con:
            rows = con.execute(
                "SELECT role, content, hits FROM messages"
                " WHERE conversation_id = ? ORDER BY id",
                (self.conversation_id,),
            ).fetchall()

        out: list[BaseMessage] = []
        for row in rows:
            hits = json.loads(row["hits"]) if row["hits"] else []
            if row["role"] == "user":
                out.append(HumanMessage(content=row["content"]))
            else:
                out.append(AIMessage(content=row["content"],
                                     additional_kwargs={"hits": hits}))
        return out

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not self.conversation_id:
            raise ValueError("Create the conversation before adding messages.")

        now = time.time()
        with _connect() as con:
            for message in messages:
                hits = message.additional_kwargs.get("hits")
                con.execute(
                    "INSERT INTO messages"
                    " (conversation_id, role, content, hits, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (self.conversation_id,
                     "user" if message.type == "human" else "assistant",
                     str(message.content),
                     json.dumps(hits) if hits else None, now),
                )
            # updated_at drives the sidebar ordering, so a reopened chat floats
            # up.
            con.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, self.conversation_id),
            )

    def clear(self) -> None:
        if not self.conversation_id:
            return
        with _connect() as con:
            con.execute("DELETE FROM messages WHERE conversation_id = ?",
                        (self.conversation_id,))


def answer_message(text: str, hits: list[dict] | None = None) -> AIMessage:
    """An assistant turn carrying the books its answer was grounded in."""
    return AIMessage(content=text, additional_kwargs={"hits": hits or []})


def rename(conversation_id: str, title: str) -> None:
    with _connect() as con:
        con.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )


def delete(conversation_id: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def delete_all() -> None:
    with _connect() as con:
        con.execute("DELETE FROM conversations")
