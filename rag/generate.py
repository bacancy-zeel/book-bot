"""Grounded answer generation.

Only this step calls a hosted model, and only once per question — which is why
a free-tier key is enough here even though it could never have embedded the
catalogue.
"""
from __future__ import annotations

import google.generativeai as genai

from . import config
from .retrieve import Hit

SYSTEM_PROMPT = """You are a book assistant. Answer using ONLY the numbered
books in the CONTEXT block below.

Rules:
- Cite every book you mention with its bracketed number, e.g. "Dune [2]".
  For several, write "[1][3]" — each number in its own brackets.
- If the context does not answer the question, say so plainly. Never invent a
  book, an author, a year or a rating that is not in the context.
- When recommending, say briefly *why* each book fits what was asked.
- Be concise. Markdown is supported.
"""

NO_CONTEXT_PROMPT = """You are a book assistant, but nothing in the catalogue
matched this question. Say so plainly and suggest the user rephrase or ask
about a different subject. Do not answer from memory and do not cite anything.
"""


class GenerationError(RuntimeError):
    pass


def _is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "quota" in text or "rate limit" in text


def _chain() -> list[str]:
    """Primary model first, then the fallbacks, without repeats."""
    seen, chain = set(), []
    for name in [config.GEMINI_CHAT_MODEL, *config.GEMINI_FALLBACK_MODELS]:
        if name and name not in seen:
            seen.add(name)
            chain.append(name)
    return chain


def _context(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        facts = " | ".join(p for p in [
            f"by {hit.authors}" if hit.authors else "",
            hit.year, f"rating {hit.rating}" if hit.rating else "",
        ] if p)
        blocks.append(
            f"[{hit.rank}] {hit.title} ({facts})\n"
            f"{hit.description or hit.text}"
        )
    rule = "=" * 60
    return f"CONTEXT\n{rule}\n" + f"\n\n{'-' * 60}\n\n".join(blocks) + f"\n{rule}\nEND CONTEXT"


def answer(question: str, hits: list[Hit], history: list[dict] | None = None) -> str:
    if not config.GEMINI_API_KEY:
        raise GenerationError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=config.GEMINI_API_KEY)

    instruction = SYSTEM_PROMPT if hits else NO_CONTEXT_PROMPT
    if hits:
        instruction = f"{instruction}\n\n{_context(hits)}"

    turns = [
        {"role": "user" if t["role"] == "user" else "model",
         "parts": [t["content"]]}
        for t in (history or [])
    ]

    chain = _chain()
    for name in chain:
        model = genai.GenerativeModel(name, system_instruction=instruction)
        try:
            chat = model.start_chat(history=turns)
            return chat.send_message(question).text
        except Exception as error:  # provider errors vary too much to enumerate
            if _is_quota_error(error):
                continue  # daily quota is per model — try the next one
            # Only the first line; the rest is a protobuf dump.
            raise GenerationError(str(error).strip().splitlines()[0]) from error

    raise GenerationError(
        "Daily free-tier quota is used up on every configured model "
        f"({', '.join(chain)}). Try again tomorrow, set GEMINI_CHAT_MODEL to "
        "another model, or enable billing on the API key."
    )
