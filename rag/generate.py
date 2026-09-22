"""Grounded answer generation, as an LCEL chain.

Only this step calls a hosted model, and only once per question — which is why
a free-tier key is enough here even though it could never have embedded the
catalogue.

The chain is `prompt | model | StrOutputParser`, one per configured model,
stacked with `with_fallbacks` so a model that has burned through its daily
free-tier quota hands over to the next one. Fallbacks fire on quota errors
only: a bad key or a malformed request should surface immediately rather than
be retried three times over.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

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

# The instruction is passed as a *value*, not as template text, so braces
# inside a book blurb are never mistaken for template variables.
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{instruction}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


class GenerationError(RuntimeError):
    pass


class _QuotaExceeded(RuntimeError):
    """Raised for a 429 so `with_fallbacks` can tell it from a real failure."""


def _is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "quota" in text or "rate limit" in text


def _models() -> list[str]:
    """Primary model first, then the fallbacks, without repeats."""
    seen, names = set(), []
    for name in [config.GEMINI_CHAT_MODEL, *config.GEMINI_FALLBACK_MODELS]:
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _guarded(runnable: Runnable, name: str) -> Runnable:
    """Sort a model's failures into 'try the next model' and 'give up'."""
    def invoke(payload: dict) -> str:
        try:
            return runnable.invoke(payload)
        except Exception as error:  # provider errors vary too much to enumerate
            if _is_quota_error(error):
                raise _QuotaExceeded(name) from error
            # Only the first line; the rest is a protobuf dump.
            raise GenerationError(str(error).strip().splitlines()[0]) from error

    return RunnableLambda(invoke)


@lru_cache(maxsize=1)
def chain() -> Runnable:
    """Built once per process; the models are fixed by config at startup."""
    links = [
        _guarded(PROMPT | ChatGoogleGenerativeAI(
            model=name,
            google_api_key=config.GEMINI_API_KEY,
            # The SDK retries a 429 six times by default, which on an exhausted
            # daily quota is half a minute of waiting for an answer that is
            # never coming. Falling through to the next model is this module's
            # whole strategy, so surface the 429 at once and let it.
            max_retries=0,
        ) | StrOutputParser(), name)
        for name in _models()
    ]
    first, *rest = links
    return first.with_fallbacks(rest, exceptions_to_handle=(_QuotaExceeded,))


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


def answer(question: str, hits: list[Hit],
           history: list[BaseMessage] | None = None) -> str:
    if not config.GEMINI_API_KEY:
        raise GenerationError("GEMINI_API_KEY is not set.")

    instruction = SYSTEM_PROMPT if hits else NO_CONTEXT_PROMPT
    if hits:
        instruction = f"{instruction}\n\n{_context(hits)}"

    try:
        return chain().invoke({
            "instruction": instruction,
            "question": question,
            "history": history or [],
        })
    except _QuotaExceeded as error:
        raise GenerationError(
            "Daily free-tier quota is used up on every configured model "
            f"({', '.join(_models())}). Try again tomorrow, set "
            "GEMINI_CHAT_MODEL to another model, or enable billing on the API "
            "key."
        ) from error
