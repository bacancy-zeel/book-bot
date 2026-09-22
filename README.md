# Book Bot

A Streamlit chatbot that answers questions grounded in a book catalogue. Ask
*"recommend a dystopian novel about surveillance"* or *"who wrote The Hunger
Games?"* and get an answer built only from the catalogue — with the exact books
it used shown alongside the answer and scored. This is **R**etrieval-**A**ugmented
**G**eneration (RAG): the model isn't answering from its general training
knowledge, it's answering from records retrieved from your catalogue at question
time.

The design decisions behind the pipeline — why embedding runs locally, what goes
into a vector and what deliberately doesn't, and how the similarity floor is
meant to be read — are in [Design notes](#design-notes) below.

## Features

- Ask questions in a chat UI; answers are grounded in the catalogue and cite every book they mention as `[1]`, `[2]`
- Every answer expands to show which books it was based on, each with a relevance score
- Conversation history in the sidebar, grouped by age, persisting across page refreshes and server restarts
- Delete one conversation (with a confirm step) or clear all of them
- Embeddings run in-process — the whole catalogue indexes with no API key, no quota and no per-record cost
- Ingestion is resumable: an interrupted run costs nothing

## Architecture at a glance

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) 1.40 (Python 3.10+) |
| Vector store | [ChromaDB](https://www.trychroma.com/) 0.5 — persisted to `./chroma_data`, cosine distance |
| Embeddings | [fastembed](https://github.com/qdrant/fastembed) (ONNX, no PyTorch) — `BAAI/bge-base-en-v1.5`, 768-dim, run locally |
| LLM (answers) | Google Gemini (`gemini-3.6-flash`) — one call per question |
| Chat history | SQLite — `./chat_history.sqlite3` |
| Source data | A Goodreads-style books CSV |

```
 Ingest → CSV row → one record per book → Embed locally (BGE) → Store (ChromaDB)
 Ask    → Embed question (BGE query prefix) → Retrieve nearest books (ChromaDB, cosine)
        → Generate cited answer (Gemini) → Persist (SQLite)
```

Only generation touches a hosted API. Embedding is local, which is what makes
indexing a few thousand books free — see [Design notes](#design-notes).

## Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier is enough) — used *only* to write answers, never to index
- A books CSV with `title`, `authors`, `description`, `original_publication_year`, `average_rating` and `book_id` columns
- ~450 MB of disk: a ~210 MB one-time model download plus ~230 MB of index for a 4,766-book catalogue

## Setup

1. **Install dependencies**

   ```bash
   python3 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt
   ```

2. **Configure the environment**

   ```bash
   cp .env.example .env
   ```

   Then fill in these values in `.env`:

   | Variable | Purpose |
   |---|---|
   | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/) → Get API key. Answer generation only |
   | `GEMINI_CHAT_MODEL` | Defaults to `gemini-3.6-flash` |
   | `GEMINI_FALLBACK_MODELS` | Tried in order on a 429 — free-tier quota is counted per model per day, so a sibling model is a fresh bucket |
   | `BOOKS_CSV` | Path to the catalogue |
   | `CHROMA_DIR` | Where the vector store is persisted |
   | `COLLECTION` | Chroma collection name |
   | `EMBED_MODEL` | Local embedding model; 768-dim by default. Changing it requires `--reset` |
   | `HISTORY_DB` | SQLite file holding the chat sidebar; defaults to `./chat_history.sqlite3` |
   | `TOP_K` | Books retrieved per question (default 5) |
   | `MIN_SIMILARITY` | Coarse junk filter, **not** the refusal mechanism — see [Design notes](#design-notes) |
   | `EMBED_BATCH` | Rows embedded per call; a memory/throughput knob, not a quota one |

   > **Gemini model availability changes over time** — if `GEMINI_CHAT_MODEL`
   > returns a 404 "model not found", list the models your key actually has
   > access to and pick a current one:
   > ```bash
   > curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY" | grep '"name"'
   > ```

3. **Index the catalogue**

   ```bash
   ./.venv/bin/python -m rag.ingest              # all rows
   ./.venv/bin/python -m rag.ingest 500          # first 500 books
   ./.venv/bin/python -m rag.ingest --reset      # wipe and rebuild
   ```

   The first run downloads the embedding model. Each book gets a deterministic
   id and is written with `upsert`, so re-running skips what is already stored
   — an interrupted run can simply be repeated.

4. **Run the app**

   ```bash
   ./.venv/bin/streamlit run app.py
   ```

   Visit **http://localhost:8501**.

## Usage

1. Ask a question in the chat box, or pick one of the example prompts on the welcome screen.
2. Expand **Sources** under any answer to see exactly which books it used, each with a percentage match.
3. Past conversations are listed in the sidebar, grouped into Today / Yesterday / Previous 7 days / Previous 30 days / Older. Click one to reopen it, or **＋ New chat** to start fresh.
4. Hover a conversation and click **✕** to delete it; you'll be asked to confirm. **Clear all history** removes every conversation at once.

History lives in SQLite rather than in Streamlit's session state, so refreshing
the page or restarting the server does not lose it.

## Inspecting the store

```bash
./.venv/bin/python browse.py stats                 # collections, dimensions, record counts
./.venv/bin/python browse.py list 20               # first 20 records
./.venv/bin/python browse.py show book-2767052     # one record, including the embedded text
./.venv/bin/python browse.py grep dystopian        # substring search over title/author/description
./.venv/bin/python browse.py search "a novel about surveillance"
```

Every command except `search` reads `chroma.sqlite3` directly in read-only mode,
so it is safe to run while an ingest is still writing and never loads the
embedding model. Only `search` needs the model (~400 MB RSS).

## Design notes

### Why embeddings run locally

Embedding is the expensive half of RAG: every book has to be embedded once. A
hosted embedding API meters each record, and on a free tier that is a hard wall
— Gemini's free tier allows **1,000 embed requests per day**, and each input
inside a batch counts separately, so a 4,766-book catalogue would take about six
days.

So embeddings run in-process via **fastembed** (ONNX, no PyTorch — which matters
on a machine with little free RAM): no key, no quota, no per-record cost, and
the whole catalogue indexes in one pass. Gemini is used *only* to write the
answer, which a free tier handles comfortably.

### Choosing the embedding model

`BAAI/bge-base-en-v1.5`, chosen by measurement rather than default. ChromaDB's
built-in `all-MiniLM-L6-v2` was tried first and did notably worse at thematic
search on this catalogue — for *"a dystopian novel about surveillance and
government control"* it surfaced spy thrillers rather than dystopias.

BGE is **asymmetric**: documents are embedded plain, queries get an instruction
prefix. `rag/store.py` applies it via `embed_query`, which is why retrieval
passes `query_embeddings` rather than Chroma's `query_texts`.

A Chroma collection is fixed-width, so changing `EMBED_MODEL` means a `--reset`
rebuild. `BAAI/bge-small-en-v1.5` is the cheaper option — 384-dim, a third of
the index size and several times faster to embed.

### What gets embedded

Only prose: `"<title> by <author>. <description>"`.

Year, rating, ids and language codes are kept as metadata but deliberately left
**out** of the vector. They are identical boilerplate on every record, so they
pull the vectors toward a common centroid and wash out what distinguishes one
book from another. It matters more than it looks: these models truncate at a few
hundred tokens, so leading metadata pushes the blurb out of the window entirely.

One record per book, never fixed-size chunks — a token window would cut a blurb
in half and glue it to an unrelated book.

### Reading MIN_SIMILARITY

These scores are **not** comparable across models. With BGE a good match sits
around 0.6–0.7 and unrelated text lands near 0.3, so the floor is deliberately
low. It exists to drop obvious junk; the prompt is what actually makes the bot
say "I don't know".

## Limitations

- Answers are only as good as the catalogue's descriptions.
- Retrieval is semantic, not filtered: *"books after 2010 rated above 4"* will
  not reliably respect the numbers, because year and rating are metadata rather
  than part of the vector.
- Thematic recall is imperfect. For *"a dystopian novel about surveillance and
  government control"* the current index returns Cryptonomicon, The Circle and
  State of Fear ahead of 1984, Brave New World and The Handmaid's Tale, all
  three of which are in the catalogue. Descriptions, not the query, are usually
  the limiting factor.
- No accounts; one shared collection, and chat history is local to the machine
  running the app.
- No automated test suite.
