# Book Bot

A FastAPI chatbot that answers questions grounded in a book catalogue. Ask
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
- The page is only one client — every action it takes is a documented JSON endpoint, browsable at `/docs`

## Architecture at a glance

| Layer | Technology |
|---|---|
| Web layer | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn (Python 3.10+) — serves the chat page and a JSON API |
| Front end | One Jinja2 template, plain CSS and vanilla JS — no build step |
| Orchestration | [LangChain](https://python.langchain.com/) — LCEL chain, `Document`s, retriever, chat message history |
| Vector store | [Pinecone](https://www.pinecone.io/) serverless via `langchain-pinecone` — 768-dim, cosine |
| Embeddings | `FastEmbedEmbeddings` over [fastembed](https://github.com/qdrant/fastembed) (ONNX, no PyTorch) — `BAAI/bge-base-en-v1.5`, 768-dim, run locally |
| LLM (answers) | Google Gemini (`gemini-3.6-flash`) via `ChatGoogleGenerativeAI` — one call per question |
| Chat history | SQLite — `./chat_history.sqlite3`, behind a LangChain `BaseChatMessageHistory` |
| Source data | A Goodreads-style books CSV |

```
 Ingest → CSV row → one Document per book → Embed locally (BGE) → Upsert (Pinecone)
 Ask    → Embed question (BGE query prefix) → Retrieve nearest books (Pinecone, cosine)
        → prompt | Gemini | StrOutputParser → cited answer → Persist (SQLite)
```

Embedding stays local even though storage is hosted: Pinecone holds the
vectors, but computing them costs nothing and is not rate limited — see
[Design notes](#design-notes).

## Prerequisites

- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier is enough) — used *only* to write answers, never to index
- A [Pinecone](https://www.pinecone.io/) API key (the free serverless tier holds this catalogue comfortably)
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
   | `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io/) → API keys |
   | `PINECONE_INDEX` | Index name; created on first use if missing (768-dim, cosine, serverless) |
   | `PINECONE_CLOUD` / `PINECONE_REGION` | Where a missing index gets created. Defaults to `aws` / `us-east-1` |
   | `PINECONE_NAMESPACE` | Optional partition within the index; empty means the default namespace |
   | `BOOKS_CSV` | Path to the catalogue |
   | `EMBED_MODEL` | Local embedding model; 768-dim by default. Changing it needs a new index |
   | `CHROMA_DIR` / `COLLECTION` | Read-only source for `rag.migrate` and `browse.py` |
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

   If you already have the catalogue in a local Chroma store, copy it across
   instead — the vectors exist, so nothing is re-embedded and the model is
   never loaded:

   ```bash
   ./.venv/bin/python -m rag.migrate             # Chroma -> Pinecone
   ./.venv/bin/python -m rag.migrate --reset     # clear the namespace first
   ```

   The first run downloads the embedding model. Each book gets a deterministic
   id, so re-running skips what is already stored — an interrupted run can
   simply be repeated.

4. **Run the app**

   ```bash
   ./.venv/bin/uvicorn main:app --reload
   ```

   Visit **http://localhost:8000**. The API's generated docs are at
   **/docs**.

## Usage

1. Ask a question in the chat box, or pick one of the example prompts on the welcome screen.
2. Expand **Sources** under any answer to see exactly which books it used, each with a percentage match.
3. Past conversations are listed in the sidebar, grouped into Today / Yesterday / Previous 7 days / Previous 30 days / Older. Click one to reopen it, or **＋ New chat** to start fresh.
4. Hover a conversation and click **✕** to delete it; you'll be asked to confirm. **Clear all history** removes every conversation at once.

History lives in SQLite rather than in the browser, so refreshing the page or
restarting the server does not lose it.

## HTTP API

The page is just one client. Every action it takes is a public JSON endpoint,
documented and testable at **/docs**:

| Method | Path | Does |
|---|---|---|
| `GET` | `/api/status` | Number of books indexed |
| `POST` | `/api/chat` | `{"question": "...", "conversation_id": null}` → the answer, its Markdown rendered to HTML, and the books it cited. Omitting `conversation_id` starts a new conversation and returns its id |
| `GET` | `/api/conversations` | Every conversation, most recent first, grouped by age |
| `GET` | `/api/conversations/{id}` | The turns of one conversation, each assistant turn carrying its sources |
| `DELETE` | `/api/conversations/{id}` | Delete one conversation |
| `DELETE` | `/api/conversations` | Delete all of them |

```bash
curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"question": "Recommend a dystopian novel about surveillance"}'
```

Endpoints are declared `def` rather than `async def` on purpose: embedding a
query and calling Gemini both block, so FastAPI runs them in a worker thread
instead of stalling the event loop. `history.py` opens a SQLite connection per
call, which is what makes that safe.

## Inspecting the store

```bash
./.venv/bin/python browse.py stats                 # collections, dimensions, record counts
./.venv/bin/python browse.py list 20               # first 20 records
./.venv/bin/python browse.py show book-2767052     # one record, including the embedded text
./.venv/bin/python browse.py grep dystopian        # substring search over title/author/description
./.venv/bin/python browse.py search "a novel about surveillance"
```

`stats`, `list`, `show` and `grep` read the local `chroma.sqlite3` copy of the
catalogue directly in read-only mode: no network and no embedding model.
`search` is the odd one out — it goes through `rag.retrieve`, so it queries the
live Pinecone index and loads the model (~400 MB RSS) to embed the question.

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

`BAAI/bge-base-en-v1.5`, chosen by measurement rather than default. The usual
default, `all-MiniLM-L6-v2`, was tried first and did notably worse at thematic
search on this catalogue — for *"a dystopian novel about surveillance and
government control"* it surfaced spy thrillers rather than dystopias.

BGE is **asymmetric**: documents are embedded plain, queries get an instruction
prefix. `FastEmbedEmbeddings` applies that split — `embed_query` goes through
fastembed's `query_embed`, `embed_documents` through its plain `embed`.

A Pinecone index is fixed-width, so changing `EMBED_MODEL` means a new index.
`BAAI/bge-small-en-v1.5` is the cheaper option — 384-dim, a third of the index
size and several times faster to embed.

### What gets embedded

Only prose: `"<title> by <author>. <description>"`.

Year, rating, ids and language codes are kept as metadata but deliberately left
**out** of the vector. They are identical boilerplate on every record, so they
pull the vectors toward a common centroid and wash out what distinguishes one
book from another. It matters more than it looks: these models truncate at a few
hundred tokens, so leading metadata pushes the blurb out of the window entirely.

One record per book, never fixed-size chunks — a token window would cut a blurb
in half and glue it to an unrelated book.

### How the pieces fit

The pipeline is assembled from LangChain parts: ingestion builds `Document`s,
the store is a `langchain-pinecone` vector store, retrieval goes through
`similarity_search_with_relevance_scores`, generation is an LCEL chain
(`prompt | model | StrOutputParser`), and the SQLite tables behind the sidebar
are exposed as a `BaseChatMessageHistory` so past turns drop straight into the
prompt's `MessagesPlaceholder`.

Two consequences worth knowing. The per-model quota fallback is `with_fallbacks`
over one chain per model, firing on 429s only, so a bad key fails immediately
instead of being retried three times over. And the books that grounded an answer
ride in that message's `additional_kwargs`, so reopening a conversation redraws
its source cards without re-running retrieval.

One sharp edge is handled in `rag/store.py`. Pinecone scores a cosine match as
a *similarity* — higher is closer — but LangChain's base vector store assumes
that number is a distance and returns `1 - score`, which inverts the ranking and
silently breaks `MIN_SIMILARITY`. `BookVectorStore` overrides the relevance
function to pass the score through untouched.

### Where the vectors live

Pinecone holds them; the embedding model does not. Computing a vector is the
part that a hosted API meters, and that still runs in-process, so re-indexing
the whole catalogue costs nothing. What the network buys is a store the app can
reach from anywhere, instead of 229 MB of index sitting next to the process.

`rag/migrate.py` is the one-way door between the two: it reads an existing
Chroma collection and upserts the vectors it already holds, so moving a built
catalogue across never loads the model or re-embeds a single book.

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
