"""Inspect the Chroma store from the command line.

Most commands read chroma.sqlite3 directly in read-only mode, so they are safe
to run while an ingest is still writing and they never load the embedding
model. Only `search` needs the model (~400MB RSS) — keep it for when the
machine has the memory to spare.

  python browse.py stats
  python browse.py list 20
  python browse.py show book-2767052
  python browse.py grep dystopian
  python browse.py search "a novel about surveillance"
"""
from __future__ import annotations

import os
import sqlite3
import sys
import textwrap

from rag import config

DB = os.path.join(config.CHROMA_DIR, "chroma.sqlite3")


def connect() -> sqlite3.Connection:
    if not os.path.exists(DB):
        sys.exit(f"No Chroma database at {DB} — run `python -m rag.ingest` first.")
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def records(con, name: str = None, limit: int = None, offset: int = 0,
            like: str = None) -> list[dict]:
    """Rebuild whole records from Chroma's key/value metadata table."""
    name = name or config.COLLECTION
    sql = """
        SELECT e.embedding_id,
               max(CASE m.key WHEN 'title' THEN m.string_value END),
               max(CASE m.key WHEN 'authors' THEN m.string_value END),
               max(CASE m.key WHEN 'year' THEN m.string_value END),
               max(CASE m.key WHEN 'rating' THEN m.float_value END),
               max(CASE m.key WHEN 'description' THEN m.string_value END),
               max(CASE m.key WHEN 'chroma:document' THEN m.string_value END)
          FROM embeddings e
          JOIN segments s ON s.id = e.segment_id
          JOIN collections c ON c.id = s.collection
          LEFT JOIN embedding_metadata m ON m.id = e.id
         WHERE c.name = ?
         GROUP BY e.id
    """
    args = [name]
    if like:
        sql += """ HAVING lower(coalesce(max(CASE m.key WHEN 'title' THEN m.string_value END), '')
                              || ' ' || coalesce(max(CASE m.key WHEN 'authors' THEN m.string_value END), '')
                              || ' ' || coalesce(max(CASE m.key WHEN 'description' THEN m.string_value END), ''))
                   LIKE ?"""
        args.append(f"%{like.lower()}%")
    sql += " ORDER BY e.id"
    if limit:
        sql += " LIMIT ? OFFSET ?"
        args += [limit, offset]

    keys = ["id", "title", "authors", "year", "rating", "description", "document"]
    return [dict(zip(keys, row)) for row in con.execute(sql, args)]


def line(rec: dict) -> str:
    facts = " · ".join(str(p) for p in [rec["authors"], rec["year"],
                                        rec["rating"] and f"★ {rec['rating']}"] if p)
    return f"  {rec['id']:<22} {(rec['title'] or 'Untitled')[:52]:<54} {facts}"


def cmd_stats(con, args) -> None:
    print(f"{DB}  ({os.path.getsize(DB) / 1e6:.1f} MB sqlite)\n")
    rows = con.execute("""
        SELECT c.name, c.dimension, count(e.id)
          FROM collections c
          LEFT JOIN segments s ON s.collection = c.id
          LEFT JOIN embeddings e ON e.segment_id = s.id
         GROUP BY c.id ORDER BY count(e.id) DESC
    """).fetchall()
    print(f"  {'COLLECTION':<28} {'DIM':>6} {'RECORDS':>9}")
    for name, dim, n in rows:
        mark = "  <- active" if name == config.COLLECTION else ""
        print(f"  {name:<28} {dim or '-':>6} {n:>9}{mark}")


def cmd_list(con, args) -> None:
    limit = int(args[0]) if args else 20
    offset = int(args[1]) if len(args) > 1 else 0
    for rec in records(con, limit=limit, offset=offset):
        print(line(rec))


def cmd_grep(con, args) -> None:
    if not args:
        sys.exit("usage: browse.py grep <text>")
    found = records(con, like=" ".join(args))
    for rec in found:
        print(line(rec))
    print(f"\n  {len(found)} match(es)")


def cmd_show(con, args) -> None:
    if not args:
        sys.exit("usage: browse.py show <book-id>")
    target = args[0]
    for rec in records(con):
        if rec["id"] == target:
            print(f"{rec['title']}\n{'=' * len(rec['title'] or '')}")
            print(f"id      : {rec['id']}")
            print(f"authors : {rec['authors']}")
            print(f"year    : {rec['year']}")
            print(f"rating  : {rec['rating']}")
            print("\nEMBEDDED TEXT (what the vector was built from)")
            print(textwrap.fill(rec["document"] or "", 78))
            print("\nDESCRIPTION (metadata, not embedded)")
            print(textwrap.fill(rec["description"] or "", 78))
            return
    sys.exit(f"{target} not found")


def cmd_search(con, args) -> None:
    if not args:
        sys.exit("usage: browse.py search <query>")
    from rag.retrieve import search  # imported late: this loads the model
    for hit in search(" ".join(args)):
        print(f"  [{hit.rank}] {hit.similarity:.0%}  {hit.title}  — {hit.authors}")


COMMANDS = {"stats": cmd_stats, "list": cmd_list, "grep": cmd_grep,
            "show": cmd_show, "search": cmd_search}

if __name__ == "__main__":
    argv = sys.argv[1:]
    command = COMMANDS.get(argv[0] if argv else "stats")
    if not command:
        sys.exit(f"unknown command; try: {', '.join(COMMANDS)}")
    command(connect(), argv[1:])
