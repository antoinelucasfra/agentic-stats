# The RAG cookbook: your documents, with receipts

Fine-tuning (Chap. 6) bakes a *skill* into weights; **RAG** (Retrieval-Augmented Generation)[^ch13-rag] answers questions *about specific documents* by retrieving the right passages at query time and letting the model cite them. [Rule of thumb: behaviour → fine-tune; knowledge that changes → RAG.]

### A fully local stack in four parts

1. **Embeddings**. A small model turns text chunks into vectors. Use a multilingual model so French and English land in the same space: `bge-m3`[^ch13-m3] or `multilingual-e5-base` run on CPU in milliseconds.
2. **Chunking**. Split documents into ~300–800 token pieces with ~15% overlap. Split on structure (headings, paragraphs) before splitting on count.
3. **Vector store**. For a lab, `sqlite-vec` (a file) is enough up to ~1M chunks; `pgvector` if you already run PostgreSQL; Qdrant when you need filtering and replication.
4. **Generation**. Any model from Chap. 2 receives the question + top-k chunks and must answer *from them*.

![Local RAG: documents are chunked and embedded once into a vector store on disk; at query time the top-k similar chunks are retrieved and answered from. Nothing leaves the machine.]

Minimal working pipeline:

```python
# pip install sentence-transformers sqlite-vec llama-cpp-python
import sqlite_vec, sqlite3
from sentence_transformers import SentenceTransformer

emb = SentenceTransformer("BAAI/bge-m3")
db = sqlite3.connect("docs.db")
db.enable_load_extension(True); sqlite_vec.load(db)
db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[1024], chunk text)")

def index(chunks):                       # chunks: list[str]
    db.executemany("INSERT INTO v(embedding, chunk) VALUES (?, ?)",
                   [(emb.encode(c).tobytes(), c) for c in chunks])

def ask(question, k=5):
    q = emb.encode(question)
    rows = db.execute("""SELECT chunk FROM v WHERE embedding MATCH ?
                       ORDER BY distance LIMIT ?""", (q.tobytes(), k)).fetchall()
    return "\n---\n".join(r[0] for r in rows)
# then: answer = llm(f"Answer only from these excerpts:\n{ask(q)}\n\nQ: {q}")

Your chat UI's web search and document features use the same mechanics internally (Chap. 4); a dataset builder can *generate* training data from those same sources when you decide a skill must move into weights (Chap. 8).

### Retrieval quality is the whole game

- **Hybrid search beats pure vectors**: add BM25 keyword scoring (SQLite FTS5) and fuse rankings. Acronyms, part numbers and names are where embeddings fail.
- **Respect lost-in-the-middle**: put the best chunks first and last (Chap. 5); 20 chunks help less than 6 good ones.
- **Evaluate retrieval separately** from generation: does the golden set's reference passage (Chap. 16) appear in the top-k? Fix retrieval before blaming the model. Remember [the routing rule].
- **Show citations**: force the model to quote chunk ids. Reviewers verify instead of trusting, which is also your GDPR transparency story (Chap. 17).

**Our take.** Every RAG disappointment we have ever debugged was a retrieval failure wearing a model's costume. Spend the week on chunking, hybrid search and a 50-question golden set before you spend a single euro on a bigger model. A mediocre model fed the right 6 chunks beats an excellent one fed 20 wrong ones, every time we have measured it.
The complete runnable version lives in `code/rag_minimal.py`; it ships here in full so this book's snippets can never drift from the companion repo:

```python
#!/usr/bin/env python3
"""Chap. 13 — minimal local RAG: sqlite-vec store + bge-m3 embeddings.

Usage:
    python rag_minimal.py index ./docs/*.md
    python rag_minimal.py ask "How do I reset my VPN token?"
"""

import argparse
import glob
import sqlite3
import sys

import sqlite_vec  # type: ignore[import-not-found]
from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

MODEL = "BAAI/bge-m3"
DB = "docs.db"

def cmd_index(paths):
    model = SentenceTransformer(MODEL)
    db = sqlite3.connect(DB)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(embedding float[1024], text text)"
    )
    n = 0
    for pattern in paths:
        for path in glob.glob(pattern):
            # ponytail: whole-file chunks; split on headings if docs exceed ~800 tokens
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as e:
                print(f"skip {path}: {e}")
                continue
            db.execute(
                "INSERT INTO chunks(embedding, text) VALUES (?, ?)",
                (model.encode(text).tolist(), text),
            )
            n += 1
    db.commit()
    print(f"indexed {n} documents into {DB}")

def cmd_ask(question, k=4):
    model = SentenceTransformer(MODEL)
    db = sqlite3.connect(DB)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    rows = db.execute(
        """SELECT text, distance FROM chunks WHERE embedding MATCH ?
           ORDER BY distance LIMIT ?""",
        (model.encode(question).tolist(), k),
    ).fetchall()
    context = "\n---\n".join(t for t, _ in rows)

    # llama-cpp-python is an optional runtime dep (see code/README.md); the
    # lazy import keeps sqlite-vec retrieval usable without it.
    from llama_cpp import Llama  # type: ignore[import-not-found]

    llm = Llama.from_pretrained(
        # any GGUF repo + quant works: swap for the model you standardised on
        repo_id="unsloth/Qwen3-8B-GGUF",
        filename="*UD-Q4_K_XL.gguf",
        n_ctx=8192,
        verbose=False,
    )
    out = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY from the context. Cite which passage. Say 'not in the documents' otherwise.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    print(out["choices"][0]["message"]["content"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("index")
    p1.add_argument("paths", nargs="+")
    p2 = sub.add_parser("ask")
    p2.add_argument("question")
    args = ap.parse_args()

    if args.cmd == "index":
        cmd_index(args.paths)
    else:
        try:
            cmd_ask(args.question)
        except FileNotFoundError:
            sys.exit(f"no index at {DB} — run `rag_minimal.py index` first")
```
