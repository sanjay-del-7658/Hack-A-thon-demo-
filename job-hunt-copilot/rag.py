"""
rag.py — Retrieval-Augmented Generation over your own experience.

Pipeline:
  knowledge/*.md  --chunk-->  small pieces  --embed-->  vectors  --store-->  ChromaDB
  query  --embed-->  vector  --nearest neighbours-->  most relevant chunks

Commands:
  python rag.py index                 # (re)build the index after editing knowledge/
  python rag.py search "REST APIs"    # test what the agent would retrieve
"""

import sys
from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DB_DIR = BASE_DIR / ".chroma"
COLLECTION = "profile"

# Chroma's default embedding model (all-MiniLM-L6-v2) runs locally and free.
# It downloads once (~80 MB) on first use.
_embedding_fn = None  # None = Chroma default. Tests swap in a fake one.


def _collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    kwargs = {"name": COLLECTION}
    if _embedding_fn is not None:
        kwargs["embedding_function"] = _embedding_fn
    return client.get_or_create_collection(**kwargs)


def chunk_file(path: Path) -> list[dict]:
    """Split a markdown file into chunks at '## ' headings.

    Why chunk? Embeddings work best on focused pieces of text. One chunk = one
    project section, so a search for 'MongoDB' returns that exact section, not
    your whole life story.
    """
    text = path.read_text(encoding="utf-8")
    title = path.stem
    chunks, heading, lines = [], "Overview", []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            chunks.append({"heading": heading, "text": body})

    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    flush()

    rel = path.relative_to(KNOWLEDGE_DIR).with_suffix("").as_posix()
    return [
        {
            "id": f"{rel}#{i}",  # stable source id the agent must cite
            "text": f"{title} — {c['heading']}\n{c['text']}",
            "meta": {"source": rel, "title": title, "heading": c["heading"]},
        }
        for i, c in enumerate(chunks)
    ]


def build_index() -> int:
    """Read every knowledge/*.md file, chunk it, embed it, store it."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch: simple & safe
    except Exception:
        pass
    col = _collection()

    chunks = []
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_file(path))

    if chunks:
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["meta"] for c in chunks],
        )
    return len(chunks)


def search(query: str, k: int = 4) -> list[dict]:
    """Return the k chunks whose meaning is closest to the query."""
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    return [
        {
            "source_id": res["ids"][0][i],
            "text": res["documents"][0][i],
            # distance: lower = more similar. Useful for spotting weak matches.
            "distance": round(res["distances"][0][i], 3),
        }
        for i in range(len(res["ids"][0]))
    ]


def source_exists(source_id: str) -> bool:
    return len(_collection().get(ids=[source_id])["ids"]) == 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "index":
        print(f"Indexed {build_index()} chunks from {KNOWLEDGE_DIR.name}/")
    elif cmd == "search" and len(sys.argv) > 2:
        for hit in search(" ".join(sys.argv[2:])):
            print(f"\n[{hit['source_id']}]  distance={hit['distance']}\n{hit['text'][:300]}")
    else:
        print(__doc__)
