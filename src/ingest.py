"""Load, chunk and embed the corpus into a persistent Chroma collection.

Every chunk stays traceable to its source: metadata carries ``doc_id``, ``source_path``,
``chunk_index`` and the SHA-256 of the source document, so an audit record that cites a chunk can
be tied back to the exact document version that produced it.

Usage::

    python -m src.ingest            # rebuild the index from settings.corpus_dir
"""

from __future__ import annotations

import argparse
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.text import split_sentences

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "corpus"
_SKIP_FILES = {"MANIFEST.md", "README.md"}


def load_documents(corpus_dir: str) -> list[dict]:
    """Read every ``.pdf`` and ``.md`` file under ``corpus_dir`` (recursively).

    Returns dicts with keys ``doc_id`` (file name), ``source_path`` (path relative to the corpus
    directory), ``text`` and ``sha256`` (of the raw file bytes). ``MANIFEST.md`` is provenance
    metadata, not content, and is skipped.
    """
    root = Path(corpus_dir)
    docs: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".pdf", ".md"} or path.name in _SKIP_FILES:
            continue
        raw = path.read_bytes()
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = raw.decode("utf-8")
        docs.append(
            {
                "doc_id": path.name,
                "source_path": str(path.relative_to(root)),
                "text": text,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most ``chunk_size`` characters.

    Strategy:
    - The atomic unit is a sentence; a chunk never ends mid-sentence. (A single sentence longer
      than ``chunk_size`` is the only exception — it is split on word boundaries.)
    - Paragraph boundaries are preferred: if the next paragraph will not fit and the current chunk
      is already at least half full, the chunk is closed at the paragraph break.
    - Consecutive chunks overlap: each new chunk starts with the trailing sentences of the previous
      chunk, up to ``overlap`` characters, so a fact that straddles a boundary is retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[tuple[int, str]] = []  # (paragraph index, sentence)
    for p_idx, paragraph in enumerate(paragraphs):
        for sentence in split_sentences(paragraph):
            if len(sentence) <= chunk_size:
                units.append((p_idx, sentence))
            else:
                units.extend((p_idx, piece) for piece in _split_long(sentence, chunk_size))

    def join(parts: list[tuple[int, str]]) -> str:
        out = ""
        for i, (p_idx, sentence) in enumerate(parts):
            if i == 0:
                out = sentence
            else:
                out += ("\n\n" if p_idx != parts[i - 1][0] else " ") + sentence
        return out

    def paragraph_length(start: int) -> int:
        p_idx = units[start][0]
        return len(join([u for u in units[start:] if u[0] == p_idx]))

    chunks: list[str] = []
    current: list[tuple[int, str]] = []
    for i, unit in enumerate(units):
        starts_paragraph = not current or unit[0] != current[-1][0]
        current_len = len(join(current))
        fits = len(join([*current, unit])) <= chunk_size
        close_at_paragraph = (
            starts_paragraph
            and current
            and current_len >= chunk_size / 2
            and current_len + 2 + paragraph_length(i) > chunk_size
        )
        if current and (not fits or close_at_paragraph):
            chunks.append(join(current))
            current = _overlap_tail(current, overlap, join)
            while current and len(join([*current, unit])) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(join(current))
    return chunks


def _overlap_tail(parts: list[tuple[int, str]], overlap: int, join) -> list[tuple[int, str]]:
    tail: list[tuple[int, str]] = []
    for part in reversed(parts):
        if len(join([part, *tail])) > overlap:
            break
        tail.insert(0, part)
    return tail


def _split_long(sentence: str, chunk_size: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > chunk_size and current:
            pieces.append(current)
            current = word[:chunk_size]
        else:
            current = candidate[:chunk_size]
    if current:
        pieces.append(current)
    return pieces


@lru_cache(maxsize=2)
def get_embedder(model_name: str) -> SentenceTransformer:
    """Load (once per process) the local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed texts as unit vectors (so cosine distance is well-behaved)."""
    vectors = get_embedder(model_name).encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def open_client(persist_dir: str | None) -> chromadb.ClientAPI:
    """A Chroma client with telemetry off; ``None`` gives an in-memory client (tests)."""
    settings = ChromaSettings(anonymized_telemetry=False)
    if persist_dir is None:
        return chromadb.EphemeralClient(settings=settings)
    return chromadb.PersistentClient(path=persist_dir, settings=settings)


def index_chunks(
    client: chromadb.ClientAPI,
    docs: list[dict],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """(Re)create ``collection_name`` and fill it with chunks of ``docs``. Returns chunk count."""
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for doc in docs:
        for idx, chunk in enumerate(chunk_text(doc["text"], chunk_size, chunk_overlap)):
            ids.append(f"{doc['doc_id']}::{idx}")
            texts.append(chunk)
            metadatas.append(
                {
                    "doc_id": doc["doc_id"],
                    "source_path": doc["source_path"],
                    "chunk_index": idx,
                    "doc_sha256": doc.get("sha256", ""),
                }
            )
    if texts:
        collection.add(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embed(texts, embedding_model)
        )
    return len(texts)


def build_index(corpus_dir: str, persist_dir: str) -> int:
    """Chunk and embed every document in ``corpus_dir`` into a persistent collection "corpus".

    Returns the number of chunks indexed. The collection is rebuilt from scratch each time so
    the index always matches the corpus exactly.
    """
    from src.config import get_settings

    settings = get_settings()
    docs = load_documents(corpus_dir)
    client = open_client(persist_dir)
    return index_chunks(
        client, docs, settings.embedding_model, settings.chunk_size, settings.chunk_overlap
    )


def main() -> None:
    from src.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the vector index from the corpus.")
    parser.add_argument("--corpus", default=settings.corpus_dir)
    parser.add_argument("--index", default=settings.index_dir)
    args = parser.parse_args()
    count = build_index(args.corpus, args.index)
    print(f"Indexed {count} chunks from {args.corpus} into {args.index}")


if __name__ == "__main__":
    main()
