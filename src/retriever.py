"""Semantic retrieval over the Chroma collection built by ``src.ingest``."""

from __future__ import annotations

import chromadb
from pydantic import BaseModel

from src.ingest import COLLECTION_NAME, embed, open_client


class RetrievedChunk(BaseModel):
    """A retrieved passage plus everything needed to trace it back to its source."""

    text: str
    doc_id: str
    source_path: str
    chunk_index: int
    score: float


class Retriever:
    """Embeds a query locally and returns the most similar chunks."""

    def __init__(
        self,
        persist_dir: str | None,
        embedding_model: str,
        collection_name: str = COLLECTION_NAME,
        client: chromadb.ClientAPI | None = None,
    ) -> None:
        self._client = client or open_client(persist_dir)
        self._embedding_model = embedding_model
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def count(self) -> int:
        """Number of chunks in the index (reported by /health)."""
        return self._collection.count()

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks, most similar first.

        Score conversion: the collection uses cosine *distance*, ``d = 1 - cos(q, c)``, which
        ranges over [0, 2]. Embeddings are unit-normalised, so the similarity score is
        ``score = max(0, 1 - d)`` — the cosine similarity clipped at zero, in the range [0, 1].
        1.0 means identical direction; 0 means unrelated (or opposed, which we treat the same).
        """
        n = min(top_k, self.count())
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=embed([query], self._embedding_model),
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        chunks = [
            RetrievedChunk(
                text=text,
                doc_id=str(meta["doc_id"]),
                source_path=str(meta["source_path"]),
                chunk_index=int(meta["chunk_index"]),
                score=round(max(0.0, 1.0 - float(distance)), 4),
            )
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
        return sorted(chunks, key=lambda c: c.score, reverse=True)
