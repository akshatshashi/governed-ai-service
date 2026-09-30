"""Retrieval ranking and traceability."""

from src.config import Settings
from src.ingest import index_chunks, open_client
from src.retriever import Retriever


def test_most_relevant_string_ranks_first():
    client = open_client(None)  # in-memory
    docs = [
        {"doc_id": "a.md", "source_path": "a.md", "text": "The cafe sells coffee and croissants."},
        {
            "doc_id": "b.md",
            "source_path": "b.md",
            "text": "Cash transactions of AUD 10,000 or more must be reported to AUSTRAC.",
        },
        {"doc_id": "c.md", "source_path": "c.md", "text": "Football season starts in March."},
    ]
    model = Settings().embedding_model
    index_chunks(client, docs, model, 800, 150, collection_name="retriever-test")
    retriever = Retriever(None, model, collection_name="retriever-test", client=client)

    results = retriever.search("What is the cash reporting threshold?", top_k=3)

    assert [r.doc_id for r in results][0] == "b.md"
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert results == sorted(results, key=lambda r: r.score, reverse=True)
    assert results[0].chunk_index == 0 and results[0].source_path == "b.md"


def test_search_over_real_corpus_is_traceable(retriever):
    results = retriever.search("suspicious matter report deadline", top_k=3)
    assert results[0].doc_id == "austrac-reporting-obligations.md"
    assert results[0].score > 0.3
