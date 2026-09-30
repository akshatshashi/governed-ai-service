"""Chunking and document loading."""

from src.ingest import chunk_text, load_documents
from src.text import split_sentences
from tests.conftest import CORPUS_DIR


def _synthetic_text() -> str:
    paragraphs = []
    for p in range(6):
        sentences = [
            f"Paragraph {p} sentence {s} talks about topic {p * 10 + s}." for s in range(6)
        ]
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


def test_chunks_respect_size_limit():
    chunks = chunk_text(_synthetic_text(), chunk_size=300, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_shares_content_between_consecutive_chunks():
    chunks = chunk_text(_synthetic_text(), chunk_size=300, overlap=100)
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert split_sentences(first)[-1] in second


def test_chunks_never_split_mid_sentence():
    chunks = chunk_text(_synthetic_text(), chunk_size=300, overlap=100)
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_no_overlap_when_overlap_is_zero():
    chunks = chunk_text(_synthetic_text(), chunk_size=300, overlap=0)
    joined = sum(len(c) for c in chunks)
    assert joined <= len(_synthetic_text())


def test_load_documents_skips_manifest_and_records_provenance():
    docs = load_documents(str(CORPUS_DIR.parent))
    names = {d["doc_id"] for d in docs}
    assert "MANIFEST.md" not in names
    assert "austrac-reporting-obligations.md" in names
    doc = next(d for d in docs if d["doc_id"] == "austrac-reporting-obligations.md")
    assert doc["source_path"] == "synthetic/austrac-reporting-obligations.md"
    assert len(doc["sha256"]) == 64
