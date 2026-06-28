# tests/test_ingestion.py
import pytest
import os
import tempfile
from pathlib import Path
from langchain_community.vectorstores import Chroma

from src.ingestion import load_pdf, build_vectorstore, load_vectorstore

# ── Fixture: example PDF ───────────────────────────────────────────
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF   = FIXTURES_DIR / "sample_technical_doc.pdf"  # PDF ~1 page


@pytest.fixture(scope="module")
def sample_pdf_path():
    """Returns path to the sample PDF fixture. Skip if not found."""
    if not SAMPLE_PDF.exists():
        pytest.skip(f"Fixture not found: {SAMPLE_PDF}. Add a sample PDF to tests/fixtures/")
    return str(SAMPLE_PDF)


@pytest.fixture
def tmp_vectorstore_dir(tmp_path):
    """Provides a clean temporary directory for each test."""
    return str(tmp_path / "vectorstore")


# ── Tests: load_pdf ──────────────────────────────────────────────────
class TestLoadPdf:

    def test_returns_non_empty_string(self, sample_pdf_path):
        text = load_pdf(sample_pdf_path)
        assert isinstance(text, str)
        assert len(text) > 100, "Extracted text is suspiciously short"

    def test_preserves_known_content(self, sample_pdf_path):
        """The sample PDF must contain the word 'specification' or 'requirement'."""
        text = load_pdf(sample_pdf_path).lower()
        assert any(kw in text for kw in ["specification", "requirement", "system"]), \
            "Expected technical keywords not found — check PDF or encoding"

    def test_raises_on_missing_file(self):
        with pytest.raises(Exception):
            load_pdf("nonexistent/path/file.pdf")


# ── Tests: build_vectorstore ─────────────────────────────────────────
class TestBuildVectorstore:

    def test_creates_vectorstore_with_documents(self, sample_pdf_path, tmp_vectorstore_dir):
        vs = build_vectorstore([sample_pdf_path], persist_dir=tmp_vectorstore_dir)
        assert vs is not None
        # ChromaDB exposes document count via its collection
        count = vs._collection.count()
        assert count > 0, "Vectorstore is empty after indexing"

    def test_chunks_respect_size_limit(self, sample_pdf_path, tmp_vectorstore_dir):
        vs = build_vectorstore([sample_pdf_path], persist_dir=tmp_vectorstore_dir)
        docs = vs._collection.get(include=["documents"])["documents"]
        chunk_size  = int(os.getenv("CHUNK_SIZE", 512))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 64))
        max_allowed = chunk_size + chunk_overlap + 50  # small tolerance
        oversized = [d for d in docs if len(d) > max_allowed]
        assert len(oversized) == 0, \
            f"{len(oversized)} chunks exceed max size of {max_allowed} chars"

    def test_chunks_have_source_metadata(self, sample_pdf_path, tmp_vectorstore_dir):
        vs = build_vectorstore([sample_pdf_path], persist_dir=tmp_vectorstore_dir)
        metadatas = vs._collection.get(include=["metadatas"])["metadatas"]
        missing = [m for m in metadatas if "source" not in m]
        assert len(missing) == 0, \
            f"{len(missing)} chunks are missing 'source' metadata"

    def test_similarity_search_returns_results(self, sample_pdf_path, tmp_vectorstore_dir):
        vs = build_vectorstore([sample_pdf_path], persist_dir=tmp_vectorstore_dir)
        results = vs.similarity_search("system requirements", k=2)
        assert len(results) > 0
        assert all(hasattr(doc, "page_content") for doc in results)


# ── Tests: load_vectorstore (persistence) ────────────────────────────
class TestVectorstorePersistence:

    def test_persists_and_reloads(self, sample_pdf_path, tmp_vectorstore_dir):
        vs_original = build_vectorstore([sample_pdf_path], persist_dir=tmp_vectorstore_dir)
        original_count = vs_original._collection.count()

        vs_reloaded = load_vectorstore(persist_dir=tmp_vectorstore_dir)
        reloaded_count = vs_reloaded._collection.count()

        assert original_count == reloaded_count, \
            "Document count changed after reload — persistence broken"

    def test_reloaded_search_is_consistent(self, sample_pdf_path, tmp_vectorstore_dir):
        query = "technical specifications"
        vs_original = build_vectorstore([sample_pdf_path], persist_dir=tmp_vectorstore_dir)
        original_results = vs_original.similarity_search(query, k=2)

        vs_reloaded = load_vectorstore(persist_dir=tmp_vectorstore_dir)
        reloaded_results = vs_reloaded.similarity_search(query, k=2)

        assert len(original_results) == len(reloaded_results)
        assert original_results[0].page_content == reloaded_results[0].page_content, \
            "Top result differs after reload"