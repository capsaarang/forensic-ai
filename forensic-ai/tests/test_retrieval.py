"""
Tests for retrieval pipeline: embedder and vector store.

Note: These tests require sentence-transformers and faiss-cpu.
Run: pip install sentence-transformers faiss-cpu
"""

import pytest
import numpy as np

try:
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="sentence-transformers or faiss-cpu not installed",
)


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def populated_store(embedder):
    from src.ingestion.section_detector import detect_sections
    from src.ingestion.chunker import chunk_sections

    sample_text = """
Item 1A. Risk Factors
The Company faces significant competition from both established companies and new entrants.
Revenue concentration in iPhone products represents a material risk to the business.

Item 7. Management's Discussion and Analysis
Net sales increased 8% year-over-year. Services revenue now represents 22% of total revenue.
Gross margin improved 150 basis points driven by services mix shift.

Item 3. Legal Proceedings
The Company is subject to class-action litigation in Delaware regarding alleged misrepresentation.
Settlement discussions are ongoing. The outcome is uncertain.
"""
    sections = detect_sections(sample_text)
    chunks = chunk_sections(sections, chunk_size=300, overlap=50)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_chunks(texts, show_progress=False)

    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embeddings)
    return store, chunks


class TestEmbedder:
    def test_embed_single_query(self, embedder):
        vec = embedder.embed_query("revenue recognition policy")
        assert vec.shape == (embedder.dim,)
        assert vec.dtype == np.float32

    def test_embed_chunks_shape(self, embedder):
        texts = ["risk factor disclosure", "debt maturity schedule", "related party transaction"]
        vecs = embedder.embed_chunks(texts, show_progress=False)
        assert vecs.shape == (3, embedder.dim)
        assert vecs.dtype == np.float32

    def test_embeddings_normalized(self, embedder):
        vec = embedder.embed_query("test query")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_empty_input(self, embedder):
        vecs = embedder.embed_chunks([], show_progress=False)
        assert vecs.shape == (0, embedder.dim)


class TestVectorStore:
    def test_search_returns_results(self, embedder, populated_store):
        store, _ = populated_store
        query_vec = embedder.embed_query("revenue growth year over year")
        results = store.search(query_vec, k=3)
        assert len(results) > 0

    def test_results_have_score(self, embedder, populated_store):
        store, _ = populated_store
        query_vec = embedder.embed_query("litigation risk")
        results = store.search(query_vec, k=2)
        for r in results:
            assert "score" in r
            assert "chunk" in r
            assert 0 <= r["score"] <= 1.1

    def test_results_sorted_by_score(self, embedder, populated_store):
        store, _ = populated_store
        query_vec = embedder.embed_query("legal proceedings class action")
        results = store.search(query_vec, k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_respected(self, embedder, populated_store):
        store, _ = populated_store
        query_vec = embedder.embed_query("competition market share")
        results = store.search(query_vec, k=2)
        assert len(results) <= 2

    def test_len(self, populated_store):
        store, chunks = populated_store
        assert len(store) == len(chunks)
