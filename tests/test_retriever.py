import pytest
from unittest.mock import MagicMock, patch
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.retriever import (
    get_base_retriever,
    get_mmr_retriever,
    get_contextual_retriever,
    get_ensemble_retriever,
    compare_retrievers
)

# ── Clases Mock Nativas de LangChain ──────────────────────────────────
class MockEmbeddings(Embeddings):
    """Falsify an embeddings model to pass Pydantic validation"""
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384

class MockRetriever(BaseRetriever):
    """Fakes a base retriever to pass Pydantic validation (Runnable)"""
    docs: List[Document] = []
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.docs

# ── Fixtures ────────────────────────────────────────────────────────
@pytest.fixture
def mock_vectorstore():
    """Simula el comportamiento de ChromaDB devolviendo nuestro Mock nativo"""
    vs = MagicMock()
    # We assign LangChain's native Mock instead of MagicMock()
    vs.as_retriever.return_value = MockRetriever(docs=[Document(page_content="resultado simulado")])
    return vs

@pytest.fixture
def sample_docs():
    """Test documents to initialize BM25"""
    return [
        Document(page_content="Requirement 1: System shall start in 5s.", metadata={"source": "doc1"}),
        Document(page_content="Requirement 2: The system shall not crash.", metadata={"source": "doc2"})
    ]

# ── Tests ───────────────────────────────────────────────────────────
def test_get_base_retriever(mock_vectorstore):
    retriever = get_base_retriever(mock_vectorstore, k=3)
    mock_vectorstore.as_retriever.assert_called_with(search_type="similarity", search_kwargs={"k": 3})
    assert retriever is not None

def test_get_mmr_retriever(mock_vectorstore):
    retriever = get_mmr_retriever(mock_vectorstore, k=3, fetch_k=15)
    mock_vectorstore.as_retriever.assert_called_with(
        search_type="mmr", 
        search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.6}
    )
    assert retriever is not None

@patch("src.retriever.HuggingFaceEmbeddings")
def test_get_contextual_retriever(mock_embeddings_class, mock_vectorstore):
    mock_embeddings_class.return_value = MockEmbeddings()
    retriever = get_contextual_retriever(mock_vectorstore, k=2)
    assert retriever is not None

def test_get_ensemble_retriever(mock_vectorstore, sample_docs):
    retriever = get_ensemble_retriever(mock_vectorstore, sample_docs, k=2)
    assert retriever is not None
    assert len(retriever.retrievers) == 2
    assert retriever.weights == [0.6, 0.4]

@patch("src.retriever.HuggingFaceEmbeddings")
def test_compare_retrievers(mock_embeddings_class, mock_vectorstore, sample_docs):
    mock_embeddings_class.return_value = MockEmbeddings()
    
    # We make the mock return the sample_docs through the MockRetriever
    mock_vectorstore.as_retriever.return_value = MockRetriever(docs=sample_docs)
    
    results = compare_retrievers(mock_vectorstore, sample_docs, query="start", k=2)
    
    assert "base_similarity" in results
    assert "mmr" in results
    assert "contextual" in results
    assert "ensemble" in results