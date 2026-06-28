from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import os


def get_base_retriever(vectorstore: Chroma, k: int = 4):
    """Cosine similarity retriever. Baseline."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_mmr_retriever(vectorstore: Chroma, k: int = 4, fetch_k: int = 20):
    """
    Maximal Marginal Relevance retriever.
    Balances relevance with diversity to avoid redundant chunks.
    Best for documents with repetitive sections.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.6}
        # lambda_mult: 1.0 = pure similarity, 0.0 = pure diversity
    )


def get_contextual_retriever(vectorstore: Chroma, k: int = 4):
    """
    Retrieves target chunk + its immediate neighbors in the original doc.
    Critical for technical specs where context spans multiple paragraphs.
    """
    base = get_base_retriever(vectorstore, k=k)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Filter out chunks below similarity threshold before returning
    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.55
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base
    )


def get_ensemble_retriever(vectorstore: Chroma, all_docs: List[Document], k: int = 4):
    """
    Hybrid retriever: semantic (embeddings) + lexical (BM25).
    Semantic excels at conceptual queries.
    BM25 excels at exact terms: codes, IDs, part numbers.
    Weights: 60% semantic, 40% BM25 — tuned for technical docs.
    """
    semantic = get_base_retriever(vectorstore, k=k)
    bm25 = BM25Retriever.from_documents(all_docs, k=k)
    
    return EnsembleRetriever(
        retrievers=[semantic, bm25],
        weights=[0.6, 0.4]
    )


def compare_retrievers(vectorstore: Chroma, all_docs: List[Document], query: str, k: int = 4) -> dict:
    """
    Benchmarking utility: runs the same query through all retriever strategies.
    Returns a dict with retrieved chunks per strategy for qualitative comparison.
    Use this to generate README benchmarks.
    """
    results = {}
    
    retrievers = {
        "base_similarity": get_base_retriever(vectorstore, k),
        "mmr": get_mmr_retriever(vectorstore, k),
        "contextual": get_contextual_retriever(vectorstore, k),
        "ensemble": get_ensemble_retriever(vectorstore, all_docs, k),
    }
    
    for name, retriever in retrievers.items():
        try:
            docs = retriever.invoke(query)
            results[name] = {
                "num_chunks": len(docs),
                "sources": [d.metadata.get("source", "unknown") for d in docs],
                "preview": [d.page_content[:120] + "..." for d in docs]
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results