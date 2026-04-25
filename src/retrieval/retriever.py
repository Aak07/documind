"""
Main retriever — orchestrates hybrid search → reranking → top-k.
"""

from typing import List, Dict, Any

from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Hybrid search (dense + BM25 with RRF fusion)
    2. Cohere reranking
    3. Return top-k most relevant documents
    """
    top_k = top_k or settings.rerank_top_k  # Final top_k after reranking

    # Step 1: Get candidates from hybrid search (fetch more than we need)
    candidates = hybrid_search(query, top_k=settings.top_k)

    # Step 2: Rerank candidates
    reranked = rerank(query, candidates, top_k=top_k)

    return reranked