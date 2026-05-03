"""
Main retriever — orchestrates hybrid search → reranking → top-k.
"""

from typing import List, Dict, Any
import time
from src.config import settings
from src.retrieval.hybrid_search import sequential_hybrid_search
from src.retrieval.reranker import rerank


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline:
    1. Hybrid search (dense + BM25 with RRF fusion)
    2. Cohere reranking
    3. Return top-k most relevant documents
    """
    top_k = top_k or settings.rerank_top_k  # Final top_k after reranking

    print("\n" + "="*40)
    print("🕵️  STARTING LATENCY PROFILER")

    # Step 1: Get candidates from hybrid search (fetch more than we need)
    t0 = time.time()
    candidates = sequential_hybrid_search(query, top_k=settings.top_k)
    t1 = time.time()
    print(f"⏱️ Sequential Hybrid Search took: {t1 - t0:.2f} seconds")

    # Step 2: Rerank candidates
    reranked = rerank(query, candidates, top_k=top_k)
    t2 = time.time()
    print(f"⏱️ Cohere Reranking API took: {t2 - t1:.2f} seconds")
    print("="*40 + "\n")

    return reranked