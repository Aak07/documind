"""
Hybrid search combining dense vectors (semantic) + BM25 (keyword).
This handles the cases where pure semantic search misses exact term matches
(e.g., searching for "SCD Type 2" — semantic search might miss, BM25 won't).
"""

"""
Hybrid search combining dense vectors (semantic) + BM25 (keyword).
This handles the cases where pure semantic search misses exact term matches
(e.g., searching for "SCD Type 2" — semantic search might miss, BM25 won't).
"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from src.config import settings
from src.ingestion.embedder import embed_query
import time

def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)

def dense_candidate_search(query: str, fetch_k: int = 50) -> List[Any]:
    """Step 1: Fetch Top-50 broad candidates extremely fast via Dense Vectors."""
    client = get_client()

    t0 = time.time()
    
    # 1. Profile HuggingFace Embeddings
    query_vector = embed_query(query)
    t1 = time.time()
    print(f"  [Micro-Profile] HF Embedding took: {t1 - t0:.2f}s")
    
    # client.query_points returns a QueryResponse object
    # 2. Profile Qdrant Database Network Ping
    response = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,   
        limit=fetch_k,        
    )
    t2 = time.time()
    print(f"  [Micro-Profile] Qdrant Network ping took: {t2 - t1:.2f}s")

    # We must return the .points attribute to get the list of ScoredPoint objects
    return response.points

def sequential_hybrid_search(query: str, top_k: int = 10, fetch_k: int = 50) -> List[Dict[str, Any]]:
    """
    Step 2: Run local BM25 ONLY on the Top-50 Dense Candidates.
    Eliminates the scroll() bottleneck completely!
    """
    # 1. Get Candidates
    candidate_points = dense_candidate_search(query, fetch_k=fetch_k)
    
    if not candidate_points:
        return []

    # 2. Extract text for BM25 processing (ensure payload is accessed safely)
    corpus = [point.payload.get("text", "") for point in candidate_points if point.payload]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    
    # 3. Run BM25 locally just on these 50 items
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # 4. Apply Reciprocal Rank Fusion (RRF)
    k = 60
    rrf_scores = {}
    
    # Score Dense results (Rank based on original fetch order)
    for rank, point in enumerate(candidate_points):
        doc_id = str(point.id)
        # Weight Dense slightly higher (0.7)
        dense_score = 0.7 * (1.0 / (k + rank + 1))
        rrf_scores[doc_id] = {"score": dense_score, "doc": point, "text": corpus[rank]}
        
    # Rank BM25 results
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    
    for rank, idx in enumerate(bm25_ranked_indices):
        if bm25_scores[idx] > 0:
            doc_id = str(candidate_points[idx].id)
            sparse_score = 0.3 * (1.0 / (k + rank + 1))
            # Use .get() to avoid KeyErrors if a doc only appeared in BM25 results
            if doc_id in rrf_scores:
                rrf_scores[doc_id]["score"] += sparse_score

    # 5. Sort by final fused RRF score
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # 6. Return Top K formatting
    final_results = []
    for item in sorted_results[:top_k]:
        point = item["doc"]
        final_results.append({
            "text": item["text"],
            "score": item["score"],
            "metadata": {
                "source": point.payload.get("source", "") if point.payload else "",
                "page": point.payload.get("page", "") if point.payload else "",
                "chunk_index": point.payload.get("chunk_index", "") if point.payload else "",
            },
            "id": point.id,
        })
        
    return final_results
