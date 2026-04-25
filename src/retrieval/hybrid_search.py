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


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def dense_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Semantic search using dense vectors."""
    client = get_client()
    query_vector = embed_query(query)

    try:
        # ✅ New Qdrant API (v1.7+)
        response = client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=query_vector,
            limit=top_k,
        )
        results = response.points
        print("[DENSE] Using query_points()")

    except AttributeError:
        # ✅ Old Qdrant API fallback
        results = client.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        print("[DENSE] Using search()")

    return [
        {
            "text": r.payload.get("text", ""),
            "score": r.score,
            "metadata": {
                "source": r.payload.get("source", ""),
                "page": r.payload.get("page", ""),
                "chunk_index": r.payload.get("chunk_index", ""),
            },
            "id": r.id,
        }
        for r in results
    ]


def bm25_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Keyword search using BM25.
    Fetches all documents from Qdrant and runs BM25 locally.
    """
    client = get_client()

    # Fetch all documents (scroll through collection)
    all_points = []
    offset = None

    while True:
        scroll_result = client.scroll(
            collection_name=settings.qdrant_collection_name,
            limit=100,
            offset=offset,
        )

        # ✅ Handle both old + new Qdrant client versions
        if isinstance(scroll_result, tuple):
            points, offset = scroll_result
        else:
            points = scroll_result.points
            offset = scroll_result.next_page_offset

        all_points.extend(points)

        if offset is None:
            break

    print(f"[BM25] Total points fetched: {len(all_points)}")

    if not all_points:
        return []

    # Build BM25 index
    corpus = [p.payload.get("text", "") for p in all_points]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Search
    tokenized_query = query.lower().split()
    print(f"[BM25] Query tokens: {tokenized_query}")

    scores = bm25.get_scores(tokenized_query)

    # Rank results
    scored_docs = list(zip(all_points, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_results = scored_docs[:top_k]

    return [
        {
            "text": point.payload.get("text", ""),
            "score": float(score),
            "metadata": {
                "source": point.payload.get("source", ""),
                "page": point.payload.get("page", ""),
                "chunk_index": point.payload.get("chunk_index", ""),
            },
            "id": point.id,
        }
        for point, score in top_results
        if score > 0
    ]


def hybrid_search(
    query: str,
    top_k: int = 10,
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Combine dense and BM25 search results using Reciprocal Rank Fusion (RRF).
    """
    k = 60  # RRF constant

    # Get results
    dense_results = dense_search(query, top_k=top_k)
    sparse_results = bm25_search(query, top_k=top_k)

    print(f"[HYBRID] Dense results: {len(dense_results)}")
    print(f"[HYBRID] BM25 results: {len(sparse_results)}")

    # Build RRF scores
    rrf_scores = {}

    for rank, doc in enumerate(dense_results):
        doc_id = str(doc["id"])
        rrf_score = dense_weight * (1.0 / (k + rank + 1))
        rrf_scores[doc_id] = {"score": rrf_score, "doc": doc}

    for rank, doc in enumerate(sparse_results):
        doc_id = str(doc["id"])
        rrf_score = bm25_weight * (1.0 / (k + rank + 1))
        if doc_id in rrf_scores:
            rrf_scores[doc_id]["score"] += rrf_score
        else:
            rrf_scores[doc_id] = {"score": rrf_score, "doc": doc}

    # Sort by score
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Final results
    final_results = []
    for item in sorted_results[:top_k]:
        doc = item["doc"]
        doc["score"] = item["score"]
        final_results.append(doc)

    return final_results