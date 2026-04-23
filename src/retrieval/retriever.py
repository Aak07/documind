"""
Basic dense retrieval from Qdrant.
This is the V1 retriever — we'll add hybrid search and reranking later.
"""

from typing import List, Dict, Any
from qdrant_client import QdrantClient

from src.config import settings
from src.ingestion.embedder import embed_query


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents for a query using dense vector search.

    Returns a list of dicts with 'text', 'score', and 'metadata' keys.
    """
    top_k = top_k or settings.top_k
    client = get_client()

    # Embed the query
    query_vector = embed_query(query)

    # Search Qdrant
    results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    # Format results
    documents = []
    for result in results:
        documents.append({
            "text": result.payload.get("text", ""),
            "score": result.score,
            "metadata": {
                "source": result.payload.get("source", ""),
                "page": result.payload.get("page", ""),
                "chunk_index": result.payload.get("chunk_index", ""),
            },
        })

    return documents