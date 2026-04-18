"""
Qdrant vector store operations.
Handles collection creation, upserting documents, and basic search.
"""

from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
Distance, VectorParams, PointStruct, Filter
)
import uuid
from src.config import settings
from src.ingestion.embedder import embed_texts

# Module-level client
_client = None

def get_client() -> QdrantClient:
    """Get or create the Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client

def create_collection(vector_size: int = 384):
    """
    Create the Qdrant collection if it doesn't exist.
    384 = dimension of all-MiniLM-L6-v2 embeddings.
    """

    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection_name not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {settings.qdrant_collection_name}")
    else:
        print(f"Collection already exists: {settings.qdrant_collection_name}")

def upsert_chunks(chunks: List[Dict[str, Any]], batch_size: int = 64):
    """
    Embed chunks and upsert into Qdrant.
    Processes in batches to handle large document sets.
    """

    client = get_client()
    create_collection()

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        # Generate embeddings for this batch
        embeddings = embed_texts(texts)

        # Create Qdrant points
        points = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                "text": chunk["text"],
                **chunk["metadata"],
                }
            )
            points.append(point)

        # Upsert batch
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
        )

        print(f" Upserted batch {i // batch_size + 1}/{(total + batch_size - 1)// batch_size}")
        print(f"Total points upserted: {total}")

def get_collection_info() -> dict:
    """Get collection statistics."""

    client = get_client()
    info = client.get_collection(settings.qdrant_collection_name)
    return {
        "name": settings.qdrant_collection_name,
        "points_count": info.points_count,
        "vectors_count": info.points_count,
        "status": info.status,
    }