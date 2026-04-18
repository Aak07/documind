"""
Embedding generation using HuggingFace sentence-transformers.
Runs locally — no API cost for embeddings.
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings

# Module-level cache so we don't reload the model every time
_embeddings_model = None

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Get or create the embeddings model (cached)."""
    global _embeddings_model

    if _embeddings_model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _embeddings_model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}, # For cosine similarity
        )
        print("Embedding model loaded.")
    return _embeddings_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    model = get_embeddings_model()

    embeddings = model.embed_documents(texts)
    return embeddings

def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query."""
    model = get_embeddings_model()
    return model.embed_query(query)