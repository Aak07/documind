"""
Embedding generation using HuggingFace sentence-transformers.
Runs locally — no API cost for embeddings.
"""

#import os
import threading
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings

class _EmbeddingService:
    """
    Thread-safe singleton embedding service.
    The model loads ONCE into memory and stays there.
    This eliminates the 24s model reload on every query.
    """
    _instance = None
    _lock = threading.Lock()
    _model: HuggingFaceEmbeddings = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:  # Double-check after acquiring lock
                    print(f"Loading embedding model: {settings.embedding_model}")
                    self._model = HuggingFaceEmbeddings(
                        model_name=settings.embedding_model,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={
                            "normalize_embeddings": True, # For cosine similarity
                            "batch_size": 64,
                        },
                    )
                    print("Embedding model loaded.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        return self._model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        self._ensure_loaded()
        return self._model.embed_query(query)

    def get_model(self) -> HuggingFaceEmbeddings:
        self._ensure_loaded()
        return self._model

# Module-level singleton
_service = _EmbeddingService()

def get_embeddings_model() -> HuggingFaceEmbeddings:
    return _service.get_model()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    return _service.embed_texts(texts)

def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query."""
    return _service.embed_query(query)

def warmup():
    """Call this at app startup to pre-load the model."""
    _service._ensure_loaded()