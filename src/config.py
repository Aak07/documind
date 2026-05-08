"""
Centralized configuration using pydantic-settings.
All env vars are loaded here — no hardcoded keys anywhere else.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
import os

# FIX: Set HuggingFace cache directory to avoid re-downloads
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    default_model: str = "llama-3.1-8b-instant"
    fallback_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_api_token: str = Field(default="", env="HUGGINGFACE_API_TOKEN")

    # Reranker
    cohere_api_key: str = Field(default="", env="COHERE_API_KEY")

    # Vector DB
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_collection_name: str = Field(default="documind",
    env="QDRANT_COLLECTION_NAME")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    rerank_top_k: int = 3

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="documind", env="LANGCHAIN_PROJECT")

    # FIX: Pydantic V2 uses model_config dict, not inner Config class
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore", # Prevents errors if your .env has extra variables
        "populate_by_name": True,
    }

# Singleton — import this everywhere
settings = Settings()