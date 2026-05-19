from pydantic_settings import BaseSettings
from pydantic import Field
import os

os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    default_model: str = "llama-3.1-8b-instant"
    fallback_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_api_token: str = Field(default="", alias="HUGGINGFACE_API_TOKEN")

    # Reranker
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")

    # Vector DB
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection_name: str = Field(default="documind", alias="QDRANT_COLLECTION_NAME")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    rerank_top_k: int = 3

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="documind", alias="LANGCHAIN_PROJECT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


settings = Settings()