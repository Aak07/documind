"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")


class SourceDocument(BaseModel):
    text: str
    score: float
    source: str = ""
    page: Optional[int] = None

    # NO BACKSLASHES HERE! Just normal quotes.
    @field_validator("page", mode="before")
    @classmethod
    def parse_page(cls, v):
        # Catch empty strings or None and convert to a clean Python None
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    is_hallucination: bool
    retry_count: int
    latency_ms: Dict[str, float]
    cost_usd: float


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_vectors: int