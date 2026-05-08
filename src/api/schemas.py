"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")


class SourceDocument(BaseModel):
    text: str
    score: float
    source: str = ""
    page: Optional[int] = None


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