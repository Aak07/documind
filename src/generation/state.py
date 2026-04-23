"""
State schema for the LangGraph RAG workflow.
Every node reads from and writes to this state.
"""

from typing import TypedDict, List, Dict, Any, Optional


class GraphState(TypedDict):
    """State that flows through the RAG graph."""
    # Input
    question: str

    # Retrieval
    documents: List[Dict[str, Any]]

    # Generation
    generation: str

    # Quality checks
    relevance_scores: List[float]
    is_hallucination: bool
    answer_is_useful: bool

    # Control flow
    route: str              # "vectorstore" | "direct"
    retry_count: int
    max_retries: int

    # Observability
    cost_usd: float
    latency_ms: Dict[str, float]