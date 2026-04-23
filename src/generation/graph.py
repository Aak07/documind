"""
LangGraph RAG workflow with self-correction.

Flow:
  retrieve → grade → generate → hallucination check
       ↑                              ↓
       └── re-retrieve if hallucination detected (max 2 retries)
"""

from langgraph.graph import StateGraph, END
from src.generation.state import GraphState
from src.generation.nodes import (
    retrieve_documents,
    grade_documents,
    generate_answer,
    check_hallucination,
)


def should_retry(state: dict) -> str:
    """Decide whether to retry retrieval or finish."""
    if state.get("is_hallucination", False):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

        if retry_count < max_retries:
            return "retry"

    return "finish"


def increment_retry(state: dict) -> dict:
    """Increment the retry counter."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    """Build and compile the RAG workflow graph."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade", grade_documents)
    graph.add_node("generate", generate_answer)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("increment_retry", increment_retry)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Add edges
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", "check_hallucination")

    # Conditional: retry or finish
    graph.add_conditional_edges(
        "check_hallucination",
        should_retry,
        {
            "retry": "increment_retry",
            "finish": END,
        }
    )

    graph.add_edge("increment_retry", "retrieve")

    return graph.compile()


# Build once, reuse
rag_chain = build_graph()


def query(question: str) -> dict:
    """
    Run a query through the RAG pipeline.
    Returns the full state including answer, sources, and metrics.
    """
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "relevance_scores": [],
        "is_hallucination": False,
        "answer_is_useful": False,
        "route": "vectorstore",
        "retry_count": 0,
        "max_retries": 2,
        "cost_usd": 0.0,
        "latency_ms": {},
    }

    result = rag_chain.invoke(initial_state)
    return result