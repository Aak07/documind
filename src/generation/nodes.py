"""
Individual nodes for the LangGraph RAG workflow.
Each function takes the state and returns updated fields.
"""

import time
from typing import Dict, Any

from langchain_groq import ChatGroq
from src.config import settings
from src.retrieval.retriever import retrieve
from src.generation.prompts import (
    RAG_PROMPT,
    RELEVANCE_GRADER_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    ANSWER_GRADER_PROMPT,
)


def get_llm(model: str = None) -> ChatGroq:
    """Create a Groq LLM instance."""
    return ChatGroq(
        model=model or settings.default_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Retrieve relevant documents from the vector store."""
    start = time.time()

    question = state["question"]
    documents = retrieve(question, top_k=settings.top_k)

    latency = state.get("latency_ms", {})
    latency["retrieval"] = (time.time() - start) * 1000

    return {
        "documents": documents,
        "latency_ms": latency,
    }


def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Grade each retrieved document for relevance."""
    start = time.time()

    question = state["question"]
    documents = state["documents"]
    llm = get_llm(settings.fallback_model)  # Use cheaper model for grading

    relevant_docs = []
    scores = []

    for doc in documents:
        prompt = RELEVANCE_GRADER_PROMPT.format(
            document=doc["text"],
            question=question,
        )
        response = llm.invoke(prompt)
        grade = response.content.strip().lower()

        if grade == "yes":
            relevant_docs.append(doc)
            scores.append(1.0)
        else:
            scores.append(0.0)

    latency = state.get("latency_ms", {})
    latency["grading"] = (time.time() - start) * 1000

    return {
        "documents": relevant_docs,
        "relevance_scores": scores,
        "latency_ms": latency,
    }


def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Generate answer using relevant documents as context."""
    start = time.time()

    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # Build context from documents
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc["metadata"].get("source", "Unknown")
        page = doc["metadata"].get("page", "")
        page_str = f" (Page {page})" if page else ""
        context_parts.append(f"[Document {i+1} — {source}{page_str}]\n{doc['text']}")

    context = "\n\n".join(context_parts)

    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)

    latency = state.get("latency_ms", {})
    latency["generation"] = (time.time() - start) * 1000

    return {
        "generation": response.content,
        "latency_ms": latency,
    }


def check_hallucination(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Check if the generated answer is grounded in source documents."""
    start = time.time()

    documents = state["documents"]
    generation = state["generation"]
    llm = get_llm(settings.fallback_model)

    doc_texts = "\n\n".join([doc["text"] for doc in documents])
    prompt = HALLUCINATION_GRADER_PROMPT.format(
        documents=doc_texts,
        generation=generation,
    )

    response = llm.invoke(prompt)
    is_grounded = response.content.strip().lower() == "yes"

    latency = state.get("latency_ms", {})
    latency["hallucination_check"] = (time.time() - start) * 1000

    return {
        "is_hallucination": not is_grounded,
        "latency_ms": latency,
    }


def check_answer_quality(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node: Check if the answer actually addresses the question."""
    question = state["question"]
    generation = state["generation"]
    llm = get_llm(settings.fallback_model)

    prompt = ANSWER_GRADER_PROMPT.format(
        question=question,
        generation=generation,
    )

    response = llm.invoke(prompt)
    is_useful = response.content.strip().lower() == "yes"

    return {"answer_is_useful": is_useful}