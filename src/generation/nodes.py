import time
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from src.config import settings
from src.retrieval.retriever import retrieve
from src.observability.cost_tracker import CostTracker
from src.observability.logger import logger
from src.generation.prompts import (
    RAG_PROMPT,
    RELEVANCE_GRADER_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    ANSWER_GRADER_PROMPT,
)

# Thread pool for parallel LLM calls (Section 7)
_grading_pool = ThreadPoolExecutor(max_workers=3)


def get_llm(model: str = None) -> ChatGroq:
    """Create a Groq LLM instance."""
    return ChatGroq(
        model=model or settings.default_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def _call_llm_tracked(
    llm: ChatGroq,
    prompt: str,
    tracker: CostTracker,
    stage: str,
    model_name: str = None,
) -> str:
    """
    Call LLM and record cost. Every LLM call in the pipeline goes through this.
    This is how observability gets wired in without touching the graph structure.
    """
    response = llm.invoke(prompt)
    output_text = response.content

    tracker.record_call(
        model=model_name or settings.default_model,
        input_text=prompt,
        output_text=output_text,
        stage=stage,
    )

    return output_text


# =====================================================
# NODE: RETRIEVE DOCUMENTS
# =====================================================

def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve relevant documents from vector store."""
    start = time.time()

    question = state["question"]
    documents = retrieve(question, top_k=settings.rerank_top_k)

    latency = state.get("latency_ms", {}).copy()
    latency["retrieval"] = round((time.time() - start) * 1000, 1)

    logger.info("retrieval_complete", num_docs=len(documents), latency_ms=latency["retrieval"])

    return {
        "documents": documents,
        "latency_ms": latency,
    }


# =====================================================
# NODE: GRADE DOCUMENTS
# =====================================================

def _grade_single_doc(
    doc: Dict[str, Any],
    question: str,
    tracker: CostTracker,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Grade a single document. Returns (doc_or_None, score).
    Uses rerank_score to skip LLM call when confidence is clear.
    """
    rerank_score = doc.get("rerank_score", -1)

    # HIGH confidence from reranker — auto-accept, no LLM call needed
    if rerank_score > 0.5:
        return (doc, 1.0)

    # LOW confidence — auto-reject
    if 0 <= rerank_score < 0.05:
        return (None, 0.0)

    # MEDIUM confidence OR no rerank score — ask LLM
    try:
        llm = get_llm(settings.fallback_model)
        output = _call_llm_tracked(
            llm=llm,
            prompt=RELEVANCE_GRADER_PROMPT.format(
                document=doc["text"][:500],  # Truncate to save tokens
                question=question,
            ),
            tracker=tracker,
            stage="grading",
            model_name=settings.fallback_model,
        )
        grade = output.strip().lower()
        if "yes" in grade:
            return (doc, 1.0)
        else:
            return (None, 0.0)
    except Exception as e:
        # On LLM failure, keep the doc (fail-open for retrieval quality)
        logger.warning("grading_llm_failed", error=str(e))
        return (doc, 0.5)


def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grade documents using parallel ThreadPoolExecutor.
    High-confidence reranker results skip LLM calls entirely.
    """
    start = time.time()
    question = state["question"]
    documents = state["documents"]

    # Initialize tracker for this query (or get existing one)
    tracker = CostTracker()
    existing_usage = state.get("token_usage", {})

    if not documents:
        return {
            "documents": [],
            "relevance_scores": [],
            "latency_ms": {**state.get("latency_ms", {}), "grading": 0},
            "token_usage": existing_usage,
        }

    # Submit all grading tasks to thread pool (Section 7)
    futures = {}
    for i, doc in enumerate(documents):
        future = _grading_pool.submit(_grade_single_doc, doc, question, tracker)
        futures[future] = i

    # Collect results IN ORDER (preserves ranking from reranker)
    results = [None] * len(documents)
    for future in as_completed(futures):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            logger.warning("grading_future_failed", index=idx, error=str(e))
            results[idx] = (documents[idx], 0.5)  # Fail-open

    relevant_docs = [r[0] for r in results if r[0] is not None]
    scores = [r[1] for r in results]

    latency = state.get("latency_ms", {}).copy()
    latency["grading"] = round((time.time() - start) * 1000, 1)

    # Merge cost tracking
    cost_summary = tracker.get_summary()
    merged_usage = {**existing_usage, "grading": cost_summary}

    logger.info(
        "grading_complete",
        total_docs=len(documents),
        relevant_docs=len(relevant_docs),
        llm_calls=cost_summary["num_calls"],
        latency_ms=latency["grading"],
    )

    return {
        "documents": relevant_docs,
        "relevance_scores": scores,
        "latency_ms": latency,
        "token_usage": merged_usage,
    }


# =====================================================
# NODE: GENERATE ANSWER
# =====================================================

def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer using relevant documents as context."""
    start = time.time()

    question = state["question"]
    documents = state["documents"]
    tracker = CostTracker()

    # Build context
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc["metadata"].get("source", "Unknown")
        page = doc["metadata"].get("page", "")
        page_str = f", Page {page}" if page else ""
        context_parts.append(f"[Document {i+1} — {source}{page_str}]\n{doc['text']}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

    prompt = RAG_PROMPT.format(context=context, question=question)

    output = _call_llm_tracked(
        llm=get_llm(),
        prompt=prompt,
        tracker=tracker,
        stage="generation",
        model_name=settings.default_model,
    )

    latency = state.get("latency_ms", {}).copy()
    latency["generation"] = round((time.time() - start) * 1000, 1)

    # Merge costs
    existing_usage = state.get("token_usage", {})
    cost_summary = tracker.get_summary()
    merged_usage = {**existing_usage, "generation": cost_summary}

    logger.info("generation_complete", latency_ms=latency["generation"])

    return {
        "generation": output,
        "latency_ms": latency,
        "cost_usd": state.get("cost_usd", 0) + cost_summary["total_cost_usd"],
        "token_usage": merged_usage,
    }


# =====================================================
# NODE: HALLUCINATION CHECK
# =====================================================

def check_hallucination(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if generated answer is grounded in source documents."""
    start = time.time()

    documents = state["documents"]
    generation = state["generation"]
    tracker = CostTracker()

    # If no documents were retrieved, flag as hallucination
    if not documents:
        return {
            "is_hallucination": True,
            "latency_ms": {**state.get("latency_ms", {}), "hallucination_check": 0},
        }

    doc_texts = "\n\n".join([doc["text"][:800] for doc in documents])  # Truncate for cost

    output = _call_llm_tracked(
        llm=get_llm(settings.fallback_model),
        prompt=HALLUCINATION_GRADER_PROMPT.format(
            documents=doc_texts,
            generation=generation,
        ),
        tracker=tracker,
        stage="hallucination_check",
        model_name=settings.fallback_model,
    )

    is_grounded = "yes" in output.strip().lower()

    latency = state.get("latency_ms", {}).copy()
    latency["hallucination_check"] = round((time.time() - start) * 1000, 1)

    existing_usage = state.get("token_usage", {})
    cost_summary = tracker.get_summary()
    merged_usage = {**existing_usage, "hallucination_check": cost_summary}

    logger.info(
        "hallucination_check_complete",
        is_grounded=is_grounded,
        latency_ms=latency["hallucination_check"],
    )

    return {
        "is_hallucination": not is_grounded,
        "latency_ms": latency,
        "cost_usd": state.get("cost_usd", 0) + cost_summary["total_cost_usd"],
        "token_usage": merged_usage,
    }