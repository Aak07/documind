"""
Reranking using Cohere's rerank API.
Takes the hybrid search results and reorders them by true relevance.
This is the single biggest quality improvement you can add to a RAG pipeline.
"""

from typing import List, Dict, Any
import cohere
from src.config import settings


def rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = None,
) -> List[Dict[str, Any]]:

    top_k = top_k or settings.rerank_top_k

    if not documents:
        return documents

    if not settings.cohere_api_key:
        print("[RERANK] WARNING: No Cohere API key — skipping reranking")
        return sorted(documents, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    try:
        # ✅ Version-safe client init
        try:
            client = cohere.ClientV2(api_key=settings.cohere_api_key)
            print("[RERANK] Using Cohere ClientV2")
        except AttributeError:
            client = cohere.Client(settings.cohere_api_key)
            print("[RERANK] Using Cohere Client (v1/v4)")

        doc_texts = [doc["text"] for doc in documents]

        print(f"[RERANK] Input docs: {len(documents)}")

        response = client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=doc_texts,
            top_n=top_k,
        )

        reranked_docs = []

        for result in response.results:
            idx = getattr(result, "index", None)
            score = getattr(result, "relevance_score", None)

            if idx is None:
                continue

            doc = documents[idx].copy()
            doc["rerank_score"] = float(score) if score is not None else 0.0
            reranked_docs.append(doc)

        print(f"[RERANK] Reranked docs: {len(reranked_docs)}")

        return reranked_docs

    except Exception as e:
        print(f"[RERANK] WARNING: Failed ({e}) — fallback to original ranking")

        return sorted(
            documents,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:top_k]