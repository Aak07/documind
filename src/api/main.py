"""
FastAPI backend for DocuMind.
Decoupled from UI — any client can call this API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi import UploadFile, File
import tempfile
import os
from src.api.schemas import QueryRequest, QueryResponse, HealthResponse
from src.generation.graph import query as rag_query
from src.ingestion.store import get_collection_info
from contextlib import asynccontextmanager
from src.ingestion.embedder import warmup as warmup_embeddings

@asynccontextmanager
async def lifespan(app):
    """Pre-load models on startup so first query is fast."""
    print("Pre-warming embedding model...")
    warmup_embeddings()
    print("Ready to serve queries.")
    yield


app = FastAPI(
    title="DocuMind API",
    description="Production RAG pipeline with hybrid search and evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Streamlit or other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Accept file upload and run ingestion pipeline."""
    try:
        # Save to temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Run ingestion
        from src.ingestion.loader import load_document
        from src.ingestion.chunker import create_chunks
        from src.ingestion.store import upsert_chunks

        docs = load_document(tmp_path)
        chunks = create_chunks(docs)
        upsert_chunks(chunks)

        os.unlink(tmp_path)

        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service and its dependencies are healthy."""
    try:
        info = get_collection_info()
        return HealthResponse(
            status="healthy",
            qdrant_connected=True,
            collection_vectors=info["points_count"],
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            qdrant_connected=False,
            collection_vectors=0,
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Run a query through the RAG pipeline."""
    try:
        result = rag_query(request.question)

        sources = []
        for doc in result.get("documents", []):
            raw_page = doc["metadata"].get("page")
            clean_page = None
            if raw_page:  # This safely ignores None, "", and 0
                try:
                    clean_page = int(raw_page)
                except (ValueError, TypeError):
                    clean_page = None

            sources.append({
                "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "score": doc.get("score", 0.0),
                "source": doc["metadata"].get("source", ""),
                "page": clean_page,
            })

        return QueryResponse(
            question=request.question,
            answer=result["generation"],
            sources=sources,
            is_hallucination=result.get("is_hallucination", False),
            retry_count=result.get("retry_count", 0),
            latency_ms=result.get("latency_ms", {}),
            cost_usd=result.get("cost_usd", 0.0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint for monitoring."""
    try:
        info = get_collection_info()
        return {
            "collection": info,
            "status": "operational",
        }
    except Exception:
        return {"status": "error"}