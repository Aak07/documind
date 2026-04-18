"""
Main ingestion script.
Usage: python -m src.ingestion.ingest --dir data/sample_docs/
"""
import argparse
import time
from src.ingestion.loader import load_directory, load_document
from src.ingestion.chunker import create_chunks
from src.ingestion.store import upsert_chunks, get_collection_info

def ingest(path: str):
    """Run the full ingestion pipeline."""
    start = time.time()

    # Step 1: Load documents
    print("\n=== Step 1: Loading Documents ===")
    import os
    if os.path.isdir(path):
        documents = load_directory(path)
    else:
        documents = load_document(path)

    if not documents:
        print("No documents found. Exiting.")
        return

    # Step 2: Chunk documents
    print("\n=== Step 2: Chunking Documents ===")
    chunks = create_chunks(documents)

    # Step 3: Embed and store
    print("\n=== Step 3: Embedding & Storing in Qdrant ===")
    upsert_chunks(chunks)

    # Summary
    elapsed = time.time() - start
    info = get_collection_info()
    print(f"\n=== Ingestion Complete ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"Collection: {info['name']}")
    print(f"Total vectors: {info['points_count']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into DocuMind")
    parser.add_argument("--dir", type=str, default="data/sample_docs/",
                        help="Path to documents directory or single file")
    args = parser.parse_args()
    ingest(args.dir)
