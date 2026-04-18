"""
Text chunking strategies for RAG.
Recursive chunking splits on natural boundaries (paragraphs, sentences)
before falling back to character-level splits.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

def create_chunks(
        documents: List[Dict[str, Any]],
        chunk_size: int = None,
        chunk_overlap: int = None,
) -> List[Dict[str, Any]]:
    """
    Split documents into overlapping chunks.
    Each chunk inherits the metadata from its parent document.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        # Split this document's text
        splits = splitter.split_text(text)

        for i, chunk_text in enumerate(splits):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                **metadata,
                "chunk_index": i,
                "chunk_total": len(splits),
                "char_count": len(chunk_text),
                }
            })
            
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks