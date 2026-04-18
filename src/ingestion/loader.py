"""
Document loaders for PDF, DOCX, and TXT files.
Uses a factory pattern — add new formats by adding a loader function.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import fitz # PyMuPDF

def load_pdf(file_path: str) -> List[Dict[str, Any]]:

    """Extract text from PDF with page-level metadata."""
    documents = []
    doc = fitz.open(file_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip(): # Skip empty pages
            documents.append({
                "text": text.strip(),
                "metadata": {
                "source": os.path.basename(file_path),
                "page": page_num + 1,
                "total_pages": len(doc),
                "file_path": file_path,
                }
            })
    doc.close() 
    return documents

def load_txt(file_path: str) -> List[Dict[str, Any]]:
    """Load plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{
    "text": text.strip(),
    "metadata": {
    "source": os.path.basename(file_path),
    "file_path": file_path,
    }
    }]

# Factory: maps file extension to loader function
LOADERS = {
".pdf": load_pdf,
".txt": load_txt,
}

def load_document(file_path: str) -> List[Dict[str, Any]]:
    """Load a document based on its file extension."""
    ext = Path(file_path).suffix.lower()

    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}. Supported:{list(LOADERS.keys())}")
    return LOADERS[ext](file_path)

def load_directory(dir_path: str) -> List[Dict[str, Any]]:
    """Load all supported documents from a directory."""
    all_documents = []

    for file_name in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file_name)
        ext = Path(file_path).suffix.lower()

        if ext in LOADERS:
            print(f"Loading: {file_name}")
            docs = load_document(file_path)
            all_documents.extend(docs)
            print(f" → Extracted {len(docs)} chunks")
            
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents