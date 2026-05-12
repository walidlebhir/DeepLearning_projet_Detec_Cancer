import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cancerlistedata"
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


def get_embedding_model(model_name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.Client:
    directory = str(persist_directory or DEFAULT_CHROMA_DIR)
    os.makedirs(directory, exist_ok=True)
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=directory))


def get_cancer_collection(client: chromadb.Client, collection_name: str = COLLECTION_NAME) -> chromadb.api.models.Collection:
    return client.get_or_create_collection(name=collection_name)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = text.replace("\n", " ")
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def extract_text_from_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required to read PDF files. Install it with pip install PyPDF2.")
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    raise ValueError(f"Unsupported file type: {ext}. Only .txt and .pdf are supported.")


def upload_document(
    file_path: str,
    persist_directory: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 500,
    overlap: int = 100,
) -> Dict[str, object]:
    text = extract_text_from_file(file_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No text extracted from the document.")

    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

    client = get_chroma_client(persist_directory)
    collection = get_cancer_collection(client, collection_name)

    ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": str(file_path), "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )
    client.persist()

    return {
        "file_path": str(file_path),
        "chunks_indexed": len(chunks),
        "collection_name": collection_name,
        "store_directory": str(persist_directory or DEFAULT_CHROMA_DIR),
    }


def handle_uploaded_file(
    file_name: str,
    file_bytes: bytes,
    upload_dir: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> Dict[str, object]:
    directory = Path(upload_dir or Path(__file__).resolve().parent.parent / "uploaded_documents")
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / file_name
    file_path.write_bytes(file_bytes)
    return upload_document(str(file_path), persist_directory=persist_directory)


def search_similar_documents(
    query: str,
    persist_directory: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    n_results: int = 5,
) -> Dict[str, object]:
    client = get_chroma_client(persist_directory)
    collection = get_cancer_collection(client, collection_name)

    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "query": query,
        "results": [
            {
                "document": doc,
                "metadata": metadata,
                "distance": distance,
            }
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ],
    }


def answer_question(
    question: str,
    persist_directory: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
    n_results: int = 5,
) -> Dict[str, object]:
    search_result = search_similar_documents(
        question,
        persist_directory=persist_directory,
        collection_name=collection_name,
        n_results=n_results,
    )

    return {
        "question": question,
        "retrieved_chunks": search_result["results"],
        "note": "Cette fonction retourne les passages recherchés pour répondre à la question. Ajoute un LLM pour formuler une réponse complète si nécessaire.",
    }
