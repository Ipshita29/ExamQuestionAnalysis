import os
import chromadb
from chromadb.utils import embedding_functions
from config import settings


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _get_embedding_fn():
    return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=settings.google_api_key,
        model_name=settings.embedding_model,
    )


def _load_docs() -> list[dict]:
    docs = []
    docs_dir = os.path.abspath(settings.rag_docs_dir)
    for fname in os.listdir(docs_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(docs_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            raw = f.read()
        chunks = _chunk_text(raw, chunk_size=800, overlap=100)
        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"{fname}_chunk_{i}",
                    "text": chunk,
                    "metadata": {"source": fname},
                }
            )
    return docs


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def seed_knowledge_base() -> None:
    client = _get_client()
    embed_fn = _get_embedding_fn()
    collection = client.get_or_create_collection(
        name=settings.rag_collection_name,
        embedding_function=embed_fn,
    )

    if collection.count() > 0:
        print(f"[RAG] Knowledge base already seeded ({collection.count()} chunks). Skipping.")
        return

    docs = _load_docs()
    if not docs:
        print("[RAG] No documents found in docs directory. Skipping seed.")
        return

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    print(f"[RAG] Knowledge base seeded with {len(docs)} chunks from {settings.rag_docs_dir}.")
