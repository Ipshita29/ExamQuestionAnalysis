import chromadb
from chromadb.utils import embedding_functions
from config import settings


def _get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embed_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=settings.google_api_key,
        model_name=settings.embedding_model,
    )
    return client.get_or_create_collection(
        name=settings.rag_collection_name,
        embedding_function=embed_fn,
    )


def retrieve(query: str, k: int | None = None) -> str:
    top_k = k if k is not None else settings.rag_top_k
    collection = _get_collection()

    if collection.count() == 0:
        return "No pedagogical knowledge base found. Please restart the server to seed it."

    results = collection.query(query_texts=[query], n_results=top_k)
    passages = results.get("documents", [[]])[0]

    if not passages:
        return "No relevant pedagogical guidance found for this query."

    return "\n\n---\n\n".join(passages)
