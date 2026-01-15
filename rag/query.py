"""Query execution with retrieval and reranking."""

from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank

from rag.llm import make_embedding, make_llm
from rag.settings import AppSettings


# Run a query against the persisted index with reranking.

def run_query(
    question: str,
    settings: AppSettings,
    llm_provider: str,
    llm_model: str,
    openai_key: str,
    mistral_key: str,
):
    persist_path = Path(settings.persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError("No index found. Run ingestion first.")

    embed_key = openai_key if settings.embedding_provider == "openai" else mistral_key
    embed_model = make_embedding(settings.embedding_provider, settings.embedding_model, embed_key)

    llm_key = openai_key if llm_provider == "openai" else mistral_key
    llm = make_llm(llm_provider, llm_model, llm_key)

    Settings.embed_model = embed_model
    Settings.llm = llm

    storage_context = StorageContext.from_defaults(persist_dir=settings.persist_dir)
    index = load_index_from_storage(storage_context)

    reranker = SentenceTransformerRerank(
        top_n=settings.rerank_top_n,
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    query_engine = index.as_query_engine(
        similarity_top_k=settings.similarity_top_k,
        node_postprocessors=[reranker],
    )
    return query_engine.query(question)
