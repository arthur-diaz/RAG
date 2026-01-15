"""Document ingestion and index maintenance."""

from pathlib import Path
from typing import List
import hashlib

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from rag.llm import make_embedding
from rag.settings import AppSettings


# Build a fingerprint of the data folder + embedding config to avoid re-ingesting.

def _fingerprint(settings: AppSettings) -> str:
    data_path = Path(settings.data_dir)
    parts = [
        settings.embedding_provider,
        settings.embedding_model,
        str(settings.chunk_size),
        str(settings.chunk_overlap),
        settings.file_types,
    ]
    exts = _parse_file_types(settings.file_types)
    if data_path.exists():
        for path in sorted(data_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue
            stat = path.stat()
            parts.append(f"{path.as_posix()}|{stat.st_size}|{int(stat.st_mtime)}")
    payload = "".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Normalize file extensions from the settings string.

def _parse_file_types(file_types: str) -> List[str]:
    exts = []
    for part in file_types.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = f".{cleaned}"
        exts.append(cleaned.lower())
    return exts or [".pdf"]


# Ingest documents and update the vector index incrementally.

def ingest_documents(settings: AppSettings, openai_key: str, mistral_key: str) -> int:
    data_path = Path(settings.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    embed_key = openai_key if settings.embedding_provider == "openai" else mistral_key
    embed_model = make_embedding(settings.embedding_provider, settings.embedding_model, embed_key)

    Settings.embed_model = embed_model
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap

    fingerprint = _fingerprint(settings)
    if settings.ingest_fingerprint == fingerprint:
        return 0

    reader = SimpleDirectoryReader(
        input_dir=str(data_path),
        recursive=True,
        filename_as_id=True,
        required_exts=_parse_file_types(settings.file_types),
    )
    documents = reader.load_data()
    if not documents:
        raise ValueError("No documents loaded. Check folder and file types.")

    persist_path = Path(settings.persist_dir)
    index_store = persist_path / "index_store.json"

    if index_store.exists():
        # Incremental update when an index already exists.
        storage_context = StorageContext.from_defaults(persist_dir=settings.persist_dir)
        index = load_index_from_storage(storage_context)
        updated = 0
        inserted = 0
        for doc in documents:
            doc_id = doc.id_
            existing_hash = index.docstore.get_document_hash(doc_id)
            if existing_hash is None:
                index.insert(doc)
                inserted += 1
            elif existing_hash != doc.hash:
                index.delete_ref_doc(doc_id, raise_error=False)
                index.insert(doc)
                updated += 1
        index.storage_context.persist(persist_dir=settings.persist_dir)
        settings.ingest_fingerprint = fingerprint
        return inserted + updated

    # First-time index creation.
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=settings.persist_dir)
    settings.ingest_fingerprint = fingerprint
    return len(documents)
