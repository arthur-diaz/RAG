"""Settings persistence for the RAG app."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

# Settings file location at the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = BASE_DIR / ".rag_settings.json"


@dataclass
class AppSettings:
    # User-configurable values with safe defaults.
    data_dir: str = "data"
    persist_dir: str = "storage"
    similarity_top_k: int = 8
    rerank_top_n: int = 4
    file_types: str = ".pdf"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    openai_model: str = "gpt-4o-mini"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    # Number of recent turns used as conversation context.
    max_turns: int = 3
    ingest_fingerprint: str = ""


def load_settings() -> AppSettings:
    # Load settings from disk, merging with defaults.
    if SETTINGS_PATH.exists():
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        merged = {**asdict(AppSettings()), **data}
        return AppSettings(**merged)
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    # Persist settings to disk for future runs.
    data = asdict(settings)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
