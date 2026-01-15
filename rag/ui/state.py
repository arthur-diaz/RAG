"""Session state and persistence helpers."""

from pathlib import Path
import json
import os
import uuid

import streamlit as st

from rag.settings import AppSettings

# Base paths for local storage (project root).
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
CONVERSATIONS_PATH = BASE_DIR / ".rag_conversations.json"
# Legacy files created in older versions (under rag/).
LEGACY_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
LEGACY_CONVERSATIONS_PATH = Path(__file__).resolve().parents[1] / ".rag_conversations.json"


def load_env_file() -> None:
    # Load key/value pairs from .env into process env (migrate from legacy if needed).
    env_path = ENV_PATH if ENV_PATH.exists() else LEGACY_ENV_PATH
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    # If we loaded from legacy, copy into the new location.
    if env_path == LEGACY_ENV_PATH and not ENV_PATH.exists():
        ENV_PATH.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")


def sync_api_key_from_env() -> None:
    # Pull OpenAI key from env into session state when available.
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key and not st.session_state.get("openai_key"):
        st.session_state.openai_key = env_key


def write_env_file(updates: dict) -> None:
    # Persist key/value pairs to .env for local use.
    current = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            current[key.strip()] = value.strip().strip('"')
    current.update({k: v for k, v in updates.items() if v is not None})
    lines = [f"{key}={value}" for key, value in current.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delete_env_key(key: str) -> None:
    # Remove a key from .env while keeping other entries.
    if not ENV_PATH.exists():
        return
    current = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, value = raw.split("=", 1)
        k = k.strip()
        if k != key:
            current[k] = value.strip().strip('"')
    lines = [f"{k}={v}" for k, v in current.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_openai_key() -> str:
    # Prefer session key, fallback to process env.
    return st.session_state.get("openai_key") or os.getenv("OPENAI_API_KEY", "")


def load_conversations_file() -> dict:
    # Load conversations from disk for persistence (migrate from legacy if needed).
    path = CONVERSATIONS_PATH if CONVERSATIONS_PATH.exists() else LEGACY_CONVERSATIONS_PATH
    if not path.exists():
        return {"order": [], "conversations": {}}
    try:
        data = path.read_text(encoding="utf-8")
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            return {"order": [], "conversations": {}}
        order = parsed.get("order", [])
        conversations = parsed.get("conversations", {})
        if not isinstance(order, list) or not isinstance(conversations, dict):
            return {"order": [], "conversations": {}}
        # If we loaded from legacy, save into the new location.
        if path == LEGACY_CONVERSATIONS_PATH and not CONVERSATIONS_PATH.exists():
            CONVERSATIONS_PATH.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        # Drop any order entries that no longer exist in the conversation map.
        cleaned_order = [conv_id for conv_id in order if conv_id in conversations]
        return {"order": cleaned_order, "conversations": conversations}
    except Exception:
        return {"order": [], "conversations": {}}


def save_conversations_file() -> None:
    # Persist conversations to disk.
    payload = {
        "order": st.session_state.conversation_order,
        "conversations": st.session_state.conversations,
    }
    CONVERSATIONS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def init_session(settings: AppSettings) -> None:
    # Initialize all session defaults once per app run.
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    if "refresh_settings" not in st.session_state:
        st.session_state.refresh_settings = False

    if "data_dir" not in st.session_state:
        st.session_state.data_dir = settings.data_dir
    if "persist_dir" not in st.session_state:
        st.session_state.persist_dir = settings.persist_dir
    if "file_types" not in st.session_state:
        st.session_state.file_types = settings.file_types
    if "similarity_top_k" not in st.session_state:
        st.session_state.similarity_top_k = settings.similarity_top_k
    if "rerank_top_n" not in st.session_state:
        st.session_state.rerank_top_n = settings.rerank_top_n
    if "embedding_provider" not in st.session_state:
        st.session_state.embedding_provider = settings.embedding_provider
    if st.session_state.embedding_provider != "openai":
        st.session_state.embedding_provider = "openai"
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = settings.embedding_model

    if "conversations" not in st.session_state:
        data = load_conversations_file()
        st.session_state.conversations = data.get("conversations", {})
        st.session_state.conversation_order = data.get("order", [])
    if "conversation_order" not in st.session_state:
        st.session_state.conversation_order = []
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = (
            st.session_state.conversation_order[0]
            if st.session_state.conversation_order
            else None
        )


def apply_settings_to_session(settings: AppSettings) -> None:
    # Sync settings values into session state.
    st.session_state.data_dir = settings.data_dir
    st.session_state.persist_dir = settings.persist_dir
    st.session_state.file_types = settings.file_types
    st.session_state.similarity_top_k = settings.similarity_top_k
    st.session_state.rerank_top_n = settings.rerank_top_n
    st.session_state.embedding_provider = settings.embedding_provider
    st.session_state.embedding_model = settings.embedding_model


def new_conversation() -> None:
    # Create a new conversation entry and select it.
    conv_id = f"conv_{uuid.uuid4().hex[:8]}"
    st.session_state.conversations[conv_id] = {"title": "New chat", "messages": []}
    st.session_state.conversation_order.insert(0, conv_id)
    st.session_state.current_conversation = conv_id
    save_conversations_file()


def delete_conversation(conv_id: str) -> None:
    # Remove a conversation and update selection.
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
    if conv_id in st.session_state.conversation_order:
        st.session_state.conversation_order.remove(conv_id)
    if st.session_state.current_conversation == conv_id:
        st.session_state.current_conversation = (
            st.session_state.conversation_order[0]
            if st.session_state.conversation_order
            else None
        )
    save_conversations_file()


def rename_conversation(conv_id: str, new_title: str) -> None:
    # Update a conversation title.
    cleaned = new_title.strip()
    if cleaned:
        st.session_state.conversations[conv_id]["title"] = cleaned[:60]
        save_conversations_file()


def current_conversation() -> dict:
    # Return the active conversation payload.
    conv_id = st.session_state.current_conversation
    return st.session_state.conversations.get(conv_id, {"title": "New chat", "messages": []})


def set_conversation_title(conv_id: str, prompt: str) -> None:
    # Set title from first user prompt.
    title = prompt.strip().split("\n", 1)[0][:40]
    if title:
        st.session_state.conversations[conv_id]["title"] = title
        save_conversations_file()
