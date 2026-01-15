"""Settings page rendering."""

import os

import streamlit as st

from rag.settings import AppSettings, save_settings, SETTINGS_PATH
from rag.ui.state import delete_env_key, write_env_file

# Render the settings page UI.

def render_settings(settings: AppSettings) -> None:
    # Top bar with a close button.
    col_close, col_title = st.columns([1, 5])
    with col_close:
        if st.button("✕", key="close_settings"):
            st.session_state.show_settings = False
            st.rerun()
    with col_title:
        st.title("Settings")

    # Indexing configuration.
    st.text_input("Documents folder", key="data_dir")
    st.text_input("Index folder", key="persist_dir")
    st.text_input("File types (comma-separated)", key="file_types")

    st.number_input("Retrieval top k", min_value=1, max_value=50, key="similarity_top_k")
    st.number_input("Rerank top n", min_value=1, max_value=20, key="rerank_top_n")

    # Embedding provider is fixed to OpenAI.
    st.selectbox(
        "Embedding provider",
        options=["openai"],
        key="embedding_provider",
    )
    st.session_state.embedding_model = "text-embedding-3-small"
    st.caption(
        "Embeddings convert text into vectors for search. "
        "Indexing runs automatically when you ask a question."
    )

    # Persist or reset settings.
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("Save settings"):
            settings.data_dir = st.session_state.data_dir
            settings.persist_dir = st.session_state.persist_dir
            settings.file_types = st.session_state.file_types
            settings.similarity_top_k = int(st.session_state.similarity_top_k)
            settings.rerank_top_n = int(st.session_state.rerank_top_n)
            settings.embedding_provider = st.session_state.embedding_provider
            settings.embedding_model = st.session_state.embedding_model
            save_settings(settings)
            st.success("Settings saved.")

    with col_reset:
        if st.button("Reset settings"):
            defaults = AppSettings()
            settings.data_dir = defaults.data_dir
            settings.persist_dir = defaults.persist_dir
            settings.file_types = defaults.file_types
            settings.similarity_top_k = defaults.similarity_top_k
            settings.rerank_top_n = defaults.rerank_top_n
            settings.embedding_provider = "openai"
            settings.embedding_model = "text-embedding-3-small"
            settings.openai_model = defaults.openai_model

            st.session_state.data_dir = defaults.data_dir
            st.session_state.persist_dir = defaults.persist_dir
            st.session_state.file_types = defaults.file_types
            st.session_state.similarity_top_k = defaults.similarity_top_k
            st.session_state.rerank_top_n = defaults.rerank_top_n
            st.session_state.embedding_provider = "openai"
            st.session_state.embedding_model = "text-embedding-3-small"

            if SETTINGS_PATH.exists():
                SETTINGS_PATH.unlink()
            st.success("Settings reset to defaults.")
            st.rerun()

    # Local API key management (stored in .env).
    st.subheader("API keys")
    st.text_input("OpenAI API key", type="password", key="openai_key")
    if os.getenv("OPENAI_API_KEY"):
        st.caption("API key loaded from .env.")

    col_key_save, col_key_clear = st.columns(2)
    with col_key_save:
        if st.button("Save API key locally"):
            if not st.session_state.openai_key:
                st.error("OpenAI API key is empty.")
            else:
                try:
                    write_env_file({"OPENAI_API_KEY": st.session_state.openai_key})
                    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
                    st.success("API key saved locally in .env.")
                except Exception as exc:
                    st.error(f"Failed to save API key: {exc}")
    with col_key_clear:
        if st.button("Clear saved API key"):
            try:
                delete_env_key("OPENAI_API_KEY")
                st.session_state.openai_key = ""
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]
                st.success("Saved API key removed.")
            except Exception as exc:
                st.error(f"Failed to clear API key: {exc}")

    st.caption("API keys are saved locally in .env (not in the settings file).")
