"""App router between settings and chat pages."""

import streamlit as st

from rag.settings import load_settings
from rag.ui.chat_page import render_chat
from rag.ui.settings_page import render_settings
from rag.ui.state import (
    apply_settings_to_session,
    init_session,
    load_env_file,
    new_conversation,
    sync_api_key_from_env,
)

# App entrypoint: choose between settings page and chat page.

def main() -> None:
    # Load environment and session state before rendering.
    load_env_file()
    settings = load_settings()
    init_session(settings)
    sync_api_key_from_env()

    if not st.session_state.conversation_order:
        new_conversation()

    # Route between the settings page and the chat page.
    if st.session_state.show_settings:
        if st.session_state.refresh_settings:
            settings = load_settings()
            apply_settings_to_session(settings)
            sync_api_key_from_env()
            st.session_state.refresh_settings = False
        render_settings(settings)
    else:
        render_chat(settings)


if __name__ == "__main__":
    st.set_page_config(page_title="RAG Expert Agent", page_icon="📚", layout="wide")
    main()
