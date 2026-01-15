"""Chat page rendering."""

from pathlib import Path

import streamlit as st

from rag.ingest import ingest_documents
from rag.models import model_ids
from rag.query import run_query
from rag.settings import AppSettings, save_settings
from rag.ui.helpers import build_context, clean_source_text, source_toggle_key
from rag.ui.state import (
    current_conversation,
    delete_conversation,
    get_openai_key,
    new_conversation,
    rename_conversation,
    save_conversations_file,
    set_conversation_title,
)

# Render the main chat page UI.

def render_chat(settings: AppSettings) -> None:
    # Sidebar contains conversation management and settings access.
    with st.sidebar:
        st.button("New chat", on_click=new_conversation)
        st.divider()
        for conv_id in st.session_state.conversation_order:
            title = st.session_state.conversations[conv_id]["title"]
            label = title if title else "New chat"
            row = st.columns([6, 1, 1])
            with row[0]:
                if st.button(label, key=f"chat_{conv_id}"):
                    st.session_state.current_conversation = conv_id
                    st.rerun()
            with row[1]:
                if st.button("✎", key=f"rename_{conv_id}"):
                    st.session_state[f"renaming_{conv_id}"] = True
            with row[2]:
                if st.button("🗑", key=f"delete_{conv_id}"):
                    delete_conversation(conv_id)
                    st.rerun()

            if st.session_state.get(f"renaming_{conv_id}"):
                new_title = st.text_input(
                    "Rename chat",
                    value=label,
                    key=f"rename_input_{conv_id}",
                )
                action_cols = st.columns([1, 1])
                with action_cols[0]:
                    if st.button("Save", key=f"rename_save_{conv_id}"):
                        rename_conversation(conv_id, new_title)
                        st.session_state[f"renaming_{conv_id}"] = False
                        st.rerun()
                with action_cols[1]:
                    if st.button("Cancel", key=f"rename_cancel_{conv_id}"):
                        st.session_state[f"renaming_{conv_id}"] = False
                        st.rerun()
            st.divider()

        if st.button("Settings"):
            st.session_state.show_settings = True
            st.session_state.refresh_settings = True
            st.rerun()

    # Header row with model selector on the left.
    top_cols = st.columns([1, 3, 1])
    with top_cols[0]:
        available_models = model_ids("openai")
        llm_model_default = settings.openai_model
        llm_options = available_models + ["custom"]
        default_index = llm_options.index(llm_model_default) if llm_model_default in llm_options else len(llm_options) - 1
        selected_model = st.selectbox("Model", options=llm_options, index=default_index)
        if selected_model == "custom":
            llm_model = st.text_input("Custom model id", value=llm_model_default)
        else:
            llm_model = selected_model
    with top_cols[1]:
        st.write("")

    # Render existing messages in the active conversation.
    conv = current_conversation()
    for msg_idx, message in enumerate(conv["messages"]):
        with st.chat_message(message["role"]):
            status = message.get("status")
            if status:
                st.caption(status)
            st.write(message["content"])
            sources = message.get("sources")
            if sources:
                with st.expander("Sources"):
                    for idx, source in enumerate(sources, start=1):
                        label = f"{idx}. {source['file_name']} (score {source['score']:.3f})"
                        st.markdown(label)
                        st.write(clean_source_text(source["text"]))
                        raw_key = source_toggle_key(
                            "raw_saved",
                            msg_idx,
                            idx,
                            source["file_name"],
                            source["text"],
                        )
                        toggle_label = "See more" if not st.session_state.get(raw_key) else "See less"
                        if st.button(toggle_label, key=f"btn_{raw_key}"):
                            st.session_state[raw_key] = not st.session_state.get(raw_key, False)
                            st.rerun()
                        if st.session_state.get(raw_key):
                            st.code(source["text"])

    # Input box for new user prompts.
    prompt = st.chat_input("Ask a question")
    if prompt:
        conv_id = st.session_state.current_conversation
        st.session_state.conversations[conv_id]["messages"].append({"role": "user", "content": prompt})
        if st.session_state.conversations[conv_id]["title"] == "New chat":
            set_conversation_title(conv_id, prompt)
        save_conversations_file()

        with st.chat_message("user"):
            st.write(prompt)

        settings.data_dir = st.session_state.data_dir
        settings.persist_dir = st.session_state.persist_dir
        settings.file_types = st.session_state.file_types
        settings.similarity_top_k = int(st.session_state.similarity_top_k)
        if int(st.session_state.rerank_top_n) > int(st.session_state.similarity_top_k):
            st.warning("Rerank top n should be <= Retrieval top k. Using Retrieval top k.")
        settings.rerank_top_n = min(int(st.session_state.rerank_top_n), int(st.session_state.similarity_top_k))
        settings.embedding_provider = "openai"
        settings.embedding_model = st.session_state.embedding_model
        settings.openai_model = llm_model

        # Run ingestion only if needed, then query the index.
        try:
            with st.spinner("Checking index..."):
                count = ingest_documents(
                    settings,
                    openai_key=get_openai_key(),
                    mistral_key="",
                )
            save_settings(settings)
            if count == 0:
                status_text = "Embeddings: up to date. No changes detected."
            else:
                status_text = f"Embeddings: updated ({count} documents)."
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            return

        # Build context and run the RAG query.
        try:
            # Use the configured context window size.
            context = build_context(conv["messages"], max_turns=settings.max_turns)
            question = prompt
            if context:
                question = (
                    "Conversation context:\n"
                    f"{context}\n\n"
                    "Current question:\n"
                    f"{prompt}"
                )
            with st.spinner("Thinking..."):
                response = run_query(
                    question=question,
                    settings=settings,
                    llm_provider="openai",
                    llm_model=llm_model,
                    openai_key=get_openai_key(),
                    mistral_key="",
                )
            answer = response.response
            sources_payload = []
            if response.source_nodes:
                for node in response.source_nodes:
                    meta = node.node.metadata or {}
                    sources_payload.append(
                        {
                            "file_name": meta.get("file_name", "unknown"),
                            "score": float(node.score) if node.score is not None else 0.0,
                            "text": node.node.get_text(),
                        }
                    )

            st.session_state.conversations[conv_id]["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_payload,
                    "status": status_text,
                }
            )
            save_conversations_file()

            with st.chat_message("assistant"):
                st.caption(status_text)
                st.write(answer)
                if sources_payload:
                    with st.expander("Sources"):
                        live_msg_idx = len(st.session_state.conversations[conv_id]["messages"]) - 1
                        for idx, source in enumerate(sources_payload, start=1):
                            label = f"{idx}. {source['file_name']} (score {source['score']:.3f})"
                            st.markdown(label)
                            st.write(clean_source_text(source["text"]))
                            raw_key = source_toggle_key(
                                "raw_live",
                                live_msg_idx,
                                idx,
                                source["file_name"],
                                source["text"],
                            )
                            toggle_label = "See more" if not st.session_state.get(raw_key) else "See less"
                            if st.button(toggle_label, key=f"btn_{raw_key}"):
                                st.session_state[raw_key] = not st.session_state.get(raw_key, False)
                                st.rerun()
                            if st.session_state.get(raw_key):
                                st.code(source["text"])
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    if not Path(st.session_state.persist_dir).exists():
        st.info("No index found. Ask a question to auto-index documents.")
