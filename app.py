"""Streamlit entrypoint for the RAG app."""

import streamlit as st

from rag.ui.main import main

# Configure the Streamlit page and run the app.
st.set_page_config(page_title="RAG Expert Agent", page_icon="🔍", layout="wide")
main()
