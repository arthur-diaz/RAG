# RAG

This app builds a local RAG pipeline with LlamaIndex, reranking, and a Streamlit UI.

## Setup (uv)

1. Create a virtual environment:

```bash
uv venv
```

2. Activate it (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
uv pip install -r requirements.txt
```

3. Provide API keys (environment variables recommended):

```
setx OPENAI_API_KEY "your_key"
```
You can also create a local `.env` file at the project root (see `.env.example`).

4. Put PDFs in the folder configured in the sidebar (default: `data`).
5. Run the app:

```bash
streamlit run app.py
```

## Notes

- The index is stored in `storage` by default.
- Settings are accessed via the "Settings" button in the sidebar.
- Reranking uses a local cross-encoder model via `sentence-transformers`.



## Parameter guide

- Documents folder: path to your data directory (default: `data`).
- Index folder: where the vector index is stored (default: `storage`).
- File types: comma-separated extensions to ingest (e.g. `.pdf,.txt`).
- Retrieval top k: number of chunks retrieved from the vector store.
- Rerank top n: how many of those chunks are re-ordered by the reranker; must be <= top k.
- Embedding model is fixed to OpenAI (`text-embedding-3-small`).
- LLM model can be changed per query from the top-left selector.

## Advanced settings

- `max_turns` in `.rag_settings.json` controls how many recent turns are included as context.

## Models and pricing

