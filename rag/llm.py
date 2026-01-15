"""LLM and embedding factory functions."""


class LLMInitError(Exception):
    """Raised when an LLM or embedding cannot be initialized."""


def make_llm(provider: str, model: str, api_key: str):
    # Create the chat model client for the selected provider.
    if provider == "openai":
        try:
            from llama_index.llms.openai import OpenAI
        except Exception as exc:
            raise LLMInitError("OpenAI LLM package not installed.") from exc
        if not api_key:
            raise LLMInitError("OpenAI API key is missing.")
        return OpenAI(model=model, api_key=api_key)

    raise LLMInitError(f"Unsupported LLM provider: {provider}")


def make_embedding(provider: str, model: str, api_key: str):
    # Create the embedding client for the selected provider.
    if provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except Exception as exc:
            raise LLMInitError("OpenAI embeddings package not installed.") from exc
        if not api_key:
            raise LLMInitError("OpenAI API key is missing for embeddings.")
        return OpenAIEmbedding(model=model, api_key=api_key)

    raise LLMInitError(f"Unsupported embedding provider: {provider}")
