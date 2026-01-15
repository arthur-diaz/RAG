"""Model registry helpers for the app UI."""

from typing import Dict, List, Optional

# Available OpenAI chat models for selection in the UI.
OPENAI_MODELS: List[Dict[str, Optional[str]]] = [
    {"id": "gpt-4o-mini"},
    {"id": "gpt-4o"},
    {"id": "gpt-5-nano"},
]


def model_ids(provider: str) -> List[str]:
    # Return IDs for the selected provider.
    if provider == "openai":
        return [model["id"] for model in OPENAI_MODELS]
    return []


def find_model(provider: str, model_id: str) -> Optional[Dict[str, Optional[str]]]:
    # Look up a model entry by id.
    if provider != "openai":
        return None
    for model in OPENAI_MODELS:
        if model["id"] == model_id:
            return model
    return None
