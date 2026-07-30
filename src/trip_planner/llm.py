"""Single place where the LLM is configured.

The project uses Gemini 2.5 Flash exclusively, so every agent builds its model
through `get_model()` and no module instantiates a chat model directly.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-2.5-flash"

load_dotenv()


def get_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return the shared Gemini model.

    Args:
        temperature: Sampling temperature. Defaults to 0 so structured
            extraction stays deterministic.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to the .env file at the project root."
        )
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature)
