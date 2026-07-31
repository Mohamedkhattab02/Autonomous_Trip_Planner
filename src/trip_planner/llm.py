"""Single place where the LLM is configured.

Every agent builds its model through `get_model()`, so no module instantiates a
chat model directly and the whole system can be switched to a different model
by changing one environment variable.

Set `GEMINI_MODEL` in `.env` to override the default, e.g. when a model is
retired or is returning 503s under load:

    GEMINI_MODEL=gemini-3.6-flash
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# The model every agent runs on. Overridable so a retired or overloaded model
# never requires a code change.
DEFAULT_MODEL = "gemini-3.5-flash"

# Gemini 3.x models use fixed sampling defaults and ignore `temperature`,
# warning loudly if it is passed. Sending it anyway would print that warning on
# every one of the many calls a full plan makes.
FIXED_SAMPLING_PREFIXES = ("gemini-3",)

load_dotenv()


def model_name() -> str:
    """Return the configured model id.

    Returns:
        The value of `GEMINI_MODEL`, or the default when it is not set.
    """
    return os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def get_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return the shared Gemini model.

    Args:
        temperature: Sampling temperature. Defaults to 0 so structured
            extraction stays deterministic. Ignored by models that use fixed
            sampling, in which case it is not sent at all.

    Returns:
        The configured chat model.

    Raises:
        RuntimeError: If GOOGLE_API_KEY is not set.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to the .env file at the project root."
        )

    name = model_name()
    if name.startswith(FIXED_SAMPLING_PREFIXES):
        return ChatGoogleGenerativeAI(model=name)
    return ChatGoogleGenerativeAI(model=name, temperature=temperature)


# Kept for modules and tests that import the model name directly.
MODEL_NAME = model_name()
