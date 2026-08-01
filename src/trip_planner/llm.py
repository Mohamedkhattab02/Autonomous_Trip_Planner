"""Single place where the LLM is configured.

Every agent builds its model through `get_model()`, so no module instantiates a
chat model directly and the whole system can be repointed by editing `.env`.

Two things are configurable:

* **The model.** `LLM_MODEL` names it, in LangChain's `provider:model` form
  (`openai:gpt-5-mini`, `google_genai:gemini-3.6-flash`, `ollama:llama3`).
  A bare name is assumed to be OpenAI.
* **Per-agent tiering.** `LLM_MODEL_<ROLE>` overrides the default for one
  agent, so cheap formatting work (intake, calendar) can run on a small model
  while the reasoning-heavy agents (critic, research) keep the strong one.
  Roles map to the agent names in `AgentName`.

Retries are configured here too: transient 429 and 503 responses are the most
common failure in practice, so every model is built with a retry budget.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# The model every agent runs on unless a role override is set.
DEFAULT_MODEL = "openai:gpt-5-mini"

# How many times the client itself retries a transient failure before the
# node-level retry in `resilience.py` takes over.
MAX_RETRIES = 3

# Model families that reject a custom `temperature`. GPT-5 and Gemini 3.x both
# use fixed sampling; sending the parameter is an error or a loud warning.
FIXED_SAMPLING = ("gpt-5", "o1", "o3", "o4", "gemini-3")

load_dotenv()


def model_name(role: str | None = None) -> str:
    """Return the model id configured for an agent.

    Args:
        role: The agent's name, e.g. "critic". None returns the default.

    Returns:
        The model id, e.g. "openai:gpt-5-mini".
    """
    if role:
        override = os.getenv(f"LLM_MODEL_{role.upper()}", "").strip()
        if override:
            return override
    return os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL


def _provider_key(model: str) -> tuple[str, str]:
    """Split a model id into its provider and the environment variable it needs.

    Args:
        model: A model id, with or without a `provider:` prefix.

    Returns:
        A `(provider, env_var)` pair.
    """
    provider = model.split(":", 1)[0] if ":" in model else "openai"
    return provider, {
        "openai": "OPENAI_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider, "")


@lru_cache(maxsize=16)
def _build(model: str, temperature: float | None) -> BaseChatModel:
    """Build and cache one chat model.

    Cached because a full plan constructs a model per agent per run, and
    rebuilding the client each time is wasted work.

    Args:
        model: The model id.
        temperature: Sampling temperature, or None to leave it at the default.

    Returns:
        The chat model.

    Raises:
        RuntimeError: If the provider's API key is not set.
    """
    provider, env_var = _provider_key(model)
    if env_var and not os.getenv(env_var):
        raise RuntimeError(
            f"{env_var} is not set, but LLM_MODEL is '{model}' ({provider}). "
            f"Add {env_var} to the .env file at the project root."
        )

    kwargs: dict = {"max_retries": MAX_RETRIES}
    if temperature is not None:
        kwargs["temperature"] = temperature

    return init_chat_model(model, **kwargs)


def get_model(role: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """Return the chat model an agent should use.

    Args:
        role: The agent's name, used to look up a per-agent override.
        temperature: Sampling temperature. Defaults to 0 so structured
            extraction stays deterministic. Dropped for model families that
            use fixed sampling and reject it.

    Returns:
        The configured chat model.

    Raises:
        RuntimeError: If the provider's API key is not set.
    """
    name = model_name(role)
    bare = name.split(":", 1)[-1]
    wants_temperature = not bare.startswith(FIXED_SAMPLING)
    return _build(name, temperature if wants_temperature else None)


def describe_models() -> str:
    """Describe the configured models, for logs and the UI.

    Returns:
        A one-line summary naming the default and any per-agent overrides.
    """
    default = model_name()
    overrides = sorted(
        (key[len("LLM_MODEL_") :].lower(), value)
        for key, value in os.environ.items()
        if key.startswith("LLM_MODEL_") and value.strip()
    )
    if not overrides:
        return default
    detail = ", ".join(f"{role}={model}" for role, model in overrides)
    return f"{default} ({detail})"


# Kept for modules and tests that import the model name directly.
MODEL_NAME = model_name()
