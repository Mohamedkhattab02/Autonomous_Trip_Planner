"""How every agent is constructed.

All eleven agents build through `build_structured_agent()` rather than calling
`create_agent()` directly, so the way structured output is requested is decided
in exactly one place.

**Why this exists.** Passing a bare Pydantic model as `response_format` makes
LangChain select `ProviderStrategy`, which hard-codes `strict=True` on every
tool for OpenAI-compatible models:

    if _is_openai_compatible_model(...) and not use_responses_api:
        bind_kwargs["strict"] = True

OpenAI's strict mode forbids free-form objects, so any tool taking a
`list[dict]` — `compare_flights`, `cluster_locations`, `validate_schedule`,
`build_daily_itinerary` and others — was rejected with a 400 before the run
even started. It also rejects third-party schemas we do not control, including
the Google Calendar MCP server's own `create-event`.

`ToolStrategy` asks for the same structured output as a forced tool call and
never sets `strict`, so those schemas are accepted. It is also provider-neutral,
which matters because `LLM_MODEL` can point at OpenAI, Gemini or Ollama.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from trip_planner.llm import get_model


def build_structured_agent(
    role: str,
    tools: list[BaseTool],
    system_prompt: str,
    response_format: type[BaseModel],
    name: str,
):
    """Build an agent that returns a validated Pydantic model.

    Args:
        role: The agent's name, used to pick a per-agent model override.
        tools: The tools this agent may call. Empty for an agent whose job is
            pure extraction — see `intake_agent`.
        system_prompt: The agent's instructions.
        response_format: The Pydantic model the agent must produce.
        name: A readable name for tracing.

    Returns:
        The agent runnable.
    """
    return create_agent(
        model=get_model(role),
        tools=tools,
        system_prompt=system_prompt,
        # ToolStrategy, not the bare model: see this module's docstring.
        response_format=ToolStrategy(response_format),
        name=name,
    )
