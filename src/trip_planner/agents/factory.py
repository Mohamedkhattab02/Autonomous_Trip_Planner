"""How every agent is constructed and invoked.

All eleven agents build through `build_structured_agent()` and run through
`run_agent()`, so the three things that decide whether a stage finishes at all
are decided in exactly one place.

**Why `build_structured_agent` exists.** Passing a bare Pydantic model as
`response_format` makes LangChain select `ProviderStrategy`, which hard-codes
`strict=True` on every tool for OpenAI-compatible models:

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

**Why `run_agent` exists.** Every node used to invoke its agent itself, and
every one of them inherited two problems from that:

* **No step budget.** A sub-agent invoked without a `recursion_limit` inherits
  the graph's (80), so a model looping on a tool burned ~40 model calls before
  anything stopped it — long enough to look like a hang, and expensive enough
  to matter. `STEP_LIMITS` now bounds each agent by the work it actually has to
  do. This is not a safety net for correct behaviour; it is the ceiling on
  incorrect behaviour.
* **`KeyError: 'structured_response'`.** LangChain only puts that key in the
  result when the model actually produced structured output. When it answered
  in prose instead, the node died on a `KeyError` that says nothing about what
  went wrong. `run_agent` raises `AgentOutputError` instead, which is what the
  run log and the UI end up showing.

Nothing else is caught here. A 503 has to reach `resilience.with_retry` to be
retried, and `interrupt()` has to reach LangGraph to suspend the graph, so
neither may be swallowed on the way up.
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from trip_planner.llm import get_model

logger = logging.getLogger(__name__)

# How many steps each agent's own loop may take. A step is one node execution,
# so a round of "model calls a tool, tool answers" costs two. The numbers are
# sized from the tool calls each agent's prompt actually asks for, with room to
# recover from a bad call or two on top — Routing schedules a day at a time and
# Calendar writes an event at a time, which is why theirs are the largest.
STEP_LIMITS: dict[str, int] = {
    "manager": 8,
    "intake": 8,
    "destination_research": 24,
    "flights": 20,
    "lodging": 24,
    "attractions": 30,
    "routing": 60,
    "budget": 24,
    "critic": 40,
    "itinerary": 16,
    "calendar": 60,
}

# Used for any role not named above, so a new agent is bounded by default
# rather than silently inheriting the graph's limit.
DEFAULT_STEP_LIMIT = 24

# Appended to the system prompt of every agent that has tools. All of our tools
# are pure functions of their arguments or reads of data that cannot change
# during a stage, so a repeated call is always wasted — and a model that reads
# an unchanged answer as "not finished yet" will repeat it until its steps run
# out. Saying so once, here, is what keeps that out of eleven prompts.
TOOL_DISCIPLINE = """

How to use your tools:
- Your tools are deterministic: the same call with the same arguments always
  returns the same answer. Never repeat a call you have already made - it
  cannot tell you anything new.
- An error, an empty result or less detail than you hoped for IS the answer.
  Report it as it stands. Calling again will not change it.
- Once you have what you need, stop calling tools and return your answer. You
  have a limited number of steps; spending them on repeat calls means the
  traveler gets nothing at all.
"""


class AgentOutputError(RuntimeError):
    """An agent finished without producing the structured output it owes.

    Raised instead of a bare `KeyError` so the failure recorded in
    `failed_agents` names the agent and says what it did wrong.
    """


def step_limit(role: str) -> int:
    """Return the step budget for one agent's own loop.

    Args:
        role: The agent's name, e.g. "critic".

    Returns:
        The configured limit, or `DEFAULT_STEP_LIMIT` for an unlisted role.
    """
    return STEP_LIMITS.get(role, DEFAULT_STEP_LIMIT)


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
    if tools:
        system_prompt = system_prompt.rstrip() + "\n" + TOOL_DISCIPLINE

    return create_agent(
        model=get_model(role),
        tools=tools,
        system_prompt=system_prompt,
        # ToolStrategy, not the bare model: see this module's docstring.
        response_format=ToolStrategy(response_format),
        name=name,
    )


def agent_config(role: str, collector=None) -> dict:
    """Build the run config for one agent invocation.

    Args:
        role: The agent's name, which decides the step budget.
        collector: Optional metrics callback.

    Returns:
        A LangChain config carrying the step limit and, when given, the
        callback that records this agent's tokens and tool calls.
    """
    config: dict = {"recursion_limit": step_limit(role)}
    if collector is not None:
        config["callbacks"] = [collector]
    return config


def run_agent(agent, brief: str, role: str, collector=None) -> BaseModel:
    """Invoke an agent on one brief and return its structured answer.

    Args:
        agent: The runnable from `build_structured_agent`.
        brief: The user-role message describing the work.
        role: The agent's name, which decides the step budget.
        collector: Optional metrics callback.

    Returns:
        The validated Pydantic model the agent produced.

    Raises:
        AgentOutputError: If the agent answered without structured output.
        Exception: Anything the provider or LangGraph raises is left to travel
            up — `resilience` retries what is retryable and contains the rest.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": brief}]},
        agent_config(role, collector),
    )

    structured = result.get("structured_response")
    if structured is None:
        raise AgentOutputError(
            f"the {role} agent finished without returning its structured "
            f"result, so there is nothing to record for this stage"
        )
    return structured
