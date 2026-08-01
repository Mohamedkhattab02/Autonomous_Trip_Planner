"""Efficiency instrumentation.

Before this existed, no claim about the system's speed or cost could be checked
— every optimisation was a guess. `AgentMetrics` records what one agent
actually did; `track` wraps a node so it records itself; `summarize` turns the
collected records into the numbers the UI and the benchmark script report.

Token counts come from the provider's own usage metadata, so they are real
rather than estimated. Cost is derived from a small price table; an unknown
model simply reports no cost rather than a wrong one.
"""

from __future__ import annotations

import time
from typing import Callable

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

# USD per 1M tokens, (input, output). Only models we actually run are listed;
# anything missing reports `cost_usd = None` rather than a fabricated number.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-nano": (0.05, 0.40),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
}


class AgentMetrics(BaseModel):
    """What one agent cost, in time, tokens and calls."""

    agent: str
    seconds: float = Field(default=0.0, description="Wall-clock duration.")
    llm_calls: int = Field(default=0, description="Model round trips.")
    tool_calls: int = Field(default=0, description="Tool executions.")
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    failed: bool = False

    @property
    def total_tokens(self) -> int:
        """Tokens in and out combined."""
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float | None:
        """Cost of this agent's calls, or None when the model is unpriced."""
        bare = self.model.split(":", 1)[-1]
        price = PRICES.get(bare)
        if price is None:
            return None
        return (
            self.input_tokens * price[0] + self.output_tokens * price[1]
        ) / 1_000_000


class UsageCollector(BaseCallbackHandler):
    """Callback handler that counts model and tool activity for one agent."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model = ""

    def on_llm_end(self, response, **kwargs) -> None:
        """Record one model round trip and its token usage.

        Args:
            response: The provider's LLMResult.
            **kwargs: Unused callback arguments.
        """
        self.llm_calls += 1
        usage = (response.llm_output or {}).get("token_usage") or {}

        if not usage:
            # Providers differ: some report usage on the message instead.
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    meta = getattr(message, "usage_metadata", None) or {}
                    self.input_tokens += meta.get("input_tokens", 0)
                    self.output_tokens += meta.get("output_tokens", 0)
                    if message is not None and not self.model:
                        self.model = (
                            getattr(message, "response_metadata", {}) or {}
                        ).get("model_name", "")
            return

        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        if not self.model:
            self.model = (response.llm_output or {}).get("model_name", "")

    def on_tool_end(self, output, **kwargs) -> None:
        """Record one tool execution.

        Args:
            output: The tool's return value.
            **kwargs: Unused callback arguments.
        """
        self.tool_calls += 1


def track(agent: str, model: str = "") -> Callable:
    """Decorate a graph node so it records its own metrics into the state.

    The collector is passed to the agent through the LangChain config, so token
    counts come from the provider rather than being estimated.

    Args:
        agent: The agent's name.
        model: The model id the agent runs on.

    Returns:
        A decorator.
    """

    def decorate(func: Callable) -> Callable:
        def wrapper(state) -> dict:
            collector = UsageCollector()
            started = time.perf_counter()
            try:
                update = func(state, collector)
            finally:
                elapsed = time.perf_counter() - started

            record = AgentMetrics(
                agent=agent,
                seconds=round(elapsed, 2),
                llm_calls=collector.llm_calls,
                tool_calls=collector.tool_calls,
                input_tokens=collector.input_tokens,
                output_tokens=collector.output_tokens,
                model=collector.model or model,
                failed=bool(update.get("failed_agents")),
            )
            return {**update, "metrics": [record]}

        wrapper.__name__ = getattr(func, "__name__", agent)
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorate


class RunSummary(BaseModel):
    """Totals across every agent in one plan."""

    seconds: float = 0.0
    wall_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    agents: list[AgentMetrics] = Field(default_factory=list)

    @property
    def parallel_saving(self) -> float:
        """Seconds saved by running agents concurrently.

        Returns:
            The gap between summed agent time and wall-clock time, which is
            what the parallel fan-out bought.
        """
        return max(0.0, round(self.seconds - self.wall_seconds, 2))


def summarize(records: list[AgentMetrics], wall_seconds: float = 0.0) -> RunSummary:
    """Total a run's per-agent metrics.

    Args:
        records: Every agent's metrics, in completion order.
        wall_seconds: Measured wall-clock time for the whole run.

    Returns:
        The run totals, including cost when every model is priced.
    """
    costs = [record.cost_usd for record in records]
    return RunSummary(
        seconds=round(sum(record.seconds for record in records), 2),
        wall_seconds=round(wall_seconds, 2),
        llm_calls=sum(record.llm_calls for record in records),
        tool_calls=sum(record.tool_calls for record in records),
        input_tokens=sum(record.input_tokens for record in records),
        output_tokens=sum(record.output_tokens for record in records),
        cost_usd=(
            round(sum(cost for cost in costs if cost is not None), 4)
            if any(cost is not None for cost in costs)
            else None
        ),
        agents=sorted(records, key=lambda record: record.seconds, reverse=True),
    )
