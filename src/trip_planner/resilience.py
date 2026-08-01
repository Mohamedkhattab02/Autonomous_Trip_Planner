"""Retry and failure containment for graph nodes.

Two problems this solves, both observed in practice rather than imagined:

* **Transient provider errors.** A single 503 or 429 used to abort the whole
  run and discard every completed stage. `with_retry` retries with exponential
  backoff, and distinguishes the two: 503 is a blip and deserves a short wait,
  while 429 is a quota signal where retrying quickly makes things worse.
* **One stage failing should not lose the other ten.** `resilient_node` catches
  whatever escapes the retries, records the agent in `failed_agents`, and lets
  the graph continue. A plan missing its flights is far more useful than no
  plan, and every downstream agent already handles a missing input.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from trip_planner.schemas import AgentName

logger = logging.getLogger(__name__)

# How many times a node is retried before it is declared failed.
MAX_ATTEMPTS = 3

# Backoff in seconds for a transient error (503, timeouts, connection resets).
TRANSIENT_BACKOFF = (2.0, 6.0)

# Backoff for a rate limit (429). Longer, because hammering a quota wall only
# pushes the reset further away.
RATE_LIMIT_BACKOFF = (20.0, 45.0)

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota")
_TRANSIENT_MARKERS = (
    "503",
    "502",
    "504",
    "unavailable",
    "timeout",
    "timed out",
    "connection",
    "overloaded",
)


def classify(error: Exception) -> str:
    """Classify an error as rate-limited, transient or permanent.

    Args:
        error: The exception raised by a node.

    Returns:
        One of "rate_limit", "transient" or "permanent".
    """
    text = str(error).lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return "rate_limit"
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    return "permanent"


def with_retry(func: Callable, name: str) -> Callable:
    """Wrap a callable so transient failures are retried with backoff.

    A permanent error is re-raised immediately: retrying a bad request just
    wastes time and quota.

    Args:
        func: The function to wrap.
        name: The agent name, for log messages.

    Returns:
        The wrapped function.
    """

    def wrapper(*args, **kwargs):
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                kind = classify(exc)
                if kind == "permanent" or attempt == MAX_ATTEMPTS - 1:
                    raise

                delays = (
                    RATE_LIMIT_BACKOFF if kind == "rate_limit" else TRANSIENT_BACKOFF
                )
                delay = delays[min(attempt, len(delays) - 1)]
                logger.warning(
                    "%s failed with a %s error (attempt %d/%d); retrying in %.0fs: %s",
                    name,
                    kind,
                    attempt + 1,
                    MAX_ATTEMPTS,
                    delay,
                    str(exc)[:200],
                )
                time.sleep(delay)
        raise last  # pragma: no cover - loop always returns or raises

    return wrapper


def resilient_node(agent: AgentName) -> Callable:
    """Decorate a graph node so a failure degrades the plan instead of killing it.

    The node is retried first. If it still fails, the agent is recorded in
    `failed_agents` and marked complete so the graph moves on. Downstream
    agents already cope with a missing slot, and the UI labels the plan as
    degraded rather than presenting it as whole.

    Args:
        agent: The agent this node runs.

    Returns:
        A decorator.
    """

    def decorate(func: Callable) -> Callable:
        retrying = with_retry(func, str(agent))

        def wrapper(state) -> dict:
            try:
                return retrying(state)
            except Exception as exc:  # noqa: BLE001 - contained deliberately
                logger.error("%s failed permanently: %s", agent, exc)
                return {
                    "failed_agents": [
                        {"agent": str(agent), "error": str(exc)[:300]}
                    ],
                    "completed_agents": [agent],
                }

        wrapper.__name__ = getattr(func, "__name__", str(agent))
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorate
