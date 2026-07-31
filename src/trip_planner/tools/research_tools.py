"""Tools for the Destination Research Agent (plan.md agent #3).

All four tools hit the real Tavily API. They differ only in the query they
build, which keeps every claim in the final research traceable to a source
and lets the agent ask one focused question at a time.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.tools import tool
from tavily import TavilyClient

# How many results each search returns. Enough for citations without
# flooding the model's context.
MAX_RESULTS = 4


@lru_cache(maxsize=1)
def _client() -> TavilyClient:
    """Return a lazily built Tavily client.

    Built on first use rather than at import time so the module can be
    imported without the API key present.

    Raises:
        RuntimeError: If TAVILY_API_KEY is not set.
    """
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to the .env file at the project root."
        )
    return TavilyClient()


def tavily_web_search(query: str) -> dict:
    """Run one Tavily search and reduce it to the fields agents use.

    Shared with the Attractions and Critic agents, which search the web for
    facts that the structured Google endpoints do not cover.

    Args:
        query: The search query.

    Returns:
        A dict with Tavily's generated `answer` and a `sources` list of
        {title, url, content} entries, or `error` when the search failed.
    """
    try:
        response = _client().search(
            query=query,
            max_results=MAX_RESULTS,
            include_answer=True,
        )
    except Exception as exc:  # noqa: BLE001 - any failure degrades the same way
        return {"error": f"Tavily search failed: {exc}", "answer": "", "sources": []}

    return {
        "answer": response.get("answer") or "",
        "sources": [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
            }
            for result in response.get("results", [])
        ],
    }


@tool
def tavily_search(query: str) -> dict:
    """Search the web for any destination fact that the other tools do not cover.

    Args:
        query: A specific search query, e.g. "best neighborhoods to stay in Lisbon".

    Returns:
        A summarized answer plus the sources it came from.
    """
    return tavily_web_search(query)


@tool
def get_weather(destination: str, travel_dates: str) -> dict:
    """Look up the expected weather at a destination during the travel dates.

    Args:
        destination: The destination, e.g. "Paris, France".
        travel_dates: The travel period, e.g. "2026-09-10 to 2026-09-15".

    Returns:
        A weather summary plus the sources it came from.
    """
    return tavily_web_search(
        f"weather in {destination} during {travel_dates} temperature and rainfall"
    )


@tool
def get_entry_requirements(destination: str, nationality: str = "unspecified") -> dict:
    """Look up visa, passport and entry rules for a destination.

    Args:
        destination: The destination country or city.
        nationality: The travelers' nationality, when known.

    Returns:
        An entry-requirements summary plus the sources it came from.
    """
    who = "" if nationality == "unspecified" else f" for {nationality} citizens"
    return tavily_web_search(f"visa and entry requirements for {destination}{who} passport rules")


@tool
def get_currency_info(destination: str) -> dict:
    """Look up the local currency and payment norms at a destination.

    Args:
        destination: The destination country or city.

    Returns:
        A currency summary plus the sources it came from.
    """
    return tavily_web_search(
        f"local currency in {destination} exchange rate and card payment acceptance"
    )


RESEARCH_TOOLS = [tavily_search, get_weather, get_entry_requirements, get_currency_info]
