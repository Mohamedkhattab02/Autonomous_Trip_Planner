"""Shared SerpApi access for the flights, lodging and attractions tools.

Those three agents all read live Google data through SerpApi, so the client
and its error handling live here rather than being repeated three times.
Like the Tavily client, it is built on first use so the module can be
imported without the API key present.
"""

from __future__ import annotations

import os
from functools import lru_cache

import serpapi


@lru_cache(maxsize=1)
def _client() -> serpapi.Client:
    """Return a lazily built SerpApi client.

    Raises:
        RuntimeError: If SERPAPI_API_KEY is not set.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SERPAPI_API_KEY is not set. Add it to the .env file at the project root."
        )
    return serpapi.Client(api_key=api_key)


def serp_search(**params) -> dict:
    """Run one SerpApi search and return the raw response as a dict.

    Network and API failures are turned into an `{"error": ...}` dict rather
    than raised, so one failed lookup degrades a single tool call instead of
    crashing the whole graph run. Tools surface the error to the agent, which
    can then say plainly that the data was unavailable.

    Args:
        **params: SerpApi parameters, including the `engine` to query.

    Returns:
        The SerpApi response as a dict, or `{"error": "..."}` on failure.
    """
    try:
        return dict(_client().search(**params))
    except Exception as exc:  # noqa: BLE001 - any failure degrades the same way
        return {"error": f"SerpApi request failed: {exc}"}
