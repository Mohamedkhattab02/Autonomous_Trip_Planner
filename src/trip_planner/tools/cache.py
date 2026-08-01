"""Disk cache for tool results.

Every search the agents run costs money and seconds. Nothing about a museum's
opening hours changes between two runs a minute apart, and during development
the same request is replanned dozens of times — so without a cache almost all
of that spend is waste.

TTLs are set by how fast the underlying answer actually moves: fares and room
rates are live prices and expire in hours, while places, opening hours and
destination research are stable for days.

Set `TRIP_CACHE_ENABLED=false` to bypass it entirely, which is what the live
tests do when they need to prove a real API still works.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable

from diskcache import Cache

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "tools"

# Seconds. Named so the intent is readable at the call site.
TTL_PRICES = 6 * 3600  # flights, hotel rates
TTL_PLACES = 7 * 24 * 3600  # places, opening hours
TTL_RESEARCH = 3 * 24 * 3600  # weather, visas, currency

_cache: Cache | None = None


def is_enabled() -> bool:
    """Report whether caching is switched on.

    Returns:
        False when `TRIP_CACHE_ENABLED` is explicitly falsey.
    """
    return os.getenv("TRIP_CACHE_ENABLED", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _store() -> Cache:
    """Return the shared on-disk cache, opening it on first use.

    Returns:
        The `diskcache.Cache` instance.
    """
    global _cache
    if _cache is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache = Cache(str(CACHE_DIR))
    return _cache


def _key(namespace: str, payload: dict) -> str:
    """Build a stable cache key from a call's arguments.

    Args:
        namespace: The tool's name.
        payload: The arguments the tool was called with.

    Returns:
        A hex digest key.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    return f"{namespace}:{hashlib.sha256(blob.encode()).hexdigest()[:32]}"


def cached(namespace: str, ttl: int) -> Callable:
    """Cache a tool function's results on disk.

    Errors are never cached: a failed lookup should be retried next time, not
    remembered for a week.

    Args:
        namespace: A stable name for this tool.
        ttl: How long a result stays fresh, in seconds.

    Returns:
        A decorator.
    """

    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_enabled():
                return func(*args, **kwargs)

            key = _key(namespace, {"args": args, "kwargs": kwargs})
            store = _store()
            hit = store.get(key)
            if hit is not None:
                logger.debug("cache hit: %s", namespace)
                return hit

            result = func(*args, **kwargs)
            if isinstance(result, dict) and "error" in result:
                return result
            store.set(key, result, expire=ttl)
            return result

        return wrapper

    return decorate


def stats() -> dict:
    """Report the cache's size, for the CLI and the efficiency panel.

    Returns:
        A dict with the entry count, size on disk and whether it is enabled.
    """
    if not is_enabled():
        return {"enabled": False, "entries": 0, "megabytes": 0.0}
    store = _store()
    return {
        "enabled": True,
        "entries": len(store),
        "megabytes": round(store.volume() / 1_048_576, 2),
    }


def clear() -> int:
    """Empty the cache.

    Returns:
        How many entries were removed.
    """
    store = _store()
    count = len(store)
    store.clear()
    return count
