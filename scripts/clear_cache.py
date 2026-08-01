"""Empty the tool-result cache.

Cached searches are keyed by their arguments and expire on their own, but a
forced clear is useful when prices have moved or a bad response was stored.

Usage:
    uv run scripts/clear_cache.py
"""

from __future__ import annotations

from trip_planner.tools.cache import clear, stats


def main() -> None:
    """Report the cache size, empty it, and confirm."""
    before = stats()
    if not before["enabled"]:
        print("Caching is disabled (TRIP_CACHE_ENABLED=false); nothing to clear.")
        return

    print(f"Before: {before['entries']} entries, {before['megabytes']} MB")
    removed = clear()
    print(f"Cleared {removed} entries.")


if __name__ == "__main__":
    main()
