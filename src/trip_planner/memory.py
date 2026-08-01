"""Traveler memory that persists between trips.

Home airport, nationality, dietary limits and preferred pace are stable facts,
but every run used to ask for them again. This stores them as a JSON file per
traveler and merges them *underneath* the current request.

The merge direction matters: **the request always wins.** Preferences supply
defaults for anything the traveler did not say this time, and never override
something they did.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from trip_planner.schemas import TravelerPreferences, TravelerProfile

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"

DEFAULT_TRAVELER = "default"


def _path(traveler: str) -> Path:
    """Return the file backing one traveler's preferences.

    Args:
        traveler: The traveler's id.

    Returns:
        The JSON path.
    """
    safe = "".join(ch for ch in traveler if ch.isalnum() or ch in "-_") or "default"
    return PROFILES_DIR / f"{safe}.json"


def load(traveler: str = DEFAULT_TRAVELER) -> TravelerPreferences:
    """Load a traveler's stored preferences.

    A missing or unreadable file yields empty preferences rather than an error:
    memory is an optimisation, and losing it must never block planning.

    Args:
        traveler: The traveler's id.

    Returns:
        The stored preferences, or empty ones.
    """
    path = _path(traveler)
    if not path.exists():
        return TravelerPreferences()
    try:
        return TravelerPreferences.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - memory is best-effort
        logger.warning("Could not read preferences for '%s': %s", traveler, exc)
        return TravelerPreferences()


def save(
    preferences: TravelerPreferences, traveler: str = DEFAULT_TRAVELER
) -> Path:
    """Write a traveler's preferences to disk.

    Args:
        preferences: The preferences to store.
        traveler: The traveler's id.

    Returns:
        The path written.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(traveler)
    path.write_text(
        preferences.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    return path


def apply(profile: TravelerProfile, preferences: TravelerPreferences) -> TravelerProfile:
    """Fill gaps in a freshly extracted profile from stored preferences.

    Only fields the traveler left unstated are filled. Interests and constraints
    are unioned, because a standing constraint ("vegetarian") still applies even
    when this trip's request adds others.

    Args:
        profile: The profile the Intake Agent extracted from this request.
        preferences: The traveler's stored defaults.

    Returns:
        A new profile with the defaults merged underneath.
    """
    merged = profile.model_copy(deep=True)

    if merged.origin is None and preferences.home_city:
        merged.origin = preferences.home_city
    if merged.budget_currency is None and preferences.default_currency:
        merged.budget_currency = preferences.default_currency

    for item in preferences.interests:
        if item not in merged.interests:
            merged.interests.append(item)
    for item in preferences.constraints:
        if item not in merged.constraints:
            merged.constraints.append(item)

    return merged


def remember(profile: TravelerProfile, traveler: str = DEFAULT_TRAVELER) -> None:
    """Update stored preferences from a completed trip profile.

    Learns only the stable facts — where the traveler departs from, what
    currency they think in, and where they have now been.

    Args:
        profile: The profile of a trip that was planned successfully.
        traveler: The traveler's id.
    """
    preferences = load(traveler)

    if profile.origin and not preferences.home_city:
        preferences.home_city = profile.origin
    if profile.budget_currency and not preferences.default_currency:
        preferences.default_currency = profile.budget_currency
    if profile.destination and profile.destination not in preferences.visited_destinations:
        preferences.visited_destinations.append(profile.destination)

    save(preferences, traveler)
