"""Tools for the Intake Agent (plan.md agent #2).

`validate_required_fields` is deliberately plain Python: which fields are
mandatory is a rule of the system, not something the model should decide.
"""

from __future__ import annotations

from langchain_core.tools import tool

# Fields every downstream agent needs before planning can start.
REQUIRED_FIELDS: tuple[str, ...] = (
    "destination",
    "start_date",
    "end_date",
    "travelers",
    "budget_amount",
)


@tool
def validate_required_fields(
    destination: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    travelers: int | None = None,
    budget_amount: float | None = None,
) -> dict:
    """Check which required trip fields are still missing.

    Call this after extracting details from the user's request to find out
    what still has to be asked about.

    Args:
        destination: Destination city and/or country.
        start_date: Trip start date in YYYY-MM-DD format.
        end_date: Trip end date in YYYY-MM-DD format.
        travelers: Number of people travelling.
        budget_amount: Total budget for the trip.

    Returns:
        A dict with `missing_fields` (list of field names) and `is_complete`.
    """
    values = {
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "travelers": travelers,
        "budget_amount": budget_amount,
    }
    missing = [name for name in REQUIRED_FIELDS if values[name] in (None, "")]
    return {"missing_fields": missing, "is_complete": not missing}


# How each field is put to a traveler. A name that is not here is asked about
# by name, which reads acceptably for a real field ("origin") — but only for a
# real one, which is why blanks are dropped rather than passed through.
READABLE_FIELDS: dict[str, str] = {
    "destination": "where you want to go",
    "start_date": "when the trip starts",
    "end_date": "when the trip ends",
    "travelers": "how many people are travelling",
    "budget_amount": "your total budget",
    "budget_currency": "which currency your budget is in",
    "origin": "which city you are flying from",
    "interests": "what you would like to do there",
    "constraints": "anything the trip has to work around",
}


@tool
def ask_clarifying_question(missing_fields: list[str]) -> str:
    """Build one question that asks the user for all missing trip details.

    Blank entries are dropped rather than asked about: a field name the caller
    could not fill in is not something a traveler can answer, and asking
    produced the empty question "Could you tell me ??".

    Args:
        missing_fields: Names of the fields that are still unknown.

    Returns:
        A single question to put to the user, or a confirmation that nothing
        is missing.
    """
    names = [str(field).strip() for field in missing_fields if str(field).strip()]
    if not names:
        return "All required trip details are present; no question is needed."

    parts = [READABLE_FIELDS.get(name, name.replace("_", " ")) for name in names]
    if len(parts) == 1:
        return f"Could you tell me {parts[0]}?"
    return f"Could you tell me {', '.join(parts[:-1])} and {parts[-1]}?"


INTAKE_TOOLS = [validate_required_fields, ask_clarifying_question]
