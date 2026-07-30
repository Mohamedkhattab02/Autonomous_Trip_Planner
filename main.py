"""Run the trip planner on a single request and print what each agent produced.

Usage:
    uv run main.py
    uv run main.py "I want 4 days in Rome in May 2027 for 2 people, 2500 EUR"
"""

from __future__ import annotations

import sys

from trip_planner.graph import plan_trip

DEFAULT_REQUEST = (
    "I want to travel to Lisbon, Portugal from 2026-09-10 to 2026-09-15. "
    "We are 2 travelers flying from Tel Aviv, our total budget is 3000 USD, "
    "and we love food, history and walking tours."
)


def main() -> None:
    """Plan a trip and print the result of each agent."""
    request = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REQUEST

    print(f"Request: {request}\n")
    final = plan_trip(request)

    completed = ", ".join(str(name) for name in final.get("completed_agents", []))
    print(f"Agents that ran: {completed or 'none'}\n")

    intake = final.get("intake")
    if intake:
        print("--- Intake ---")
        print(intake.profile.model_dump_json(indent=2))
        if intake.clarifying_question:
            print(f"\nNeeds from the traveler: {intake.clarifying_question}")
        print()

    research = final.get("research")
    if research:
        print("--- Destination Research ---")
        print(f"Summary:      {research.summary}")
        print(f"Weather:      {research.weather}")
        print(f"Safety:       {research.safety}")
        print(f"Currency:     {research.currency}")
        print(f"Transport:    {research.transportation}")
        print(f"Entry:        {research.entry_requirements}")
        print("\nSources:")
        for source in research.sources:
            print(f"  - {source.title}: {source.url}")
        print()

    decision = final.get("manager_decision")
    if decision:
        print(f"--- Manager ---\n{decision.reasoning}")


if __name__ == "__main__":
    main()
