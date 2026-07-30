"""Stage 1 checks: Travel Manager + Intake + Destination Research.

Two layers:

* Structure tests - fast, no network. They check the graph shape, the routing
  rules and the deterministic tools. Run on every change.
* Live tests - marked `live`. They call the real Gemini and Tavily APIs and
  prove the agents actually work end to end.

    uv run pytest tests -m "not live"   # fast structure checks
    uv run pytest tests                 # everything, including live API calls
"""

from __future__ import annotations

from datetime import date

import pytest

from trip_planner.agents.manager_agent import route_from_manager
from trip_planner.graph import INTAKE, MANAGER, RESEARCH, app, plan_trip
from trip_planner.schemas import (
    AgentName,
    DestinationResearch,
    IntakeResult,
    ManagerDecision,
    TravelerProfile,
)
from trip_planner.tools.intake_tools import (
    ask_clarifying_question,
    validate_required_fields,
)

COMPLETE_REQUEST = (
    "I want to travel to Lisbon, Portugal from 2026-09-10 to 2026-09-15. "
    "We are 2 travelers flying from Tel Aviv, our total budget is 3000 USD, "
    "and we love food, history and walking tours."
)

INCOMPLETE_REQUEST = "I'd like to go somewhere in Italy sometime next year."


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """The graph is wired the way plan.md describes."""

    def test_graph_has_the_three_stage1_nodes(self):
        nodes = set(app.get_graph().nodes)
        assert {MANAGER, INTAKE, RESEARCH} <= nodes

    def test_specialists_report_back_to_the_manager(self):
        edges = {(edge.source, edge.target) for edge in app.get_graph().edges}
        assert (INTAKE, MANAGER) in edges
        assert (RESEARCH, MANAGER) in edges

    def test_manager_can_reach_every_specialist(self):
        edges = {(edge.source, edge.target) for edge in app.get_graph().edges}
        assert (MANAGER, INTAKE) in edges
        assert (MANAGER, RESEARCH) in edges

    def test_graph_renders_as_mermaid(self):
        """The drawing used for graph.png must be produceable offline."""
        mermaid = app.get_graph().draw_mermaid()
        assert MANAGER in mermaid
        assert INTAKE in mermaid
        assert RESEARCH in mermaid


# ---------------------------------------------------------------------------
# Routing rules
# ---------------------------------------------------------------------------


class TestRouting:
    """`route_from_manager` turns a decision into the next node safely."""

    def test_routes_to_the_chosen_agent(self):
        state = {
            "manager_decision": ManagerDecision(
                next_agent=AgentName.INTAKE, reasoning="nothing has run yet"
            ),
            "completed_agents": [],
        }
        assert route_from_manager(state) == INTAKE

    def test_finishes_when_no_agent_is_chosen(self):
        state = {
            "manager_decision": ManagerDecision(next_agent=None, reasoning="all done"),
            "completed_agents": [AgentName.INTAKE, AgentName.DESTINATION_RESEARCH],
        }
        assert route_from_manager(state) == "finish"

    def test_never_re_runs_a_completed_agent(self):
        """Guards against the manager looping the graph forever."""
        state = {
            "manager_decision": ManagerDecision(
                next_agent=AgentName.INTAKE, reasoning="confused"
            ),
            "completed_agents": [AgentName.INTAKE],
        }
        assert route_from_manager(state) == "finish"

    def test_finishes_when_there_is_no_decision(self):
        assert route_from_manager({"completed_agents": []}) == "finish"


# ---------------------------------------------------------------------------
# Deterministic tools
# ---------------------------------------------------------------------------


class TestIntakeTools:
    """The intake tools decide completeness by rule, not by model judgement."""

    def test_reports_every_missing_field(self):
        result = validate_required_fields.invoke({})
        assert result["is_complete"] is False
        assert set(result["missing_fields"]) == {
            "destination",
            "start_date",
            "end_date",
            "travelers",
            "budget_amount",
        }

    def test_reports_complete_when_all_fields_are_present(self):
        result = validate_required_fields.invoke(
            {
                "destination": "Lisbon, Portugal",
                "start_date": "2026-09-10",
                "end_date": "2026-09-15",
                "travelers": 2,
                "budget_amount": 3000.0,
            }
        )
        assert result == {"missing_fields": [], "is_complete": True}

    def test_reports_only_what_is_actually_missing(self):
        result = validate_required_fields.invoke(
            {"destination": "Rome, Italy", "travelers": 2}
        )
        assert set(result["missing_fields"]) == {
            "start_date",
            "end_date",
            "budget_amount",
        }

    def test_asks_one_question_covering_all_missing_fields(self):
        question = ask_clarifying_question.invoke(
            {"missing_fields": ["start_date", "budget_amount"]}
        )
        assert question.count("?") == 1
        assert "when the trip starts" in question
        assert "your total budget" in question

    def test_asks_nothing_when_nothing_is_missing(self):
        question = ask_clarifying_question.invoke({"missing_fields": []})
        assert "no question is needed" in question


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    """The structured messages agents pass to each other."""

    def test_profile_starts_empty_so_nothing_is_invented(self):
        profile = TravelerProfile()
        assert profile.destination is None
        assert profile.interests == []

    def test_intake_result_is_complete_only_without_missing_fields(self):
        assert IntakeResult(profile=TravelerProfile()).is_complete is True
        assert (
            IntakeResult(
                profile=TravelerProfile(), missing_fields=["destination"]
            ).is_complete
            is False
        )


# ---------------------------------------------------------------------------
# Live runs against the real APIs
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveAgents:
    """Real Gemini + real Tavily. Slow, and needs the API keys in .env."""

    def test_intake_extracts_a_complete_profile(self):
        from trip_planner.agents.intake_agent import intake_node

        result = intake_node({"user_request": COMPLETE_REQUEST})
        intake: IntakeResult = result["intake"]

        assert intake.is_complete, f"unexpected missing fields: {intake.missing_fields}"
        assert "lisbon" in (intake.profile.destination or "").lower()
        assert intake.profile.start_date == date(2026, 9, 10)
        assert intake.profile.end_date == date(2026, 9, 15)
        assert intake.profile.travelers == 2
        assert intake.profile.budget_amount == 3000
        assert result["completed_agents"] == [AgentName.INTAKE]

    def test_intake_asks_instead_of_inventing_missing_details(self):
        from trip_planner.agents.intake_agent import intake_node

        result = intake_node({"user_request": INCOMPLETE_REQUEST})
        intake: IntakeResult = result["intake"]

        assert not intake.is_complete
        assert intake.clarifying_question, "should have asked the traveler a question"
        # It must not fabricate dates or a budget that were never stated.
        assert intake.profile.start_date is None
        assert intake.profile.budget_amount is None

    def test_research_returns_cited_facts_from_the_web(self):
        from trip_planner.agents.research_agent import research_node

        state = {
            "intake": IntakeResult(
                profile=TravelerProfile(
                    destination="Lisbon, Portugal",
                    start_date=date(2026, 9, 10),
                    end_date=date(2026, 9, 15),
                    travelers=2,
                    interests=["food", "history"],
                )
            )
        }
        research: DestinationResearch = research_node(state)["research"]

        assert "lisbon" in research.destination.lower()
        for field in ("summary", "weather", "safety", "currency", "transportation"):
            assert getattr(research, field).strip(), f"{field} is empty"
        assert research.sources, "research must cite its sources"
        assert all(s.url.startswith("http") for s in research.sources)

    def test_full_graph_runs_manager_intake_and_research(self):
        """The whole stage-1 pipeline, orchestrated by the manager."""
        final = plan_trip(COMPLETE_REQUEST)

        assert final["completed_agents"] == [
            AgentName.INTAKE,
            AgentName.DESTINATION_RESEARCH,
        ], "the manager must run intake first, then research, each exactly once"
        assert final["intake"].is_complete
        assert final["research"].sources
        assert final["manager_decision"].next_agent is None, "should end deliberately"

    def test_graph_stops_when_the_request_is_too_vague(self):
        """With fields missing, the manager must stop rather than research blindly."""
        final = plan_trip(INCOMPLETE_REQUEST)

        assert AgentName.INTAKE in final["completed_agents"]
        assert AgentName.DESTINATION_RESEARCH not in final["completed_agents"]
        assert final["intake"].clarifying_question
