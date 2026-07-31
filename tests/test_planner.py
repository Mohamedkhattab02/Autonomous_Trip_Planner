"""Checks for the eleven-agent trip planner.

Two layers:

* Structure tests - fast, no network. They check the graph shape, the routing
  rules and every deterministic tool. Run on every change.
* Live tests - marked `live`. They call the real Gemini, Tavily and SerpApi
  APIs and prove the agents actually work end to end.

    uv run pytest tests -m "not live"   # fast structure checks
    uv run pytest tests                 # everything, including live API calls
"""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from trip_planner.agents.manager_agent import (
    MAX_REVISIONS,
    next_required_agent,
    route_from_manager,
)
from trip_planner.graph import (
    ATTRACTIONS,
    BUDGET,
    CALENDAR,
    CRITIC,
    FLIGHTS,
    INTAKE,
    ITINERARY,
    LODGING,
    MANAGER,
    RESEARCH,
    ROUTING,
    SPECIALISTS,
    app,
    plan_trip,
)
from trip_planner.mcp_client import (
    calendar_server_config,
    is_calendar_mcp_enabled,
    load_calendar_tools,
    run_async,
)
from trip_planner.render import (
    PIPELINE,
    progress_html,
    render_budget,
    render_calendar,
    render_critic,
    render_flights,
    render_itinerary,
    render_lodging,
    render_places,
    render_profile,
)
from trip_planner.schemas import (
    AgentName,
    AttractionsResult,
    BudgetLine,
    BudgetResult,
    CriticResult,
    DestinationResearch,
    FlightLeg,
    FlightOption,
    FlightsResult,
    IntakeResult,
    Issue,
    ManagerDecision,
    Severity,
    TravelerProfile,
)
from trip_planner.tools.budget_tools import (
    calculate_total_cost,
    estimate_food_cost,
    suggest_cheaper_alternatives,
)
from trip_planner.tools.calendar_tools import (
    create_calendar_event,
    export_ics,
    reset_calendar,
    update_event,
)
from trip_planner.tools.critic_tools import validate_budget, validate_schedule
from trip_planner.tools.flight_tools import compare_flights
from trip_planner.tools.geo import suggest_mode, travel_minutes
from trip_planner.tools.intake_tools import (
    ask_clarifying_question,
    validate_required_fields,
)
from trip_planner.tools.itinerary_tools import build_daily_itinerary, format_trip_plan
from trip_planner.tools.lodging_tools import check_hotel_location, compare_hotels
from trip_planner.tools.manager_tools import AGENT_SEQUENCE
from trip_planner.tools.routing_tools import (
    calculate_distance,
    check_opening_hours,
    cluster_locations,
)

COMPLETE_REQUEST = (
    "I want to travel to Lisbon, Portugal from 2026-09-10 to 2026-09-15. "
    "We are 2 travelers flying from Tel Aviv, our total budget is 3000 USD, "
    "and we love food, history and walking tours."
)

INCOMPLETE_REQUEST = "I'd like to go somewhere in Italy sometime next year."

COMPLETE_PROFILE = IntakeResult(
    profile=TravelerProfile(destination="Lisbon, Portugal"), missing_fields=[]
)


def _lisbon_intake() -> IntakeResult:
    """A complete Lisbon profile, for live tests of a single agent.

    Returns:
        An `IntakeResult` the downstream agents can consume directly.
    """
    return IntakeResult(
        profile=TravelerProfile(
            destination="Lisbon, Portugal",
            origin="Tel Aviv, Israel",
            start_date=date(2026, 9, 10),
            end_date=date(2026, 9, 15),
            travelers=2,
            budget_amount=3000,
            budget_currency="USD",
            interests=["food", "history", "walking tours"],
        )
    )


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """The graph is wired the way plan.md describes."""

    def test_graph_has_all_eleven_agents(self):
        nodes = set(app.get_graph().nodes)
        expected = {
            MANAGER,
            INTAKE,
            RESEARCH,
            FLIGHTS,
            LODGING,
            ATTRACTIONS,
            ROUTING,
            BUDGET,
            CRITIC,
            ITINERARY,
            CALENDAR,
        }
        assert expected <= nodes
        assert len(SPECIALISTS) == 10, "ten specialists plus the manager"

    def test_every_specialist_reports_back_to_the_manager(self):
        edges = {(edge.source, edge.target) for edge in app.get_graph().edges}
        for name in SPECIALISTS:
            assert (name, MANAGER) in edges, f"{name} must return to the manager"

    def test_manager_can_reach_every_specialist(self):
        edges = {(edge.source, edge.target) for edge in app.get_graph().edges}
        for name in SPECIALISTS:
            assert (MANAGER, name) in edges, f"the manager must be able to run {name}"

    def test_graph_renders_as_mermaid(self):
        """The drawing used for graph.png must be produceable offline."""
        mermaid = app.get_graph().draw_mermaid()
        for name in (MANAGER, *SPECIALISTS):
            assert name in mermaid


# ---------------------------------------------------------------------------
# Routing rules
# ---------------------------------------------------------------------------


class TestSequencing:
    """`next_required_agent` fixes the stage order by rule, not by model whim."""

    def test_starts_with_intake(self):
        assert next_required_agent({"completed_agents": []}) == AgentName.INTAKE

    def test_follows_the_declared_sequence(self):
        completed: list[AgentName] = []
        for expected in AGENT_SEQUENCE:
            state = {"completed_agents": list(completed), "intake": COMPLETE_PROFILE}
            assert next_required_agent(state) == expected
            completed.append(expected)

    def test_finishes_when_every_stage_is_done(self):
        state = {"completed_agents": list(AGENT_SEQUENCE), "intake": COMPLETE_PROFILE}
        assert next_required_agent(state) is None

    def test_stops_when_intake_is_incomplete(self):
        """Nothing downstream can run without a destination and dates."""
        state = {
            "completed_agents": [AgentName.INTAKE],
            "intake": IntakeResult(
                profile=TravelerProfile(), missing_fields=["destination"]
            ),
        }
        assert next_required_agent(state) is None

    def test_a_rejected_plan_goes_back_to_routing(self):
        state = {
            "completed_agents": [a for a in AGENT_SEQUENCE if a != AgentName.CALENDAR],
            "intake": COMPLETE_PROFILE,
            "critic": CriticResult(
                issues=[
                    Issue(
                        severity=Severity.BLOCKER,
                        category="schedule",
                        description="two stops overlap",
                    )
                ],
                approved=False,
                reasoning="not usable",
            ),
            "revision_count": 0,
        }
        assert next_required_agent(state) == AgentName.ROUTING

    def test_revisions_are_capped_so_the_graph_cannot_loop_forever(self):
        state = {
            "completed_agents": [
                a
                for a in AGENT_SEQUENCE
                if a not in (AgentName.ITINERARY, AgentName.CALENDAR)
            ],
            "intake": COMPLETE_PROFILE,
            "critic": CriticResult(
                issues=[
                    Issue(
                        severity=Severity.BLOCKER,
                        category="schedule",
                        description="still broken",
                    )
                ],
                approved=False,
                reasoning="still not usable",
            ),
            "revision_count": MAX_REVISIONS,
        }
        assert next_required_agent(state) == AgentName.ITINERARY

    def test_an_approved_plan_moves_on_to_the_itinerary(self):
        state = {
            "completed_agents": [
                a
                for a in AGENT_SEQUENCE
                if a not in (AgentName.ITINERARY, AgentName.CALENDAR)
            ],
            "intake": COMPLETE_PROFILE,
            "critic": CriticResult(issues=[], approved=True, reasoning="looks good"),
            "revision_count": 0,
        }
        assert next_required_agent(state) == AgentName.ITINERARY


class TestRouting:
    """`route_from_manager` turns a decision into the next node."""

    def test_routes_to_the_chosen_agent(self):
        state = {
            "manager_decision": ManagerDecision(
                next_agent=AgentName.FLIGHTS, reasoning="time to book"
            )
        }
        assert route_from_manager(state) == FLIGHTS

    def test_finishes_when_no_agent_is_chosen(self):
        state = {
            "manager_decision": ManagerDecision(next_agent=None, reasoning="all done")
        }
        assert route_from_manager(state) == "finish"

    def test_finishes_when_there_is_no_decision(self):
        assert route_from_manager({}) == "finish"


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

    def test_asks_one_question_covering_all_missing_fields(self):
        question = ask_clarifying_question.invoke(
            {"missing_fields": ["start_date", "budget_amount"]}
        )
        assert question.count("?") == 1
        assert "when the trip starts" in question
        assert "your total budget" in question


class TestFlightTools:
    """Flight ranking is arithmetic, so it must be reproducible."""

    def test_prefers_the_cheaper_flight_when_times_are_similar(self):
        result = compare_flights.invoke(
            {
                "options": [
                    {
                        "option_id": "flight-1",
                        "price": 900,
                        "total_duration_minutes": 400,
                        "stops": 0,
                    },
                    {
                        "option_id": "flight-2",
                        "price": 600,
                        "total_duration_minutes": 420,
                        "stops": 0,
                    },
                ]
            }
        )
        assert result["best_option_id"] == "flight-2"

    def test_penalizes_a_much_longer_flight_despite_a_lower_price(self):
        result = compare_flights.invoke(
            {
                "options": [
                    {
                        "option_id": "cheap-but-endless",
                        "price": 580,
                        "total_duration_minutes": 1800,
                        "stops": 2,
                    },
                    {
                        "option_id": "direct",
                        "price": 620,
                        "total_duration_minutes": 400,
                        "stops": 0,
                    },
                ]
            }
        )
        assert result["best_option_id"] == "direct"

    def test_handles_options_with_no_prices(self):
        result = compare_flights.invoke({"options": [{"option_id": "x"}]})
        assert result["best_option_id"] is None


class TestLodgingTools:
    """Location and value checks are computed, not guessed."""

    def test_measures_distance_from_the_places_that_matter(self):
        result = check_hotel_location.invoke(
            {
                "hotel_latitude": 38.7223,
                "hotel_longitude": -9.1393,
                "reference_points": [
                    {"name": "Alfama", "latitude": 38.7139, "longitude": -9.1259},
                    {"name": "Belem", "latitude": 38.6970, "longitude": -9.2065},
                ],
            }
        )
        assert len(result["distances"]) == 2
        # Nearest first, so the agent can read centrality straight off.
        assert result["distances"][0]["name"] == "Alfama"
        assert result["average_km"] > 0

    def test_ranks_a_cheaper_better_rated_stay_first(self):
        result = compare_hotels.invoke(
            {
                "options": [
                    {
                        "option_id": "stay-1",
                        "name": "Pricey",
                        "total_price": 1200,
                        "rating": 4.2,
                    },
                    {
                        "option_id": "stay-2",
                        "name": "Value",
                        "total_price": 600,
                        "rating": 4.6,
                    },
                ]
            }
        )
        assert result["best_option_id"] == "stay-2"


class TestRoutingTools:
    """Geometry decides the day plan, so these must be exact."""

    def test_measures_distance_and_suggests_walking_when_close(self):
        result = calculate_distance.invoke(
            {
                "from_latitude": 38.7139,
                "from_longitude": -9.1394,
                "to_latitude": 38.7154,
                "to_longitude": -9.1350,
            }
        )
        assert 0 < result["distance_km"] < 1
        assert result["suggested_mode"] == "walking"

    def test_suggests_transit_when_far(self):
        assert suggest_mode(7.5) == "transit"

    def test_travel_time_grows_with_distance(self):
        assert travel_minutes(1.0, "walking") > travel_minutes(1.0, "taxi")
        assert travel_minutes(5.0, "walking") > travel_minutes(1.0, "walking")

    def test_splits_places_into_one_cluster_per_day(self):
        places = [
            {
                "place_id": f"place-{index}",
                "name": f"P{index}",
                "latitude": 38.70 + 0.01 * index,
                "longitude": -9.14 - 0.01 * index,
            }
            for index in range(6)
        ]
        result = cluster_locations.invoke({"places": places, "days": 3})
        assert len(result["clusters"]) == 3
        scheduled = sum(len(cluster["places"]) for cluster in result["clusters"])
        assert scheduled == 6, "no place may be silently dropped"

    def test_reports_places_that_have_no_coordinates(self):
        result = cluster_locations.invoke(
            {
                "places": [
                    {
                        "place_id": "place-1",
                        "name": "A",
                        "latitude": 38.7,
                        "longitude": -9.1,
                    },
                    {"place_id": "place-2", "name": "B"},
                ],
                "days": 1,
            }
        )
        assert result["unassigned"] == ["place-2"]

    def test_catches_a_visit_on_a_closing_day(self):
        # 2026-09-15 is a Tuesday.
        result = check_opening_hours.invoke(
            {
                "place_name": "Gulbenkian Museum",
                "closed_days": ["Tuesday"],
                "visit_date": "2026-09-15",
            }
        )
        assert result["is_open"] is False
        assert result["weekday"] == "Tuesday"

    def test_allows_a_visit_on_an_open_day(self):
        result = check_opening_hours.invoke(
            {
                "place_name": "Gulbenkian Museum",
                "closed_days": ["Tuesday"],
                "visit_date": "2026-09-16",
            }
        )
        assert result["is_open"] is True


class TestBudgetTools:
    """The totals must be arithmetic, never a model's estimate."""

    def test_totals_every_category(self):
        result = calculate_total_cost.invoke(
            {
                "flights": 1200,
                "lodging": 800,
                "activities": 150,
                "food": 600,
                "local_transport": 100,
                "budget_amount": 3000,
            }
        )
        assert result["total_cost"] == 2850
        assert result["within_budget"] is True
        assert result["overage"] == 0

    def test_reports_the_overage_when_over_budget(self):
        result = calculate_total_cost.invoke(
            {"flights": 2000, "lodging": 1500, "budget_amount": 3000}
        )
        assert result["within_budget"] is False
        assert result["overage"] == 500

    def test_uses_itemized_meal_costs_when_they_exceed_the_allowance(self):
        result = estimate_food_cost.invoke(
            {
                "travelers": 2,
                "days": 5,
                "style": "budget",
                "known_restaurant_costs": 900,
            }
        )
        assert result["estimated_total_usd"] == 900

    def test_suggests_cuts_only_when_over_budget(self):
        result = suggest_cheaper_alternatives.invoke(
            {"overage": 0, "currency": "USD", "lodging_cost": 800}
        )
        assert result["suggestions"] == []

    def test_targets_the_largest_category_first(self):
        result = suggest_cheaper_alternatives.invoke(
            {
                "overage": 200,
                "currency": "USD",
                "flights_cost": 500,
                "lodging_cost": 1500,
                "activities_cost": 100,
            }
        )
        assert result["suggestions"][0]["category"] == "lodging"


class TestCriticTools:
    """The Critic's checks are the safety net, so they must be exact."""

    def test_accepts_a_workable_day(self):
        days = [
            {
                "day_number": 1,
                "date": "2026-09-10",
                "stops": [
                    {
                        "name": "Museum",
                        "start_time": "10:00",
                        "end_time": "12:00",
                        "travel_minutes_from_previous": 0,
                    },
                    {
                        "name": "Lunch",
                        "start_time": "12:30",
                        "end_time": "13:30",
                        "travel_minutes_from_previous": 15,
                    },
                ],
            }
        ]
        assert validate_schedule.invoke({"days": days})["valid"] is True

    def test_catches_overlapping_stops(self):
        days = [
            {
                "day_number": 1,
                "date": "2026-09-10",
                "stops": [
                    {
                        "name": "A",
                        "start_time": "10:00",
                        "end_time": "12:00",
                        "travel_minutes_from_previous": 0,
                    },
                    {
                        "name": "B",
                        "start_time": "11:30",
                        "end_time": "13:00",
                        "travel_minutes_from_previous": 15,
                    },
                ],
            }
        ]
        result = validate_schedule.invoke({"days": days})
        assert result["valid"] is False
        assert any("before" in problem["problem"] for problem in result["problems"])

    def test_catches_a_stop_that_ignores_travel_time(self):
        """Leaving at 12:00 with 40 minutes of travel cannot arrive at 12:10."""
        days = [
            {
                "day_number": 1,
                "date": "2026-09-10",
                "stops": [
                    {
                        "name": "A",
                        "start_time": "10:00",
                        "end_time": "12:00",
                        "travel_minutes_from_previous": 0,
                    },
                    {
                        "name": "B",
                        "start_time": "12:10",
                        "end_time": "13:00",
                        "travel_minutes_from_previous": 40,
                    },
                ],
            }
        ]
        result = validate_schedule.invoke({"days": days})
        assert result["valid"] is False
        assert any("arriving no earlier" in p["problem"] for p in result["problems"])

    def test_catches_a_stop_outside_the_travelers_day(self):
        days = [
            {
                "day_number": 1,
                "date": "2026-09-10",
                "stops": [
                    {
                        "name": "Very early",
                        "start_time": "05:00",
                        "end_time": "06:00",
                        "travel_minutes_from_previous": 0,
                    }
                ],
            }
        ]
        assert validate_schedule.invoke({"days": days})["valid"] is False

    def test_catches_an_empty_day(self):
        result = validate_schedule.invoke(
            {"days": [{"day_number": 1, "date": "2026-09-10", "stops": []}]}
        )
        assert result["valid"] is False

    def test_confirms_a_budget_breach(self):
        result = validate_budget.invoke(
            {"total_cost": 3500, "budget_amount": 3000, "currency": "USD"}
        )
        assert result["within_budget"] is False
        assert result["overage"] == 500

    def test_passes_when_no_budget_was_stated(self):
        result = validate_budget.invoke({"total_cost": 5000, "budget_amount": None})
        assert result["within_budget"] is True


class TestItineraryTools:
    """The write-up layout is a rule of the system, not improvised each run."""

    def test_orders_days_and_stops_by_time(self):
        result = build_daily_itinerary.invoke(
            {
                "days": [
                    {
                        "day_number": 2,
                        "date": "2026-09-11",
                        "stops": [
                            {
                                "name": "Late",
                                "start_time": "15:00",
                                "end_time": "16:00",
                            },
                            {
                                "name": "Early",
                                "start_time": "09:00",
                                "end_time": "10:00",
                            },
                        ],
                    },
                    {"day_number": 1, "date": "2026-09-10", "stops": []},
                ]
            }
        )
        assert [day["day_number"] for day in result["days"]] == [1, 2]
        assert result["days"][1]["stops"][0]["name"] == "Early"
        assert result["totals"]["stops"] == 2

    def test_renders_markdown_with_every_section(self):
        markdown = format_trip_plan.invoke(
            {
                "title": "5 Days in Lisbon",
                "overview": "A food and history trip.",
                "days": [
                    {
                        "day_number": 1,
                        "date": "2026-09-10",
                        "summary": "Old town",
                        "stops": [
                            {
                                "name": "Alfama walk",
                                "start_time": "10:00",
                                "end_time": "12:00",
                                "travel_minutes_from_previous": 0,
                            }
                        ],
                    }
                ],
                "flight_summary": "TAP direct from Tel Aviv.",
                "lodging_summary": "Hotel in Baixa.",
                "budget_summary": "2850 of 3000 USD.",
                "practical_notes": ["Bring a light jacket."],
            }
        )
        assert markdown.startswith("# 5 Days in Lisbon")
        for section in (
            "## Flights",
            "## Day by Day",
            "## Budget",
            "## Practical Notes",
        ):
            assert section in markdown
        assert "Alfama walk" in markdown


class TestCalendarTools:
    """The .ics export has to be a real file a calendar app can import."""

    def setup_method(self):
        reset_calendar()

    def test_creates_events_and_exports_a_valid_ics(self, tmp_path, monkeypatch):
        import trip_planner.tools.calendar_tools as calendar_tools

        monkeypatch.setattr(calendar_tools, "EXPORT_DIR", tmp_path)

        create_calendar_event.invoke(
            {
                "event_id": "event-1",
                "title": "TLV to LIS",
                "start": "2026-09-10T16:05:00",
                "end": "2026-09-10T20:20:00",
                "category": "flight",
            }
        )
        create_calendar_event.invoke(
            {
                "event_id": "event-2",
                "title": "Gulbenkian Museum",
                "start": "2026-09-11T10:00:00",
                "end": "2026-09-11T12:00:00",
                "category": "activity",
            }
        )
        result = export_ics.invoke({"filename": "trip.ics"})

        assert result["event_count"] == 2
        content = (tmp_path / "trip.ics").read_text(encoding="utf-8")
        assert content.startswith("BEGIN:VCALENDAR")
        assert content.count("BEGIN:VEVENT") == 2
        assert "Gulbenkian Museum" in content

    def test_rejects_an_event_that_ends_before_it_starts(self):
        result = create_calendar_event.invoke(
            {
                "event_id": "event-1",
                "title": "Impossible",
                "start": "2026-09-11T14:00:00",
                "end": "2026-09-11T13:00:00",
            }
        )
        assert "error" in result

    def test_rejects_unparseable_times(self):
        result = create_calendar_event.invoke(
            {
                "event_id": "event-1",
                "title": "Vague",
                "start": "next tuesday",
                "end": "later",
            }
        )
        assert "error" in result

    def test_updating_an_unknown_event_is_an_error(self):
        assert "error" in update_event.invoke({"event_id": "nope", "title": "X"})

    def test_exporting_nothing_is_an_error(self):
        assert "error" in export_ics.invoke({})


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------


class TestMcpLayer:
    """The calendar backend must degrade rather than fail."""

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CALENDAR_MCP_ENABLED", raising=False)
        assert calendar_server_config() is None
        assert is_calendar_mcp_enabled() is False

    def test_builds_a_stdio_config_when_enabled(self, monkeypatch):
        monkeypatch.setenv("CALENDAR_MCP_ENABLED", "true")
        monkeypatch.setenv("CALENDAR_MCP_COMMAND", "node")
        config = calendar_server_config()
        assert config is not None
        assert config["transport"] == "stdio"
        assert config["command"].lower().endswith(("node", "node.exe"))

    def test_falls_back_when_the_command_is_missing(self, monkeypatch):
        """A misconfigured server must not take the planner down with it."""
        monkeypatch.setenv("CALENDAR_MCP_ENABLED", "true")
        monkeypatch.setenv("CALENDAR_MCP_COMMAND", "definitely-not-a-real-binary")
        assert calendar_server_config() is None

    def test_falls_back_to_the_local_ics_tools(self, monkeypatch):
        monkeypatch.delenv("CALENDAR_MCP_ENABLED", raising=False)
        tools, using_mcp = load_calendar_tools(force_reload=True)
        assert using_mcp is False
        assert {tool.name for tool in tools} == {
            "create_calendar_event",
            "update_event",
            "export_ics",
        }

    def test_run_async_works_without_a_running_loop(self):
        async def answer() -> int:
            return 42

        assert run_async(answer()) == 42

    def test_run_async_works_inside_a_running_loop(self):
        """The Gradio server already owns a loop, so this path must work too."""

        async def answer() -> int:
            return 42

        async def outer() -> int:
            return run_async(answer())

        assert asyncio.run(outer()) == 42


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    """The frontend must render partial state without crashing."""

    def test_every_panel_handles_a_missing_stage(self):
        """Panels are drawn while the run is still in progress."""
        assert render_profile(None)
        assert render_flights(None)
        assert render_lodging(None)
        assert render_places(None)
        assert render_budget(None)
        assert render_critic(None)
        assert render_itinerary(None, None)
        assert render_calendar(None, "local")

    def test_renders_a_flight_with_its_recommendation(self):
        flights = FlightsResult(
            options=[
                FlightOption(
                    option_id="flight-1",
                    legs=[
                        FlightLeg(
                            airline="TAP",
                            flight_number="TP 1",
                            departure_airport="TLV",
                            arrival_airport="LIS",
                            departure_time="2026-09-10 16:05",
                            arrival_time="2026-09-10 20:20",
                            duration_minutes=315,
                        )
                    ],
                    stops=0,
                    total_duration_minutes=315,
                    price=617,
                    currency="USD",
                )
            ],
            recommended_option_id="flight-1",
            reasoning="Cheapest direct.",
        )
        output = render_flights(flights)
        assert "617" in output
        assert "Direct" in output
        assert "Recommended" in output

    def test_flags_an_over_budget_trip(self):
        budget = BudgetResult(
            lines=[BudgetLine(category="flights", amount=3500)],
            total_cost=3500,
            budget_amount=3000,
            currency="USD",
            within_budget=False,
            overage=500,
            reasoning="over",
        )
        assert "Over budget" in render_budget(budget)

    def test_progress_tracker_marks_done_active_and_idle(self):
        html = progress_html([AgentName.INTAKE, AgentName.FLIGHTS], active="lodging")
        assert html.count("chip-done") == 2
        assert html.count("chip-active") == 1
        assert html.count("chip-idle") == len(PIPELINE) - 3


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

    def test_critic_separates_blockers_from_lesser_issues(self):
        critic = CriticResult(
            issues=[
                Issue(
                    severity=Severity.BLOCKER,
                    category="facts",
                    description="place does not exist",
                ),
                Issue(
                    severity=Severity.WARNING,
                    category="schedule",
                    description="tight connection",
                ),
                Issue(severity=Severity.INFO, category="budget", description="fyi"),
            ],
            approved=False,
            reasoning="one blocker",
        )
        assert len(critic.blockers) == 1
        assert critic.blockers[0].category == "facts"


# ---------------------------------------------------------------------------
# Live runs against the real APIs
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveAgents:
    """Real Gemini, Tavily and SerpApi. Slow, and needs the API keys in .env."""

    def test_intake_extracts_a_complete_profile(self):
        from trip_planner.agents.intake_agent import intake_node

        result = intake_node({"user_request": COMPLETE_REQUEST})
        intake: IntakeResult = result["intake"]

        assert intake.is_complete, f"unexpected missing fields: {intake.missing_fields}"
        assert "lisbon" in (intake.profile.destination or "").lower()
        assert intake.profile.start_date == date(2026, 9, 10)
        assert intake.profile.travelers == 2
        assert intake.profile.budget_amount == 3000

    def test_intake_asks_instead_of_inventing_missing_details(self):
        from trip_planner.agents.intake_agent import intake_node

        result = intake_node({"user_request": INCOMPLETE_REQUEST})
        intake: IntakeResult = result["intake"]

        assert not intake.is_complete
        assert intake.clarifying_question
        assert intake.profile.start_date is None
        assert intake.profile.budget_amount is None

    def test_research_returns_cited_facts_from_the_web(self):
        from trip_planner.agents.research_agent import research_node

        research: DestinationResearch = research_node({"intake": _lisbon_intake()})[
            "research"
        ]

        assert "lisbon" in research.destination.lower()
        for field in ("summary", "weather", "safety", "currency", "transportation"):
            assert getattr(research, field).strip(), f"{field} is empty"
        assert research.sources
        assert all(source.url.startswith("http") for source in research.sources)

    def test_flights_returns_real_priced_options(self):
        from trip_planner.agents.flights_agent import flights_node

        flights = flights_node({"intake": _lisbon_intake()})["flights"]

        assert flights.options, "should have found real flights"
        assert all(option.price > 0 for option in flights.options)
        assert all(option.legs for option in flights.options)
        assert flights.recommended_option_id in {
            option.option_id for option in flights.options
        }

    def test_lodging_returns_real_priced_stays(self):
        from trip_planner.agents.lodging_agent import lodging_node

        lodging = lodging_node({"intake": _lisbon_intake()})["lodging"]

        assert lodging.options, "should have found real stays"
        assert all(option.total_price > 0 for option in lodging.options)
        assert any(
            option.coordinates for option in lodging.options
        ), "routing needs coordinates on at least some stays"

    def test_attractions_returns_places_with_coordinates(self):
        from trip_planner.agents.attractions_agent import attractions_node

        attractions: AttractionsResult = attractions_node({"intake": _lisbon_intake()})[
            "attractions"
        ]

        assert len(attractions.places) >= 4
        assert all(
            place.coordinates for place in attractions.places
        ), "every place must be locatable, or routing cannot schedule it"

    def test_full_graph_plans_a_whole_trip(self):
        """Every stage, orchestrated by the manager, end to end."""
        final = plan_trip(COMPLETE_REQUEST)

        completed = final["completed_agents"]
        for name in AGENT_SEQUENCE:
            assert name in completed, f"{name} never ran"

        assert final["intake"].is_complete
        assert final["research"].sources
        assert final["routing"].days, "the plan must have days"
        assert final["budget"].total_cost > 0
        assert final["itinerary"].markdown, "the traveler needs a readable plan"
        assert final["calendar"].events, "the approved plan must reach the calendar"
        assert final["manager_decision"].next_agent is None, "should end deliberately"

    def test_graph_stops_when_the_request_is_too_vague(self):
        """With fields missing, the planner must stop rather than plan blindly."""
        final = plan_trip(INCOMPLETE_REQUEST)

        assert AgentName.INTAKE in final["completed_agents"]
        assert AgentName.FLIGHTS not in final["completed_agents"]
        assert final["intake"].clarifying_question
