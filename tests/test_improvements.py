"""Checks for the efficiency, correctness and robustness work in TODOLIST.md.

Two layers, both offline:

* **Unit** — the new deterministic pieces: retry classification, metrics,
  memory merging, the time-window scheduler, exports.
* **Node integration** — real node functions driven by a stubbed agent, so the
  state handling that used to be untestable without API keys (plan versioning,
  memory merge, failure containment) is now covered in CI.

    uv run pytest tests/test_improvements.py -q
"""

from __future__ import annotations

from datetime import date, time

import pytest

from trip_planner.agents.manager_agent import (
    MAX_REVISIONS,
    PARALLEL_AFTER_INTAKE,
    next_required_agents,
)
from trip_planner.export import day_directions_link, maps_link, to_pdf
from trip_planner.memory import apply as apply_preferences
from trip_planner.memory import load as load_preferences
from trip_planner.memory import save as save_preferences
from trip_planner.metrics import AgentMetrics, UsageCollector, summarize, track
from trip_planner.resilience import classify, resilient_node, with_retry
from trip_planner.schemas import (
    AgentName,
    AttractionsResult,
    Coordinates,
    CriticResult,
    DayPlan,
    IntakeResult,
    Issue,
    ItineraryResult,
    Place,
    RoutingResult,
    ScheduledStop,
    Severity,
    TravelerPreferences,
    TravelerProfile,
)
from trip_planner.tools.cache import _key, cached, is_enabled
from trip_planner.tools.routing_tools import _parse_window, schedule_day

COMPLETE = IntakeResult(
    profile=TravelerProfile(destination="Lisbon, Portugal"), missing_fields=[]
)
REJECTED = CriticResult(
    issues=[
        Issue(severity=Severity.BLOCKER, category="schedule", description="overlap")
    ],
    approved=False,
    reasoning="not usable",
)
APPROVED = CriticResult(issues=[], approved=True, reasoning="fine")


def _fresh() -> dict:
    """Return a starting state for sequencing tests.

    Returns:
        A state dict with the version counters initialised.
    """
    return {
        "intake": COMPLETE,
        "completed_agents": [],
        "plan_version": 0,
        "budget_version": 0,
        "critic_version": 0,
        "revision_count": 0,
    }


def _advance(state: dict, agents: list[AgentName], critic_verdict=None) -> None:
    """Apply one dispatch group to the state, as the real nodes would.

    Args:
        state: The state to mutate.
        agents: The agents that just ran.
        critic_verdict: What the Critic returned, when it was one of them.
    """
    for agent in agents:
        if agent is AgentName.ROUTING:
            state["plan_version"] += 1
            if state.get("critic") and not state["critic"].approved:
                state["revision_count"] += 1
        if agent is AgentName.BUDGET:
            state["budget_version"] = state["plan_version"]
        if agent is AgentName.CRITIC:
            state["critic_version"] = state["plan_version"]
            state["critic"] = critic_verdict
        if agent not in state["completed_agents"]:
            state["completed_agents"] = state["completed_agents"] + [agent]


# ---------------------------------------------------------------------------
# Item 3 — parallel fan-out
# ---------------------------------------------------------------------------


class TestParallelDispatch:
    """The four profile-only agents must be dispatched together."""

    def test_intake_runs_alone_first(self):
        assert next_required_agents({"completed_agents": []}) == [AgentName.INTAKE]

    def test_four_independent_agents_are_dispatched_at_once(self):
        state = _fresh()
        _advance(state, [AgentName.INTAKE])
        assert set(next_required_agents(state)) == set(PARALLEL_AFTER_INTAKE)
        assert len(next_required_agents(state)) == 4

    def test_routing_waits_for_attractions(self):
        """Routing needs the place pool, so it must not start early."""
        state = _fresh()
        _advance(state, [AgentName.INTAKE, AgentName.FLIGHTS, AgentName.LODGING])
        assert AgentName.ROUTING not in next_required_agents(state)

    def test_nothing_runs_on_an_incomplete_profile(self):
        state = _fresh()
        state["intake"] = IntakeResult(
            profile=TravelerProfile(), missing_fields=["destination"]
        )
        _advance(state, [AgentName.INTAKE])
        assert next_required_agents(state) == []


# ---------------------------------------------------------------------------
# Item 1 — the Critic revision loop
# ---------------------------------------------------------------------------


class TestRevisionLoop:
    """A rejected plan must be rebuilt, re-costed and re-checked."""

    def _run(self, verdicts: list) -> list[list[str]]:
        """Drive the sequencer to completion, feeding the Critic verdicts.

        Args:
            verdicts: What the Critic returns on each of its runs.

        Returns:
            The dispatch groups, in order.
        """
        state = _fresh()
        groups: list[list[str]] = []
        pending = list(verdicts)
        for _ in range(40):
            agents = next_required_agents(state)
            if not agents:
                break
            groups.append([str(agent) for agent in agents])
            verdict = (
                pending.pop(0) if AgentName.CRITIC in agents and pending else APPROVED
            )
            _advance(state, agents, verdict)
        return groups

    def test_critic_rechecks_the_revised_plan(self):
        """The bug this fixes: the Critic used to run exactly once."""
        flat = [agent for group in self._run([REJECTED, APPROVED]) for agent in group]
        assert flat.count("critic") == 2, "the revised plan must be re-reviewed"

    def test_budget_is_recomputed_after_a_revision(self):
        """Costs must belong to the plan that actually shipped."""
        flat = [agent for group in self._run([REJECTED, APPROVED]) for agent in group]
        assert flat.count("budget") == 2

    def test_routing_does_not_run_three_times_in_a_row(self):
        """The old loop dispatched routing repeatedly with nothing in between."""
        groups = self._run([REJECTED, APPROVED])
        flat = [agent for group in groups for agent in group]
        assert flat.count("routing") == 2
        for first, second in zip(flat, flat[1:]):
            assert not (first == "routing" and second == "routing")

    def test_revisions_stop_at_the_cap(self):
        """A Critic that never approves must not loop forever."""
        groups = self._run([REJECTED] * 10)
        flat = [agent for group in groups for agent in group]
        assert flat.count("routing") <= MAX_REVISIONS + 1
        assert "itinerary" in flat, "the plan still ships, with issues reported"

    def test_a_stale_verdict_forces_a_recheck(self):
        state = _fresh()
        for agent in (AgentName.INTAKE, *PARALLEL_AFTER_INTAKE):
            _advance(state, [agent])
        _advance(state, [AgentName.ROUTING])
        _advance(state, [AgentName.BUDGET])
        _advance(state, [AgentName.CRITIC], APPROVED)

        # Routing runs again: both downstream stages are now behind.
        _advance(state, [AgentName.ROUTING])
        assert next_required_agents(state) == [AgentName.BUDGET]


# ---------------------------------------------------------------------------
# Item 4b / 11 — retry and failure containment
# ---------------------------------------------------------------------------


class TestResilience:
    """Transient failures retry; permanent ones degrade rather than crash."""

    def test_classifies_the_errors_seen_in_practice(self):
        assert classify(Exception("503 UNAVAILABLE, high demand")) == "transient"
        assert classify(Exception("429 RESOURCE_EXHAUSTED")) == "rate_limit"
        assert classify(Exception("quota exceeded")) == "rate_limit"
        assert classify(Exception("invalid request schema")) == "permanent"

    def test_retries_a_transient_failure_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("trip_planner.resilience.time.sleep", lambda _: None)
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("503 unavailable")
            return "ok"

        assert with_retry(flaky, "test")() == "ok"
        assert attempts["n"] == 3

    def test_does_not_retry_a_permanent_failure(self, monkeypatch):
        monkeypatch.setattr("trip_planner.resilience.time.sleep", lambda _: None)
        attempts = {"n": 0}

        def broken():
            attempts["n"] += 1
            raise ValueError("invalid schema")

        with pytest.raises(ValueError):
            with_retry(broken, "test")()
        assert attempts["n"] == 1, "a bad request must not be retried"

    def test_a_failing_node_degrades_instead_of_crashing(self, monkeypatch):
        monkeypatch.setattr("trip_planner.resilience.time.sleep", lambda _: None)

        @resilient_node(AgentName.FLIGHTS)
        def node(state):
            raise RuntimeError("SerpApi is down")

        update = node({})
        assert update["failed_agents"][0]["agent"] == "flights"
        assert update["completed_agents"] == [AgentName.FLIGHTS]

    def test_the_graph_continues_past_a_failed_stage(self):
        """A failed agent counts as complete, so the pipeline moves on."""
        state = _fresh()
        _advance(state, [AgentName.INTAKE, *PARALLEL_AFTER_INTAKE])
        state["failed_agents"] = [{"agent": "flights", "error": "down"}]
        assert next_required_agents(state) == [AgentName.ROUTING]


# ---------------------------------------------------------------------------
# Item 5 — efficiency instrumentation
# ---------------------------------------------------------------------------


class TestMetrics:
    """The numbers that make every other optimisation checkable."""

    def test_totals_across_agents(self):
        summary = summarize(
            [
                AgentMetrics(
                    agent="flights",
                    seconds=12.0,
                    llm_calls=3,
                    tool_calls=4,
                    input_tokens=8000,
                    output_tokens=1000,
                    model="gpt-5-mini",
                ),
                AgentMetrics(
                    agent="lodging",
                    seconds=8.0,
                    llm_calls=2,
                    tool_calls=2,
                    input_tokens=4000,
                    output_tokens=500,
                    model="gpt-5-mini",
                ),
            ],
            wall_seconds=13.0,
        )
        assert summary.seconds == 20.0
        assert summary.llm_calls == 5
        assert summary.cost_usd is not None and summary.cost_usd > 0
        assert summary.agents[0].agent == "flights", "slowest first"

    def test_reports_what_parallelism_saved(self):
        """Agent time minus wall time is exactly the concurrency benefit."""
        summary = summarize(
            [
                AgentMetrics(agent="a", seconds=10.0),
                AgentMetrics(agent="b", seconds=10.0),
            ],
            wall_seconds=11.0,
        )
        assert summary.parallel_saving == 9.0

    def test_an_unpriced_model_reports_no_cost_rather_than_a_wrong_one(self):
        record = AgentMetrics(agent="x", model="some-unknown-model", input_tokens=1000)
        assert record.cost_usd is None

    def test_track_records_a_node_run(self):
        @track("flights", "gpt-5-mini")
        def node(state, collector):
            return {"flights": "result"}

        update = node({})
        assert update["flights"] == "result"
        assert update["metrics"][0].agent == "flights"
        assert update["metrics"][0].seconds >= 0

    def test_usage_collector_counts_tool_calls(self):
        collector = UsageCollector()
        collector.on_tool_end("x")
        collector.on_tool_end("y")
        assert collector.tool_calls == 2


# ---------------------------------------------------------------------------
# Item 15 — traveler memory
# ---------------------------------------------------------------------------


class TestMemory:
    """Preferences fill gaps; the current request always wins."""

    def test_fills_only_what_the_request_left_out(self):
        preferences = TravelerPreferences(
            home_city="Tel Aviv", default_currency="USD", constraints=["vegetarian"]
        )
        merged = apply_preferences(
            TravelerProfile(destination="Rome", constraints=["no early flights"]),
            preferences,
        )
        assert merged.origin == "Tel Aviv"
        assert merged.budget_currency == "USD"
        assert set(merged.constraints) == {"vegetarian", "no early flights"}

    def test_the_request_overrides_a_stored_preference(self):
        preferences = TravelerPreferences(home_city="Tel Aviv")
        merged = apply_preferences(
            TravelerProfile(destination="Rome", origin="Haifa"), preferences
        )
        assert merged.origin == "Haifa"

    def test_round_trips_through_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("trip_planner.memory.PROFILES_DIR", tmp_path)
        save_preferences(
            TravelerPreferences(home_airport="TLV", pace="relaxed"), "tester"
        )
        assert load_preferences("tester").home_airport == "TLV"

    def test_a_missing_profile_is_empty_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("trip_planner.memory.PROFILES_DIR", tmp_path)
        assert load_preferences("nobody").home_airport is None


# ---------------------------------------------------------------------------
# Item 12 — time-window scheduling
# ---------------------------------------------------------------------------


class TestScheduling:
    """The day scheduler respects real opening hours and travel time."""

    PLACES = [
        {
            "place_id": "p1",
            "name": "Museum",
            "latitude": 38.7378,
            "longitude": -9.1535,
            "visit_duration_minutes": 120,
            "opening_hours": "10 AM-6 PM",
            "closed_days": ["Tuesday"],
        },
        {
            "place_id": "p2",
            "name": "Market",
            "latitude": 38.7071,
            "longitude": -9.1459,
            "visit_duration_minutes": 90,
            "opening_hours": "10 AM-12 AM",
        },
    ]

    def test_parses_the_hour_formats_google_returns(self):
        assert _parse_window("10 AM-6 PM") == (600, 1080)
        assert _parse_window("10:00-18:00") == (600, 1080)
        assert _parse_window("Closed") is None
        assert _parse_window("Open 24 hours") == (0, 1440)

    def test_never_schedules_a_place_on_its_closing_day(self):
        # 2026-09-15 is a Tuesday.
        result = schedule_day.invoke(
            {"places": self.PLACES, "visit_date": "2026-09-15"}
        )
        assert result["weekday"] == "Tuesday"
        assert [stop["place_id"] for stop in result["stops"]] == ["p2"]
        assert result["unscheduled"][0]["reason"] == "closed on Tuesday"

    def test_stops_never_overlap_and_allow_for_travel(self):
        result = schedule_day.invoke(
            {"places": self.PLACES, "visit_date": "2026-09-16"}
        )
        stops = result["stops"]
        assert len(stops) == 2
        for previous, following in zip(stops, stops[1:]):
            assert following["start_time"] >= previous["end_time"]
            assert following["travel_minutes_from_previous"] > 0

    def test_respects_the_opening_hour(self):
        """Nothing may start before the place actually opens."""
        result = schedule_day.invoke(
            {"places": [self.PLACES[0]], "visit_date": "2026-09-16"}
        )
        assert result["stops"][0]["start_time"] >= "10:00"

    def test_reports_what_did_not_fit(self):
        cramped = [
            {**self.PLACES[0], "place_id": f"p{i}", "visit_duration_minutes": 240}
            for i in range(5)
        ]
        result = schedule_day.invoke({"places": cramped, "visit_date": "2026-09-16"})
        assert result["unscheduled"], "places that do not fit must be reported"


# ---------------------------------------------------------------------------
# Item 14 — actionable output
# ---------------------------------------------------------------------------


class TestExport:
    """The plan must be bookable, navigable and portable."""

    PLACE = Place(
        place_id="p1",
        name="Jerónimos Monastery",
        category="historic site",
        coordinates=Coordinates(latitude=38.6979, longitude=-9.2065),
    )

    def test_builds_a_maps_link_from_coordinates(self):
        assert "38.6979,-9.2065" in maps_link(self.PLACE)

    def test_falls_back_to_the_name_without_coordinates(self):
        place = Place(place_id="p2", name="Some Place", category="x")
        assert "Some+Place" in maps_link(place)

    def test_builds_day_directions_covering_every_stop(self):
        second = Place(
            place_id="p2",
            name="Belém Tower",
            category="historic site",
            coordinates=Coordinates(latitude=38.6916, longitude=-9.2160),
        )
        day = DayPlan(
            day_number=1,
            date=date(2026, 9, 10),
            stops=[
                ScheduledStop(
                    place_id="p1", name="A", start_time=time(10, 0), end_time=time(11, 0)
                ),
                ScheduledStop(
                    place_id="p2", name="B", start_time=time(12, 0), end_time=time(13, 0)
                ),
            ],
        )
        link = day_directions_link(day, {"p1": self.PLACE, "p2": second})
        assert link.startswith("https://www.google.com/maps/dir/")
        assert "origin=38.6979" in link and "destination=38.6916" in link

    def test_no_directions_link_for_a_single_stop(self):
        day = DayPlan(
            day_number=1,
            date=date(2026, 9, 10),
            stops=[
                ScheduledStop(
                    place_id="p1", name="A", start_time=time(10, 0), end_time=time(11, 0)
                )
            ],
        )
        assert day_directions_link(day, {"p1": self.PLACE}) == ""

    def test_writes_a_real_pdf(self, tmp_path, monkeypatch):
        monkeypatch.setattr("trip_planner.export.EXPORT_DIR", tmp_path)
        itinerary = ItineraryResult(
            title="5 Days in Lisbon",
            overview="Food and history.",
            days=[
                DayPlan(
                    day_number=1,
                    date=date(2026, 9, 10),
                    summary="Old town",
                    stops=[
                        ScheduledStop(
                            place_id="p1",
                            name="Alfama",
                            start_time=time(10, 0),
                            end_time=time(12, 0),
                        )
                    ],
                )
            ],
            practical_notes=["Bring a jacket"],
        )
        path = to_pdf(itinerary, "trip.pdf")
        data = (tmp_path / "trip.pdf").read_bytes()
        assert path.endswith("trip.pdf")
        assert data.startswith(b"%PDF-"), "must be a real PDF, not text"


# ---------------------------------------------------------------------------
# Item 6 — caching
# ---------------------------------------------------------------------------


class TestCache:
    """Repeated identical lookups must not repeat the network call."""

    def test_the_key_depends_on_the_arguments(self):
        assert _key("t", {"q": "a"}) == _key("t", {"q": "a"})
        assert _key("t", {"q": "a"}) != _key("t", {"q": "b"})

    def test_a_second_identical_call_is_served_from_cache(self, monkeypatch):
        monkeypatch.setenv("TRIP_CACHE_ENABLED", "true")
        calls = {"n": 0}

        @cached("unit-test-hits", ttl=60)
        def lookup(query: str) -> dict:
            calls["n"] += 1
            return {"answer": query}

        lookup("lisbon weather")
        lookup("lisbon weather")
        assert calls["n"] == 1, "the second call must be served from cache"

    def test_errors_are_never_cached(self, monkeypatch):
        monkeypatch.setenv("TRIP_CACHE_ENABLED", "true")
        calls = {"n": 0}

        @cached("unit-test-errors", ttl=60)
        def failing(query: str) -> dict:
            calls["n"] += 1
            return {"error": "upstream down"}

        failing("x")
        failing("x")
        assert calls["n"] == 2, "a failure must be retried, not remembered"

    def test_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("TRIP_CACHE_ENABLED", "false")
        assert is_enabled() is False


# ---------------------------------------------------------------------------
# Item 13 — node integration without live APIs
# ---------------------------------------------------------------------------


class StubAgent:
    """A stand-in for a built agent, returning a canned structured response."""

    def __init__(self, response) -> None:
        self.response = response
        self.calls = 0

    def with_config(self, **kwargs):
        """Accept the metrics callback the node attaches.

        Args:
            **kwargs: Ignored config.

        Returns:
            This same stub.
        """
        return self

    def invoke(self, payload, config=None):
        """Return the canned response.

        Args:
            payload: Ignored message payload.
            config: Ignored run config.

        Returns:
            A dict shaped like a real agent result.
        """
        self.calls += 1
        return {"structured_response": self.response}


class TestNodeIntegration:
    """Real node logic, driven by a stub agent — no API keys, no network."""

    def test_routing_bumps_the_plan_version(self, monkeypatch):
        from trip_planner.agents import routing_agent

        routed = RoutingResult(days=[], reasoning="done")
        monkeypatch.setattr(
            routing_agent, "build_routing_agent", lambda: StubAgent(routed)
        )

        update = routing_agent.routing_node(
            {
                "intake": COMPLETE,
                "attractions": AttractionsResult(places=[], reasoning="x"),
                "plan_version": 2,
            }
        )
        assert update["plan_version"] == 3, "a rebuilt plan is a new version"

    def test_a_revision_counts_but_does_not_re_complete(self, monkeypatch):
        from trip_planner.agents import routing_agent

        monkeypatch.setattr(
            routing_agent,
            "build_routing_agent",
            lambda: StubAgent(RoutingResult(days=[], reasoning="fixed")),
        )
        update = routing_agent.routing_node(
            {
                "intake": COMPLETE,
                "attractions": AttractionsResult(places=[], reasoning="x"),
                "critic": REJECTED,
                "plan_version": 1,
                "revision_count": 0,
            }
        )
        assert update["revision_count"] == 1
        assert "completed_agents" not in update, "routing is already recorded"

    def test_critic_records_the_version_it_judged(self, monkeypatch):
        from trip_planner.agents import critic_agent

        monkeypatch.setattr(
            critic_agent, "build_critic_agent", lambda: StubAgent(APPROVED)
        )
        update = critic_agent.critic_node(
            {"intake": COMPLETE, "plan_version": 4, "completed_agents": []}
        )
        assert update["critic_version"] == 4

    def test_budget_records_the_version_it_costed(self, monkeypatch):
        from trip_planner.agents import budget_agent
        from trip_planner.schemas import BudgetResult

        costed = BudgetResult(
            total_cost=100, currency="USD", within_budget=True, reasoning="ok"
        )
        monkeypatch.setattr(
            budget_agent, "build_budget_agent", lambda: StubAgent(costed)
        )
        update = budget_agent.budget_node(
            {"intake": COMPLETE, "plan_version": 4, "completed_agents": []}
        )
        assert update["budget_version"] == 4

    def test_intake_merges_stored_preferences(self, tmp_path, monkeypatch):
        from trip_planner.agents import intake_agent

        monkeypatch.setattr("trip_planner.memory.PROFILES_DIR", tmp_path)
        save_preferences(TravelerPreferences(home_city="Tel Aviv"), "default")

        extracted = IntakeResult(
            profile=TravelerProfile(destination="Rome"), missing_fields=[]
        )
        monkeypatch.setattr(
            intake_agent, "build_intake_agent", lambda: StubAgent(extracted)
        )
        update = intake_agent.intake_node({"user_request": "4 days in Rome"})
        assert update["intake"].profile.origin == "Tel Aviv"

    def test_a_preference_can_clear_a_missing_field(self, tmp_path, monkeypatch):
        """If memory supplies the gap, the planner should stop asking about it."""
        from trip_planner.agents import intake_agent

        monkeypatch.setattr("trip_planner.memory.PROFILES_DIR", tmp_path)
        save_preferences(TravelerPreferences(default_currency="EUR"), "default")

        extracted = IntakeResult(
            profile=TravelerProfile(destination="Rome"),
            missing_fields=["budget_currency"],
        )
        monkeypatch.setattr(
            intake_agent, "build_intake_agent", lambda: StubAgent(extracted)
        )
        update = intake_agent.intake_node({"user_request": "Rome"})
        assert update["intake"].missing_fields == []


# ---------------------------------------------------------------------------
# Regression — OpenAI strict-mode tool schemas
# ---------------------------------------------------------------------------


class TestStrictModeRegression:
    """Guards the 400 that killed six agents on the first real GPT-5 run.

    Passing a bare Pydantic model as `response_format` makes LangChain choose
    `ProviderStrategy`, which hard-codes `strict=True` on every tool. OpenAI's
    strict mode forbids free-form objects, so every tool taking a `list[dict]`
    was rejected before the run began — as was the Google Calendar MCP server's
    own schema, which is not ours to fix.
    """

    def test_the_factory_wraps_the_schema_in_a_tool_strategy(self):
        """ToolStrategy is the branch that does not force strict."""
        import inspect

        from langchain.agents.structured_output import ToolStrategy

        from trip_planner.agents import factory

        source = inspect.getsource(factory.build_structured_agent)
        assert "ToolStrategy(" in source
        assert ToolStrategy is not None

    def test_no_agent_bypasses_the_factory(self):
        """A direct `create_agent(response_format=Model)` reintroduces the bug."""
        import pathlib

        agents_dir = pathlib.Path(factory_dir())
        offenders = []
        for path in agents_dir.glob("*_agent.py"):
            source = path.read_text(encoding="utf-8")
            if "create_agent(" in source and "build_structured_agent" not in source:
                offenders.append(path.name)
        assert offenders == [], f"these bypass the factory: {offenders}"

    def test_every_agent_still_builds(self):
        """All eleven must construct without touching the network."""
        from trip_planner.agents.attractions_agent import build_attractions_agent
        from trip_planner.agents.budget_agent import build_budget_agent
        from trip_planner.agents.critic_agent import build_critic_agent
        from trip_planner.agents.flights_agent import build_flights_agent
        from trip_planner.agents.intake_agent import build_intake_agent
        from trip_planner.agents.itinerary_agent import build_itinerary_agent
        from trip_planner.agents.lodging_agent import build_lodging_agent
        from trip_planner.agents.research_agent import build_research_agent
        from trip_planner.agents.routing_agent import build_routing_agent

        for build in (
            build_intake_agent,
            build_research_agent,
            build_flights_agent,
            build_lodging_agent,
            build_attractions_agent,
            build_routing_agent,
            build_budget_agent,
            build_critic_agent,
        ):
            assert build() is not None, build.__name__

        assert build_itinerary_agent({}) is not None

    def test_the_free_form_tools_are_still_the_known_set(self):
        """Documents which tools rely on strict being off.

        If this set grows, those tools also need `strict=False` — which the
        factory already guarantees. If it shrinks to empty, strict mode could
        be reconsidered for our own tools (never for the MCP ones).
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        from trip_planner.tools import (
            CRITIC_TOOLS,
            FLIGHT_TOOLS,
            ITINERARY_TOOLS,
            LODGING_TOOLS,
            ROUTING_TOOLS,
        )

        def has_free_form_object(tool) -> bool:
            schema = convert_to_openai_tool(tool)["function"].get("parameters", {})
            found = []

            def walk(node):
                if isinstance(node, dict):
                    if node.get("type") == "object" and not node.get("properties"):
                        found.append(True)
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)

            walk(schema)
            return bool(found)

        free_form = {
            tool.name
            for tools in (
                FLIGHT_TOOLS,
                LODGING_TOOLS,
                ROUTING_TOOLS,
                CRITIC_TOOLS,
                ITINERARY_TOOLS,
            )
            for tool in tools
            if has_free_form_object(tool)
        }
        assert free_form == {
            "compare_flights",
            "check_hotel_location",
            "compare_hotels",
            "cluster_locations",
            "schedule_day",
            "validate_schedule",
            "build_daily_itinerary",
            "format_trip_plan",
        }


def factory_dir() -> str:
    """Return the directory holding the agent modules.

    Returns:
        The path to `trip_planner/agents`.
    """
    import trip_planner.agents as agents_package

    return str(__import__("pathlib").Path(agents_package.__file__).parent)


# ---------------------------------------------------------------------------
# Regression — human-in-the-loop must survive the resilience wrapper
# ---------------------------------------------------------------------------


class TestInterruptRegression:
    """Guards the cascade that turned one clarifying question into 11 failures.

    `interrupt()` suspends the graph by raising `GraphInterrupt`. The retry and
    failure-containment wrappers caught it as an ordinary error, so intake was
    recorded as failed, its result slot stayed empty, and every downstream
    agent then died with `KeyError: 'intake'`.
    """

    def test_an_interrupt_is_not_treated_as_a_failure(self, monkeypatch):
        monkeypatch.setattr("trip_planner.resilience.time.sleep", lambda _: None)
        from langgraph.errors import GraphInterrupt

        @resilient_node(AgentName.INTAKE)
        def node(state):
            raise GraphInterrupt(("need the dates",))

        with pytest.raises(GraphInterrupt):
            node({})

    def test_an_interrupt_is_never_retried(self, monkeypatch):
        """Retrying a suspend would ask the traveler the same question again."""
        monkeypatch.setattr("trip_planner.resilience.time.sleep", lambda _: None)
        from langgraph.errors import GraphInterrupt

        attempts = {"n": 0}

        def asking():
            attempts["n"] += 1
            raise GraphInterrupt(("need the dates",))

        with pytest.raises(GraphInterrupt):
            with_retry(asking, "intake")()
        assert attempts["n"] == 1

    def test_real_errors_are_still_contained(self, monkeypatch):
        """The fix must not stop ordinary failures from degrading gracefully."""
        monkeypatch.setattr("trip_planner.resilience.time.sleep", lambda _: None)

        @resilient_node(AgentName.FLIGHTS)
        def node(state):
            raise RuntimeError("SerpApi is down")

        assert node({})["failed_agents"][0]["agent"] == "flights"

    def test_nothing_is_dispatched_when_intake_produced_no_profile(self):
        """The cascade: an empty `intake` slot must stop the pipeline dead."""
        state = _fresh()
        state["intake"] = None
        state["completed_agents"] = [AgentName.INTAKE]
        state["failed_agents"] = [{"agent": "intake", "error": "boom"}]
        assert next_required_agents(state) == []

    def test_the_real_graph_suspends_and_asks(self, tmp_path, monkeypatch):
        """End to end: the graph must pause with the question, not fail.

        Only the manager and intake run before the suspend, and neither touches
        the network here — the manager's routing is templated and intake is
        stubbed — so this exercises the real graph offline.
        """
        from langgraph.checkpoint.memory import MemorySaver

        from trip_planner.agents import intake_agent
        from trip_planner.graph import compile_graph, initial_state

        monkeypatch.setattr("trip_planner.memory.PROFILES_DIR", tmp_path)
        monkeypatch.setenv("TRIP_ASK_USER", "true")

        incomplete = IntakeResult(
            profile=TravelerProfile(destination="Rome"),
            missing_fields=["start_date", "end_date"],
            clarifying_question="When does the trip start and end?",
        )
        monkeypatch.setattr(
            intake_agent, "build_intake_agent", lambda: StubAgent(incomplete)
        )

        graph = compile_graph(MemorySaver())
        config = {"configurable": {"thread_id": "test-hitl"}, "recursion_limit": 20}

        chunks = list(
            graph.stream(
                initial_state("a week in Rome"), config=config, stream_mode="updates"
            )
        )

        interrupts = [c for c in chunks if "__interrupt__" in c]
        assert interrupts, "the graph must suspend, not finish"

        payload = interrupts[0]["__interrupt__"][0].value
        assert payload["missing_fields"] == ["start_date", "end_date"]
        assert "start" in payload["question"].lower()

        # Nothing may have been recorded as failed.
        failures = [
            entry
            for chunk in chunks
            for update in chunk.values()
            if isinstance(update, dict)
            for entry in update.get("failed_agents", [])
        ]
        assert failures == [], f"a suspend is not a failure: {failures}"

    def test_resuming_feeds_the_answer_back_in(self, tmp_path, monkeypatch):
        """`Command(resume=...)` must continue the same run, not restart it."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        from trip_planner.agents import intake_agent
        from trip_planner.graph import compile_graph, initial_state

        monkeypatch.setattr("trip_planner.memory.PROFILES_DIR", tmp_path)
        monkeypatch.setenv("TRIP_ASK_USER", "true")

        seen: list[str] = []
        incomplete = IntakeResult(
            profile=TravelerProfile(destination="Rome"),
            missing_fields=["start_date", "end_date"],
            clarifying_question="When does the trip start and end?",
        )

        class RecordingAgent(StubAgent):
            def invoke(self, payload, config=None):
                seen.append(payload["messages"][0]["content"])
                return super().invoke(payload, config)

        monkeypatch.setattr(
            intake_agent, "build_intake_agent", lambda: RecordingAgent(incomplete)
        )

        graph = compile_graph(MemorySaver())
        config = {"configurable": {"thread_id": "test-resume"}, "recursion_limit": 20}
        list(
            graph.stream(
                initial_state("a week in Rome"), config=config, stream_mode="updates"
            )
        )
        list(
            graph.stream(
                Command(resume="10-15 September 2026"),
                config=config,
                stream_mode="updates",
            )
        )

        # LangGraph replays an interrupted node from the top when it resumes,
        # so intake is invoked more than twice. What matters is the last call:
        # it must carry the traveler's answer *and* the original request.
        assert len(seen) >= 2, "intake should have re-run after the answer"
        assert "10-15 September 2026" in seen[-1], "the answer must reach the agent"
        assert "a week in Rome" in seen[-1], "the original request must be kept"
