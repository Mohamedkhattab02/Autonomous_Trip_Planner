"""Gradio frontend for the Autonomous Trip Planner.

Streams the LangGraph run so the traveler watches each of the eleven agents
finish in turn, rather than staring at a spinner for two minutes.

Usage:
    uv run app.py
"""

from __future__ import annotations

import time

import gradio as gr

from trip_planner.graph import RECURSION_LIMIT, app as graph_app, initial_state
from trip_planner.llm import describe_models
from trip_planner.mcp_client import describe_calendar_backend
from trip_planner.render import (
    PIPELINE,
    progress_html,
    render_degraded,
    render_efficiency,
    render_budget,
    render_calendar,
    render_critic,
    render_flights,
    render_itinerary,
    render_lodging,
    render_places,
    render_profile,
    render_research,
)

EXAMPLES = [
    "I want to travel to Lisbon, Portugal from 2026-09-10 to 2026-09-15. "
    "We are 2 travelers flying from Tel Aviv, our total budget is 3000 USD, "
    "and we love food, history and walking tours.",
    "4 days in Rome from 2027-05-03 to 2027-05-07, 2 people from Tel Aviv, "
    "2500 EUR total, we care about art, ancient history and good pasta.",
    "A week in Tokyo from 2027-04-01 to 2027-04-08 for 1 traveler from London, "
    "4000 GBP, interested in temples, ramen and photography.",
]

CSS = """
:root {
    --tp-ink: #0f1729;
    --tp-muted: #64748b;
    --tp-line: rgba(148, 163, 184, 0.22);
    --tp-accent: #6366f1;
}

.gradio-container { max-width: 1240px !important; }

/* ---------- header ---------- */
#tp-header {
    background: linear-gradient(125deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 20px;
    color: #fff;
    box-shadow: 0 18px 40px -18px rgba(79, 70, 229, 0.75);
}
#tp-header h1 {
    margin: 0;
    font-size: 1.95rem;
    font-weight: 750;
    letter-spacing: -0.028em;
    color: #fff;
}
#tp-header p {
    margin: 8px 0 0;
    opacity: 0.9;
    font-size: 0.97rem;
    line-height: 1.5;
}
#tp-header .tp-tags { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
#tp-header .tp-tag {
    background: rgba(255, 255, 255, 0.17);
    border: 1px solid rgba(255, 255, 255, 0.26);
    border-radius: 999px;
    padding: 4px 13px;
    font-size: 0.78rem;
    font-weight: 500;
    backdrop-filter: blur(6px);
}

/* ---------- pipeline tracker ---------- */
.pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 4px 0 2px;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 550;
    border: 1px solid var(--tp-line);
    transition: all 0.25s ease;
}
.chip-icon { font-size: 0.95rem; line-height: 1; }
.chip-idle { color: var(--tp-muted); opacity: 0.55; }
.chip-active {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-color: transparent;
    color: #fff;
    box-shadow: 0 6px 18px -6px rgba(99, 102, 241, 0.9);
    animation: tp-pulse 1.4s ease-in-out infinite;
}
.chip-done {
    background: rgba(16, 185, 129, 0.13);
    border-color: rgba(16, 185, 129, 0.42);
    color: #059669;
}
.chip-done .chip-label::after { content: " ✓"; font-weight: 700; }
@keyframes tp-pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.045); opacity: 0.9; }
}
.dark .chip-done { color: #34d399; }

/* ---------- panels ---------- */
#tp-status {
    font-size: 0.87rem;
    color: var(--tp-muted);
    padding: 2px 0 0 2px;
    min-height: 20px;
}
.tp-panel {
    border: 1px solid var(--tp-line);
    border-radius: 16px;
    padding: 20px 24px;
    background: var(--block-background-fill);
}

/* ---------- shared building blocks ----------
   Colours are the validated data-viz palette: status hues for state,
   categorical slots 1-5 for the budget breakdown. Every status also carries
   an icon and a label, so colour never carries meaning alone. */
.tp-panel {
    --tp-good: #0ca30c;
    --tp-warning: #fab219;
    --tp-critical: #d03b3b;
    --tp-s1: #2a78d6;  /* flights  */
    --tp-s2: #eb6834;  /* lodging  */
    --tp-s3: #1baf7a;  /* activities */
    --tp-s4: #eda100;  /* food     */
    --tp-s5: #e87ba4;  /* transport */
    --tp-ring: rgba(11, 11, 11, 0.10);
    --tp-soft: rgba(148, 163, 184, 0.10);
}
.dark .tp-panel {
    --tp-s1: #3987e5;
    --tp-s2: #d95926;
    --tp-s3: #199e70;
    --tp-s4: #c98500;
    --tp-s5: #d55181;
    --tp-ring: rgba(255, 255, 255, 0.10);
    --tp-soft: rgba(148, 163, 184, 0.09);
}

.tp-series-1 { background: var(--tp-s1); }
.tp-series-2 { background: var(--tp-s2); }
.tp-series-3 { background: var(--tp-s3); }
.tp-series-4 { background: var(--tp-s4); }
.tp-series-5 { background: var(--tp-s5); }
.tp-series-8 { background: var(--tp-muted); }

.tp-dim { color: var(--tp-muted); font-weight: 400; }

.tp-empty {
    display: flex; align-items: center; gap: 10px;
    padding: 26px; border: 1px dashed var(--tp-line); border-radius: 14px;
    color: var(--tp-muted); font-size: 0.92rem; justify-content: center;
}

/* hero number + lede */
.tp-hero {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em;
    line-height: 1.1; margin: 2px 0 6px;
}
.tp-lede { font-size: 1.02rem; line-height: 1.6; margin-bottom: 18px; opacity: 0.9; }

/* stat tiles */
.tp-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px; margin-bottom: 20px;
}
.tp-stat {
    border: 1px solid var(--tp-line); border-radius: 14px;
    padding: 13px 16px; background: var(--tp-soft);
}
.tp-stat-label {
    font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--tp-muted); font-weight: 600;
}
.tp-stat-value {
    font-size: 1.45rem; font-weight: 650; letter-spacing: -0.02em; margin-top: 3px;
}
.tp-stat-sub { font-size: 0.78rem; color: var(--tp-muted); margin-top: 2px; }

/* sections */
.tp-section { margin: 22px 0; }
.tp-h3 {
    font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--tp-muted); font-weight: 700; margin: 0 0 12px !important;
}

/* cards */
.tp-card {
    border: 1px solid var(--tp-line); border-radius: 15px;
    padding: 16px 18px; margin-bottom: 12px; background: var(--block-background-fill);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.tp-card:hover { border-color: rgba(99, 102, 241, 0.45); }
.tp-card-chosen {
    border-color: rgba(99, 102, 241, 0.55);
    box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.25), 0 10px 26px -18px rgba(99,102,241,0.9);
}
.tp-card-head {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 14px; flex-wrap: wrap;
}
.tp-card-title { font-weight: 650; font-size: 1.02rem; }
.tp-card-meta { font-size: 0.83rem; color: var(--tp-muted); margin-top: 3px; }
.tp-card-why { font-size: 0.87rem; margin-top: 8px; opacity: 0.9; }
.tp-hours { font-size: 0.78rem; margin-top: 6px; }
.tp-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
}
.tp-grid .tp-card { margin-bottom: 0; }

/* badges */
.tp-badges { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 10px; }
.tp-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 550;
    border: 1px solid var(--tp-line); color: var(--tp-muted); white-space: nowrap;
}
.tp-badge-accent {
    background: rgba(99, 102, 241, 0.13); border-color: rgba(99, 102, 241, 0.4);
    color: #4f46e5;
}
.dark .tp-badge-accent { color: #a5b4fc; }
.tp-badge-good {
    background: rgba(12, 163, 12, 0.12); border-color: rgba(12, 163, 12, 0.4);
    color: #0a7c0a;
}
.dark .tp-badge-good { color: #4ade80; }
.tp-badge-warning {
    background: rgba(250, 178, 25, 0.15); border-color: rgba(250, 178, 25, 0.45);
    color: #92610a;
}
.dark .tp-badge-warning { color: #fbbf24; }
.tp-badge-critical {
    background: rgba(208, 59, 59, 0.12); border-color: rgba(208, 59, 59, 0.42);
    color: #b3302f;
}
.dark .tp-badge-critical { color: #f87171; }
.tp-badge-info { background: var(--tp-soft); }
.tp-chips { display: flex; flex-wrap: wrap; gap: 7px; }

.tp-stars { font-size: 0.82rem; color: var(--tp-s4); letter-spacing: 1px; }
.tp-stars-num { color: var(--tp-muted); margin-left: 5px; letter-spacing: 0; }

/* callouts */
.tp-callout {
    border-radius: 14px; padding: 14px 18px; margin: 14px 0;
    border: 1px solid var(--tp-line); border-left-width: 3px; background: var(--tp-soft);
}
.tp-callout p { margin: 5px 0 0; font-size: 0.9rem; opacity: 0.9; }
.tp-callout-title { font-weight: 650; font-size: 0.94rem; }
.tp-callout-good { border-left-color: var(--tp-good); }
.tp-callout-warning { border-left-color: var(--tp-warning); }
.tp-callout-critical { border-left-color: var(--tp-critical); }

/* agent reasoning */
.tp-note {
    display: flex; gap: 9px; margin-top: 18px; padding: 13px 16px;
    border-radius: 12px; background: var(--tp-soft);
    font-size: 0.88rem; color: var(--tp-muted); line-height: 1.55;
}
.tp-note-icon { flex-shrink: 0; }

/* flights */
.tp-price { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.025em; }
.tp-price-sm { font-size: 1.15rem; font-weight: 650; white-space: nowrap; }
.tp-legs { margin-top: 12px; display: flex; flex-direction: column; gap: 10px; }
.tp-leg { padding-left: 2px; }
.tp-leg-route { display: flex; align-items: center; gap: 10px; }
.tp-iata { font-weight: 700; font-size: 0.95rem; letter-spacing: 0.04em; }
.tp-leg-line {
    flex: 1; height: 1px; background: var(--tp-line); position: relative; max-width: 190px;
}
.tp-leg-plane {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: var(--block-background-fill); padding: 0 6px;
    font-size: 0.8rem; color: var(--tp-accent);
}
.tp-leg-meta { font-size: 0.8rem; color: var(--tp-muted); margin-top: 3px; }
.tp-tradeoffs { margin: 12px 0 0; padding-left: 2px; list-style: none; }
.tp-tradeoffs li { font-size: 0.85rem; margin: 4px 0; opacity: 0.9; }

/* budget meter + stacked breakdown */
.tp-meter { margin: 16px 0 6px; }
.tp-meter-track {
    height: 12px; border-radius: 999px; overflow: hidden;
    background: rgba(42, 120, 214, 0.16);
}
.tp-meter-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
.tp-meter-good { background: var(--tp-good); }
.tp-meter-critical { background: var(--tp-critical); }
.tp-meter-scale {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; color: var(--tp-muted); margin-top: 6px;
}
.tp-stack {
    display: flex; height: 16px; border-radius: 8px; overflow: hidden;
    gap: 2px; margin-bottom: 14px;   /* 2px surface gap between segments */
}
.tp-seg { height: 100%; min-width: 2px; }
.tp-legend { display: flex; flex-direction: column; gap: 2px; }
.tp-legend-row {
    display: grid; grid-template-columns: 14px 1fr auto; gap: 10px;
    align-items: center; padding: 7px 2px; border-bottom: 1px solid var(--tp-line);
    font-size: 0.89rem;
}
.tp-legend-row:last-child { border-bottom: none; }
.tp-swatch { width: 11px; height: 11px; border-radius: 3px; }
.tp-legend-value { font-weight: 600; white-space: nowrap; }
.tp-legend-detail {
    grid-column: 2 / -1; font-size: 0.79rem; color: var(--tp-muted); margin-top: -2px;
}

/* itinerary timeline */
.tp-day { padding-bottom: 8px; }
.tp-day-head {
    display: flex; align-items: baseline; gap: 10px;
    flex-wrap: wrap; padding-bottom: 12px; margin-bottom: 4px;
    border-bottom: 1px solid var(--tp-line);
}
.tp-day-num { font-weight: 700; font-size: 1.05rem; }
.tp-day-date { font-size: 0.85rem; color: var(--tp-muted); }
.tp-day-theme {
    font-size: 0.82rem; color: var(--tp-accent); font-weight: 550;
    margin-left: auto;
}
.tp-timeline { position: relative; padding-top: 6px; }
.tp-stop {
    display: grid; grid-template-columns: 60px 16px 1fr;
    gap: 10px; align-items: start; padding: 7px 0;
}
.tp-stop-time {
    font-size: 0.8rem; font-weight: 650; text-align: right;
    display: flex; flex-direction: column; line-height: 1.35;
    font-variant-numeric: tabular-nums;
}
.tp-stop-time .tp-dim { font-weight: 400; font-size: 0.74rem; }
.tp-stop-dot { position: relative; height: 100%; }
.tp-stop-dot::before {
    content: ""; position: absolute; left: 4px; top: 5px;
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--tp-accent); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.16);
}
.tp-stop-dot::after {
    content: ""; position: absolute; left: 8px; top: 16px; bottom: -14px;
    width: 1px; background: var(--tp-line);
}
.tp-stop:last-child .tp-stop-dot::after { display: none; }
.tp-stop-name { font-weight: 600; font-size: 0.95rem; }
.tp-travel {
    font-size: 0.76rem; color: var(--tp-muted);
    padding: 2px 0 2px 86px; font-style: italic;
}

/* research briefing */
.tp-brief { display: flex; gap: 13px; align-items: flex-start; }
.tp-brief-icon { font-size: 1.2rem; flex-shrink: 0; line-height: 1.4; }
.tp-brief-title { font-weight: 650; font-size: 0.93rem; margin-bottom: 3px; }
.tp-brief-body { font-size: 0.86rem; line-height: 1.55; opacity: 0.9; }
.tp-sources { display: flex; flex-direction: column; gap: 5px; }
.tp-source {
    font-size: 0.84rem; text-decoration: none; color: var(--tp-accent);
    padding: 3px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tp-source:hover { text-decoration: underline; }

/* critic issues */
.tp-issue { border-left: 3px solid var(--tp-line); }
.tp-issue .tp-badges { margin-top: 0; }
.tp-issue-critical { border-left-color: var(--tp-critical); }
.tp-issue-warning { border-left-color: var(--tp-warning); }
.tp-issue-info { border-left-color: var(--tp-s1); }
.tp-issue-body { font-size: 0.92rem; margin-top: 9px; line-height: 1.55; }
.tp-issue-fix {
    font-size: 0.85rem; margin-top: 7px; color: var(--tp-muted);
    padding: 8px 12px; border-radius: 9px; background: var(--tp-soft);
}
.tp-notes { margin: 0; padding-left: 20px; }
.tp-notes li { font-size: 0.89rem; margin: 6px 0; line-height: 1.55; }

/* calendar */
.tp-backend {
    display: flex; gap: 12px; align-items: center; margin-bottom: 18px;
    padding: 13px 16px; border-radius: 13px;
    border: 1px solid var(--tp-line); background: var(--tp-soft);
}
.tp-backend-icon { font-size: 1.3rem; }
.tp-backend-title { font-weight: 650; font-size: 0.93rem; }
.tp-events { display: flex; flex-direction: column; gap: 3px; }
.tp-event {
    display: flex; gap: 12px; align-items: flex-start; padding: 10px 4px;
    border-bottom: 1px solid var(--tp-line);
}
.tp-event:last-child { border-bottom: none; }
.tp-event-icon { font-size: 1rem; line-height: 1.35; }
.tp-event-title { font-weight: 600; font-size: 0.92rem; }

/* places */
.tp-place .tp-badges { margin-top: 8px; }

/* deep links out to booking and maps */
.tp-book {
    display: inline-block; margin-top: 10px; font-size: 0.83rem; font-weight: 550;
    text-decoration: none; color: var(--tp-accent);
    border: 1px solid var(--tp-line); border-radius: 9px; padding: 5px 12px;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.tp-book:hover {
    border-color: rgba(99, 102, 241, 0.55);
    background: rgba(99, 102, 241, 0.08);
}
.tp-book:focus-visible { outline: 2px solid var(--tp-accent); outline-offset: 2px; }

/* efficiency: one horizontal bar per agent, longest first */
.tp-bars { display: flex; flex-direction: column; gap: 4px; }
.tp-bar-row {
    display: grid; grid-template-columns: 120px 1fr 56px; gap: 10px;
    align-items: center; padding: 8px 2px;
    border-bottom: 1px solid var(--tp-line);
}
.tp-bar-row:last-child { border-bottom: none; }
.tp-bar-label { font-size: 0.85rem; font-weight: 600; }
.tp-bar-track {
    height: 10px; border-radius: 999px; overflow: hidden;
    background: rgba(42, 120, 214, 0.15);
}
.tp-bar-fill { display: block; height: 100%; border-radius: 999px; }
.tp-bar-failed { background: var(--tp-critical); }
.tp-bar-value {
    font-size: 0.83rem; font-weight: 650; text-align: right;
    font-variant-numeric: tabular-nums;
}
.tp-bar-meta {
    grid-column: 2 / -1; font-size: 0.76rem; color: var(--tp-muted); margin-top: -2px;
}

footer { display: none !important; }
"""

HEADER = """
<div id="tp-header">
  <h1>✈️ Autonomous Trip Planner</h1>
  <p>Eleven specialist agents research, book, route, cost and fact-check your
     trip — then put it straight on your calendar.</p>
  <div class="tp-tags">
    <span class="tp-tag">🕸 LangGraph</span>
    <span class="tp-tag">🧠 GPT-5 mini</span>
    <span class="tp-tag">🔌 MCP · Google Calendar</span>
    <span class="tp-tag">🌐 Live flight &amp; hotel data</span>
    <span class="tp-tag">⚡ Parallel agents</span>
  </div>
</div>
"""


def _sections(state: dict, backend: str, wall: float = 0.0) -> list[str]:
    """Render every result panel from the accumulated state.

    Args:
        state: The merged graph state so far.
        backend: Description of the active calendar backend.

    Returns:
        The Markdown for each tab, in the order the UI expects.
    """
    return [
        render_degraded(state.get("failed_agents", []))
        + render_itinerary(state.get("itinerary"), state.get("routing")),
        render_flights(state.get("flights")),
        render_lodging(state.get("lodging")),
        render_places(state.get("attractions")),
        render_budget(state.get("budget")),
        render_critic(state.get("critic")),
        render_calendar(state.get("calendar"), backend),
        render_profile(state.get("intake")),
        render_research(state.get("research")),
        render_efficiency(state.get("metrics", []), wall, describe_models()),
    ]


def plan(request: str):
    """Run the planner, yielding UI updates as each agent finishes.

    Args:
        request: The traveler's free-text request.

    Yields:
        The progress tracker, a status line, every result panel, and the
        downloadable .ics file once the Calendar Agent has run.
    """
    labels = {str(name): label for name, label, _ in PIPELINE}
    backend = describe_calendar_backend()
    state: dict = {}
    started = time.perf_counter()

    if not request.strip():
        yield (
            progress_html([]),
            "⚠️ Tell me where you want to go first.",
            *_sections({}, backend),
            gr.update(visible=False),
        )
        return

    yield (
        progress_html([], active="intake"),
        "🚀 Starting — the Travel Manager is deciding what to do first…",
        *_sections({}, backend),
        gr.update(visible=False),
    )

    try:
        stream = graph_app.stream(
            initial_state(request),
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        )

        for chunk in stream:
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue

                # Merge this node's partial update into our view of the state.
                for key, value in update.items():
                    if key == "completed_agents":
                        state.setdefault("completed_agents", [])
                        state["completed_agents"] += value
                    else:
                        state[key] = value

                completed = state.get("completed_agents", [])
                if node == "manager":
                    decision = state.get("manager_decision")
                    group = decision.next_agents if decision else []
                    if len(group) > 1:
                        names = ", ".join(labels.get(str(a), str(a)) for a in group)
                        status = f"🧭 Manager → **{names}** running in parallel ⚡"
                    elif group:
                        status = f"🧭 Manager → **{labels.get(str(group[0]), group[0])}**"
                    else:
                        status = "🧭 Manager: the plan is complete."
                    active = str(group[0]) if group else None
                else:
                    status = f"✅ {labels.get(node, node)} finished."
                    active = None

                if state.get("revision_count"):
                    status += f"  ·  🔄 revision {state['revision_count']}"

                yield (
                    progress_html(completed, active),
                    status,
                    *_sections(state, backend, time.perf_counter() - started),
                    gr.update(visible=False),
                )

    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
        message = str(exc)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            note = (
                "🛑 **Gemini quota exhausted.** A free-tier key allows 20 requests "
                "per day, and a full eleven-agent run needs more than that. "
                "Enable billing on the key, or try again tomorrow."
            )
        else:
            note = f"🛑 **The run failed:** {message[:400]}"

        yield (
            progress_html(state.get("completed_agents", [])),
            note,
            *_sections(state, backend, time.perf_counter() - started),
            gr.update(visible=False),
        )
        return

    # Offer the .ics for download when one was written locally.
    calendar = state.get("calendar")
    ics = getattr(calendar, "ics_path", None) if calendar else None
    intake = state.get("intake")

    if intake is not None and not intake.is_complete:
        final = "❓ I need a few more details before I can plan this trip."
    else:
        final = "🎉 **Your trip is ready.** " + (
            "Events were added to your Google Calendar."
            if calendar and not ics and calendar.events
            else "Download the calendar file below."
            if ics
            else ""
        )

    yield (
        progress_html(state.get("completed_agents", [])),
        final,
        *_sections(state, backend, time.perf_counter() - started),
        gr.update(value=ics, visible=bool(ics)),
    )


# Fonts must be Font objects, not bare strings: Gradio compares the theme
# against its built-ins at launch, and mixing the two breaks that check.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=(
        gr.themes.GoogleFont("Inter"),
        gr.themes.Font("Segoe UI"),
        gr.themes.Font("system-ui"),
        gr.themes.Font("sans-serif"),
    ),
)


def _stopped():
    """Report a cancelled run and restore the Plan button.

    Returns:
        The status line plus the two button states.
    """
    return (
        "■ **Stopped.** Whatever finished before you stopped is still shown below.",
        gr.update(visible=True),
        gr.update(visible=False),
    )


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface.

    The theme and CSS are applied at launch, which is where Gradio 6 expects
    them, so they are not passed to `Blocks` here.

    Returns:
        The Blocks app, ready to launch.
    """
    with gr.Blocks(title="Autonomous Trip Planner") as demo:
        gr.HTML(HEADER)

        with gr.Row():
            with gr.Column(scale=2):
                request = gr.Textbox(
                    label="Describe your trip",
                    placeholder=(
                        "Where to, when, how many of you, what budget, and what "
                        "you enjoy — one sentence is enough."
                    ),
                    lines=4,
                    autofocus=True,
                )
            with gr.Column(scale=1, min_width=190):
                plan_button = gr.Button(
                    "✨ Plan my trip", variant="primary", size="lg", scale=2
                )
                stop_button = gr.Button(
                    "■ Stop", variant="stop", size="lg", visible=False
                )
                gr.Markdown(
                    f"<div id='tp-status'>🧠 {describe_models()}<br>"
                    f"📅 {describe_calendar_backend()}</div>"
                )

        gr.Examples(examples=EXAMPLES, inputs=request, label="Try one of these")

        progress = gr.HTML(progress_html([]))
        status = gr.Markdown("", elem_id="tp-status")

        # Panels are gr.HTML, not gr.Markdown: each agent's result is rendered
        # as cards, stat tiles and meters, which Markdown cannot express.
        with gr.Tabs():
            with gr.Tab("📖 Itinerary"):
                itinerary = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("✈️ Flights"):
                flights = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("🏨 Stay"):
                lodging = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("📍 Places"):
                places = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("💰 Budget"):
                budget = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("🔍 Review"):
                critic = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("📅 Calendar"):
                calendar = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("📋 Your request"):
                profile = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("🌍 Destination"):
                research = gr.HTML(elem_classes="tp-panel")
            with gr.Tab("⚡ Efficiency"):
                efficiency = gr.HTML(elem_classes="tp-panel")

        ics_file = gr.File(label="📎 Add to your calendar (.ics)", visible=False)

        outputs = [
            progress,
            status,
            itinerary,
            flights,
            lodging,
            places,
            budget,
            critic,
            calendar,
            profile,
            research,
            efficiency,
            ics_file,
        ]
        # Swap the buttons around the run so Stop is only offered while a plan
        # is actually in flight.
        def _running():
            """Show Stop, hide Plan."""
            return gr.update(visible=False), gr.update(visible=True)

        def _idle():
            """Show Plan, hide Stop."""
            return gr.update(visible=True), gr.update(visible=False)

        buttons = [plan_button, stop_button]

        run_events = []
        for trigger in (plan_button.click, request.submit):
            event = trigger(_running, outputs=buttons, queue=False).then(
                plan, inputs=request, outputs=outputs
            )
            event.then(_idle, outputs=buttons, queue=False)
            run_events.append(event)

        # `cancels` stops the generator mid-plan; the partial results already
        # streamed into the panels stay on screen.
        stop_button.click(
            _stopped,
            outputs=[status, *buttons],
            cancels=run_events,
            queue=False,
        )

    return demo


def main() -> None:
    """Launch the trip planner UI in a browser."""
    build_ui().launch(theme=THEME, css=CSS, inbrowser=True)


if __name__ == "__main__":
    main()
