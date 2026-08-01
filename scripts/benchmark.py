"""Measure what one plan costs in time, tokens and money.

Every optimisation in TODOLIST.md is supposed to be a measured delta, not an
opinion. This runs a fixed request end to end and writes a JSON report, so the
effect of a change is the difference between two reports.

Usage:
    uv run scripts/benchmark.py
    uv run scripts/benchmark.py --request "4 days in Rome, 2 people, 2000 EUR"
    uv run scripts/benchmark.py --label after-parallel-fanout
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from trip_planner.graph import plan_trip
from trip_planner.llm import describe_models
from trip_planner.metrics import summarize
from trip_planner.tools.cache import stats as cache_stats

REPORT_DIR = Path(__file__).resolve().parent.parent / "benchmarks"

DEFAULT_REQUEST = (
    "I want to travel to Lisbon, Portugal from 2026-09-10 to 2026-09-15. "
    "We are 2 travelers flying from Tel Aviv, our total budget is 3000 USD, "
    "and we love food, history and walking tours."
)


def main() -> None:
    """Run one plan and write a benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default=DEFAULT_REQUEST, help="Trip request to plan.")
    parser.add_argument("--label", default="", help="Name for this report.")
    args = parser.parse_args()

    print(f"Model: {describe_models()}")
    print(f"Cache: {cache_stats()}")
    print(f"Request: {args.request}\n")

    started = time.perf_counter()
    final = plan_trip(args.request)
    wall = time.perf_counter() - started

    summary = summarize(final.get("metrics", []), wall_seconds=wall)

    print(f"{'agent':<24}{'secs':>8}{'llm':>6}{'tools':>7}{'tokens':>10}{'cost':>10}")
    print("-" * 65)
    for record in summary.agents:
        cost = f"${record.cost_usd:.4f}" if record.cost_usd is not None else "—"
        flag = "  FAILED" if record.failed else ""
        print(
            f"{record.agent:<24}{record.seconds:>8.1f}{record.llm_calls:>6}"
            f"{record.tool_calls:>7}{record.total_tokens:>10,}{cost:>10}{flag}"
        )
    print("-" * 65)
    total_cost = f"${summary.cost_usd:.4f}" if summary.cost_usd is not None else "—"
    print(
        f"{'TOTAL':<24}{summary.seconds:>8.1f}{summary.llm_calls:>6}"
        f"{summary.tool_calls:>7}{summary.input_tokens + summary.output_tokens:>10,}"
        f"{total_cost:>10}"
    )
    print(f"\nWall clock:      {summary.wall_seconds:.1f}s")
    print(f"Saved by parallelism: {summary.parallel_saving:.1f}s")

    failed = final.get("failed_agents", [])
    if failed:
        print(f"\nDegraded — {len(failed)} agent(s) failed:")
        for entry in failed:
            print(f"  - {entry['agent']}: {entry['error'][:120]}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = args.label or "run"
    path = REPORT_DIR / f"{stamp}-{label}.json"
    path.write_text(
        json.dumps(
            {
                "label": label,
                "timestamp": stamp,
                "model": describe_models(),
                "request": args.request,
                "summary": summary.model_dump(mode="json"),
                "completed_agents": [str(a) for a in final.get("completed_agents", [])],
                "failed_agents": failed,
                "revisions": final.get("revision_count", 0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport: {path}")


if __name__ == "__main__":
    main()
