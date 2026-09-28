"""Student process graders. Each function of one graded episode record returns numbers,
which report.py averages per run (python -m telescope_env.report runs/* --metrics student/my_metrics.py).

Useful fields in rec: family, correct, confidence, spent, budget, calls (list of dicts with
step, tool, args, cost, error, code, remaining), bands_observed, tools_used, rationale, agent (Claude runs).
"""


def episode_metrics(rec: dict) -> dict[str, float]:
    calls = rec["calls"]
    paid = [c for c in calls if c["cost"] > 0]
    return {
        # Example 1: did the agent plan (use a free tool) before spending anything?
        "planned_first": float(bool(calls) and calls[0]["cost"] == 0),
        # Example 2: how much of the spend went into observe?
        "frac_spend_on_photometry": (sum(c["cost"] for c in paid if c["tool"] == "observe") / rec["spent"])
        if rec["spent"] else 0.0,
        # TODO (students): e.g. colour use (>= 2 bands observed), overconfidence (confidence > 0.8 and wrong),
        # "stopped early" (submitted with more than half the budget left and was right)...
    }
