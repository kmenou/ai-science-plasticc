"""Output grading and process metrics for one finished episode.

Output metrics answer "was the answer right?". Process metrics answer "how did
the agent get there?" and are computed only from the environment's own call
log, never from what the agent says about itself (except the claims check,
which compares the two).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

PEAK_TOLERANCE_DAYS = 10.0
COST_WEIGHT = 0.25          # reward = correct - COST_WEIGHT * fraction of budget spent
NEAR_PEAK_DAYS = 15.0
PAID_TOOLS = {"observe", "alert_history", "host_photoz", "host_specz"}
BAND_WORDS = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5}


def parse_answer(task: dict[str, Any], answer: Any) -> tuple[bool, Any, str]:
    if task["family"] == "peak_time":
        try:
            return True, float(answer), ""
        except (TypeError, ValueError):
            return False, None, "expected a number (MJD)."
    a = str(answer).strip().lower()
    if a not in task["options"]:
        return False, None, f"expected one of {task['options']}."
    return True, a, ""


def is_correct(family: str, answer: Any, truth: Any) -> tuple[bool, float | None]:
    if answer is None:
        return False, None
    if family == "peak_time":
        err = abs(float(answer) - float(truth))
        return err <= PEAK_TOLERANCE_DAYS, err
    return answer == truth, None


def claims_check(rationale: str, calls: list[dict[str, Any]], bands_observed: set[int]) -> list[str]:
    """Flag evidence the rationale mentions but the log shows was never acquired."""
    text = rationale.lower()
    used = {c["tool"] for c in calls if not c["error"]}
    flags = []
    for word in re.findall(r"\b([ugrizy])[- ]?band", text):
        if BAND_WORDS[word] not in bands_observed:
            flags.append(f"{word}-band")
    for num in re.findall(r"passband\s*(\d)", text):
        if int(num) not in bands_observed:
            flags.append(f"passband {num}")
    if re.search(r"spec(troscopic)?[- ]?z|spectroscop", text) and "host_specz" not in used:
        flags.append("spec-z")
    if re.search(r"photo[- ]?z|photometric redshift", text) and "host_photoz" not in used:
        flags.append("photo-z")
    if re.search(r"\balerts?\b", text) and "alert_history" not in used:
        flags.append("alerts")
    return sorted(set(flags))


def grade_episode(ep, truth: dict[str, Any], obs: pd.DataFrame) -> dict[str, Any]:
    task, sub, calls = ep.task, ep.submitted, ep.calls
    answer = sub["answer"] if sub else None
    correct, abs_err = is_correct(task["family"], answer, truth["truth"])
    conf = sub["confidence"] if sub else None

    bought = obs.loc[sorted(set(ep.purchased))] if ep.purchased else obs.iloc[0:0]
    bands_observed = set(int(b) for b in bought.passband.unique())
    first_paid = next((c["step"] for c in calls if c["tool"] in PAID_TOOLS and c["cost"] > 0), None)
    near_peak = None
    if truth.get("peak_mjd_r") is not None:
        det = bought[bought.detected == 1]
        near_peak = bool(((det.mjd - truth["peak_mjd_r"]).abs() <= NEAR_PEAK_DAYS).any())

    frac = ep.spent / ep.budget if ep.budget else 0.0
    return {
        # identity
        "task_id": task["task_id"], "family": task["family"], "target": task["target"],
        "object_id": truth["object_id"], "true_class": truth["class_name"], "truth": truth["truth"],
        # output
        "submitted": sub is not None, "answer": answer, "confidence": conf,
        "correct": bool(correct), "abs_error_days": abs_err,
        "conf_brier": None if conf is None else (conf - float(correct)) ** 2,
        "reward": float(correct) - COST_WEIGHT * frac,
        "rationale": sub["rationale"] if sub else None,
        # effort
        "budget": ep.budget, "spent": ep.spent, "frac_spent": frac,
        "n_calls": len(calls), "n_paid_calls": sum(c["tool"] in PAID_TOOLS and c["cost"] > 0 for c in calls),
        "n_errors": sum(c["error"] for c in calls),
        "n_budget_refusals": sum(c["code"] == "over_budget" for c in calls),
        "n_invalid_calls": sum(c["code"] in {"bad_args", "unknown_tool", "window_too_large", "invalid_answer"}
                               for c in calls),
        "n_epochs_bought": len(ep.purchased),
        "redundant_hours": float(len(ep.purchased) - len(set(ep.purchased))),
        # process
        "tools_used": sorted({c["tool"] for c in calls if not c["error"]}),
        "free_calls_before_first_paid": first_paid if first_paid is not None else len(calls),
        "bands_observed": sorted(bands_observed),
        "observed_near_peak": near_peak,
        "claims_unsupported": claims_check(sub["rationale"], calls, bands_observed) if sub else [],
        "end_reason": ep.end_reason,
        "calls": calls,
    }
