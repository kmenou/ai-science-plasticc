"""Telescope-time environment: a gym-style episode over one PLAsTiCC object.

The agent starts knowing almost nothing about a target and must *buy* data
(photometry, host redshifts, alert history) from a fixed budget of telescope
hours before submitting an answer. The environment is pure Python: it has no
LLM dependency, so scripted policies and Claude drive it through the same
``call(tool, **args)`` interface.

Episode lifecycle::

    env = TelescopeEnv(catalog)
    first_message = env.reset(task)        # task: dict from tasks/*.jsonl
    result = env.call("observe", passband=2, mjd_start=60100, mjd_end=60160)
    ...
    env.call("submit", answer="supernova", confidence=0.7, rationale="...")
    record = env.finalize()                # graded episode record (dict)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .catalog import Catalog
from . import scoring

BANDS = "ugrizy"

# Telescope-hour prices. Free tools are "planning" tools.
COSTS = {
    "observe_per_epoch": 1.0,
    "observe_min": 1.0,
    "alert_history": 2.0,
    "host_photoz": 3.0,
    "host_specz": 10.0,
}
MAX_EPOCHS_PER_OBSERVE = 30
DEFAULT_BUDGET = 30.0
DEFAULT_MAX_CALLS = 30


@dataclass
class ToolResult:
    text: str                      # what the agent sees
    cost: float = 0.0
    error: bool = False
    done: bool = False
    code: str | None = None        # machine-readable error kind, logged for process metrics

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "cost": self.cost, "error": self.error, "done": self.done, "code": self.code}


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[..., ToolResult]


@dataclass
class Episode:
    task: dict[str, Any]
    budget: float
    max_calls: int
    spent: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)
    purchased: list[int] = field(default_factory=list)       # observation row ids, with repeats
    submitted: dict[str, Any] | None = None
    done: bool = False
    end_reason: str | None = None
    t_start: float = field(default_factory=time.time)

    @property
    def remaining(self) -> float:
        return self.budget - self.spent


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props or {}, "required": required or [],
            "additionalProperties": False}


class TelescopeEnv:
    """One environment instance runs one episode at a time; create one per concurrent agent."""

    def __init__(self, catalog: Catalog, budget: float | None = None, max_calls: int = DEFAULT_MAX_CALLS):
        self.catalog = catalog
        self.budget_override = budget
        self.max_calls = max_calls
        self.ep: Episode | None = None
        self._obj: pd.DataFrame | None = None
        self._meta: dict[str, Any] | None = None
        self._tools = self._build_tools()

    # ------------------------------------------------------------------ API
    def reset(self, task: dict[str, Any]) -> str:
        budget = float(self.budget_override or task.get("budget", DEFAULT_BUDGET))
        self.ep = Episode(task=task, budget=budget, max_calls=self.max_calls)
        oid = self.catalog.object_id_for(task["target"])
        self._obj = self.catalog.observations(oid)
        self._meta = self.catalog.metadata(oid)
        return self.briefing()

    def briefing(self) -> str:
        t = self.ep.task
        lines = [
            f"Target: {t['target']}",
            f"Question: {t['question']}",
            f"Answer format: {t['answer_format']}",
            f"Budget: {self.ep.budget:g} telescope hours (max {self.ep.max_calls} tool calls).",
            "Prices: observe = 1 h per epoch returned (min 1 h, at most "
            f"{MAX_EPOCHS_PER_OBSERVE} epochs per call); alert_history = 2 h; "
            "host_photoz = 3 h; host_specz = 10 h; everything else is free.",
            "Passbands: 0=u 1=g 2=r 3=i 4=z 5=y. Fluxes are calibrated and can be negative.",
            "Finish by calling submit exactly once.",
        ]
        return "\n".join(lines)

    def tool_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def call(self, name: str, **args: Any) -> ToolResult:
        ep = self._require_episode()
        if ep.done:
            res = ToolResult("Episode is over; no further calls are accepted.", error=True, done=True,
                             code="episode_over")
            self._log(name, args, res)
            return res
        if name not in self._tools:
            res = ToolResult(f"Unknown tool '{name}'. Available: {', '.join(self._tools)}.", error=True,
                             code="unknown_tool")
        else:
            try:
                res = self._tools[name].fn(**args)
            except TypeError as exc:
                res = ToolResult(f"Bad arguments for {name}: {exc}", error=True, code="bad_args")
        ep.spent += res.cost
        self._log(name, args, res)
        if not ep.done and len(ep.calls) >= ep.max_calls:
            ep.done, ep.end_reason = True, "max_calls"
            res.text += f"\n[Environment] Call limit ({ep.max_calls}) reached; episode ended without submission."
            res.done = True
        return res

    def finalize(self) -> dict[str, Any]:
        ep = self._require_episode()
        if not ep.done:
            ep.done, ep.end_reason = True, "agent_stopped"
        truth = self.catalog.answer(ep.task["task_id"])
        return scoring.grade_episode(ep, truth, self._obj)

    # ------------------------------------------------------------ internals
    def _require_episode(self) -> Episode:
        if self.ep is None:
            raise RuntimeError("call reset(task) first")
        return self.ep

    def _log(self, name: str, args: dict[str, Any], res: ToolResult) -> None:
        ep = self.ep
        ep.calls.append({
            "step": len(ep.calls), "t": round(time.time() - ep.t_start, 3), "tool": name,
            "args": args, "cost": res.cost, "error": res.error,
            "remaining": round(ep.remaining, 3), "code": res.code, "result_chars": len(res.text),
        })

    def _charge(self, cost: float) -> ToolResult | None:
        if cost > self.ep.remaining + 1e-9:
            return ToolResult(f"Refused: costs {cost:g} h but only {self.ep.remaining:g} h remain. "
                              "Nothing was charged.", error=True, code="over_budget")
        return None

    def _build_tools(self) -> dict[str, ToolSpec]:
        specs = [
            ToolSpec("budget", "Remaining telescope hours and tool calls. Free.", _schema(), self._t_budget),
            ToolSpec("sky_position", "Target position: ra, decl, Galactic l and b, Milky Way extinction "
                     "E(B-V), and whether it lies in a Deep Drilling Field. Free.", _schema(), self._t_sky),
            ToolSpec("observing_log", "Observing seasons for the target: MJD range and number of scheduled "
                     "epochs per passband in each season. No fluxes. Free.", _schema(), self._t_log),
            ToolSpec("alert_history", "Times of significant detections (alerts), grouped into clusters, "
                     "with the passbands involved. No fluxes. Costs 2 h.", _schema(), self._t_alerts),
            ToolSpec("observe", "Photometry of the target in one passband within [mjd_start, mjd_end]: "
                     "mjd, flux, flux_err, detected. Costs 1 h per epoch returned (minimum 1 h). "
                     f"Windows with more than {MAX_EPOCHS_PER_OBSERVE} epochs are refused at no cost.",
                     _schema({"passband": {"type": "integer", "minimum": 0, "maximum": 5},
                              "mjd_start": {"type": "number"}, "mjd_end": {"type": "number"}},
                             ["passband", "mjd_start", "mjd_end"]), self._t_observe),
            ToolSpec("host_photoz", "Photometric redshift of the host galaxy (with error). A value of 0 "
                     "means no extragalactic host was found. Costs 3 h.", _schema(), self._t_photoz),
            ToolSpec("host_specz", "Spectroscopic redshift of the host galaxy. Costs 10 h.",
                     _schema(), self._t_specz),
            ToolSpec("submit", "Submit the final answer and end the episode. confidence is your probability "
                     "(0-1) that the answer is correct; rationale is 1-3 sentences on the evidence used.",
                     _schema({"answer": {"type": "string"}, "confidence": {"type": "number"},
                              "rationale": {"type": "string"}}, ["answer", "confidence", "rationale"]),
                     self._t_submit),
        ]
        return {s.name: s for s in specs}

    # ---------------------------------------------------------------- tools
    def _t_budget(self) -> ToolResult:
        ep = self.ep
        return ToolResult(f"Remaining: {ep.remaining:g} of {ep.budget:g} h; "
                          f"calls used {len(ep.calls)} of {ep.max_calls}.")

    def _t_sky(self) -> ToolResult:
        m = self._meta
        return ToolResult(json.dumps({
            "ra": round(m["ra"], 4), "decl": round(m["decl"], 4), "gal_l": round(m["gal_l"], 3),
            "gal_b": round(m["gal_b"], 3), "mwebv": round(m["mwebv"], 3), "deep_field": bool(m["ddf"])}))

    def _t_log(self) -> ToolResult:
        lines = []
        for i, s in enumerate(_seasons(self._obj), 1):
            counts = s.groupby("passband").size()
            per = " ".join(f"{BANDS[b]}:{int(counts.get(b, 0))}" for b in range(6))
            lines.append(f"season {i}: MJD {s.mjd.min():.1f}-{s.mjd.max():.1f}; epochs {per}")
        return ToolResult("\n".join(lines))

    def _t_alerts(self) -> ToolResult:
        if (r := self._charge(COSTS["alert_history"])):
            return r
        det = self._obj[self._obj.detected == 1].sort_values("mjd")
        if det.empty:
            return ToolResult("No alerts: the target was never significantly detected.",
                              cost=COSTS["alert_history"])
        lines = [f"{len(det)} detections in total."]
        for i, c in enumerate(_clusters(det, gap=30.0), 1):
            bands = "".join(BANDS[b] for b in sorted(c.passband.unique()))
            lines.append(f"cluster {i}: MJD {c.mjd.min():.1f}-{c.mjd.max():.1f}, "
                         f"{len(c)} detections, bands {bands}")
        if len(lines) > 16:
            lines = lines[:15] + [f"... {len(lines) - 15} more clusters"]
        return ToolResult("\n".join(lines), cost=COSTS["alert_history"])

    def _t_observe(self, passband: int, mjd_start: float, mjd_end: float) -> ToolResult:
        passband, mjd_start, mjd_end = int(passband), float(mjd_start), float(mjd_end)
        if not 0 <= passband <= 5 or mjd_end < mjd_start:
            return ToolResult("Refused: passband must be 0-5 and mjd_end >= mjd_start. Nothing was charged.",
                              error=True, code="bad_args")
        o = self._obj
        sel = o[(o.passband == passband) & (o.mjd >= mjd_start) & (o.mjd <= mjd_end)]
        if len(sel) > MAX_EPOCHS_PER_OBSERVE:
            return ToolResult(f"Refused: window contains {len(sel)} epochs (> {MAX_EPOCHS_PER_OBSERVE}). "
                              "Narrow it. Nothing was charged.", error=True,
                              code="window_too_large")
        cost = max(COSTS["observe_min"], COSTS["observe_per_epoch"] * len(sel))
        if (r := self._charge(cost)):
            return r
        self.ep.purchased.extend(int(i) for i in sel.index)
        if sel.empty:
            return ToolResult(f"No scheduled epochs in {BANDS[passband]} between MJD {mjd_start:.1f} and "
                              f"{mjd_end:.1f}. Charged {cost:g} h.", cost=cost)
        rows = ["mjd,flux,flux_err,detected"] + [
            f"{r.mjd:.3f},{r.flux:.2f},{r.flux_err:.2f},{int(r.detected)}" for r in sel.itertuples()]
        return ToolResult(f"{BANDS[passband]}-band, {len(sel)} epochs, charged {cost:g} h\n" + "\n".join(rows),
                          cost=cost)

    def _t_photoz(self) -> ToolResult:
        if (r := self._charge(COSTS["host_photoz"])):
            return r
        m = self._meta
        if m["hostgal_photoz"] == 0:
            txt = "photo-z = 0: no extragalactic host galaxy found."
        else:
            txt = f"photo-z = {m['hostgal_photoz']:.3f} +/- {m['hostgal_photoz_err']:.3f}"
        return ToolResult(txt, cost=COSTS["host_photoz"])

    def _t_specz(self) -> ToolResult:
        if (r := self._charge(COSTS["host_specz"])):
            return r
        z = self._meta["hostgal_specz"]
        txt = ("spec-z unavailable." if pd.isna(z) else
               "spec-z = 0: no extragalactic host galaxy." if z == 0 else f"spec-z = {z:.4f}")
        return ToolResult(txt, cost=COSTS["host_specz"])

    def _t_submit(self, answer: Any, confidence: Any, rationale: str = "") -> ToolResult:
        ok, parsed, msg = scoring.parse_answer(self.ep.task, answer)
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = float("nan")
        if not ok:
            return ToolResult(f"Invalid answer: {msg} Not recorded; you may submit again.", error=True,
                              code="invalid_answer")
        if not 0 <= conf <= 1:
            return ToolResult("Invalid confidence: must be a number in [0, 1]. Not recorded.", error=True,
                              code="invalid_answer")
        self.ep.submitted = {"answer": parsed, "raw_answer": answer, "confidence": conf,
                             "rationale": str(rationale)}
        self.ep.done, self.ep.end_reason = True, "submitted"
        return ToolResult("Answer recorded. Episode over.", done=True)


def _seasons(obs: pd.DataFrame, gap: float = 100.0) -> list[pd.DataFrame]:
    return _clusters(obs.sort_values("mjd"), gap)


def _clusters(df: pd.DataFrame, gap: float) -> list[pd.DataFrame]:
    if df.empty:
        return []
    breaks = np.flatnonzero(np.diff(df.mjd.to_numpy()) > gap) + 1
    return [df.iloc[a:b] for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(df)])]
