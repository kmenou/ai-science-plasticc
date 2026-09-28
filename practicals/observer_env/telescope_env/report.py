"""Aggregate graded episodes from one or more runs into CSV tables and a standalone HTML report.

    python -m telescope_env.report runs/*                      # all runs
    python -m telescope_env.report runs/a runs/b --metrics student/my_metrics.py --out report.html

--metrics points to a Python file defining ``episode_metrics(rec: dict) -> dict[str, float]``.
Its values are averaged per run and added to the summary table: the students' process graders.
"""

from __future__ import annotations

import argparse
import base64
import html
import importlib.util
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load(run_dirs: list[Path]) -> pd.DataFrame:
    frames = [pd.read_json(d / "episodes.jsonl", lines=True) for d in run_dirs if (d / "episodes.jsonl").exists()]
    if not frames:
        raise SystemExit("no episodes.jsonl found")
    return pd.concat(frames, ignore_index=True)


def add_student_metrics(df: pd.DataFrame, path: Path | None) -> list[str]:
    if not path:
        return []
    spec = importlib.util.spec_from_file_location("student_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    extra = pd.DataFrame([mod.episode_metrics(r) for r in df.to_dict("records")], index=df.index)
    for c in extra:
        df[c] = extra[c].astype(float)
    return list(extra.columns)


def summarize(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run", sort=False):
        by_task = g.groupby("task_id").correct
        agent = g["agent"] if "agent" in g else pd.Series([None] * len(g), index=g.index)
        agent = agent.map(lambda a: a if isinstance(a, dict) else {})
        tokens = agent.map(lambda a: sum((a.get("usage") or {}).get(k, 0) or 0 for k in
                                         ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                          "cache_creation_input_tokens")))
        row = {
            "run": run, "episodes": len(g), "k": int(g.rep.max() + 1),
            "accuracy": g.correct.mean(), "pass@k": by_task.any().mean(), "pass^k": by_task.all().mean(),
            "reward": g.reward.mean(), "hours": g.spent.mean(), "frac_budget": g.frac_spent.mean(),
            "submitted": g.submitted.mean(), "confidence": g.confidence.mean(), "conf_brier": g.conf_brier.mean(),
            "calls": g.n_calls.mean(), "invalid_calls": g.n_invalid_calls.mean(),
            "budget_refusals": g.n_budget_refusals.mean(), "redundant_h": g.redundant_hours.mean(),
            "free_calls_first": g.free_calls_before_first_paid.mean(),
            "near_peak(peak_time)": g.loc[g.family == "peak_time", "observed_near_peak"].astype(float).mean(),
            "unsupported_claims": g.claims_unsupported.map(len).gt(0).mean(),
        }
        for fam, gf in g.groupby("family"):
            row[f"acc:{fam}"] = gf.correct.mean()
        if agent.map(bool).any():
            row["turns"] = agent.map(lambda a: a.get("num_turns")).astype(float).mean()
            row["wall_s"] = agent.map(lambda a: a.get("wall_s")).astype(float).mean()
            row["ktokens"] = tokens.mean() / 1e3
            row["sdk_errors"] = int(agent.map(lambda a: bool(a.get("sdk_error"))).sum())
        for c in extra:
            row[c] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def figures(df: pd.DataFrame) -> list[tuple[str, str]]:
    runs = list(dict.fromkeys(df.run))
    col = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(runs)}
    out = []

    fig, ax = plt.subplots(figsize=(7, 4.2))
    marks = {"coarse_class": "o", "is_snia": "s", "peak_time": "^"}
    for r in runs:
        g = df[df.run == r]
        ax.scatter(g.spent.mean(), g.correct.mean(), s=140, color=col[r], edgecolor="k", zorder=3, label=r)
        for fam, gf in g.groupby("family"):
            ax.scatter(gf.spent.mean(), gf.correct.mean(), s=40, marker=marks.get(fam, "x"),
                       color=col[r], alpha=0.6, zorder=2)
    for fam, m in marks.items():
        ax.scatter([], [], marker=m, color="grey", label=fam)
    ax.set(xlabel="mean telescope hours spent per episode", ylabel="accuracy", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    out.append(("Accuracy vs cost", _png(fig) + "<p>Large markers show all tasks; small markers show one task family. "
                "Up and to the left is better.</p>"))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 6)
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    for r in runs:
        g = df[(df.run == r) & df.confidence.notna()]
        if g.empty:
            continue
        idx = np.clip(np.digitize(g.confidence, bins) - 1, 0, len(bins) - 2)
        cal = g.groupby(idx).agg(c=("confidence", "mean"), a=("correct", "mean"), n=("correct", "size"))
        ax.plot(cal.c, cal.a, "-o", color=col[r], label=r, ms=4)
    ax.set(xlabel="stated confidence", ylabel="fraction correct", xlim=(0, 1), ylim=(-0.03, 1.03))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    out.append(("Calibration", _png(fig) + "<p>A well-calibrated agent lies on the dashed line.</p>"))

    tools = ["sky_position", "observing_log", "budget", "alert_history", "host_photoz", "host_specz", "observe", "submit"]
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(runs)))
    mat = np.array([[np.mean([sum(c["tool"] == t for c in calls) for calls in df[df.run == r].calls]) for t in tools]
                    for r in runs])
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(tools)), tools, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(runs)), runs, fontsize=7)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="k" if v < mat.max() * 0.6 else "w")
    out.append(("Tool usage (mean calls per episode)", _png(fig)))
    return out


def page(summary: pd.DataFrame, figs: list[tuple[str, str]]) -> str:
    fmt = summary.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    table = fmt.to_html(index=False, escape=True, border=0, classes="t")
    body = "".join(f"<h2>{html.escape(t)}</h2>{f}" for t, f in figs)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Telescope-time eval</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
.t{{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}}
.t td,.t th{{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}
.t td:first-child,.t th:first-child{{text-align:left}} img{{max-width:100%}}</style></head><body>
<h1>Telescope-time agentic evaluation</h1>
<p>pass@k is the fraction of tasks solved in at least one of k repeats; pass^k is the fraction solved in all
k repeats. reward = correct &minus; 0.25 &times; fraction of budget spent. conf_brier = mean (confidence &minus;
correct)<sup>2</sup>. unsupported_claims is the fraction of rationales that cite evidence the log shows was never
acquired.</p>{table}{body}</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("report.html"))
    a = ap.parse_args()
    df = load(a.runs)
    extra = add_student_metrics(df, a.metrics)
    summary = summarize(df, extra)
    summary.to_csv(a.out.with_suffix(".csv"), index=False)
    a.out.write_text(page(summary, figures(df)))
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(summary[["run", "episodes", "accuracy", "pass@k", "pass^k", "hours", "reward", "conf_brier"]]
              .round(2).to_string(index=False))
    print(f"-> {a.out} and {a.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
