"""Diagrams for the student guide. Run from practicals/observer_env/: python docs/figures.py"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/figures")
RUNS = Path("runs")
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 10})

INK = "#1f2a37"
MUTED = "#6b7280"
BLUE, ORANGE, GREEN, AMBER, PINK, PURPLE, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"
FILL = {BLUE: "#e7f0fb", ORANGE: "#fdefe7", GREEN: "#e4f6ef", AMBER: "#fdf4dc", PURPLE: "#ecebf7",
        RED: "#fce8e9", MUTED: "#f3f4f6"}


def box(ax, x, y, w, h, title, body="", color=BLUE, title_size=11, body_size=8.8, mono=False, dashed=False,
        align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=FILL.get(color, "white"), ec=color, lw=1.6, ls="--" if dashed else "-"))
    ty = y + h - 0.28 if body else y + h / 2
    tx = x + w / 2 if align == "center" else x + 0.18
    ax.text(tx, ty, title, ha=align, va="center" if not body else "top", fontsize=title_size,
            fontweight="bold", color=INK)
    if body:
        ax.text(tx, ty - 0.52, body, ha=align, va="top", fontsize=body_size, color=INK, linespacing=1.45,
                family="Menlo" if mono else None)


def region(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                fc="none", ec=color, lw=1.2, ls=(0, (4, 3))))
    ax.text(x + 0.2, y + h - 0.12, label, ha="left", va="top", fontsize=9.5, color=color, fontweight="bold")


def arrow(ax, p, q, label="", color=INK, rad=0.0, lx=0.0, ly=0.0, size=8.6, both=False, ha="center",
          mono=False, lw=1.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=13, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, ha=ha, va="center", fontsize=size,
                color=color, linespacing=1.35, family="Menlo" if mono else None,
                bbox=dict(fc="white", ec="none", pad=1.5))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, w * 2)
    ax.set_ylim(0, h * 2)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1 --------------------------------------------------------------------------- the Gym loop
def gym_loop():
    fig, ax = canvas(7.2, 3.5)
    box(ax, 0.3, 1.5, 4.6, 3.7, "Agent", "Claude (via the Agent SDK)\nor a Python script\n\n"
        "reads the latest observation,\ndecides the next action", ORANGE)
    box(ax, 9.5, 1.5, 4.6, 3.7, "Environment", "TelescopeEnv  (env.py)\n\nhidden state:\n"
        "budget left, data bought,\ncall log, the true object", BLUE)
    arrow(ax, (5.0, 4.3), (9.4, 4.3), "action = a tool call\nobserve(passband=2, 60100, 60160)", ORANGE, ly=0.62)
    arrow(ax, (9.4, 2.4), (5.0, 2.4), "observation = the tool's text result\n\"r-band, 12 epochs, charged 12 h\"",
          BLUE, ly=-0.62)
    ax.text(7.2, 3.35, "repeat until done", ha="center", va="center", fontsize=9, color=MUTED, style="italic")
    box(ax, 0.3, 5.9, 4.6, 0.8, "env.reset(task)  ->  briefing", color=MUTED, title_size=9.5)
    arrow(ax, (2.6, 5.9), (2.6, 5.25), color=MUTED)
    box(ax, 8.6, 0.1, 5.5, 0.8, "submit  ->  done  ->  env.finalize()", color=GREEN, title_size=9.5)
    arrow(ax, (11.8, 1.48), (11.8, 0.93), color=GREEN)
    ax.text(8.4, 0.5, "graded episode record", ha="right", va="center", fontsize=8.8, color=GREEN)
    save(fig, "fig1_gym_loop.png")


# 2 --------------------------------------------------------------------------- architecture
def architecture():
    fig, ax = canvas(7.4, 5.0)
    region(ax, 0.1, 0.1, 7.0, 9.1, "Your Python process  (run.py)", BLUE)
    region(ax, 8.4, 3.6, 6.3, 5.6, "Claude Code subprocess  (query())", ORANGE)
    ax.text(7.4, 9.65, "The agent reaches the data only through the tool server.", ha="center", va="center",
            fontsize=9.5, color=INK, style="italic")

    box(ax, 0.4, 5.1, 3.0, 3.3, "TelescopeEnv", "one per episode\nloads the task\n\nbudget left\n"
        "data bought\ncall log", BLUE, body_size=8.4)
    box(ax, 3.8, 5.1, 3.0, 3.3, "Tool server", "\"telescope\" (MCP)\n\n8 tools, e.g.\nobserve, host_photoz,\n"
        "submit; each one\ncalls env.call()", BLUE, body_size=8.4)
    box(ax, 0.4, 0.4, 3.0, 3.9, "Grader", "env.finalize()\n\nscores output\nand process\nfrom the call log",
        GREEN, body_size=8.4)
    box(ax, 3.8, 2.6, 3.0, 1.9, "Answer key", "*_answers.jsonl\nread by grader only", RED, body_size=8.2)
    box(ax, 3.8, 0.4, 3.0, 1.85, "Outputs", "episodes.jsonl\ntranscripts/", GREEN, body_size=8.2, title_size=10)

    box(ax, 8.8, 5.1, 5.5, 3.3, "Agent loop", "model turn  ->  tool call?\n->  run tool, return result text\n"
        "->  repeat until submit\n\nsystem prompt + briefing to start", ORANGE, body_size=8.4)
    box(ax, 8.8, 3.9, 4.1, 0.8, "built-in Bash, Read, Write ...: off", color=MUTED, title_size=8.2, dashed=True)
    box(ax, 9.3, 0.4, 5.2, 2.2, "Claude model", "Haiku / Sonnet / Opus\nruns on Anthropic servers,\n"
        "uses your Claude login", PURPLE, body_size=8.4)

    arrow(ax, (3.8, 6.4), (3.45, 6.4), "", BLUE)
    arrow(ax, (1.9, 5.1), (1.9, 4.35), "finalize", GREEN, lx=0.55, ha="left")
    arrow(ax, (3.8, 3.5), (3.45, 3.5), "", RED)
    arrow(ax, (3.45, 1.15), (3.8, 1.15), "", GREEN)
    arrow(ax, (6.85, 7.7), (8.75, 7.7), "query()", ORANGE, ly=0.3, size=8.2)
    arrow(ax, (8.75, 6.1), (6.85, 6.1), "", ORANGE, both=True)
    ax.text(7.8, 5.75, "tool calls /\nresults", ha="center", va="top", fontsize=8.2, color=ORANGE)
    arrow(ax, (13.4, 5.08), (13.4, 2.65), "", PURPLE, both=True)
    ax.text(13.2, 3.15, "messages", ha="right", va="center", fontsize=8.3, color=PURPLE)
    save(fig, "fig2_architecture.png")


# 3 --------------------------------------------------------------------------- one episode, budget over time
def episode_timeline(task="dev-006"):
    def calls(run):
        for l in (RUNS / run / "episodes.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["task_id"] == task:
                return r
    haiku, heur = calls("pilot-haiku-dev"), calls("heuristic-dev")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    label = {"sky_position": "sky", "observing_log": "log", "budget": "budget", "alert_history": "alerts",
             "host_photoz": "photo-z", "observe": "obs", "submit": "submit"}
    for rec, color, name in [(haiku, ORANGE, "Claude Haiku  (answered SN Ia: wrong)"),
                             (heur, BLUE, "heuristic script  (answered not SN Ia: right)")]:
        xs, ys = [0], [rec["budget"]]
        for c in rec["calls"]:
            xs.append(c["step"] + 1)
            ys.append(c["remaining"])
        ax.step(xs, ys, where="post", color=color, lw=2.2, label=name)
        for c in rec["calls"]:
            x = c["step"] + 1
            t = label[c["tool"]]
            if c["tool"] == "observe":
                t = "obs " + "ugrizy"[c["args"]["passband"]]
            if c["error"]:
                t += "\nrefused"
            is_pz = c["tool"] == "host_photoz"
            ax.plot(x, c["remaining"], "o", ms=7 if is_pz else 4.5, color=RED if is_pz else color,
                    zorder=4, mec="white", mew=0.8)
            dy = (2.0 if "\n" in t else 1.3) if rec is haiku else -1.5
            tx, ty, ha = x + 0.5, c["remaining"] + dy, "center"
            if is_pz and rec is haiku:
                tx, ty, ha = x - 0.15, c["remaining"], "right"
            ax.text(tx, ty, t, ha=ha, va="center", fontsize=7.3,
                    color=RED if is_pz else color, fontweight="bold" if is_pz else None)
    ax.annotate("photo-z = 0  ->  Galactic!\nHaiku bought it only after\nspending 24 h on photometry",
                xy=(9, 1), xytext=(9.3, 12), fontsize=8.4, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1))
    ax.annotate("the script buys photo-z first\nand stops: 3 h in total", xy=(2, 26.5), xytext=(0.2, 14),
                fontsize=8.4, color=BLUE, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1))
    ax.set(xlabel="tool call number", ylabel="telescope hours left", ylim=(-3, 33), xlim=(-0.4, 12.4))
    ax.set_xticks(range(0, 12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", frameon=False, fontsize=8.6, bbox_to_anchor=(0.0, 0.0))
    ax.set_title(f"Episode {task}: is target T-073f a SN Ia?  (truly a microlensing event)", fontsize=10,
                 loc="left", color=INK)
    save(fig, "fig3_episode_dev006.png")


# 4 --------------------------------------------------------------------------- evaluation pipeline
def pipeline():
    fig, ax = canvas(8.2, 2.0)
    y, h = 1.05, 2.75
    boxes = [(0.2, 2.7, "Tasks", "12 dev or 30 eval\ntasks,  x k repeats", MUTED),
             (3.5, 3.0, "Episodes", "agent <-> env loop,\none per task\nand repeat", ORANGE),
             (7.1, 3.5, "Three records", "call log  (env)\ntranscript  (SDK)\nusage & turns  (SDK)", BLUE),
             (11.2, 2.6, "Grade", "each episode:\noutput + process", GREEN),
             (14.4, 1.8, "Report", "tables, plots,\nreplay", PURPLE)]
    for x, w, t, b, c in boxes:
        box(ax, x, y, w, h, t, b, c, body_size=8.4)
    for (x, w, *_), (x2, *__) in zip(boxes, boxes[1:]):
        arrow(ax, (x + w + 0.02, y + h / 2), (x2 - 0.02, y + h / 2))
    ax.text(8.2, 0.5, "accuracy  |  pass@k, pass^k  |  hours spent  |  reward  |  calibration  |  "
            "wasted hours  |  unsupported claims", ha="center", va="center", fontsize=8.6, color=INK)
    save(fig, "fig4_pipeline.png")


# 5 --------------------------------------------------------------------------- pilot results
def results():
    import pandas as pd
    runs = [("random-dev", "random", MUTED), ("heuristic-dev", "heuristic script", BLUE),
            ("pilot-haiku-dev", "Claude Haiku", ORANGE), ("answer_key-dev", "cheater (reads answers)", RED)]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    import numpy as np
    h = np.linspace(0, 30, 50)
    for r in [0.2, 0.4, 0.6, 0.8]:
        ax.plot(h, r + 0.25 * h / 30, color="#d1d5db", lw=0.9, ls="--", zorder=0)
        ax.text(30.4, r + 0.25, f"reward {r:.1f}", fontsize=7.5, color=MUTED, va="center")
    for run, name, color in runs:
        d = pd.read_json(RUNS / run / "episodes.jsonl", lines=True)
        x, y = d.spent.mean(), d.correct.mean()
        hollow = run.startswith("answer_key")
        ax.scatter(x, y, s=150, color="white" if hollow else color, edgecolor=color, lw=2, zorder=3)
        ax.annotate(name, (x, y), xytext=(8, -12 if hollow else 8), textcoords="offset points", fontsize=8.8,
                    color=color, fontweight="bold")
    ax.set(xlabel="mean telescope hours spent per episode", ylabel="accuracy", xlim=(-1.5, 34), ylim=(0.25, 1.1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Dev set, 12 tasks. Dashed lines: equal reward. Up and to the left is better.", fontsize=10, loc="left", color=INK)
    save(fig, "fig5_results.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gym_loop(); architecture(); episode_timeline(); pipeline(); results()
    print("figures ->", OUT)
