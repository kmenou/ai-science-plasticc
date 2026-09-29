"""Print one episode as a timeline: what the agent said, what it bought, what it cost.

    python -m telescope_env.replay runs/<run> dev-003 [--rep 0]
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("task_id")
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()

    recs = [json.loads(l) for l in (a.run / "episodes.jsonl").read_text().splitlines()]
    rec = next(r for r in recs if r["task_id"] == a.task_id and r["rep"] == a.rep)
    print(f"{rec['task_id']} rep {rec['rep']}  [{rec['family']}]  target {rec['target']}  "
          f"(truly {rec['true_class']}, answer key: {rec['truth']})")
    print(f"budget {rec['budget']:g} h, spent {rec['spent']:g} h, end: {rec['end_reason']}\n")

    tpath = a.run / "transcripts" / f"{a.task_id}__r{a.rep}.json"
    said = []
    if tpath.exists():
        for m in json.loads(tpath.read_text()):
            if m["role"] == "assistant":
                for b in m["content"]:
                    if b["type"] == "text" and b["text"].strip():
                        said.append(b["text"].strip())
                    elif b["type"] == "tool_use":
                        said.append(None)             # marks the position of the next env call
    calls = iter(rec["calls"])
    items = said if said else [None] * len(rec["calls"])
    for s in items:
        if s is not None:
            print(textwrap.indent(textwrap.fill(s, 100), "  | "))
            continue
        c = next(calls, None)
        if c:
            args = ", ".join(f"{k}={v}" for k, v in c["args"].items())
            flag = f"  ERROR:{c['code']}" if c["error"] else ""
            print(f"{c['step']:>3}  {c['tool']}({args})  -{c['cost']:g} h  -> {c['remaining']:g} h left{flag}")
    for c in calls:                                   # any calls not matched to transcript tool_use blocks
        print(f"{c['step']:>3}  {c['tool']}({c['args']})  -{c['cost']:g} h{'  ERROR' if c['error'] else ''}")

    print(f"\nanswer: {rec['answer']}  (confidence {rec['confidence']})  correct: {rec['correct']}")
    if rec.get("rationale"):
        print(textwrap.fill("rationale: " + rec["rationale"], 100))
    if rec["claims_unsupported"]:
        print(f"UNSUPPORTED CLAIMS: {rec['claims_unsupported']}")
    if rec.get("agent"):
        ag = rec["agent"]
        print(f"turns {ag.get('num_turns')}, {ag.get('wall_s')} s, sdk_error: {ag.get('sdk_error')}")


if __name__ == "__main__":
    main()
