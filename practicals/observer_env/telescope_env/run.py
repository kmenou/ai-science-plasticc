"""Run a policy over a task set, k repeats per task, and write graded episodes.

Examples:
    python -m telescope_env.run --policy heuristic --tasks tasks/dev.jsonl
    python -m telescope_env.run --policy claude --model haiku --prompt prompts/default.md \\
        --tasks tasks/dev.jsonl --k 3 --concurrency 3

Output: runs/<name>/config.json, episodes.jsonl (one graded record per episode),
        transcripts/<task_id>__r<rep>.json (Claude runs only)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from .catalog import Catalog, answers_path_for, load_tasks
from .env import TelescopeEnv


def _write(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def run_scripted(args, tasks, catalog, out: Path, meta: dict) -> None:
    from .policies import POLICIES
    policy = POLICIES[args.policy]
    for task in tasks:
        for rep in range(args.k):
            env = TelescopeEnv(catalog, budget=args.budget, max_calls=args.max_calls)
            env.reset(task)
            policy(env, task, rep)
            _write(out / "episodes.jsonl", {**meta, "rep": rep, **env.finalize()})


async def run_claude(args, tasks, catalog, out: Path, meta: dict) -> None:
    from .claude_agent import load_prompt, run_episode
    system_prompt = load_prompt(args.prompt)
    sem = asyncio.Semaphore(args.concurrency)
    (out / "transcripts").mkdir(exist_ok=True)
    jobs = [(t, r) for r in range(args.k) for t in tasks]
    done = 0

    async def one(task, rep):
        nonlocal done
        async with sem:
            env = TelescopeEnv(catalog, budget=args.budget, max_calls=args.max_calls)
            transcript, info = await run_episode(env, task, model=args.model, system_prompt=system_prompt,
                                                 max_turns=args.max_turns, effort=args.effort)
            rec = {**meta, "rep": rep, **env.finalize(), "agent": info}
            _write(out / "episodes.jsonl", rec)
            (out / "transcripts" / f"{task['task_id']}__r{rep}.json").write_text(
                json.dumps(transcript, indent=1, default=str))
            done += 1
            flag = "ok " if rec["correct"] else "BAD"
            err = f"  [{info['sdk_error'][:80]}]" if info["sdk_error"] else ""
            print(f"[{done}/{len(jobs)}] {flag} {task['task_id']} r{rep} {task['family']:<12} "
                  f"spent {rec['spent']:>4g}h calls {rec['n_calls']:>2} {info['wall_s']:>5}s{err}", flush=True)

    await asyncio.gather(*(one(t, r) for t, r in jobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, choices=["claude", "random", "heuristic", "answer_key"])
    ap.add_argument("--tasks", type=Path, default=Path("tasks/dev.jsonl"))
    ap.add_argument("--dataset", default="tiny", choices=["tiny", "full"])
    ap.add_argument("--k", type=int, default=1, help="repeats per task")
    ap.add_argument("--budget", type=float, default=None, help="override the task budget (hours)")
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--family", default=None, help="only run tasks of this family")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--model", default="haiku", help="claude only: haiku | sonnet | opus | full model id")
    ap.add_argument("--prompt", type=Path, default=Path("prompts/default.md"), help="claude only")
    ap.add_argument("--effort", default=None, choices=[None, "low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--name", default=None, help="run folder name (default: derived from settings)")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    if args.family:
        tasks = [t for t in tasks if t["family"] == args.family]
    tasks = tasks[: args.limit]
    catalog = Catalog(args.dataset, answers_path_for(args.tasks))

    budget_tag = f"b{args.budget:g}" if args.budget else "btask"
    agent_tag = f"claude-{args.model}-{args.prompt.stem}" if args.policy == "claude" else args.policy
    name = args.name or f"{agent_tag}-{budget_tag}-{args.tasks.stem}-{time.strftime('%m%d-%H%M%S')}"
    out = args.runs_dir / name
    out.mkdir(parents=True, exist_ok=False)
    meta = {"run": name, "policy": args.policy, "model": args.model if args.policy == "claude" else None,
            "prompt": args.prompt.stem if args.policy == "claude" else None, "task_set": args.tasks.stem}
    (out / "config.json").write_text(json.dumps({**vars(args), "n_tasks": len(tasks)}, indent=1, default=str))

    t0 = time.time()
    if args.policy == "claude":
        asyncio.run(run_claude(args, tasks, catalog, out, meta))
    else:
        run_scripted(args, tasks, catalog, out, meta)
    print(f"{len(tasks) * args.k} episodes in {time.time() - t0:.0f}s -> {out}/episodes.jsonl")


if __name__ == "__main__":
    main()
