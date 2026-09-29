# Observer environment: evaluating an AI agent with a telescope-time budget

In this practical, Claude acts as an observer with a limited allocation of telescope time. In each
episode it must *buy* data about one hidden PLAsTiCC object (alerts, photometry windows, host
redshifts) and then answer a question about it. You measure two things:
- **what it answers:** accuracy and calibration;
- **how it gets there:** what it bought, in what order, what it wasted, and whether its rationale
  matches the evidence it actually acquired.

Read **[GUIDE.md](GUIDE.md)** first. It explains the environment, its rules, how Claude is
connected through the Claude Agent SDK, one episode step by step, and how episodes are graded.
The same guide, with diagrams, is also available as
[docs/Telescope_Eval_Student_Guide.docx](docs/Telescope_Eval_Student_Guide.docx).

## Contents

```
telescope_env/
  env.py           TelescopeEnv: reset(task) -> call(tool, **args) -> finalize(); tools, prices, call log
  catalog.py       read-only data + answer key (the agent sees only aliases like T-3f9a)
  tasks.py         task families and seeded task-set generation
  scoring.py       output grading + process metrics (from the env's log, not the agent's words)
  policies.py      random / heuristic / answer_key baselines
  claude_agent.py  Agent SDK adapter: env tools -> in-process MCP tools; built-in tools disabled
  run.py           run a policy over tasks x k repeats -> <runs-dir>/<name>/episodes.jsonl (+ transcripts)
  report.py        aggregate runs -> report.html + report.csv (accepts your metric file)
  replay.py        print one episode as a timeline
tasks/             dev (12 tasks) and eval (30 tasks) sets, with *_answers.jsonl read only by the grader
prompts/default.md the agent's default system prompt
student/my_metrics.py  template for your own process metrics
tests/             pytest checks of the environment (no model needed)
docs/              the guide's diagrams and the scripts that build them
```

The answer files are in the repository so that the grader can run on your machine. Do not
read them, and do not let your prompts or policies use them. The `answer_key` baseline is there
precisely to show what such a cheat looks like in the process log.

## Setup

1. **Install the Python dependencies.** From the repository root, run the usual course setup
   (`uv sync`). This installs `claude-agent-sdk`, which runs the agent on your Claude
   subscription login. No API key is needed.
2. **Log in once.** Use the `claude` command (Claude Code) in a normal terminal:
   - type `/login`, follow the browser prompt, then `/exit`.

   If you do not have Claude Code installed, use the copy bundled with the SDK. This prints its
   path:

   ```bash
   uv run python -c "import claude_agent_sdk, pathlib; print(pathlib.Path(claude_agent_sdk.__file__).parent / '_bundled' / 'claude')"
   ```

   Run that path in place of `claude`.
3. **Check** that a one-line request works. It must print `ok`:

   ```bash
   claude -p "Reply with the single word: ok" --model haiku --max-turns 1
   ```

Run everything in a normal terminal. Commands launched from *inside* another Claude Code session
inherit that session's settings and fail with `401`. If you get `401 OAuth access token is
invalid`, run `/login` again.

## Where to put your work

Keep your prompts, metrics, runs and reports in `submissions/<your-slug>/observer_env/`. Do not
edit the files in this folder. Start with:

```bash
mkdir -p ../../submissions/<your-slug>/observer_env
cp prompts/default.md ../../submissions/<your-slug>/observer_env/my_prompt.md
cp student/my_metrics.py ../../submissions/<your-slug>/observer_env/my_metrics.py
```

Then follow section 9 of [GUIDE.md](GUIDE.md) for the commands. Run them from this folder.

## Suggested practical (about 1 hour)

1. **Warm-up (10 min).** Run the heuristic and one Claude pilot on `dev` with `--k 1`. Use
   `replay` to read one episode, and name one good and one bad decision.
2. **Build a process grader (15 min).** Add two or three metrics to your `my_metrics.py`, for example:
   - colour use (at least 2 bands observed);
   - overconfidence (confidence above 0.8 and wrong);
   - wasted spend after a decisive clue (photo-z = 0, then more purchases).
3. **Run one experiment (20 min).** Run `dev` with `--k 3` (36 episodes), changing a single factor:
   - a system prompt variant (e.g. "plan before you spend");
   - a model (haiku vs sonnet);
   - a budget (15 vs 30 vs 60).
4. **Report (15 min).** Build `report.html` over your runs plus the baselines, and answer:
   - Does Claude beat the heuristic?
   - Where does it spend its hours?
   - Is it calibrated?
   - Do pass@3 and pass^3 differ?
   - Did any rationale claim evidence the log shows was never acquired?

Every Claude episode counts against your subscription limits (about 50 s and 12 model turns
each). Develop with `--limit 3 --k 1` and switch to the full set only for final numbers.
