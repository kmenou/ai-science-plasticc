# Evaluating an AI agent in a "telescope-time" environment

This guide explains the small evaluation system you will use this week: what the environment is,
how Claude is connected to it, what happens during one run, and how the runs are graded. It
assumes you know the PLAsTiCC dataset (objects, passbands, light curves, host redshifts, classes).

---

## 1. Three words: agent, environment, evaluation

- **Agent.** A language model that can *act* by calling tools, read the results, and decide
  what to do next, in a loop, until it stops. Here the agent is Claude (Haiku, Sonnet or Opus).
- **Environment.** The world the agent acts in. It defines which actions exist, what they cost,
  what the agent observes after each action, and when the interaction ends. Here it is a
  simulated telescope that sells observations of one PLAsTiCC object.
- **Evaluation.** Running the agent on many tasks, several times each, and measuring:
  - the **output**: was the final answer right?
  - the **process**: how did it get there? What did it buy, in what order, how much did it
    waste, and did it describe its evidence truthfully?

The key design choice is that **the agent can only affect the world through the environment's
tools.** It has no shell, no files and no internet. So every action it takes passes through code
we wrote, and every action is logged. That is what makes the process measurable.

---

## 2. The "Gym" pattern, in five lines

Reinforcement-learning research uses a standard interface for environments, known as the
*Gym* interface:

```
observation = env.reset(task)                      # start an episode
while not done:
    action = agent.choose(observation)             # the agent decides
    observation, reward, done = env.step(action)   # the environment responds
```

- An **episode** is one attempt at one task, from `reset` to `done`.
- The environment holds the **state**: what has been bought and how much budget is left.
- The agent only sees **observations**: the text each tool returns.

Our environment follows this pattern, with names adapted to tool calling:

| Gym | Here | What it does |
|---|---|---|
| `reset(task)` | `env.reset(task)` | Loads the target and returns the **briefing** (question, budget, prices) |
| `step(action)` | `env.call(tool, **args)` | Runs one tool, charges its cost, logs it, returns text |
| `done` | `submit(...)`, or the call limit | Ends the episode |
| `reward` | `env.finalize()` | Grades the finished episode (section 6) |

The same environment can be driven by Claude **or** by a plain Python script. We use scripts as
baselines (section 7).

---

## 3. The environment's rules

**Setting.** You are an observer with a limited allocation of telescope time. Each episode asks
one question about one PLAsTiCC object. The object is hidden behind an alias such as `T-3f9a`,
so the agent cannot look it up or recognise its `object_id`.

**Budget.**
- 30 telescope hours and at most 30 tool calls per episode.
- A typical object has about 176 epochs, so the budget buys roughly one sixth of its light curve.

**Tools (the only possible actions):**

| Tool | Price | Returns |
|---|---|---|
| `budget` | free | hours and calls left |
| `sky_position` | free | ra, dec, Galactic l and b, Milky Way extinction E(B−V), deep-field flag |
| `observing_log` | free | observing seasons and epochs per band in each season (no fluxes) |
| `alert_history` | 2 h | when the object was significantly detected, grouped into clusters (no fluxes) |
| `observe(passband, mjd_start, mjd_end)` | 1 h per epoch (min 1 h) | mjd, flux, flux_err, detected for that band and window |
| `host_photoz` | 3 h | host photometric redshift; **0 means no extragalactic host, i.e. a Galactic object** |
| `host_specz` | 10 h | host spectroscopic redshift |
| `submit(answer, confidence, rationale)` | free | the final answer, a probability of being right, and 1–3 sentences on the evidence |

**Rules the environment enforces:**
- **Too expensive:** a call that costs more than the remaining budget is refused and nothing is charged.
- **Window too large:** an `observe` window with more than 30 epochs is refused, again free of
  charge. The agent must narrow the window.
- **Paying twice:** buying the same epochs twice costs twice. The waste is logged.
- **Invalid answers:** an answer that is not one of the options, or a confidence outside
  [0, 1], is rejected, and the agent may submit again.
- **End of episode:** after a valid `submit`, or after 30 calls, the episode is over and
  further calls are refused.

**Three task families:**

| Family | Question | Counted correct if |
|---|---|---|
| `is_snia` | Is the target a SN Ia? | exact match. Most "no" cases are *other* supernovae, so this is hard |
| `coarse_class` | galactic_variable / supernova / agn / rare_transient | exact match |
| `peak_time` | MJD of peak brightness in r | within ±10 days of the brightest detected r-band epoch |

Task sets are generated with a fixed random seed:
- `tasks/dev.jsonl`: 12 tasks, for trying things out.
- `tasks/eval.jsonl`: 30 tasks, for final numbers.

The answers are kept in separate `*_answers.jsonl` files, which only the grader reads.

---

## 4. How Claude is connected: the Agent SDK

We use the **Claude Agent SDK** (`pip install claude-agent-sdk`). It is Claude Code packaged as a
Python library, and it authenticates with your Claude subscription login.

When our code calls `query(...)`, the SDK starts Claude Code as a **subprocess**. That subprocess
runs the agent loop for us:
1. It sends the conversation to the model.
2. When the model asks for a tool, it runs the tool.
3. It sends the result back to the model, and repeats.

We never write this loop ourselves. Our job is to supply the tools and the rules.

**Step 1: turn each environment tool into an SDK tool.** This is from `telescope_env/claude_agent.py`:

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

def make_tools(env):
    tools = []
    for spec in env.tool_specs():                       # name, description, JSON schema
        async def handler(args, _name=spec.name):
            res = env.call(_name, **args)               # <- the only door into the environment
            return {"content": [{"type": "text", "text": res.text}], "is_error": res.error}
        tools.append(tool(spec.name, spec.description, spec.schema)(handler))
    return tools

server = create_sdk_mcp_server("telescope", tools=make_tools(env))
```

- Each `handler` is a *closure* bound to one `env` object. Several episodes can therefore run in
  parallel, each with its own environment and budget.
- The tools are served as an in-process **MCP server**. MCP (Model Context Protocol) is the
  standard way to plug tools into Claude.
- Claude sees each tool under a name like `mcp__telescope__observe`, together with its
  description and input schema. **The description is all the model knows about a tool**, so
  wording it well matters.

**Step 2: start the agent in a locked room.**

```python
from claude_agent_sdk import ClaudeAgentOptions, query

briefing = env.reset(task)
opts = ClaudeAgentOptions(
    system_prompt=open("prompts/default.md").read(),   # the agent's standing instructions
    mcp_servers={"telescope": server}, strict_mcp_config=True,   # only our tool server
    tools=[],                                           # disable ALL built-in tools (Bash, Read, ...)
    allowed_tools=["mcp__telescope__observe", ...],     # pre-approve our tools
    setting_sources=[], cwd=empty_temp_dir,             # ignore local settings; nothing on disk
    model="haiku", max_turns=40,
)
async for message in query(prompt=briefing, options=opts):
    transcript.append(message)                          # everything the model says and does
record = env.finalize()                                 # grade the episode
```

Each isolation option closes a loophole:
- `tools=[]`: without it, the agent could run shell commands and read the answer file.
- `strict_mcp_config`: keeps out any other tool servers configured on your machine.
- `setting_sources=[]`: ignores local settings and `CLAUDE.md` files.
- empty `cwd`: there is nothing on disk to discover.

---

## 5. One episode, step by step

Here is what actually happens, using real episode `dev-006` from the Haiku pilot. The target is
really a **microlensing event**, which is Galactic.

```
our code        env.reset(task) -> briefing: "Target T-073f. Is the target a SN Ia? Budget 30 h ..."
SDK -> model    system prompt + briefing + tool list
model           "Let me start with the free tools."      -> tool_use sky_position()
SDK -> handler  env.call("sky_position")                 -> text back to the model      (0 h)
model           -> observing_log(), budget()                                            (0 h)
model           -> alert_history()                        "4 detection clusters ..."    (-2 h, 28 left)
model           -> observe(u, g, r, i over one season)    four light-curve tables       (-24 h, 4 left)
model           -> host_photoz()                          "photo-z = 0: no host found"  (-3 h, 1 left)
model           "puzzling ... the host may be too faint"
model           -> observe(r, next season)                REFUSED: over budget          (0 h)
model           -> submit("yes", 0.68, "... photo-z failure likely reflects the bright transient ...")
env             episode over -> finalize(): correct = False
```

Things to notice:
- **Everything the model knows came back as tool text.** It never saw the data file.
- **The loop is the model's own sequence of decisions.** Our code only answers each call.
- **The costly mistake is in the process, not just the output.** The decisive clue (photo-z = 0,
  so the object is Galactic) cost only 3 h. The agent bought it *after* spending 24 h on
  photometry, and then argued it away. A 40-line Python heuristic that buys photo-z first gets
  this task right for 3 h.

Three records are kept for every episode:

| Record | Written by | Contains | Trust it for |
|---|---|---|---|
| **Call log** (`episodes.jsonl` → `calls`) | the environment | every call: tool, arguments, cost, error code, budget left | what the agent *did* |
| **Transcript** (`transcripts/*.json`) | the SDK | the model's text, its tool calls, the tool results | what the agent *said* and *saw* |
| **Result message** | the SDK | number of turns, tokens, wall time, errors | cost of running the agent |

Process metrics are computed from the call log, never from the agent's own account.

---

## 6. Grading an episode

`env.finalize()` turns an episode into one graded record.

**Output metrics:**
- `correct`: see the task table in section 3.
- `conf_brier` = (confidence − correct)²: is the stated confidence honest?
- `reward` = correct − 0.25 × (fraction of budget spent): a single number that trades accuracy
  against cost.

**Process metrics, computed from the call log:**
- `spent`, `n_calls`, `n_budget_refusals` (tried to overspend), `n_invalid_calls`.
- `redundant_hours`: epochs bought more than once.
- `free_calls_before_first_paid`: did it plan before spending?
- `bands_observed`, and `observed_near_peak`: for transients, did it buy a detected point within
  15 days of the true peak?
- `unsupported_claims`: does the rationale cite evidence (a band, a redshift, alerts) that the log
  shows was never acquired? This is a crude text match: it can raise false alarms, and it misses
  wrong *reasoning*.

**Across repeats.** The model is not deterministic, so each task is run *k* times:
- **pass@k**: the fraction of tasks solved in *at least one* of k tries.
- **pass^k**: the fraction of tasks solved in *all* k tries.

A large gap between the two means the agent is unreliable.

---

## 7. Baselines: why we also run dumb scripts

A score only means something next to a reference. Three scripted policies use exactly the same
tools as Claude (`telescope_env/policies.py`):

| Policy | What it does | Why it is there |
|---|---|---|
| `random` | buys random windows, then guesses | the floor |
| `heuristic` | ~40 lines of astronomer rules: photo-z first, then alerts, then one r-band window | the bar an agent should clear |
| `answer_key` | reads the grader's answer file and submits | a **cheater**: perfect output score, zero evidence in the log |

Pilot results on the dev set (12 tasks, one run each):

| | accuracy | hours spent | reward | mean confidence |
|---|---|---|---|---|
| random | 0.42 | 11.6 | 0.32 | 0.28 |
| heuristic | 0.75 | 6.4 | **0.70** | 0.59 |
| Claude Haiku | 0.75 | 25.0 | 0.54 | 0.86 |

The cheater shows why output metrics alone are not enough. Only the process log (0 hours, one
call) reveals that it never looked at the data.

---

## 8. How it was built (and why in this order)

1. **Environment first, as plain Python** (`env.py`): tools, prices, rules and a call log, with
   no model involved. Every tool returns text, because text is what a language model reads.
2. **Tasks with hidden answers** (`tasks.py`, `catalog.py`): seeded sampling, balanced classes,
   aliases instead of object ids, and answer keys in separate files.
3. **Grader** (`scoring.py`): output metrics plus process metrics from the log.
4. **Scripted policies** (`policies.py`) and **tests** (`tests/`). These check that the rules
   and graders work *before* spending any model time. For example: the cheater must score 100%,
   and a call over budget must be refused and not charged.
5. **The Claude adapter** (`claude_agent.py`): only now is a model plugged in, through the same
   `env.call` door as the scripts.
6. **Runner, report and replay** (`run.py`, `report.py`, `replay.py`): run many episodes, then
   aggregate and inspect them.

This mirrors how real agent benchmarks are built. If the environment is not trustworthy without
the model, the model's scores mean nothing.

---

## 9. Commands

Run from the `practicals/observer_env/` folder, in a normal terminal, with the course environment
active (or put `uv run` in front of each command). Runs launched from inside a Claude Code session
fail to authenticate.

Keep your own files (prompts, metrics, runs, reports) in `submissions/<your-slug>/observer_env/`.
Below, `$MY` stands for `../../submissions/<your-slug>/observer_env`.

```bash
python -m pytest -q tests                                              # the environment works
python -m telescope_env.run --policy heuristic --tasks tasks/dev.jsonl --runs-dir $MY/runs
python -m telescope_env.run --policy claude --model haiku --tasks tasks/dev.jsonl --k 3 \
       --prompt $MY/my_prompt.md --runs-dir $MY/runs                   # about 12 min
python -m telescope_env.replay $MY/runs/<run-name> dev-006             # read one episode
python -m telescope_env.report $MY/runs/* --metrics $MY/my_metrics.py --out $MY/report.html
```

Useful switches for `run`:
- `--budget 15`: change the budget.
- `--family is_snia`: one task family only.
- `--model sonnet`: a different model.
- `--limit 3`: only the first 3 tasks.
- `--prompt my_prompt.md`: your own system prompt (start from a copy of `prompts/default.md`).

Every Claude episode uses your subscription quota (about 50 s and 12 model turns each), so
develop with `--limit` and `--k 1`.
