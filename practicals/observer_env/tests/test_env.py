import json
from pathlib import Path

import pytest

from telescope_env.catalog import Catalog, answers_path_for, load_tasks
from telescope_env.env import TelescopeEnv
from telescope_env.policies import POLICIES

TASKS = Path(__file__).resolve().parents[1] / "tasks" / "dev.jsonl"


@pytest.fixture(scope="module")
def catalog():
    return Catalog("tiny", answers_path_for(TASKS))


@pytest.fixture
def task():
    return next(t for t in load_tasks(TASKS) if t["family"] == "peak_time")


def test_briefing_hides_identity(catalog, task):
    env = TelescopeEnv(catalog)
    text = env.reset(task)
    oid = catalog.object_id_for(task["target"])
    assert str(oid) not in text and task["target"] in text


def test_costs_and_refusals(catalog, task):
    env = TelescopeEnv(catalog, budget=5)
    env.reset(task)
    assert env.call("observing_log").cost == 0
    r = env.call("host_specz")                       # 10 h > 5 h budget
    assert r.error and r.cost == 0 and env.ep.spent == 0
    obs = catalog.observations(catalog.object_id_for(task["target"]))
    r = env.call("observe", passband=int(obs.passband.mode()[0]), mjd_start=0, mjd_end=1e6)
    assert r.error and r.code in {"window_too_large", "over_budget"} and env.ep.spent == 0
    r = env.call("observe", passband=2, mjd_start=0, mjd_end=1)   # empty window still costs the minimum
    assert r.cost == 1 and env.ep.spent == 1
    r = env.call("host_photoz")
    assert r.cost == 3 and env.ep.remaining == 1


def test_window_cap(catalog):
    env = TelescopeEnv(catalog, budget=1000)
    for task in load_tasks(TASKS):
        obs = catalog.observations(catalog.object_id_for(task["target"]))
        counts = obs.passband.value_counts()
        if counts.max() > 30:
            env.reset(task)
            r = env.call("observe", passband=int(counts.idxmax()), mjd_start=0, mjd_end=1e6)
            assert r.code == "window_too_large" and r.cost == 0
            return
    pytest.skip("no object with > 30 epochs in one band")


def test_redundant_purchases_are_counted(catalog, task):
    env = TelescopeEnv(catalog)
    env.reset(task)
    obs = catalog.observations(catalog.object_id_for(task["target"]))
    t = obs[obs.passband == 2].mjd.sort_values().iloc[:3]
    for _ in range(2):
        env.call("observe", passband=2, mjd_start=t.iloc[0], mjd_end=t.iloc[-1])
    rec = env.finalize()
    assert rec["n_epochs_bought"] == 6 and rec["redundant_hours"] == 3


def test_submit_validation_and_episode_end(catalog):
    task = next(t for t in load_tasks(TASKS) if t["family"] == "coarse_class")
    env = TelescopeEnv(catalog)
    env.reset(task)
    assert env.call("submit", answer="quasar", confidence=0.5, rationale="").error
    assert not env.ep.done
    assert env.call("submit", answer="agn", confidence=0.5, rationale="").done
    assert env.call("budget").error                  # nothing after submit
    rec = env.finalize()
    assert rec["submitted"] and rec["n_invalid_calls"] == 1 and rec["end_reason"] == "submitted"


def test_call_limit(catalog, task):
    env = TelescopeEnv(catalog, max_calls=3)
    env.reset(task)
    for _ in range(3):
        env.call("budget")
    assert env.ep.done and env.finalize()["end_reason"] == "max_calls"


def test_claims_check(catalog, task):
    env = TelescopeEnv(catalog)
    env.reset(task)
    env.call("submit", answer=60000, confidence=0.5,
             rationale="The g-band and r-band light curves and the spectroscopic redshift show a peak.")
    assert set(env.finalize()["claims_unsupported"]) == {"g-band", "r-band", "spec-z"}


@pytest.mark.parametrize("name", list(POLICIES))
def test_policies_complete_every_task(catalog, name):
    for task in load_tasks(TASKS):
        env = TelescopeEnv(catalog)
        env.reset(task)
        POLICIES[name](env, task, 0)
        rec = env.finalize()
        assert rec["submitted"] and rec["spent"] <= rec["budget"]
        json.dumps(rec, default=str)
    if name == "answer_key":
        assert rec["correct"]


def test_claude_adapter_tools_drive_env(catalog, task):
    pytest.importorskip("claude_agent_sdk")
    import asyncio
    from telescope_env.claude_agent import build_server

    env = TelescopeEnv(catalog)
    env.reset(task)
    server, allowed = build_server(env)
    assert "mcp__telescope__observe" in allowed and len(allowed) == len(env.tool_specs())
    assert server["type"] == "sdk"
    from telescope_env.claude_agent import make_tools
    tools = {t.name: t for t in make_tools(env)}
    out = asyncio.run(tools["observe"].handler({"passband": 2, "mjd_start": 0, "mjd_end": 1}))
    assert out["is_error"] is False and env.ep.spent == 1          # closure is bound to this env
    out = asyncio.run(tools["submit"].handler({"answer": "nope", "confidence": 0.5, "rationale": ""}))
    assert out["is_error"] is True and not env.ep.done
    asyncio.run(tools["submit"].handler({"answer": 60000.0, "confidence": 0.5, "rationale": "x"}))
    assert env.ep.done and env.finalize()["n_calls"] == 3
