"""Scripted (non-LLM) policies. They use exactly the same text interface as the agent.

- random:     buys random windows until half the budget is gone, then guesses. Lower bound.
- heuristic:  ~40 lines of astronomer rules. The bar an LLM agent should clear.
- answer_key: cheats by reading the grader's answer key. Used to test the graders, and as a
              worked example of a perfect output score with zero evidence in the process log.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from .env import TelescopeEnv


def _submit(env, answer, confidence, rationale):
    env.call("submit", answer=answer, confidence=confidence, rationale=rationale)


def _seasons(text: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in re.findall(r"MJD ([\d.]+)-([\d.]+);", text)]


def _clusters(text: str) -> list[tuple[float, float, int]]:
    return [(float(a), float(b), int(n)) for a, b, n in
            re.findall(r"MJD ([\d.]+)-([\d.]+), (\d+) detections", text)]


def _photometry(text: str) -> np.ndarray:
    rows = re.findall(r"^([\d.]+),(-?[\d.]+),([\d.]+),([01])$", text, flags=re.M)
    return np.array(rows, dtype=float).reshape(-1, 4)


def _observe(env, band, t0, t1):
    res = env.call("observe", passband=band, mjd_start=t0, mjd_end=t1)
    if "window contains" in res.text:            # too many epochs: halve the window once
        res = env.call("observe", passband=band, mjd_start=t0, mjd_end=(t0 + t1) / 2)
    return _photometry(res.text)


def random_policy(env: TelescopeEnv, task: dict, rep: int = 0) -> None:
    rng = np.random.default_rng(zlib.crc32(f"{task['task_id']}:{rep}".encode()))
    seasons = _seasons(env.call("observing_log").text)
    lo, hi = seasons[0][0], seasons[-1][1]
    for _ in range(8):
        if env.ep.remaining < env.ep.budget / 2:
            break
        t0 = rng.uniform(lo, hi - 50)
        _observe(env, int(rng.integers(6)), t0, t0 + 50)
    if task["family"] == "peak_time":
        _submit(env, float(rng.uniform(lo, hi)), 0.1, "Random guess.")
    else:
        _submit(env, str(rng.choice(task["options"])), 1 / len(task["options"]), "Random guess.")


def heuristic_policy(env: TelescopeEnv, task: dict, rep: int = 0) -> None:
    fam = task["family"]
    env.call("observing_log")
    if fam != "peak_time":
        z_text = env.call("host_photoz").text
        if "photo-z = 0:" in z_text:
            _submit(env, "no" if fam == "is_snia" else "galactic_variable", 0.9,
                    "Host photo-z is 0, so the source is Galactic.")
            return
    clusters = _clusters(env.call("alert_history").text)
    if not clusters:
        ans = {"is_snia": "no", "coarse_class": "supernova"}.get(fam)
        _submit(env, ans if ans else 0.0, 0.2, "No alerts, so there is little to go on.")
        return
    main = max(clusters, key=lambda c: c[2])
    span = clusters[-1][1] - clusters[0][0]
    if fam == "coarse_class":
        if len(clusters) >= 3 or span > 300:
            _submit(env, "agn", 0.6, "Alerts recur across seasons: stochastic variability, like an AGN.")
        elif main[1] - main[0] < 15:
            _submit(env, "rare_transient", 0.4, "A single, very short cluster of alerts: a fast transient.")
        else:
            _submit(env, "supernova", 0.6, "A single cluster of alerts lasting weeks: a supernova-like transient.")
        return
    r = _observe(env, 2, main[0] - 20, main[0] + 60)
    det = r[r[:, 1] > 0] if len(r) else r
    if len(det) == 0:
        _submit(env, "no" if fam == "is_snia" else main[0] + 10, 0.2, "No usable r-band flux near the alerts.")
        return
    i_pk = int(np.argmax(det[:, 1]))
    t_pk, f_pk = det[i_pk, 0], det[i_pk, 1]
    if fam == "peak_time":
        _submit(env, float(t_pk), 0.6, "Brightest r-band epoch near the first alert cluster.")
        return
    late = r[(r[:, 0] > t_pk + 20) & (r[:, 0] < t_pk + 45)]
    if len(late) == 0 or f_pk <= 0:
        _submit(env, "no", 0.5, "Not enough r-band data after the peak to measure a decline.")
        return
    ratio = float(np.median(late[:, 1]) / f_pk)
    is_ia = ratio < 0.55
    _submit(env, "yes" if is_ia else "no", 0.6,
            f"r-band flux 20-45 d after the peak is {ratio:.2f} of the peak value "
            f"({'fast' if is_ia else 'slow'} decline).")


def answer_key_policy(env: TelescopeEnv, task: dict, rep: int = 0) -> None:
    truth = env.catalog.answer(task["task_id"])["truth"]
    _submit(env, truth, 1.0, "I am confident.")


POLICIES = {"random": random_policy, "heuristic": heuristic_policy, "answer_key": answer_key_policy}
