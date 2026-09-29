"""Task families and seeded task-set generation.

Each task is one question about one target. A set is written as two files:

    tasks/<name>.jsonl          what the agent is told (task_id, family, target alias, question, ...)
    tasks/<name>_answers.jsonl  the answer key (object_id, truth), read only by the grader

Usage:
    python -m telescope_env.tasks --dataset tiny --seed 0 --out tasks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plasticc_course.data import load_metadata, load_observations

SNIA = {90}
SN = {42, 52, 62, 67, 90, 95}
COARSE = {
    "galactic_variable": {6, 16, 53, 65, 92},
    "supernova": SN,
    "agn": {88},
    "rare_transient": {15, 64},
}
CLASS_NAMES = {6: "microlens", 15: "TDE", 16: "eclipsing binary", 42: "SN II", 52: "SN Iax",
               53: "Mira", 62: "SN Ibc", 64: "kilonova", 65: "M-dwarf flare", 67: "SN Ia-91bg",
               88: "AGN", 90: "SN Ia", 92: "RR Lyrae", 95: "SLSN-I"}
PEAK_TOLERANCE_DAYS = 10.0

FAMILIES = {
    "is_snia": {
        "question": "Is the target a type Ia supernova (SN Ia)?",
        "answer_format": "'yes' or 'no'",
        "options": ["yes", "no"],
    },
    "coarse_class": {
        "question": "Which broad class does the target belong to?",
        "answer_format": ("one of 'galactic_variable' (Milky Way star: eclipsing binary, RR Lyrae, Mira, "
                          "M-dwarf flare, microlensing), 'supernova' (any supernova type, including "
                          "superluminous), 'agn' (active galactic nucleus), 'rare_transient' (tidal "
                          "disruption event or kilonova)"),
        "options": list(COARSE),
    },
    "peak_time": {
        "question": "At what MJD did the target reach its peak brightness in the r band (passband 2)?",
        "answer_format": f"a number (MJD); counted correct within +/-{PEAK_TOLERANCE_DAYS:g} days",
        "options": None,
    },
}


def r_peak(obs: pd.DataFrame) -> float | None:
    """MJD of the brightest detected r-band point, if it is bracketed by r epochs within 30 d."""
    r = obs[obs.passband == 2]
    det = r[r.detected == 1]
    if len(det) < 3:
        return None
    t = float(det.loc[det.flux.idxmax(), "mjd"])
    before = ((r.mjd < t) & (r.mjd > t - 30)).any()
    after = ((r.mjd > t) & (r.mjd < t + 30)).any()
    return t if (before and after) else None


def truth_for(family: str, target: int, peak: float | None):
    if family == "is_snia":
        return "yes" if target in SNIA else "no"
    if family == "coarse_class":
        return next(k for k, v in COARSE.items() if target in v)
    return peak


def _pick(rng, pool: pd.DataFrame, n: int, used: set) -> list[int]:
    pool = pool[~pool.object_id.isin(used)]
    ids = rng.choice(pool.object_id.to_numpy(), size=min(n, len(pool)), replace=False).tolist()
    used.update(ids)
    return ids


def make_sets(dataset: str, seed: int, sizes: dict[str, int], out: Path) -> None:
    rng = np.random.default_rng(seed)
    meta = load_metadata(dataset)
    obs = load_observations(dataset)
    peaks = {oid: r_peak(g) for oid, g in obs.groupby("object_id")}
    meta = meta.assign(peak=meta.object_id.map(peaks))
    used: set[int] = set()
    aliases: set[str] = set()
    out.mkdir(parents=True, exist_ok=True)

    for set_name, n in sizes.items():            # n tasks per family
        chosen: list[tuple[str, int]] = []
        # is_snia: half SN Ia; negatives mostly other supernovae (the hard case)
        n_pos = n // 2
        n_hard = int(round(0.6 * (n - n_pos)))
        chosen += [("is_snia", i) for i in _pick(rng, meta[meta.target.isin(SNIA)], n_pos, used)]
        chosen += [("is_snia", i) for i in _pick(rng, meta[meta.target.isin(SN - SNIA)], n_hard, used)]
        chosen += [("is_snia", i) for i in _pick(rng, meta[~meta.target.isin(SN)], n - n_pos - n_hard, used)]
        # coarse_class: as balanced as n allows
        groups = list(COARSE)
        per = [n // 4 + (1 if k < n % 4 else 0) for k in range(4)]
        for g, k in zip(rng.permutation(groups), per):
            chosen += [("coarse_class", i) for i in _pick(rng, meta[meta.target.isin(COARSE[g])], k, used)]
        # peak_time: supernovae with a bracketed r-band peak
        pool = meta[meta.target.isin(SN) & meta.peak.notna()]
        chosen += [("peak_time", i) for i in _pick(rng, pool, n, used)]

        order = rng.permutation(len(chosen))
        tasks, answers = [], []
        for j, idx in enumerate(order):
            fam, oid = chosen[idx]
            row = meta.loc[meta.object_id == oid].iloc[0]
            while (alias := f"T-{rng.integers(16**4):04x}") in aliases:
                pass
            aliases.add(alias)
            task_id = f"{set_name}-{j:03d}"
            f = FAMILIES[fam]
            tasks.append({"task_id": task_id, "family": fam, "target": alias, "question": f["question"],
                          "answer_format": f["answer_format"], "options": f["options"], "budget": 30})
            peak = None if pd.isna(row.peak) else float(row.peak)
            answers.append({"task_id": task_id, "target": alias, "object_id": int(oid),
                            "true_class": int(row.target), "class_name": CLASS_NAMES[int(row.target)],
                            "truth": truth_for(fam, int(row.target), peak), "peak_mjd_r": peak})
        (out / f"{set_name}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
        (out / f"{set_name}_answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))
        print(f"{set_name}: {len(tasks)} tasks -> {out / (set_name + '.jsonl')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="tiny", choices=["tiny", "full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", type=int, default=4, help="tasks per family in the dev set")
    ap.add_argument("--eval", type=int, default=10, help="tasks per family in the eval set")
    ap.add_argument("--out", type=Path, default=Path("tasks"))
    a = ap.parse_args()
    make_sets(a.dataset, a.seed, {"dev": a.dev, "eval": a.eval}, a.out)


if __name__ == "__main__":
    main()
