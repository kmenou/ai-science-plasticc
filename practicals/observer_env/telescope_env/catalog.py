"""Read-only data access shared by all episodes: observations, metadata and the answer key.

The agent never touches this module. It only sees tool outputs, and targets are
referred to by anonymous aliases (e.g. ``T-3f9a``) instead of PLAsTiCC object ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from plasticc_course.data import load_metadata, load_observations


class Catalog:
    def __init__(self, dataset: str, answers_path: str | Path):
        self.dataset = dataset
        meta = load_metadata(dataset)
        self._meta = meta.set_index("object_id")
        obs = load_observations(dataset)
        self._obs = {oid: g for oid, g in obs.groupby("object_id", sort=False)}
        self._answers = {}
        self._alias = {}
        for line in Path(answers_path).read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                self._answers[a["task_id"]] = a
                self._alias[a["target"]] = a["object_id"]

    def object_id_for(self, alias: str) -> int:
        return self._alias[alias]

    def observations(self, object_id: int) -> pd.DataFrame:
        return self._obs[object_id]

    def metadata(self, object_id: int) -> dict[str, Any]:
        return self._meta.loc[object_id].to_dict()

    def answer(self, task_id: str) -> dict[str, Any]:
        return self._answers[task_id]


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def answers_path_for(tasks_path: str | Path) -> Path:
    p = Path(tasks_path)
    return p.with_name(p.stem + "_answers.jsonl")
