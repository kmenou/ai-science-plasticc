import os
os.environ.setdefault("MPLBACKEND", "Agg")

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow

from plasticc_course.data import load_metadata, load_observations


ROOT_ERROR = "Run this command from the repository root, where `pyproject.toml` is located."
HASH_ERROR = (
    "The course data differs from the current repository version. Run `git pull` and "
    "rerun the preflight. If the problem remains, contact the instructor."
)
METADATA_COLUMNS = [
    "object_id",
    "ra",
    "decl",
    "gal_l",
    "gal_b",
    "ddf",
    "hostgal_specz",
    "hostgal_photoz",
    "hostgal_photoz_err",
    "distmod",
    "mwebv",
    "target",
]
OBSERVATION_COLUMNS = ["object_id", "mjd", "passband", "flux", "flux_err", "detected"]


class DataHashError(RuntimeError):
    """A committed tiny data file does not match its recorded hash."""


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _summary(metadata: pd.DataFrame, observations: pd.DataFrame) -> dict[str, object]:
    return {
        "metadata_rows": len(metadata),
        "observation_rows": len(observations),
        "unique_objects": metadata["object_id"].nunique(),
        "passbands": sorted(int(value) for value in observations["passband"].unique()),
        "target_classes": sorted(int(value) for value in metadata["target"].unique()),
        "detected_fraction": float((observations["detected"] == 1).sum() / len(observations)),
        "negative_flux_fraction": float((observations["flux"] < 0).sum() / len(observations)),
    }


def _check_hashes(repo_root: Path, truth: dict[str, object], manifest: dict[str, object]) -> None:
    for name, record in truth["tiny_files"].items():
        path = repo_root / "data" / name
        if not path.is_file():
            raise DataHashError(f"Missing required file: data/{name}")
        actual = _sha256(path)
        expected = record["sha256"]
        manifest_expected = manifest["outputs"][name]["sha256"]
        if actual != expected or expected != manifest_expected:
            raise DataHashError(f"SHA-256 mismatch for data/{name}")


def _check_summary(actual: dict[str, object], truth: dict[str, object]) -> None:
    for key, record in truth["summary"].items():
        expected = record["value"]
        value = actual[key]
        if key in {"detected_fraction", "negative_flux_fraction"}:
            matches = math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-6)
        else:
            matches = value == expected
        if not matches:
            raise RuntimeError(f"Unexpected {key}: expected {expected!r}, got {value!r}")


def _check_plot(observations: pd.DataFrame, object_id: int) -> None:
    selected = observations.loc[observations["object_id"] == object_id]
    with tempfile.TemporaryDirectory(prefix="plasticc-preflight-") as temporary:
        output = Path(temporary) / "preflight.png"
        figure, axis = plt.subplots()
        for passband, group in selected.groupby("passband", sort=True):
            axis.errorbar(group["mjd"], group["flux"], yerr=group["flux_err"], fmt="o", label=passband)
        axis.set(xlabel="MJD", ylabel="Flux", title=f"Object {object_id}")
        axis.legend()
        figure.savefig(output)
        plt.close(figure)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Matplotlib did not create a non-empty PNG")


def run_preflight(repo_root: Path) -> None:
    truth = json.loads((repo_root / "practicals" / "week01" / "ground_truth.json").read_text())
    manifest = json.loads((repo_root / "data" / "manifest.json").read_text())
    _check_hashes(repo_root, truth, manifest)

    metadata = load_metadata("tiny")
    observations = load_observations("tiny")
    if list(metadata.columns) != METADATA_COLUMNS:
        raise RuntimeError(f"Unexpected tiny metadata columns: {list(metadata.columns)!r}")
    if list(observations.columns) != OBSERVATION_COLUMNS:
        raise RuntimeError(f"Unexpected tiny observation columns: {list(observations.columns)!r}")
    if len(metadata) != 300 or metadata["object_id"].nunique() != 300:
        raise RuntimeError("Tiny metadata must contain exactly 300 unique objects")
    if sorted(int(value) for value in observations["passband"].unique()) != [0, 1, 2, 3, 4, 5]:
        raise RuntimeError("Tiny observations must contain passbands 0 through 5")

    anchor = truth["anchor_object"]
    anchor_id = int(anchor["object_id"])
    anchor_metadata = metadata.loc[metadata["object_id"] == anchor_id]
    anchor_observations = observations.loc[observations["object_id"] == anchor_id]
    if len(anchor_metadata) != 1 or int(anchor_metadata.iloc[0]["target"]) != 90:
        raise RuntimeError("The recorded target-90 anchor is absent from tiny metadata")
    if sorted(int(value) for value in anchor_observations["passband"].unique()) != [0, 1, 2, 3, 4, 5]:
        raise RuntimeError("The anchor does not contain all six passbands")

    _check_summary(_summary(metadata, observations), truth)
    if matplotlib.get_backend().lower() != "agg":
        raise RuntimeError(f"Expected the Agg backend, got {matplotlib.get_backend()}")
    _check_plot(observations, anchor_id)

    # Referencing these modules makes the required-import check explicit.
    if not all(module is not None for module in (np, pd, pyarrow, matplotlib)):
        raise RuntimeError("A required package could not be imported")


def main() -> int:
    repo_root = Path.cwd()
    if not (repo_root / "pyproject.toml").is_file():
        print(ROOT_ERROR)
        return 1
    try:
        run_preflight(repo_root)
    except DataHashError as error:
        print(HASH_ERROR)
        print(error)
        return 1
    except Exception as error:
        print(f"Preflight failed: {error}")
        return 1
    print("Preflight passed: imports, tiny data, hashes, summaries, anchor, and Agg plotting are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

