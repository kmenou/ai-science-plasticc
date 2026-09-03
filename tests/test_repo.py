import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from plasticc_course.data import load_metadata, load_object, load_observations


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = json.loads(
    (REPO_ROOT / "practicals" / "week01" / "ground_truth.json").read_text()
)
MANIFEST = json.loads((REPO_ROOT / "data" / "manifest.json").read_text())


VALID_IMPLEMENTATION = r'''

# Reference implementation appended only to a copied temporary submission.
def dataset_summary(method: str = "pandas") -> dict:
    if method == "pandas":
        metadata = pd.read_parquet(TINY_METADATA_PATH)
        observations = pd.read_parquet(TINY_OBSERVATIONS_PATH)
    elif method == "course":
        metadata = course_data.load_metadata("tiny")
        observations = course_data.load_observations("tiny")
    else:
        raise ValueError("method must be 'pandas' or 'course'")
    return {
        "metadata_rows": len(metadata),
        "observation_rows": len(observations),
        "unique_objects": metadata["object_id"].nunique(),
        "passbands": sorted(int(value) for value in observations["passband"].unique()),
        "target_classes": sorted(int(value) for value in metadata["target"].unique()),
        "detected_fraction": float((observations["detected"] == 1).sum() / len(observations)) + 5e-7,
        "negative_flux_fraction": float((observations["flux"] < 0).sum() / len(observations)) - 5e-7,
    }


def plot_lightcurve(object_id: int, output_path: str | Path) -> None:
    observations = course_data.load_object(object_id, "tiny")
    colors = ["#7f3c8d", "#11a579", "#3969ac", "#f2b701", "#e73f74", "#80ba5a"]
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for passband, color in enumerate(colors):
        band = observations[observations["passband"] == passband]
        for detected, fill in ((1, color), (0, "none")):
            points = band[band["detected"] == detected]
            if points.empty:
                continue
            axis.errorbar(
                points["mjd"], points["flux"], yerr=points["flux_err"],
                linestyle="none", marker="o", markersize=4,
                markerfacecolor=fill, markeredgecolor=color, ecolor=color,
                alpha=0.85 if detected else 0.45,
                label=f"band {passband} ({'detected' if detected else 'not detected'})",
            )
    axis.set_xlabel("MJD")
    axis.set_ylabel("Flux")
    axis.set_title(f"PLAsTiCC object {object_id}")
    axis.legend(ncols=2, fontsize=8)
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
'''


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def run_python(script: Path, *args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("MPLBACKEND", None)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("temporary_week01_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def temporary_valid_submission():
    submissions = REPO_ROOT / "submissions"
    with tempfile.TemporaryDirectory(prefix="temporary-student-", dir=submissions) as temporary:
        submission = Path(temporary) / "week01"
        submission.mkdir()
        analysis = submission / "analysis.py"
        shutil.copy2(REPO_ROOT / "practicals" / "week01" / "starter.py", analysis)
        with analysis.open("a", encoding="utf-8") as handle:
            handle.write(VALID_IMPLEMENTATION)
        (submission / "REPORT.md").write_text(
            "# Week 1 report\n\nI inspected `lightcurve.png`: yes\n", encoding="utf-8"
        )
        module = load_module(analysis)
        module.plot_lightcurve(
            int(GROUND_TRUTH["anchor_object"]["object_id"]), submission / "lightcurve.png"
        )
        yield submission, module


def test_loaders_support_tiny_full_and_file_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiny_metadata = load_metadata()
    tiny_observations = load_observations()
    full_metadata = load_metadata("full")
    full_observations = load_observations("full")
    assert (len(tiny_metadata), len(tiny_observations)) == (300, 52_763)
    assert (len(full_metadata), len(full_observations)) == (7_848, 1_421_705)
    anchor = int(GROUND_TRUTH["anchor_object"]["object_id"])
    assert len(load_object(anchor)) == GROUND_TRUTH["anchor_object"]["observation_rows"]
    with pytest.raises(ValueError, match="tiny.*full"):
        load_metadata("other")
    with pytest.raises(KeyError, match="object_id"):
        load_object(-1)


def test_manifest_data_and_ground_truth_agree():
    frames: dict[str, pd.DataFrame] = {}
    for name, record in MANIFEST["outputs"].items():
        path = REPO_ROOT / "data" / name
        assert path.stat().st_size == record["size_bytes"]
        assert sha256(path) == record["sha256"]
        frame = pd.read_parquet(path)
        frames[name] = frame
        assert len(frame) == record["row_count"]
        assert frame["object_id"].nunique() == record["object_count"]
        assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == record["dtypes"]

    metadata = frames["plasticc_train_metadata.parquet"]
    observations = frames["plasticc_train_observations.parquet"]
    tiny_metadata = frames["plasticc_tiny300_metadata.parquet"]
    tiny_observations = frames["plasticc_tiny300_observations.parquet"]
    assert metadata["object_id"].is_unique
    assert set(observations["object_id"]).issubset(set(metadata["object_id"]))
    assert set(tiny_observations["object_id"]) == set(tiny_metadata["object_id"])
    assert len(tiny_observations) == len(
        observations[observations["object_id"].isin(tiny_metadata["object_id"])]
    )
    assert MANIFEST["tiny_subset"]["selected_object_ids"] == tiny_metadata["object_id"].tolist()
    for name, record in GROUND_TRUTH["tiny_files"].items():
        assert record["sha256"] == MANIFEST["outputs"][name]["sha256"]


def test_exact_summary_definitions_and_anchor():
    metadata = load_metadata()
    observations = load_observations()
    actual = {
        "metadata_rows": len(metadata),
        "observation_rows": len(observations),
        "unique_objects": metadata["object_id"].nunique(),
        "passbands": sorted(int(value) for value in observations["passband"].unique()),
        "target_classes": sorted(int(value) for value in metadata["target"].unique()),
        "detected_fraction": float((observations["detected"] == 1).sum() / len(observations)),
        "negative_flux_fraction": float((observations["flux"] < 0).sum() / len(observations)),
    }
    expected = {key: record["value"] for key, record in GROUND_TRUTH["summary"].items()}
    for key in {"detected_fraction", "negative_flux_fraction"}:
        assert actual[key] == pytest.approx(expected[key], abs=1e-6, rel=0.0)
    for key in set(expected) - {"detected_fraction", "negative_flux_fraction"}:
        assert actual[key] == expected[key]
    anchor = GROUND_TRUTH["anchor_object"]
    row = metadata.loc[metadata["object_id"] == anchor["object_id"]].iloc[0]
    selected = observations.loc[observations["object_id"] == anchor["object_id"]]
    assert int(row["target"]) == 90
    assert len(selected) == anchor["observation_rows"]
    assert int((selected["detected"] == 1).sum()) == anchor["detected_rows"]
    assert sorted(int(value) for value in selected["passband"].unique()) == anchor["passbands"]


def test_preflight_and_headless_first_lines():
    preflight = REPO_ROOT / "scripts" / "preflight.py"
    checker = REPO_ROOT / "practicals" / "week01" / "check_submission.py"
    expected_prefix = 'import os\nos.environ.setdefault("MPLBACKEND", "Agg")\n'
    assert preflight.read_text().startswith(expected_prefix)
    assert checker.read_text().startswith(expected_prefix)
    result = run_python(preflight)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Preflight passed" in result.stdout


def test_helpful_wrong_directory_and_missing_analysis_messages(tmp_path):
    preflight = REPO_ROOT / "scripts" / "preflight.py"
    checker = REPO_ROOT / "practicals" / "week01" / "check_submission.py"
    expected = "Run this command from the repository root, where `pyproject.toml` is located."
    wrong_preflight = run_python(preflight, cwd=tmp_path)
    wrong_checker = run_python(checker, "missing", cwd=tmp_path)
    assert wrong_preflight.returncode != 0 and wrong_preflight.stdout.splitlines()[0] == expected
    assert wrong_checker.returncode != 0 and wrong_checker.stdout.splitlines()[0] == expected

    with tempfile.TemporaryDirectory(prefix="missing-analysis-", dir=REPO_ROOT / "submissions") as temp:
        submission = Path(temp) / "week01"
        submission.mkdir()
        missing = run_python(checker, str(submission.relative_to(REPO_ROOT)))
        assert missing.returncode != 0
        assert "cp practicals/week01/starter.py" in missing.stdout
        assert f"{submission.relative_to(REPO_ROOT)}/analysis.py" in missing.stdout
        assert "Traceback" not in missing.stdout + missing.stderr


def test_temporary_copied_submission_checker_plot_path_and_fraction_tolerance(tmp_path):
    checker = REPO_ROOT / "practicals" / "week01" / "check_submission.py"
    with temporary_valid_submission() as (submission, module):
        assert "submissions" in submission.parts and submission.name == "week01"
        assert module.__file__ != str(REPO_ROOT / "practicals" / "week01" / "starter.py")
        supplied_path = tmp_path / "caller-selected.png"
        assert module.plot_lightcurve(
            int(GROUND_TRUTH["anchor_object"]["object_id"]), supplied_path
        ) is None
        assert supplied_path.stat().st_size > 0
        assert module.plt.get_backend().lower() == "agg"
        result = run_python(checker, str(submission.relative_to(REPO_ROOT)))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Submission check passed" in result.stdout


def test_repository_has_no_workflows_or_extra_week_files():
    workflows = REPO_ROOT / ".github" / "workflows"
    assert not workflows.exists() or not any(workflows.iterdir())
    for week in range(2, 7):
        directory = REPO_ROOT / "practicals" / f"week{week:02d}"
        assert [path.name for path in directory.iterdir()] == ["README.md"]

