import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path


ROOT_ERROR = "Run this command from the repository root, where `pyproject.toml` is located."
EXPECTED_KEYS = {
    "metadata_rows",
    "observation_rows",
    "unique_objects",
    "passbands",
    "target_classes",
    "detected_fraction",
    "negative_flux_fraction",
}
FRACTION_KEYS = {"detected_fraction", "negative_flux_fraction"}


def _copy_command(submission: Path) -> str:
    try:
        display = submission.relative_to(Path.cwd())
    except ValueError:
        display = submission
    return f"cp practicals/week01/starter.py \\\n  {display}/analysis.py"


def _load_analysis(path: Path):
    spec = importlib.util.spec_from_file_location("week01_student_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not prepare analysis.py for import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_summary(module, method: str, expected: dict[str, object]) -> list[str]:
    failures: list[str] = []
    try:
        actual = module.dataset_summary(method)
    except Exception as error:
        return [f"dataset_summary({method!r}) failed: {error}"]
    if not isinstance(actual, dict):
        return [f"dataset_summary({method!r}) must return a dict"]
    if set(actual) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(actual))
        extra = sorted(set(actual) - EXPECTED_KEYS)
        return [f"dataset_summary({method!r}) keys are wrong; missing={missing}, extra={extra}"]
    for key, expected_value in expected.items():
        value = actual[key]
        if key in FRACTION_KEYS:
            try:
                matches = math.isclose(
                    float(value), float(expected_value), rel_tol=0.0, abs_tol=1e-6
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = value == expected_value
        if not matches:
            failures.append(
                f"dataset_summary({method!r}) returned {value!r} for {key}; "
                f"expected {expected_value!r}"
            )
    return failures


def check_submission(submission: Path, repo_root: Path) -> list[str]:
    analysis_path = submission / "analysis.py"
    if not analysis_path.is_file():
        return [
            f"Missing {analysis_path}.",
            "Copy the starter first:",
            _copy_command(submission),
        ]

    try:
        module = _load_analysis(analysis_path)
    except Exception as error:
        return [f"Could not import {analysis_path}: {error}"]
    for function_name in ("dataset_summary", "plot_lightcurve"):
        if not callable(getattr(module, function_name, None)):
            return [f"analysis.py must define callable {function_name}()"]

    truth = json.loads(
        (repo_root / "practicals" / "week01" / "ground_truth.json").read_text()
    )
    expected = {key: record["value"] for key, record in truth["summary"].items()}
    failures: list[str] = []
    for method in ("pandas", "course"):
        failures.extend(_check_summary(module, method, expected))

    anchor_id = int(truth["anchor_object"]["object_id"])
    with tempfile.TemporaryDirectory(prefix="plasticc-check-") as temporary:
        temporary_plot = Path(temporary) / "checker-output.png"
        try:
            result = module.plot_lightcurve(anchor_id, temporary_plot)
            if result is not None:
                failures.append("plot_lightcurve() must return None")
        except Exception as error:
            failures.append(f"plot_lightcurve({anchor_id}, output_path) failed: {error}")
        if not temporary_plot.is_file() or temporary_plot.stat().st_size == 0:
            failures.append("plot_lightcurve() did not create a non-empty PNG at the supplied path")

    submitted_plot = submission / "lightcurve.png"
    if not submitted_plot.is_file() or submitted_plot.stat().st_size == 0:
        failures.append("Create and visually inspect a non-empty lightcurve.png in your submission folder")
    report = submission / "REPORT.md"
    if not report.is_file() or report.stat().st_size == 0:
        failures.append("Complete REPORT.md in your submission folder")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check one Week 1 submission folder")
    parser.add_argument("submission", type=Path)
    return parser.parse_args()


def main() -> int:
    repo_root = Path.cwd()
    if not (repo_root / "pyproject.toml").is_file():
        print(ROOT_ERROR)
        return 1
    args = parse_args()
    failures = check_submission(args.submission.resolve(), repo_root)
    if failures:
        print("Submission check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Submission check passed for both loading methods and the plotting function.")
    print("Human review still needs to confirm the submitted plot and report are clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

