#!/usr/bin/env python3
"""Build and validate the full and teaching-subset PLAsTiCC Parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Iterable
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".cache" / "plasticc"
DATA_DIR = REPO_ROOT / "data"
GROUND_TRUTH_PATH = REPO_ROOT / "practicals" / "week01" / "ground_truth.json"
PREVIEW_PATH = REPO_ROOT / ".cache" / "anchor_preview.png"

SEED = 2026
TINY_OBJECT_COUNT = 300
PASSBANDS = [0, 1, 2, 3, 4, 5]

SOURCE_FILES = {
    "plasticc_train_metadata.csv.gz": {
        "url": "https://zenodo.org/api/records/2539456/files/plasticc_train_metadata.csv.gz/content",
        "md5": "8c6b00fd503d6cf3d9a42bfb53046e0f",
    },
    "plasticc_train_lightcurves.csv.gz": {
        "url": "https://zenodo.org/api/records/2539456/files/plasticc_train_lightcurves.csv.gz/content",
        "md5": "1aa1605908b5a6398bd46bf9120b6400",
    },
}

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

SOURCE_METADATA_COLUMNS = [
    "object_id",
    "ra",
    "decl",
    "ddf_bool",
    "hostgal_specz",
    "hostgal_photoz",
    "hostgal_photoz_err",
    "distmod",
    "mwebv",
    "target",
]
SOURCE_OBSERVATION_COLUMNS = [
    "object_id",
    "mjd",
    "passband",
    "flux",
    "flux_err",
    "detected_bool",
]

OUTPUT_NAMES = [
    "plasticc_train_metadata.parquet",
    "plasticc_train_observations.parquet",
    "plasticc_tiny300_metadata.parquet",
    "plasticc_tiny300_observations.parquet",
]


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_sources() -> dict[str, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, details in SOURCE_FILES.items():
        destination = CACHE_DIR / name
        if destination.exists() and digest(destination, "md5") == details["md5"]:
            print(f"Using verified source: {destination.relative_to(REPO_ROOT)}")
            paths[name] = destination
            continue

        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)
        print(f"Downloading {name} ...")
        try:
            with urllib.request.urlopen(details["url"]) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        actual = digest(partial, "md5")
        if actual != details["md5"]:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {name}: expected {details['md5']}, got {actual}"
            )
        partial.replace(destination)
        print(f"Verified MD5 {actual}")
        paths[name] = destination
    return paths


def load_sources(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(
        paths["plasticc_train_metadata.csv.gz"], usecols=SOURCE_METADATA_COLUMNS
    )
    observations = pd.read_csv(
        paths["plasticc_train_lightcurves.csv.gz"], usecols=SOURCE_OBSERVATION_COLUMNS
    )

    # The unblinded Zenodo release uses *_bool names and omits the Galactic
    # coordinates present in the challenge metadata schema. Compute Galactic
    # longitude and latitude from J2000 ICRS RA/Dec with the standard rotation
    # matrix, without adding an astronomy-package dependency.
    metadata = metadata.rename(columns={"ddf_bool": "ddf"})
    observations = observations.rename(columns={"detected_bool": "detected"})
    ra = np.deg2rad(metadata["ra"].to_numpy(dtype=np.float64))
    decl = np.deg2rad(metadata["decl"].to_numpy(dtype=np.float64))
    equatorial = np.vstack(
        [np.cos(decl) * np.cos(ra), np.cos(decl) * np.sin(ra), np.sin(decl)]
    )
    rotation = np.array(
        [
            [-0.0548755604, -0.8734370902, -0.4838350155],
            [0.4941094279, -0.4448296300, 0.7469822445],
            [-0.8676661490, -0.1980763734, 0.4559837762],
        ],
        dtype=np.float64,
    )
    galactic = rotation @ equatorial
    metadata["gal_l"] = np.mod(np.rad2deg(np.arctan2(galactic[1], galactic[0])), 360.0)
    metadata["gal_b"] = np.rad2deg(np.arcsin(np.clip(galactic[2], -1.0, 1.0)))
    metadata = metadata[METADATA_COLUMNS]
    observations = observations[OBSERVATION_COLUMNS]

    metadata = metadata.astype(
        {
            "object_id": "int64",
            "ra": "float64",
            "decl": "float64",
            "gal_l": "float64",
            "gal_b": "float64",
            "ddf": "int8",
            "hostgal_specz": "float32",
            "hostgal_photoz": "float32",
            "hostgal_photoz_err": "float32",
            "distmod": "float32",
            "mwebv": "float32",
            "target": "int16",
        }
    )
    observations = observations.astype(
        {
            "object_id": "int64",
            "mjd": "float64",
            "passband": "int8",
            "flux": "float32",
            "flux_err": "float32",
            "detected": "int8",
        }
    )
    metadata = metadata.sort_values("object_id", kind="mergesort").reset_index(drop=True)
    observations = observations.sort_values(
        ["object_id", "mjd", "passband"], kind="mergesort"
    ).reset_index(drop=True)
    return metadata, observations


def allocate_sample(class_counts: pd.Series) -> dict[int, int]:
    targets = sorted(int(value) for value in class_counts.index)
    if any(int(class_counts.loc[target]) < 10 for target in targets):
        raise RuntimeError("Every class must contain at least 10 source objects")
    minimum = 10 * len(targets)
    if minimum > TINY_OBJECT_COUNT:
        raise RuntimeError("The 10-per-class minimum exceeds the subset size")

    remaining = TINY_OBJECT_COUNT - minimum
    total = int(class_counts.sum())
    quotas = {target: remaining * int(class_counts.loc[target]) / total for target in targets}
    allocations = {target: 10 + int(np.floor(quotas[target])) for target in targets}
    unallocated = TINY_OBJECT_COUNT - sum(allocations.values())
    remainder_order = sorted(targets, key=lambda target: (-(quotas[target] % 1), target))
    for target in remainder_order[:unallocated]:
        allocations[target] += 1

    if sum(allocations.values()) != TINY_OBJECT_COUNT:
        raise AssertionError("Largest-remainder allocation did not total 300")
    if any(allocations[target] > int(class_counts.loc[target]) for target in targets):
        raise RuntimeError("A sample allocation exceeds its source class size")
    return allocations


def sample_metadata(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    class_counts = metadata.groupby("target", sort=True)["object_id"].count()
    allocations = allocate_sample(class_counts)
    rng = np.random.default_rng(SEED)
    chosen: list[int] = []
    for target in sorted(allocations):
        candidates = np.sort(
            metadata.loc[metadata["target"] == target, "object_id"].to_numpy(copy=True)
        )
        sampled = rng.choice(candidates, size=allocations[target], replace=False)
        chosen.extend(int(value) for value in sampled)
    tiny = metadata[metadata["object_id"].isin(chosen)].copy()
    tiny = tiny.sort_values("object_id", kind="mergesort").reset_index(drop=True)
    return tiny, allocations


def choose_anchor(
    tiny_metadata: pd.DataFrame, tiny_observations: pd.DataFrame
) -> dict[str, object]:
    target_ids = set(
        int(value)
        for value in tiny_metadata.loc[tiny_metadata["target"] == 90, "object_id"]
    )
    candidates: list[dict[str, object]] = []
    for object_id, group in tiny_observations[tiny_observations["object_id"].isin(target_ids)].groupby(
        "object_id", sort=True
    ):
        passbands = sorted(int(value) for value in group["passband"].unique())
        if passbands != PASSBANDS:
            continue
        candidates.append(
            {
                "object_id": int(object_id),
                "target": 90,
                "class_name": "SNIa",
                "observation_rows": int(len(group)),
                "detected_rows": int((group["detected"] == 1).sum()),
                "passbands": passbands,
            }
        )
    if not candidates:
        raise RuntimeError("The tiny subset has no target-90 object with all six passbands")

    eligible = [
        row
        for row in candidates
        if int(row["observation_rows"]) >= 40 and int(row["detected_rows"]) >= 20
    ]
    ranked = sorted(
        eligible or candidates,
        key=lambda row: (
            -int(row["detected_rows"]),
            -int(row["observation_rows"]),
            int(row["object_id"]),
        ),
    )
    anchor = ranked[0]
    missed: list[str] = []
    if int(anchor["observation_rows"]) < 40:
        missed.append("at least 40 total observation rows")
    if int(anchor["detected_rows"]) < 20:
        missed.append("at least 20 detected rows")
    anchor["thresholds_met"] = not missed
    anchor["thresholds_missed"] = missed
    anchor["selection_criteria"] = (
        "Within the completed tiny subset: target == 90, all six passbands; prefer at "
        "least 40 observations and 20 detected rows; then rank by detected rows "
        "descending, total rows descending, and object_id ascending."
    )
    anchor["description"] = (
        "Well-sampled target-90 Type Ia supernova selected by the documented anchor rule."
    )
    return anchor


def write_parquet(frame: pd.DataFrame, path: Path, dictionary_columns: Iterable[str]) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=list(dictionary_columns),
        write_statistics=True,
    )


def render_preview(observations: pd.DataFrame, anchor: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ["#7f3c8d", "#11a579", "#3969ac", "#f2b701", "#e73f74", "#80ba5a"]
    selected = observations[observations["object_id"] == anchor["object_id"]]
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for passband, color in zip(PASSBANDS, colors, strict=True):
        band = selected[selected["passband"] == passband]
        for detected, marker, fill in [(1, "o", color), (0, "o", "none")]:
            points = band[band["detected"] == detected]
            if points.empty:
                continue
            axis.errorbar(
                points["mjd"],
                points["flux"],
                yerr=points["flux_err"],
                linestyle="none",
                marker=marker,
                markersize=4.5,
                markerfacecolor=fill,
                markeredgecolor=color,
                ecolor=color,
                alpha=0.85 if detected else 0.45,
                label=f"band {passband} ({'detected' if detected else 'not detected'})",
            )
    axis.axhline(0, color="0.7", linewidth=0.8)
    axis.set_xlabel("MJD")
    axis.set_ylabel("Flux")
    axis.set_title(f"PLAsTiCC target 90 (SNIa), object {anchor['object_id']}")
    axis.legend(ncols=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Anchor preview was not created")


def class_count_dict(metadata: pd.DataFrame) -> dict[str, int]:
    counts = metadata.groupby("target", sort=True)["object_id"].count()
    return {str(int(target)): int(count) for target, count in counts.items()}


def dtype_dict(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}


def file_record(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": digest(path, "sha256"),
        "row_count": int(len(frame)),
        "object_count": int(frame["object_id"].nunique()),
        "dtypes": dtype_dict(frame),
        "class_counts": class_count_dict(frame) if "target" in frame else None,
    }


def summary_values(metadata: pd.DataFrame, observations: pd.DataFrame) -> dict[str, object]:
    return {
        "metadata_rows": int(len(metadata)),
        "observation_rows": int(len(observations)),
        "unique_objects": int(metadata["object_id"].nunique()),
        "passbands": sorted(int(value) for value in observations["passband"].unique()),
        "target_classes": sorted(int(value) for value in metadata["target"].unique()),
        "detected_fraction": float((observations["detected"] == 1).sum() / len(observations)),
        "negative_flux_fraction": float((observations["flux"] < 0).sum() / len(observations)),
    }


SUMMARY_DESCRIPTIONS = {
    "metadata_rows": "len(metadata) for the tiny metadata table.",
    "observation_rows": "len(observations) for the tiny observations table.",
    "unique_objects": "metadata['object_id'].nunique() from the tiny metadata table, not observations.",
    "passbands": "Sorted unique integer values in observations['passband'].",
    "target_classes": "Sorted unique integer values in metadata['target'].",
    "detected_fraction": "Rows with detected == 1 divided by all observation rows; no rows are dropped.",
    "negative_flux_fraction": "Rows with flux < 0, strictly, divided by all observation rows; zero is not negative and no rows are dropped.",
}


def validate_frames(
    metadata: pd.DataFrame,
    observations: pd.DataFrame,
    tiny_metadata: pd.DataFrame,
    tiny_observations: pd.DataFrame,
    anchor: dict[str, object],
) -> None:
    if metadata["object_id"].duplicated().any():
        raise RuntimeError("Metadata object_id values are not unique")
    metadata_ids = set(int(value) for value in metadata["object_id"])
    observation_ids = set(int(value) for value in observations["object_id"])
    if not observation_ids.issubset(metadata_ids):
        raise RuntimeError("An observation object_id is absent from metadata")
    if sorted(int(value) for value in observations["passband"].unique()) != PASSBANDS:
        raise RuntimeError("The source observations do not contain exactly six passbands")
    if len(tiny_metadata) != TINY_OBJECT_COUNT or tiny_metadata["object_id"].nunique() != TINY_OBJECT_COUNT:
        raise RuntimeError("Tiny metadata does not contain exactly 300 unique objects")
    if set(tiny_metadata["target"].unique()) != set(metadata["target"].unique()):
        raise RuntimeError("Tiny metadata does not contain every training class")
    tiny_ids = set(int(value) for value in tiny_metadata["object_id"])
    expected_tiny_observations = observations[observations["object_id"].isin(tiny_ids)]
    if len(expected_tiny_observations) != len(tiny_observations):
        raise RuntimeError("Tiny observations do not contain all source rows for selected objects")
    if set(int(value) for value in tiny_observations["object_id"]) != tiny_ids:
        raise RuntimeError("Tiny observations contain the wrong object IDs")
    if int(anchor["object_id"]) not in tiny_ids or int(anchor["target"]) != 90:
        raise RuntimeError("Anchor must be a target-90 object in the tiny subset")


def validate_written(
    paths: dict[str, Path],
    source_row_counts: tuple[int, int],
) -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    if len(frames["plasticc_train_metadata.parquet"]) != source_row_counts[0]:
        raise RuntimeError("Source and full metadata row counts differ")
    if len(frames["plasticc_train_observations.parquet"]) != source_row_counts[1]:
        raise RuntimeError("Source and full observation row counts differ")
    pair_size = sum(paths[name].stat().st_size for name in OUTPUT_NAMES[:2])
    limit = 50 * 1024 * 1024
    if pair_size >= limit:
        sizes = ", ".join(f"{name}={paths[name].stat().st_size} bytes" for name in OUTPUT_NAMES[:2])
        raise RuntimeError(f"Full Parquet pair is not below 50 MiB: {sizes}")
    return frames


def build_manifest(
    staged_paths: dict[str, Path],
    frames: dict[str, pd.DataFrame],
    allocations: dict[int, int],
    tiny_metadata: pd.DataFrame,
    anchor: dict[str, object],
) -> dict[str, object]:
    return {
        "sources": {
            name: {"url": details["url"], "md5": details["md5"]}
            for name, details in SOURCE_FILES.items()
        },
        "data_license": {
            "name": "Creative Commons Attribution 4.0 International",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "title": "Unblinded Data for PLAsTiCC Classification Challenge",
            "creators": "PLAsTiCC Team and PLAsTiCC modelers",
            "doi": "10.5281/zenodo.2539456",
            "paper": "PLAsTiCC Team et al., arXiv:1810.00001",
        },
        "conversion": {
            "date": date.today().isoformat(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "matplotlib": matplotlib.__version__,
            "compression": "zstd",
            "flat_tables": True,
            "schema_normalization": {
                "ddf": "Renamed from the source column ddf_bool.",
                "detected": "Renamed from the source column detected_bool.",
                "gal_l_gal_b": (
                    "Derived deterministically from J2000 ICRS ra/decl using the "
                    "standard ICRS-to-Galactic rotation matrix."
                ),
            },
        },
        "outputs": {
            name: file_record(staged_paths[name], frames[name]) for name in OUTPUT_NAMES
        },
        "tiny_subset": {
            "seed": SEED,
            "object_count": TINY_OBJECT_COUNT,
            "minimum_per_class": 10,
            "allocation_rule": (
                "Allocate 10 per target, then allocate remaining slots proportional to full "
                "class counts by largest remainders, breaking ties by ascending target. "
                "Within each ascending target, sample sorted object IDs without replacement "
                "from one NumPy default_rng seeded with 2026."
            ),
            "allocations": {str(target): count for target, count in sorted(allocations.items())},
            "selected_object_ids": [int(value) for value in tiny_metadata["object_id"]],
        },
        "anchor_object": anchor,
    }


def ground_truth(
    staged_paths: dict[str, Path],
    tiny_metadata: pd.DataFrame,
    tiny_observations: pd.DataFrame,
    anchor: dict[str, object],
) -> dict[str, object]:
    values = summary_values(tiny_metadata, tiny_observations)
    anchor_fields = {
        key: anchor[key]
        for key in [
            "object_id",
            "target",
            "class_name",
            "selection_criteria",
            "observation_rows",
            "detected_rows",
            "passbands",
            "thresholds_met",
            "thresholds_missed",
            "description",
        ]
    }
    return {
        "tiny_files": {
            name: {"sha256": digest(staged_paths[name], "sha256")}
            for name in OUTPUT_NAMES[2:]
        },
        "summary": {
            key: {"value": value, "description": SUMMARY_DESCRIPTIONS[key]}
            for key, value in values.items()
        },
        "anchor_object": anchor_fields,
    }


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=False) + "\n").encode("utf-8")


def install_outputs(
    staged_paths: dict[str, Path],
    manifest: dict[str, object],
    truth: dict[str, object],
    replace: bool,
) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_outputs = [DATA_DIR / name for name in OUTPUT_NAMES]
    existing_complete = (
        all(path.exists() for path in existing_outputs)
        and (DATA_DIR / "manifest.json").exists()
        and GROUND_TRUTH_PATH.exists()
    )
    staged_hashes = {name: digest(path, "sha256") for name, path in staged_paths.items()}
    existing_match = existing_complete and all(
        digest(DATA_DIR / name, "sha256") == staged_hashes[name] for name in OUTPUT_NAMES
    )

    if existing_match:
        return "unchanged"

    any_existing = any(path.exists() for path in existing_outputs) or (DATA_DIR / "manifest.json").exists()
    if any_existing and not replace:
        raise RuntimeError(
            "Generated artifacts differ from existing files; rerun with --replace to overwrite them."
        )

    for name, staged in staged_paths.items():
        shutil.copyfile(staged, DATA_DIR / name)
    (DATA_DIR / "manifest.json").write_bytes(json_bytes(manifest))
    GROUND_TRUTH_PATH.write_bytes(json_bytes(truth))
    return "replaced" if any_existing else "created"


def validate_installed(manifest: dict[str, object], truth: dict[str, object]) -> None:
    installed_paths = {name: DATA_DIR / name for name in OUTPUT_NAMES}
    installed = validate_written(
        installed_paths,
        (
            int(manifest["outputs"][OUTPUT_NAMES[0]]["row_count"]),
            int(manifest["outputs"][OUTPUT_NAMES[1]]["row_count"]),
        ),
    )
    for name, path in installed_paths.items():
        expected = manifest["outputs"][name]["sha256"]
        if digest(path, "sha256") != expected:
            raise RuntimeError(f"Installed SHA-256 mismatch for {name}")
    for name in OUTPUT_NAMES[2:]:
        if truth["tiny_files"][name]["sha256"] != digest(installed_paths[name], "sha256"):
            raise RuntimeError(f"Ground-truth SHA-256 mismatch for {name}")
    anchor = manifest["anchor_object"]
    validate_frames(
        installed[OUTPUT_NAMES[0]],
        installed[OUTPUT_NAMES[1]],
        installed[OUTPUT_NAMES[2]],
        installed[OUTPUT_NAMES[3]],
        anchor,
    )


def report(manifest: dict[str, object]) -> None:
    print("\nPhase A outputs")
    for name in OUTPUT_NAMES:
        item = manifest["outputs"][name]
        print(
            f"- {name}: rows={item['row_count']:,}, objects={item['object_count']:,}, "
            f"size={item['size_bytes']:,} bytes, sha256={item['sha256'][:12]}"
        )
        print(f"  dtypes={item['dtypes']}")
        if item["class_counts"] is not None:
            print(f"  class_counts={item['class_counts']}")
    anchor = manifest["anchor_object"]
    print(
        "- anchor: "
        f"object_id={anchor['object_id']}, target={anchor['target']} ({anchor['class_name']}), "
        f"rows={anchor['observation_rows']}, detected={anchor['detected_rows']}, "
        f"passbands={anchor['passbands']}, thresholds_met={anchor['thresholds_met']}"
    )
    if anchor["thresholds_missed"]:
        print(f"  thresholds_missed={anchor['thresholds_missed']}")
    print(f"- preview: {PREVIEW_PATH.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace", action="store_true", help="replace existing artifacts when generated content differs"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = download_sources()
    metadata, observations = load_sources(source_paths)
    tiny_metadata, allocations = sample_metadata(metadata)
    tiny_ids = set(int(value) for value in tiny_metadata["object_id"])
    tiny_observations = observations[observations["object_id"].isin(tiny_ids)].copy()
    tiny_observations = tiny_observations.sort_values(
        ["object_id", "mjd", "passband"], kind="mergesort"
    ).reset_index(drop=True)
    anchor = choose_anchor(tiny_metadata, tiny_observations)
    validate_frames(metadata, observations, tiny_metadata, tiny_observations, anchor)

    with tempfile.TemporaryDirectory(prefix="plasticc-build-", dir=CACHE_DIR) as temporary:
        stage = Path(temporary)
        staged_paths = {name: stage / name for name in OUTPUT_NAMES}
        write_parquet(metadata, staged_paths[OUTPUT_NAMES[0]], ["target", "ddf"])
        write_parquet(observations, staged_paths[OUTPUT_NAMES[1]], ["object_id", "passband"])
        write_parquet(tiny_metadata, staged_paths[OUTPUT_NAMES[2]], ["target", "ddf"])
        write_parquet(tiny_observations, staged_paths[OUTPUT_NAMES[3]], ["object_id", "passband"])
        frames = validate_written(staged_paths, (len(metadata), len(observations)))
        manifest = build_manifest(staged_paths, frames, allocations, tiny_metadata, anchor)
        truth = ground_truth(staged_paths, tiny_metadata, tiny_observations, anchor)
        disposition = install_outputs(staged_paths, manifest, truth, args.replace)

    if disposition == "unchanged":
        manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    render_preview(tiny_observations, anchor, PREVIEW_PATH)
    validate_installed(manifest, truth)
    print(f"Artifacts {disposition}; all Phase A checks passed.")
    report(manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
