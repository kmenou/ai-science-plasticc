"""Copy this file to your submission folder, then complete both functions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plasticc_course import data as course_data


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
TINY_METADATA_PATH = DATA_DIR / "plasticc_tiny300_metadata.parquet"
TINY_OBSERVATIONS_PATH = DATA_DIR / "plasticc_tiny300_observations.parquet"
ANCHOR_OBJECT_ID = 252646


def dataset_summary(method: str = "pandas") -> dict:
    """Return the seven required summary values for the tiny dataset."""

    if method not in {"pandas", "course"}:
        raise ValueError("method must be 'pandas' or 'course'")

    # TODO: load the tiny metadata and observations using the requested method.
    # TODO: return exactly the seven values defined in the Week 1 README.
    raise NotImplementedError("Complete Exercise 1 in your copied starter")


def plot_lightcurve(object_id: int, output_path: str | Path) -> None:
    """Save the required six-passband light curve to output_path."""

    # TODO: load one object's observations, plot the required distinctions and
    # error bars, label the figure, and save it to the caller's output_path.
    raise NotImplementedError("Complete Exercise 3 in your copied starter")


if __name__ == "__main__":
    print(dataset_summary("pandas"))
    plot_lightcurve(ANCHOR_OBJECT_ID, Path(__file__).with_name("lightcurve.png"))

