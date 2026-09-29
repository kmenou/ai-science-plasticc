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

    if method == "pandas":
        metadata = pd.read_parquet(TINY_METADATA_PATH)
        observations = pd.read_parquet(TINY_OBSERVATIONS_PATH)
    else:
        metadata = course_data.load_metadata()
        observations = course_data.load_observations()

    return {
        "metadata_rows": len(metadata),
        "observation_rows": len(observations),
        "unique_objects": metadata["object_id"].nunique(),
        "passbands": sorted(observations["passband"].unique().tolist()),
        "target_classes": sorted(metadata["target"].unique().tolist()),
        "detected_fraction": (observations["detected"] == 1).sum() / len(observations),
        "negative_flux_fraction": (observations["flux"] < 0).sum() / len(observations),
    }


def plot_lightcurve(object_id: int, output_path: str | Path) -> None:
    """Save the required six-passband light curve to output_path."""

    observations = course_data.load_object(object_id)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for passband in sorted(observations["passband"].unique()):
        band = observations[observations["passband"] == passband]
        color = colors[passband % len(colors)]
        for detected, marker, label_suffix in ((1, "o", ""), (0, "x", " (non-detected)")):
            subset = band[band["detected"] == detected]
            if subset.empty:
                continue
            ax.errorbar(
                subset["mjd"],
                subset["flux"],
                yerr=subset["flux_err"],
                fmt=marker,
                color=color,
                markersize=5,
                linestyle="none",
                capsize=2,
                label=f"passband {passband}{label_suffix}",
            )

    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux")
    ax.set_title(f"Object {object_id} multiband light curve")
    ax.legend(fontsize=12, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    print(dataset_summary("pandas"))
    plot_lightcurve(ANCHOR_OBJECT_ID, Path(__file__).with_name("lightcurve.png"))

