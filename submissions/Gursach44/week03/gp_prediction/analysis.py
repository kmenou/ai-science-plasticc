"""Week 3: single-passband light curves for a Type Ia supernova and an AGN."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plasticc_course import data as course_data


# PLAsTiCC passband integers follow the LSST filter order u, g, r, i, z, y.
PASSBAND_NAMES = {0: "u", 1: "g", 2: "r", 3: "i", 4: "z", 5: "y"}
PASSBAND_INDEX = {name: index for index, name in PASSBAND_NAMES.items()}

# (object_id, class label, passband name). Object 713 is only in the full set.
TARGETS = [
    (252646, "SN Ia", "i"),
    (713, "AGN", "r"),
]


def plot_single_band(ax, object_id: int, label: str, band_name: str) -> None:
    """Plot one object's flux versus MJD in a single passband on ax."""

    observations = course_data.load_object(object_id, dataset="full")
    band = observations[observations["passband"] == PASSBAND_INDEX[band_name]]

    for detected, marker, fill, legend in (
        (1, "o", None, "detected"),
        (0, "o", "none", "non-detected"),
    ):
        subset = band[band["detected"] == detected]
        if subset.empty:
            continue
        ax.errorbar(
            subset["mjd"],
            subset["flux"],
            yerr=subset["flux_err"],
            fmt=marker,
            markerfacecolor=fill,
            color="C0",
            markersize=5,
            linestyle="none",
            capsize=2,
            label=legend,
        )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux")
    ax.set_title(f"{label} — object {object_id}, {band_name} band")
    ax.legend()


def plot_lightcurves(output_path: str | Path) -> None:
    """Save the two single-band light curves as stacked panels to output_path."""

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(9, 8))
    for ax, (object_id, label, band_name) in zip(axes, TARGETS):
        plot_single_band(ax, object_id, label, band_name)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    plot_lightcurves(Path(__file__).with_name("lightcurves.png"))
