"""Task 3 - flagging anomalous observations inside individual light curves.

Six objects, one per class, each inspected by eye after flagging. Three
independent detectors run on every observation:

  A  residual from a smooth fit
     A leave-one-out Gaussian-kernel regression is fitted per object and per
     passband, with the kernel bandwidth chosen per band by leave-one-out
     cross-validation. Leave-one-out matters twice over: a smoother that can
     see the point it is predicting will chase an outlier and hide it, and a
     bandwidth chosen on in-sample residuals will collapse toward zero width
     for the same reason. The local model is a line rather than an average,
     because an average is biased on any sloped stretch of light curve and a
     supernova decline is nothing but slope; the script reports both so the
     size of that effect is measured rather than claimed. The residual is
     divided by the quoted flux error and then by the robust spread of the
     residuals among each point's neighbours in time, so that the threshold
     tightens where the fit is good and loosens where the fit cannot track the
     data.

  B  detection flag against measured signal-to-noise
     `detected` is the survey pipeline's own call. Across the full training
     set it corresponds to |flux / flux_err| of roughly 3 to 5, but not
     exactly: 17,117 undetected observations exceed |SNR| 5 and 1,048 detected
     ones sit below |SNR| 3. Those disagreements are places where the catalogue
     contradicts itself, which is worth surfacing even though neither value is
     provably wrong.

  C  outsized photometric error
     flux_err has a runaway tail - the largest value in the training set is
     2.2e6, against a median of 4.7. These are unusable measurements taken
     through cloud or moonlight, and they are the easiest anomaly to catch.

No metric is optimised here and nothing is held out, because there is no
ground-truth label for "bad observation" in this catalogue. That is the point
of the task: the assessment is visual, and the report says where each detector
was right, where it was wrong, and why.

Run from the repository root:
    uv run --with lightgbm --with scikit-learn python \\
      submissions/rorikang/week02/hw2_task3_outliers.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw2_common import (BAND_COLORS, BAND_NAMES, INK, INK_MUTED, N_BANDS,
                        OUT_DIR, PLASTICC_CLASSES, load_metadata,
                        load_observations, style_axes)

# One object from each of six classes, chosen to span the kinds of variability
# the survey contains: a single transient burst, a slow superluminous event, a
# stochastic active nucleus, and three kinds of repeating stellar variability.
SHOWCASE_TARGETS = [90, 95, 88, 92, 16, 65]

SMOOTH_SIGMA = 5.0      # robust-sigma threshold for detector A
SNR_HIGH = 5.0          # undetected but this significant -> flag (detector B)
SNR_LOW = 3.0           # detected but less significant than this -> flag
ERROR_RATIO = 10.0      # flux_err this many times the band median -> flag (C)
MIN_POINTS_PER_BAND = 8


def choose_objects(metadata: pd.DataFrame, observations: pd.DataFrame) -> list:
    """Pick the best-sampled deep-drilling object in each showcase class.

    Deep-drilling fields are observed far more often than the wide survey, so
    their light curves are dense enough that a smooth fit is meaningful and a
    human can actually judge whether a flagged point is wrong. Choosing the
    most-detected object makes the selection deterministic and reproducible.
    """
    counts = (observations[observations["detected"] == 1]
              .groupby("object_id").size().rename("n_detected"))
    table = metadata.join(counts, on="object_id")
    table = table[(table["ddf"] == 1) & table["n_detected"].notna()]

    chosen = []
    for target in SHOWCASE_TARGETS:
        candidates = table[table["target"] == target]
        if candidates.empty:
            continue
        chosen.append(int(candidates.sort_values(
            ["n_detected", "object_id"], ascending=[False, True])
            .iloc[0]["object_id"]))
    return chosen


def smooth_leave_one_out(time, flux, error, bandwidth, degree=1):
    """Local regression evaluated without each point's own value.

    Weights combine a Gaussian kernel in time with the inverse variance of each
    measurement, so a noisy neighbour counts for less than a precise one.

    `degree` selects the local model. Degree 0 is a kernel-weighted average
    (Nadaraya-Watson); degree 1 fits a straight line in the local window and
    takes its value at the point. The difference is not cosmetic. A weighted
    average is biased wherever the light curve has a slope, because it mixes in
    neighbours from the brighter side, and the bias grows with the slope. On a
    supernova decline that bias is many times the quoted flux error, so a
    degree-0 fit reports the entire decline as anomalous. A local line absorbs
    the slope exactly and leaves only curvature, which is second order. Both
    are kept here so the report can quantify the difference rather than assert
    it.
    """
    separation = time[None, :] - time[:, None]      # row i, column j: t_j - t_i
    weight = np.exp(-0.5 * (separation / bandwidth) ** 2) / error[None, :] ** 2
    np.fill_diagonal(weight, 0.0)

    s0 = weight.sum(axis=1)
    t0 = weight @ flux
    if degree == 0:
        return np.divide(t0, s0, out=np.full_like(flux, np.nan), where=s0 > 0)

    s1 = (weight * separation).sum(axis=1)
    s2 = (weight * separation ** 2).sum(axis=1)
    t1 = (weight * separation * flux[None, :]).sum(axis=1)
    determinant = s0 * s2 - s1 ** 2
    # Where the local window is degenerate (too few neighbours, or all at the
    # same time) the line is not identified; fall back to the average there.
    usable = determinant > 1e-12 * np.abs(s0 * s2)
    prediction = np.divide(t0, s0, out=np.full_like(flux, np.nan), where=s0 > 0)
    prediction[usable] = ((s2 * t0 - s1 * t1)[usable]
                          / determinant[usable])
    return prediction


def choose_bandwidth(time, flux, error, degree=1):
    """Leave-one-out bandwidth selection, on a robust criterion.

    The criterion is the median absolute normalised residual rather than the
    mean squared one. A least-squares criterion would widen the bandwidth to
    accommodate the very outliers this analysis is trying to find; a median
    criterion lets the fit track the bulk of the light curve and leave the
    outliers sticking out. The grid spans a day to a year because the six
    objects vary on timescales that differ by two orders of magnitude, and
    fixing one bandwidth for all of them would guarantee failure on most.
    """
    grid = np.geomspace(1.0, 365.0, 40)
    best_bandwidth, best_score = grid[0], np.inf
    for bandwidth in grid:
        prediction = smooth_leave_one_out(time, flux, error, bandwidth,
                                          degree=degree)
        valid = np.isfinite(prediction)
        if valid.sum() < MIN_POINTS_PER_BAND:
            continue
        score = np.median(
            np.abs((flux[valid] - prediction[valid]) / error[valid]))
        if score < best_score:
            best_bandwidth, best_score = bandwidth, score
    return best_bandwidth


def local_scale(time, pull, neighbours=15):
    """Robust spread of the pulls among each point's nearest neighbours in time.

    A single spread per passband is not enough. The pull is
    (flux - fit) / flux_err, so it measures measurement noise only where the
    fit is good. On a supernova rise the fit lags the data by more than the
    quoted error, and on a sub-day periodic variable it cannot track the curve
    at all; in both cases the pull is dominated by how well the model fits, not
    by the noise. A per-band spread is then set by whichever regime covers most
    of the light curve, and the other regime either floods with false flags or
    goes silent.

    Measuring the spread locally lets the threshold follow the fit quality:
    strict on the flat pre-explosion baseline, lenient through the peak. The
    floor of 1 keeps an unusually quiet stretch from flagging ordinary noise.
    Neighbours exclude the point itself, so one bad measurement cannot raise
    the threshold that is about to be applied to it.
    """
    scale = np.ones_like(pull)
    order = np.argsort(time)
    finite = np.isfinite(pull)
    for position, index in enumerate(order):
        if not finite[index]:
            continue
        low = max(0, position - neighbours)
        high = min(len(order), position + neighbours + 1)
        window = [j for j in order[low:high] if j != index and finite[j]]
        if len(window) < 5:
            continue
        values = pull[window]
        spread = 1.4826 * np.median(np.abs(values - np.median(values)))
        scale[index] = max(spread, 1.0)
    return scale


def flag_object(observations: pd.DataFrame, degree: int = 1) -> pd.DataFrame:
    """Run all three detectors over one object's observations."""
    frame = observations.sort_values("mjd").reset_index(drop=True).copy()
    frame["snr"] = frame["flux"] / frame["flux_err"]
    frame["smooth"] = np.nan
    frame["pull"] = np.nan
    frame["robust_z"] = np.nan
    frame["local_scale"] = np.nan
    frame["steepness"] = np.nan
    frame["bandwidth"] = np.nan

    for band in range(N_BANDS):
        rows = frame.index[frame["passband"] == band]
        if len(rows) < MIN_POINTS_PER_BAND:
            continue
        time = frame.loc[rows, "mjd"].to_numpy(dtype=float)
        flux = frame.loc[rows, "flux"].to_numpy(dtype=float)
        error = frame.loc[rows, "flux_err"].to_numpy(dtype=float)

        bandwidth = choose_bandwidth(time, flux, error, degree=degree)
        prediction = smooth_leave_one_out(time, flux, error, bandwidth,
                                          degree=degree)
        pull = (flux - prediction) / error
        scale = local_scale(time, pull)

        # How far the light curve moves across one bandwidth, in units of the
        # measurement error. This is the quantity that controls how badly a
        # locally-constant fit is biased, so it is what the two local models
        # are compared against.
        order = np.argsort(time)
        slope = np.full_like(flux, np.nan)
        if np.ptp(time[order]) > 0:
            slope[order] = np.gradient(prediction[order], time[order])
        steepness = np.abs(slope) * bandwidth / error

        frame.loc[rows, "smooth"] = prediction
        frame.loc[rows, "pull"] = pull
        frame.loc[rows, "robust_z"] = pull / scale
        frame.loc[rows, "local_scale"] = scale
        frame.loc[rows, "steepness"] = steepness
        frame.loc[rows, "bandwidth"] = bandwidth

    band_median_error = frame.groupby("passband")["flux_err"].transform("median")

    frame["flag_smooth"] = frame["robust_z"].abs() > SMOOTH_SIGMA
    frame["flag_detection"] = (
        ((frame["detected"] == 0) & (frame["snr"].abs() > SNR_HIGH))
        | ((frame["detected"] == 1) & (frame["snr"].abs() < SNR_LOW)))
    frame["flag_error"] = frame["flux_err"] > ERROR_RATIO * band_median_error
    frame["flagged"] = (frame["flag_smooth"] | frame["flag_detection"]
                        | frame["flag_error"])
    return frame


def bias_on_steep_stretches(constant_fit, linear_fit) -> dict:
    """Compare the two local models where a constant fit is expected to fail.

    Counting flags does not settle this: the better-fitting model has smaller
    local residuals, so its threshold tightens and it can end up flagging more
    points. What distinguishes bias from sensitivity is *where* the residuals
    sit. A locally-constant fit averages in neighbours from the brighter side
    of a slope, so its residuals carry a systematic offset that grows with the
    slope, while a local line absorbs the slope exactly. Splitting on steepness
    and comparing the typical residual measures that directly.
    """
    steepness = linear_fit["steepness"]
    valid = steepness.notna()
    if valid.sum() < 20:
        return {}
    steep = valid & (steepness >= steepness[valid].quantile(0.9))
    flat = valid & (steepness <= steepness[valid].quantile(0.5))
    result = {}
    for label, fit in (("kernel_average", constant_fit),
                       ("local_line", linear_fit)):
        pull = fit["pull"]
        result[f"abs_pull_steep_{label}"] = float(pull[steep].abs().median())
        result[f"abs_pull_flat_{label}"] = float(pull[flat].abs().median())
    return result


def main() -> None:
    metadata = load_metadata()
    observations = load_observations()
    object_ids = choose_objects(metadata, observations)

    host = metadata.set_index("object_id")
    print("Objects inspected (most-detected deep-drilling object per class):")
    for object_id in object_ids:
        row = host.loc[object_id]
        print(f"  {object_id:>10}  {PLASTICC_CLASSES[int(row['target'])]:<17} "
              f"z_spec {row['hostgal_specz']:.3f}")
    print()

    flagged = {}
    summary_rows = []
    comparison_rows = []
    for object_id in object_ids:
        subset = observations[observations["object_id"] == object_id]
        result = flag_object(subset, degree=1)
        constant_fit = flag_object(subset, degree=0)
        flagged[object_id] = result
        target = int(host.loc[object_id, "target"])
        comparison_rows.append({
            "object_id": object_id,
            "class": PLASTICC_CLASSES[target],
            "flags_kernel_average": int(constant_fit["flag_smooth"].sum()),
            "flags_local_line": int(result["flag_smooth"].sum()),
            **bias_on_steep_stretches(constant_fit, result),
        })
        summary_rows.append({
            "object_id": object_id,
            "class": PLASTICC_CLASSES[target],
            "n_obs": len(result),
            "A_smooth_residual": int(result["flag_smooth"].sum()),
            "B_detection_snr": int(result["flag_detection"].sum()),
            "C_large_error": int(result["flag_error"].sum()),
            "any_flag": int(result["flagged"].sum()),
            "flagged_percent": 100 * result["flagged"].mean(),
            "median_bandwidth_days": float(
                result["bandwidth"].dropna().median()),
        })

    summary = pd.DataFrame(summary_rows).set_index("object_id")
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))

    comparison = pd.DataFrame(comparison_rows).set_index("object_id")
    print("\nDetector A, the two local models compared.")
    print("`abs_pull_steep` is the median |flux - fit| / flux_err over the "
          "steepest\ndecile of each light curve - the stretches where a "
          "locally-constant fit is\nbiased. `abs_pull_flat` is the same over "
          "the flattest half, where neither\nmodel should be biased.")
    print(comparison.to_string(float_format=lambda v: f"{v:.2f}"))
    comparison.to_csv(OUT_DIR / "task3_local_model_comparison.csv")

    detail = pd.concat(
        [f.loc[f["flagged"], ["object_id", "mjd", "passband", "flux",
                              "flux_err", "detected", "snr", "smooth",
                              "pull", "robust_z", "local_scale", "steepness",
                              "flag_smooth", "flag_detection", "flag_error"]]
         for f in flagged.values()])
    detail.to_csv(OUT_DIR / "task3_flagged_observations.csv", index=False)
    summary.to_csv(OUT_DIR / "task3_summary.csv")

    print(f"\n{len(detail)} flagged observations in total across "
          f"{len(object_ids)} objects.")
    print("Bandwidth chosen per band varies from "
          f"{summary['median_bandwidth_days'].min():.1f} to "
          f"{summary['median_bandwidth_days'].max():.1f} days across these "
          f"objects - one fixed smoothing scale would not have worked.")

    plot(flagged, host, OUT_DIR / "task3_outliers.png")
    plot_zoom(flagged, host, OUT_DIR / "task3_outliers_zoom.png")
    print(f"\nWrote task3_outliers.png, task3_outliers_zoom.png, "
          f"task3_summary.csv, task3_flagged_observations.csv, "
          f"task3_local_model_comparison.csv to {OUT_DIR.name}/")


def draw_object(ax, frame, object_id, host, show_smooth=True, window=None):
    """One light curve, all bands, with flags marked by detector."""
    if window is not None:
        frame = frame[frame["mjd"].between(*window)]
    for band in range(N_BANDS):
        band_rows = frame[frame["passband"] == band]
        if band_rows.empty:
            continue
        ax.errorbar(band_rows["mjd"], band_rows["flux"],
                    yerr=band_rows["flux_err"], fmt="o", markersize=2.4,
                    linewidth=0, elinewidth=0.5, color=BAND_COLORS[band],
                    alpha=0.75, label=BAND_NAMES[band])
        if show_smooth:
            ordered = band_rows.sort_values("mjd")
            # Break the curve across seasonal gaps. Drawn straight through,
            # a 200-day gap becomes a diagonal line that looks like data.
            smooth = ordered["smooth"].to_numpy(dtype=float).copy()
            gap = ordered["mjd"].diff().to_numpy()
            smooth[gap > 30] = np.nan
            ax.plot(ordered["mjd"], smooth, color=BAND_COLORS[band],
                    linewidth=1.0, alpha=0.65)

    markers = (("flag_smooth", "o", "A: smooth-fit residual", "#0b0b0b"),
               ("flag_detection", "s", "B: detection vs SNR", "#4a3aa7"),
               ("flag_error", "X", "C: outsized error", "#eb6834"))
    for column, marker, label, color in markers:
        hits = frame[frame[column]]
        if hits.empty:
            continue
        ax.scatter(hits["mjd"], hits["flux"], s=70, marker=marker,
                   facecolors="none", edgecolors=color, linewidths=1.3,
                   label=f"{label} ({len(hits)})", zorder=5)

    # Scale to the measurements, not to the fit. Where the local line
    # extrapolates across a gap it can shoot far outside the data; letting that
    # set the axis would squash the light curve into a flat line.
    low = (frame["flux"] - frame["flux_err"]).min()
    high = (frame["flux"] + frame["flux_err"]).max()
    padding = 0.06 * (high - low) if high > low else 1.0
    ax.set_ylim(low - padding, high + padding)

    target = int(host.loc[object_id, "target"])
    redshift = host.loc[object_id, "hostgal_specz"]
    label = "Galactic" if redshift == 0 else f"z = {redshift:.3f}"
    ax.set_title(f"{object_id} - {PLASTICC_CLASSES[target]} ({label})",
                 fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("MJD", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel("Flux", fontsize=9.5, color=INK_MUTED)
    ax.legend(loc="best", frameon=False, fontsize=6.6, ncol=2, labelcolor=INK)
    style_axes(ax)


def plot(flagged, host, output_path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    for ax, (object_id, frame) in zip(axes.ravel(), flagged.items()):
        draw_object(ax, frame, object_id, host)
    fig.suptitle("Task 3 - anomalous observations in six light curves; lines "
                 "are the leave-one-out smooth fit", fontsize=13, color=INK,
                 y=0.997)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zoom(flagged, host, output_path) -> None:
    """Close-ups around the single most extreme flag in each light curve.

    The full-range panels compress a 1,000-day baseline into 8 cm, which is
    enough to see that a point was flagged but not enough to judge whether it
    should have been. These panels are what the written assessment is based on.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for ax, (object_id, frame) in zip(axes.ravel(), flagged.items()):
        flags = frame[frame["flagged"]]
        if flags.empty:
            ax.set_visible(False)
            continue
        worst = flags.loc[flags["robust_z"].abs().fillna(0).idxmax()] \
            if flags["robust_z"].notna().any() else flags.iloc[0]
        centre = float(worst["mjd"])
        draw_object(ax, frame, object_id, host,
                    window=(centre - 60, centre + 60))
        ax.axvline(centre, color=INK_MUTED, linestyle=":", linewidth=0.9)
    fig.suptitle("Task 3 - the same light curves, zoomed to +/-60 days around "
                 "their most extreme flag", fontsize=13, color=INK, y=0.997)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
