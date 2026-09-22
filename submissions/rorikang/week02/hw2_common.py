"""Shared pieces for AST3101 Homework 2: features, metrics, and plot style.

Everything here is imported by the four task scripts. Nothing in this file
trains a model or writes a figure.

Two properties of the catalogue drive most of the design decisions below, and
both are documented where they are used:

1. `hostgal_specz == hostgal_photoz == 0` and `distmod == -9` are sentinels
   meaning "Galactic object, no host galaxy", not measurements. All 2,325 such
   objects are dropped from every redshift task.
2. `distmod` is a deterministic function of `hostgal_photoz` in this
   catalogue. It is photo-z re-expressed in magnitudes, so it is excluded from
   the redshift feature sets.

Run from the repository root.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
FEATURE_CACHE = REPO_ROOT / ".cache" / "hw2_lightcurve_features.parquet"

METADATA_PATH = DATA_DIR / "plasticc_train_metadata.parquet"
OBSERVATIONS_PATH = DATA_DIR / "plasticc_train_observations.parquet"

RANDOM_STATE = 42
N_BANDS = 6

# PLAsTiCC photometric zeropoint: mag = ZP - 2.5 * log10(flux).
ZEROPOINT = 27.5

# The standard photometric-redshift catastrophic-failure definition.
OUTLIER_CUT = 0.15

PLASTICC_CLASSES = {
    6: "mu-lens single", 15: "TDE", 16: "eclipsing binary", 42: "SN II",
    52: "SN Iax", 53: "Mira", 62: "SN Ibc", 64: "kilonova", 65: "M-dwarf",
    67: "SN Ia-91bg", 88: "AGN", 90: "SN Ia", 92: "RR Lyrae", 95: "SLSN-I",
}

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, GREEN, GOLD, PURPLE = ("#2a78d6", "#eb6834", "#1baf7a",
                                     "#eda100", "#4a3aa7")
BAND_COLORS = ["#4a3aa7", "#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#8c1d40"]
BAND_NAMES = ["u", "g", "r", "i", "z", "y"]


def style_axes(ax):
    """House plot style: light grid, no top/right spines, muted ticks."""
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def load_metadata() -> pd.DataFrame:
    """Full training metadata, one row per object."""
    return pd.read_parquet(METADATA_PATH)


def load_observations() -> pd.DataFrame:
    """Full training photometry, many rows per object."""
    return pd.read_parquet(OBSERVATIONS_PATH)


def extragalactic_mask(metadata: pd.DataFrame) -> pd.Series:
    """True for objects with a real host-galaxy redshift.

    Galactic objects carry 0 in both redshift columns and -9 in distmod as
    sentinels. Both conditions are checked rather than one, so that a change in
    how the sentinels are encoded shows up as a disagreement instead of
    silently admitting 2,325 fake z = 0 objects into a regression.
    """
    by_photoz = metadata["hostgal_photoz"] == 0
    by_distmod = metadata["distmod"] == -9
    if not (by_photoz == by_distmod).all():
        raise ValueError("hostgal_photoz and distmod sentinels disagree")
    return ~by_photoz


# --- light-curve features -----------------------------------------------------

def build_lightcurve_features(observations: pd.DataFrame) -> pd.DataFrame:
    """One feature row per object, from photometry alone.

    No metadata enters here, so the same table can be used for classification
    (Task 1) and for redshift regression (Tasks 2 and 4) without any risk of a
    metadata column leaking into a model that is not supposed to see it.

    The features are chosen for the two physical handles that photometry has on
    redshift: brightness (peak flux, a distance proxy for standard candles) and
    timescale (an intrinsic light curve of duration t is observed to last
    t * (1 + z)). Colours carry the third handle, the redshifted spectral
    energy distribution.
    """
    obs = observations.copy()
    obs["snr"] = obs["flux"] / obs["flux_err"]

    by_object = obs.groupby("object_id")
    features = pd.DataFrame(index=by_object.size().index)
    features.index.name = "object_id"

    # --- whole-light-curve behaviour ---
    features["n_obs"] = by_object.size()
    features["flux_max"] = by_object["flux"].max()
    features["flux_min"] = by_object["flux"].min()
    features["flux_std"] = by_object["flux"].std()
    features["flux_median"] = by_object["flux"].median()
    features["flux_skew"] = by_object["flux"].skew()
    features["flux_err_median"] = by_object["flux_err"].median()
    features["snr_max"] = by_object["snr"].max()
    features["snr_std"] = by_object["snr"].std()
    features["detected_fraction"] = by_object["detected"].mean()
    features["positive_flux_fraction"] = by_object["flux"].apply(
        lambda f: float((f > 0).mean()))
    features["amplitude"] = features["flux_max"] - features["flux_min"]
    features["amplitude_over_error"] = (
        features["amplitude"] / by_object["flux_err"].median())

    # Peak brightness as a magnitude. For a standard candle this is the single
    # most distance-sensitive number in the light curve.
    peak = features["flux_max"].where(features["flux_max"] > 0)
    features["peak_mag"] = ZEROPOINT - 2.5 * np.log10(peak)

    # chi-square against a constant flux: how significant the variability is,
    # independent of how bright the object happens to be.
    weight = 1.0 / obs["flux_err"] ** 2
    obs["_w"] = weight
    obs["_wf"] = weight * obs["flux"]
    sum_w = obs.groupby("object_id")["_w"].sum()
    weighted_mean = obs.groupby("object_id")["_wf"].sum() / sum_w
    obs["_chi"] = ((obs["flux"] - obs["object_id"].map(weighted_mean))
                   / obs["flux_err"]) ** 2
    features["reduced_chi2_constant"] = (
        obs.groupby("object_id")["_chi"].sum() / (features["n_obs"] - 1))

    # --- timescales, from detections only ---
    # A supernova detects in one burst; a variable star detects throughout.
    # The observed duration of a transient scales as (1 + z), which is the
    # cleanest redshift signal photometry alone can offer.
    detected = obs[obs["detected"] == 1]
    by_detected = detected.groupby("object_id")
    total_span = by_object["mjd"].max() - by_object["mjd"].min()
    detected_span = by_detected["mjd"].max() - by_detected["mjd"].min()
    features["n_detected"] = by_detected.size().reindex(features.index).fillna(0)
    features["detected_span"] = detected_span.reindex(features.index)
    features["detected_span_fraction"] = (
        detected_span / total_span).reindex(features.index)
    features["detected_mjd_std"] = (
        by_detected["mjd"].std().reindex(features.index))

    # Flux-weighted duration: the second moment of the detected light curve in
    # time. Robust to a few stray detections in a way that the raw span is not.
    positive_detected = detected[detected["flux"] > 0].copy()
    pd_group = positive_detected.groupby("object_id")
    positive_detected["_wf_t"] = (
        positive_detected["flux"] * positive_detected["mjd"])
    flux_sum = pd_group["flux"].sum()
    mean_time = (positive_detected.groupby("object_id")["_wf_t"].sum() / flux_sum)
    positive_detected["_var_t"] = (
        positive_detected["flux"]
        * (positive_detected["mjd"]
           - positive_detected["object_id"].map(mean_time)) ** 2)
    weighted_duration = np.sqrt(
        positive_detected.groupby("object_id")["_var_t"].sum() / flux_sum)
    features["flux_weighted_duration"] = weighted_duration.reindex(features.index)

    # Rise and fall time about the brightest detection.
    peak_time = pd_group.apply(
        lambda g: g.loc[g["flux"].idxmax(), "mjd"], include_groups=False)
    first_detection = by_detected["mjd"].min()
    last_detection = by_detected["mjd"].max()
    features["rise_time"] = (peak_time - first_detection).reindex(features.index)
    features["fall_time"] = (last_detection - peak_time).reindex(features.index)

    # --- per-passband block ---
    band_group = obs.groupby(["object_id", "passband"])
    band_table = band_group.agg(
        flux_max=("flux", "max"),
        flux_std=("flux", "std"),
        flux_median=("flux", "median"),
        flux_err_median=("flux_err", "median"),
        snr_max=("snr", "max"),
        n_detected=("detected", "sum"),
    )
    for column in band_table.columns:
        wide = band_table[column].unstack("passband").reindex(features.index)
        for band in range(N_BANDS):
            if band in wide.columns:
                features[f"{column}_band{band}"] = wide[band]

    # Colours: adjacent-band flux ratios in magnitudes. Redshift moves the
    # spectral energy distribution through the filters, so colour is the third
    # physical handle on z alongside brightness and timescale.
    band_peak = band_table["flux_max"].unstack("passband").reindex(features.index)
    band_peak = band_peak.where(band_peak > 0)
    for band in range(N_BANDS - 1):
        if band in band_peak.columns and band + 1 in band_peak.columns:
            features[f"color_{BAND_NAMES[band]}{BAND_NAMES[band + 1]}"] = (
                -2.5 * np.log10(band_peak[band] / band_peak[band + 1]))

    # Share of total peak flux per band: the same information as colour but
    # defined even when one band is at or below zero flux.
    peak_total = band_peak.sum(axis=1)
    for band in range(N_BANDS):
        if band in band_peak.columns:
            features[f"flux_share_band{band}"] = band_peak[band] / peak_total
    # A handful of objects never reach positive flux in any band; they have no
    # brightest band, so the feature is left missing rather than invented.
    any_positive = band_peak.notna().any(axis=1)
    brightest = pd.Series(np.nan, index=features.index, dtype="float")
    brightest.loc[any_positive] = band_peak.loc[any_positive].idxmax(axis=1)
    features["brightest_band"] = brightest

    return features.replace([np.inf, -np.inf], np.nan)


def load_lightcurve_features(rebuild: bool = False) -> pd.DataFrame:
    """Build the feature table once, then reuse it from the gitignored cache."""
    if FEATURE_CACHE.exists() and not rebuild:
        print(f"Reusing cached features from {FEATURE_CACHE.relative_to(REPO_ROOT)}")
        return pd.read_parquet(FEATURE_CACHE)
    observations = load_observations()
    print(f"Building light-curve features from {len(observations):,} "
          f"observations ...")
    features = build_lightcurve_features(observations)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_CACHE)
    print(f"Cached {features.shape[1]} features for {len(features):,} objects")
    return features


# --- metrics ------------------------------------------------------------------

def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Sample weights inversely proportional to class frequency, mean 1.

    These turn LightGBM's ordinary multi-class log loss into the class-balanced
    version used as Task 1's primary metric, so the model is trained on the
    same objective it is scored on.
    """
    counts = pd.Series(y).value_counts()
    weights = (1.0 / counts.reindex(y).to_numpy()).astype(float)
    return weights / weights.mean()


def photoz_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """The metrics photometric-redshift work actually reports.

    Scatter is quoted on dz = (z_pred - z_spec) / (1 + z_spec), because
    redshift errors grow with redshift and a raw residual would let the
    high-z objects dominate any average. sigma_NMAD is a median-based scatter
    that ignores the catastrophic tail; the outlier fraction counts that tail
    separately. RMSE is reported too, but on its own it hides which of the two
    is the problem.
    """
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    dz = (prediction - truth) / (1 + truth)
    bias = float(np.median(dz))
    return {
        "bias": bias,
        "sigma_nmad": float(1.4826 * np.median(np.abs(dz - bias))),
        "outlier_frac": float(np.mean(np.abs(dz) > OUTLIER_CUT)),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "mad": float(np.median(np.abs(prediction - truth))),
        "n": int(len(truth)),
    }
