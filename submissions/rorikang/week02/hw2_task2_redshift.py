"""Task 2 - spectroscopic redshift from photo-z plus light-curve features.

Pipeline:

  sample     5,523 extragalactic objects (z = 0 is a Galactic sentinel)
  features   70 light-curve features + hostgal_photoz + hostgal_photoz_err
  split      random 60 / 20 / 20 train / validation / test, fixed seed
  metric     catastrophic-outlier fraction (primary), with sigma_NMAD as a
             guard on the core and RMSE alongside, all on
             dz = (z_pred - z_spec) / (1 + z_spec)
  model      LightGBM regressor, early-stopped on validation
  baseline   the photometric redshift itself, used unchanged
  final      one evaluation on the test set

Metric choice. Redshift errors grow with redshift, so the residual is quoted
normalised by (1 + z_spec); this is the standard in photo-z work and stops the
few z > 2 objects from dominating. The error distribution is heavy-tailed, so
one number will not do, and the two that matter pull in opposite directions:
sigma_NMAD measures the core scatter and the outlier fraction (|dz| > 0.15)
measures the tail.

The outlier fraction is the primary metric, fixed before any model was fitted,
for a physical reason. A redshift that is wrong by 0.02 costs a little
precision on a luminosity; a redshift that is wrong by 0.5 puts an object in
the wrong epoch of the universe and corrupts anything built on it. At 12% the
catastrophic rate is the dominant systematic in this catalogue, so it is what a
model should be asked to reduce. sigma_NMAD is carried as a guard rather than a
target: a method that halves the tail while doubling the core scatter has not
obviously helped, and the results below show that this is exactly the trade
the plain regressors make. RMSE is reported because it is what the least-
squares training loss optimises, but it is dominated by the tail and so says
little on its own.

Split choice. The split is random rather than stratified: the target is
continuous, the sample is 5,523 objects, and a 20% test fold covers the
redshift range densely enough that stratifying by redshift bin changes the
metrics in the third decimal. Task 4 deliberately breaks this assumption.

Baseline choice. The baseline is not a constant or the training mean. It is
hostgal_photoz used as-is, because that is what an astronomer already has for
free. A model only earns its place if it beats "trust the photo-z".

Run from the repository root:
    uv run --with lightgbm --with scikit-learn python \\
      submissions/rorikang/week02/hw2_task2_redshift.py
"""

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from hw2_common import (BLUE, GREEN, INK, INK_MUTED, ORANGE, OUTLIER_CUT,
                        OUT_DIR, PURPLE, RANDOM_STATE, extragalactic_mask,
                        load_lightcurve_features, load_metadata, photoz_metrics,
                        style_axes)

TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.25  # of the remaining 80%, giving 60 / 20 / 20

LGB_PARAMS = dict(objective="regression", n_estimators=3000, learning_rate=0.03,
                  num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


def build_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Extragalactic feature matrix and spectroscopic-redshift target."""
    metadata = load_metadata()
    features = load_lightcurve_features()
    host = metadata.set_index("object_id").loc[features.index]

    keep = extragalactic_mask(host).to_numpy()
    print(f"{keep.sum():,} extragalactic objects; {(~keep).sum():,} Galactic "
          f"objects dropped (their z = 0 is a sentinel, not a measurement)")
    host, features = host[keep], features[keep]

    report_distmod_redundancy(host)

    features = features.assign(
        hostgal_photoz=host["hostgal_photoz"].astype(float),
        hostgal_photoz_err=host["hostgal_photoz_err"].astype(float))
    return (features.replace([np.inf, -np.inf], np.nan),
            host["hostgal_specz"].astype(float))


def report_distmod_redundancy(host: pd.DataFrame) -> None:
    """Show, rather than assert, that distmod carries no new information.

    The photo-z to distmod relation is monotonic but non-linear, so a Pearson
    correlation understates it. The honest checks are the rank correlation and
    the scatter of distmod inside narrow photo-z bins.
    """
    photoz = host["hostgal_photoz"].astype(float).to_numpy()
    distmod = host["distmod"].astype(float).to_numpy()
    rank = spearmanr(photoz, distmod).statistic
    order = np.argsort(photoz)
    within = np.median([distmod[b].std() for b in np.array_split(order, 300)
                        if len(b) > 3])
    print(f"spearman(hostgal_photoz, distmod) = {rank:.6f}; distmod scatter "
          f"inside narrow photo-z bins {within:.4f} mag against "
          f"{distmod.std():.3f} mag overall.")
    print("distmod is therefore excluded: it is photo-z re-expressed in "
          "magnitudes, not independent information.\n")


def fit(x_train, y_train, x_val, y_val, columns=None):
    """LightGBM regressor early-stopped on the validation fold."""
    columns = list(x_train.columns) if columns is None else columns
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(x_train[columns], y_train,
              eval_X=x_val[columns], eval_y=y_val, eval_metric="l2",
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
    return model, columns


def main() -> None:
    features, y = build_sample()

    x_rest, x_test, y_rest, y_test = train_test_split(
        features, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE)
    print(f"Train {len(x_train):,} | validation {len(x_val):,} | "
          f"test {len(x_test):,}\n")

    photoz_columns = ["hostgal_photoz", "hostgal_photoz_err"]

    # Three models, all chosen on the validation fold only.
    #
    # The residual model predicts dz = z_spec - z_photo rather than z_spec
    # itself. Trees cannot extrapolate: a model predicting z directly can never
    # output a value outside its training range, which flattens the high-z end.
    # Learning the correction sidesteps that, since the correction is small and
    # centred near zero everywhere.
    direct, _ = fit(x_train, y_train, x_val, y_val)
    photoz_only, _ = fit(x_train, y_train, x_val, y_val, photoz_columns)
    residual = lgb.LGBMRegressor(**LGB_PARAMS)
    residual.fit(x_train, y_train - x_train["hostgal_photoz"],
                 eval_X=x_val, eval_y=y_val - x_val["hostgal_photoz"],
                 eval_metric="l2",
                 callbacks=[lgb.early_stopping(100, verbose=False),
                            lgb.log_evaluation(0)])

    # The regressors below turn out to trade core precision for tail
    # robustness. The hybrid keeps both: it leaves the photometric redshift
    # alone unless the model disagrees with it by more than a threshold, so the
    # photo-z's excellent core survives and only the suspect objects are
    # overwritten. The threshold is picked on validation, never on test.
    def hybrid(frame, model_prediction, threshold):
        photoz = frame["hostgal_photoz"].to_numpy()
        disagreement = np.abs(model_prediction - photoz) / (1 + photoz)
        return np.where(disagreement > threshold, model_prediction, photoz)

    grid = np.round(np.arange(0.02, 0.301, 0.005), 3)
    val_direct = direct.predict(x_val)
    sweep = pd.DataFrame(
        [photoz_metrics(y_val.to_numpy(), hybrid(x_val, val_direct, t))
         | {"threshold": t} for t in grid]).set_index("threshold")
    threshold = float(sweep["outlier_frac"].idxmin())
    print(f"Hybrid disagreement threshold chosen on validation: "
          f"{threshold:.3f} (validation outlier fraction "
          f"{sweep.loc[threshold, 'outlier_frac']:.4f})\n")

    def predict(frame: pd.DataFrame) -> dict:
        model_prediction = direct.predict(frame)
        return {
            "Photo-z as-is (baseline)": frame["hostgal_photoz"].to_numpy(),
            "LightGBM, photo-z only": photoz_only.predict(frame[photoz_columns]),
            "LightGBM, photo-z + light curves": model_prediction,
            "LightGBM on residual, photo-z + light curves":
                frame["hostgal_photoz"].to_numpy() + residual.predict(frame),
            f"Hybrid: photo-z, overwritten if model disagrees > {threshold:g}":
                hybrid(frame, model_prediction, threshold),
        }

    validation_table = pd.DataFrame(
        {name: photoz_metrics(y_val.to_numpy(), p)
         for name, p in predict(x_val).items()}).T
    validation_table.index.name = "model (validation)"
    print(validation_table.to_string(float_format=lambda v: f"{v:.4f}"))

    # Selection rule, fixed in advance: lowest catastrophic-outlier fraction
    # among candidates that do not let the core scatter blow out past 1.5x the
    # baseline. The guard is what stops a model from "winning" by smearing
    # every redshift toward the middle of the distribution.
    candidates = validation_table.drop("Photo-z as-is (baseline)")
    guard = 1.5 * validation_table.loc["Photo-z as-is (baseline)", "sigma_nmad"]
    eligible = candidates[candidates["sigma_nmad"] <= guard]
    if eligible.empty:
        eligible = candidates
        print("No candidate stayed inside the sigma_NMAD guard; selecting on "
              "outlier fraction alone.")
    # Differences in outlier fraction smaller than one binomial standard error
    # on the validation fold are not real, so ties within that band are broken
    # on the guard metric instead of on row order.
    lowest = eligible["outlier_frac"].min()
    noise = float(np.sqrt(lowest * (1 - lowest) / len(y_val)))
    tied = eligible[eligible["outlier_frac"] <= lowest + noise]
    best_name = tied["sigma_nmad"].idxmin()
    print(f"\nChosen on validation by outlier fraction (subject to "
          f"sigma_NMAD <= {guard:.4f}), with ties inside one standard error "
          f"({noise:.4f}) broken on sigma_NMAD:\n  {best_name}")
    print(f"Trees kept: direct {direct.best_iteration_}, "
          f"residual {residual.best_iteration_}, "
          f"photo-z only {photoz_only.best_iteration_}\n")

    sweep.to_csv(OUT_DIR / "task2_hybrid_threshold_sweep.csv")

    # --- final: the test set, read once ---
    test_predictions = predict(x_test)
    test_table = pd.DataFrame(
        {name: photoz_metrics(y_test.to_numpy(), p)
         for name, p in test_predictions.items()}).T
    test_table.index.name = "model (test)"
    print("--- final test-set evaluation (one look, nothing tuned after) ---")
    print(test_table.to_string(float_format=lambda v: f"{v:.4f}"))

    baseline = test_table.loc["Photo-z as-is (baseline)"]
    best = test_table.loc[best_name]
    print("\nChosen model against the photo-z baseline on the test set "
          "(positive = the model reduced the number, which is better for all "
          "three):")
    for key, label in (("outlier_frac", "outlier fraction"),
                       ("sigma_nmad", "sigma_NMAD    "),
                       ("rmse", "RMSE          ")):
        reduction = 1 - best[key] / baseline[key]
        print(f"  {label}  {baseline[key]:.4f} -> {best[key]:.4f}  "
              f"({reduction:+.1%})")

    truth = y_test.to_numpy()
    was_catastrophic = np.abs(
        (test_predictions["Photo-z as-is (baseline)"] - truth)
        / (1 + truth)) > OUTLIER_CUT
    now_fine = np.abs(
        (test_predictions[best_name] - truth) / (1 + truth)) <= OUTLIER_CUT
    print(f"\nOf {was_catastrophic.sum()} photo-z catastrophic failures in the "
          f"test set, {(was_catastrophic & now_fine).sum()} are recovered.")
    print(f"The model introduces {((~was_catastrophic) & ~now_fine).sum()} new "
          f"failures on objects the photo-z had right.")

    importance = pd.Series(direct.feature_importances_,
                           index=features.columns).sort_values(ascending=False)
    print(f"\nTop features (direct model): "
          f"{', '.join(importance.head(6).index)}")

    pd.concat([validation_table, test_table]).to_csv(
        OUT_DIR / "task2_metrics.csv")
    importance.to_csv(OUT_DIR / "task2_feature_importance.csv",
                      header=["importance"])
    plot(truth, test_predictions, best_name, test_table, importance,
         OUT_DIR / "task2_redshift.png")
    print(f"\nWrote task2_redshift.png, task2_metrics.csv, "
          f"task2_feature_importance.csv to {OUT_DIR.name}/")


def plot(truth, predictions, best_name, table, importance, output_path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    limits = (0, 3.6)

    # Row 1: predicted against true, baseline and model side by side.
    for ax, name in ((axes[0, 0], "Photo-z as-is (baseline)"),
                     (axes[0, 1], best_name)):
        prediction = predictions[name]
        outlier = np.abs((prediction - truth) / (1 + truth)) > OUTLIER_CUT
        ax.scatter(truth[~outlier], prediction[~outlier], s=7, alpha=0.35,
                   color=BLUE, linewidths=0, label="Within 0.15")
        ax.scatter(truth[outlier], prediction[outlier], s=14, alpha=0.8,
                   color=ORANGE, linewidths=0,
                   label=f"Catastrophic ({outlier.sum()})")
        ax.plot(limits, limits, color=INK_MUTED, linestyle="--", linewidth=1)
        line = np.linspace(*limits, 50)
        for sign in (1, -1):
            ax.plot(line, line + sign * OUTLIER_CUT * (1 + line),
                    color=INK_MUTED, linestyle=":", linewidth=0.9)
        row = table.loc[name]
        ax.set(xlim=limits, ylim=limits)
        ax.set_xlabel("Spectroscopic redshift", fontsize=10, color=INK_MUTED)
        ax.set_ylabel("Predicted redshift", fontsize=10, color=INK_MUTED)
        ax.set_title(f"{name}\nsigma_NMAD {row['sigma_nmad']:.4f}   "
                     f"outliers {row['outlier_frac']:.2%}   "
                     f"RMSE {row['rmse']:.4f}",
                     fontsize=10.5, color=INK, loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
        style_axes(ax)

    # Residuals against true redshift, for the chosen model.
    ax = axes[0, 2]
    dz = (predictions[best_name] - truth) / (1 + truth)
    outlier = np.abs(dz) > OUTLIER_CUT
    ax.scatter(truth[~outlier], dz[~outlier], s=7, alpha=0.35, color=BLUE,
               linewidths=0)
    ax.scatter(truth[outlier], dz[outlier], s=14, alpha=0.8, color=ORANGE,
               linewidths=0)
    ax.axhline(0, color=INK_MUTED, linestyle="--", linewidth=1)
    for sign in (1, -1):
        ax.axhline(sign * OUTLIER_CUT, color=INK_MUTED, linestyle=":",
                   linewidth=0.9)
    edges = np.array([0, 0.15, 0.3, 0.5, 0.8, 1.2, 3.6])
    centres = 0.5 * (edges[:-1] + edges[1:])
    index = np.clip(np.digitize(truth, edges) - 1, 0, len(edges) - 2)
    running = [np.median(dz[index == b]) if (index == b).sum() > 5 else np.nan
               for b in range(len(centres))]
    ax.plot(centres, running, color=PURPLE, linewidth=2, marker="o",
            markersize=4, label="Median residual in bin")
    ax.set(xlim=limits, ylim=(-0.75, 0.75))
    ax.set_xlabel("Spectroscopic redshift", fontsize=10, color=INK_MUTED)
    ax.set_ylabel(r"$\Delta z / (1 + z_{spec})$", fontsize=10, color=INK_MUTED)
    ax.set_title("Residuals of the chosen model", fontsize=10.5, color=INK,
                 loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
    style_axes(ax)

    # Row 2: residual distribution, scatter against redshift, importances.
    ax = axes[1, 0]
    bins = np.linspace(-0.6, 0.6, 90)
    for name, color in (("Photo-z as-is (baseline)", ORANGE),
                        (best_name, BLUE)):
        values = (predictions[name] - truth) / (1 + truth)
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, color=color,
                label=f"{name}\nsigma_NMAD "
                      f"{table.loc[name, 'sigma_nmad']:.4f}")
    for sign in (1, -1):
        ax.axvline(sign * OUTLIER_CUT, color=INK_MUTED, linestyle=":",
                   linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta z / (1 + z_{spec})$", fontsize=10, color=INK_MUTED)
    ax.set_ylabel("Objects (log scale)", fontsize=10, color=INK_MUTED)
    ax.set_title("Residual distribution; dotted lines are the 0.15 cut",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK)
    style_axes(ax)

    ax = axes[1, 1]
    positions = np.arange(len(edges) - 1)
    counts = [int((index == b).sum()) for b in positions]
    for offset, (name, color) in zip((-0.2, 0.2),
                                     (("Photo-z as-is (baseline)", ORANGE),
                                      (best_name, BLUE))):
        values = [photoz_metrics(truth[index == b],
                                 predictions[name][index == b])["sigma_nmad"]
                  if counts[b] > 5 else np.nan for b in positions]
        ax.bar(positions + offset, values, width=0.38, color=color,
               label=name.split(",")[0])
    ax.set_xticks(positions, [f"{lo:g}-{hi:g}\nn={n}" for lo, hi, n
                              in zip(edges[:-1], edges[1:], counts)],
                  fontsize=8)
    ax.set_xlabel("Spectroscopic redshift bin", fontsize=10, color=INK_MUTED)
    ax.set_ylabel("sigma_NMAD in bin", fontsize=10, color=INK_MUTED)
    ax.set_title("Where the gain comes from", fontsize=10.5, color=INK,
                 loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    style_axes(ax)

    ax = axes[1, 2]
    top = importance.head(14)[::-1]
    colors = [GREEN if name.startswith("hostgal") else BLUE
              for name in top.index]
    ax.barh(range(len(top)), top.to_numpy(), color=colors)
    ax.set_yticks(range(len(top)), top.index, fontsize=8)
    ax.set_xlabel("LightGBM split count", fontsize=10, color=INK_MUTED)
    ax.set_title("What the model uses (green = host metadata)",
                 fontsize=10.5, color=INK, loc="left")
    style_axes(ax)

    fig.suptitle("Task 2 - spectroscopic redshift from photo-z and light curves "
                 "(test set)", fontsize=13, color=INK, y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
