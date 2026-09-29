"""Task 4 - the Task 2 redshift regression under a deliberate distribution shift.

Two shifts, run with the same features, model and metrics as Task 2:

  survey depth  train on deep-drilling-field objects, test on wide-field ones
  redshift      train on z_spec < 0.4, test on z_spec >= 0.4

Measuring a "drop" needs care, because two different things can make a shifted
test set score worse, and only one of them is a transfer failure:

  1. the target domain is intrinsically harder - wide-field light curves are
     sparser and noisier, and high-redshift objects are fainter;
  2. the model genuinely fails to carry over what it learned.

Reporting the shifted score alone cannot tell these apart. Each scenario
therefore runs a matched control: the same number of training objects, drawn at
random from everything outside the test set, evaluated on exactly the same test
objects. The control absorbs (1), so the gap between the shifted model and the
control is (2). The photo-z baseline is carried through as well, since it does
not learn anything and so tracks how hard the test domain is by itself.

The validation fold is always carved out of the *training* distribution, never
the target one. Using target-domain objects to early-stop would be assuming
away the problem: in the real setting that motivates this task - training on
the objects that have spectra and applying the model to the ones that do not -
target-domain labels are exactly what is missing.

Run from the repository root:
    uv run --with lightgbm --with scikit-learn python \\
      submissions/rorikang/week02/hw2_task4_shift.py
"""

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hw2_common import (BLUE, GREEN, INK, INK_MUTED, ORANGE, OUTLIER_CUT,
                        OUT_DIR, PURPLE, RANDOM_STATE, extragalactic_mask,
                        load_lightcurve_features, load_metadata, photoz_metrics,
                        style_axes)
from hw2_task2_redshift import LGB_PARAMS

REDSHIFT_CUT = 0.4
TARGET_TEST_FRACTION = 0.5   # half the target domain is held out for testing
VALIDATION_FRACTION = 0.2    # carved from whichever training set is in use

SHIFT_DIAGNOSTIC_FEATURES = ["n_obs", "n_detected", "flux_err_median",
                             "detected_span", "snr_max", "hostgal_photoz"]


def build_sample():
    """Extragalactic features, target, and the two domain labels."""
    metadata = load_metadata()
    features = load_lightcurve_features()
    host = metadata.set_index("object_id").loc[features.index]
    keep = extragalactic_mask(host).to_numpy()
    host, features = host[keep], features[keep]

    features = features.assign(
        hostgal_photoz=host["hostgal_photoz"].astype(float),
        hostgal_photoz_err=host["hostgal_photoz_err"].astype(float))
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, host["hostgal_specz"].astype(float), host


def fit_pair(x_train, y_train):
    """Direct and residual regressors, early-stopped inside the training domain.

    Both are reported in every scenario because they fail differently. A tree
    ensemble predicting z directly can only ever output a value it saw in
    training, so under a redshift shift it is capped at the training maximum
    however clear the evidence. The residual model predicts a correction to be
    added to the photometric redshift, which is not capped, so on the face of
    it it should survive extrapolation.

    It does under the survey-depth shift and it does not under the redshift
    shift, and `explain_redshift_shift` below measures why: selecting a
    training set on the target variable poisons the correction itself. Every
    training object with a high photometric redshift is, by construction of the
    cut, a photo-z overestimate, so the model learns that a high photo-z must
    be corrected sharply downward - and then applies that lesson to the
    genuinely high-redshift objects it was never shown.
    """
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE)
    stop = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]

    direct = lgb.LGBMRegressor(**LGB_PARAMS)
    direct.fit(x_fit, y_fit, eval_X=x_val, eval_y=y_val, eval_metric="l2",
               callbacks=stop)

    residual = lgb.LGBMRegressor(**LGB_PARAMS)
    residual.fit(x_fit, y_fit - x_fit["hostgal_photoz"],
                 eval_X=x_val, eval_y=y_val - x_val["hostgal_photoz"],
                 eval_metric="l2", callbacks=stop)
    return direct, residual


def predictions_for(direct, residual, x_test) -> dict:
    return {
        "Photo-z as-is (baseline)": x_test["hostgal_photoz"].to_numpy(),
        "LightGBM direct": direct.predict(x_test),
        "LightGBM on residual":
            x_test["hostgal_photoz"].to_numpy() + residual.predict(x_test),
    }


def run_scenario(name, source_mask, target_mask, features, y, rng):
    """One shift: shifted model, matched control, shared test set."""
    source_ids = features.index[source_mask]
    target_ids = features.index[target_mask]

    test_ids = pd.Index(rng.choice(
        target_ids, size=int(round(TARGET_TEST_FRACTION * len(target_ids))),
        replace=False))
    x_test, y_test = features.loc[test_ids], y.loc[test_ids]

    # Shifted training set: the whole source domain, which shares no object
    # with the test set by construction.
    shifted_ids = source_ids
    # Control: the same number of objects drawn at random from everything that
    # is not being tested on, so it contains target-domain objects too.
    available = features.index.difference(test_ids)
    control_ids = pd.Index(rng.choice(
        available, size=min(len(shifted_ids), len(available)), replace=False))

    print(f"\n=== {name} ===")
    print(f"  shifted training set {len(shifted_ids):,} (source domain only)")
    print(f"  control training set {len(control_ids):,} (random, same size, "
          f"contains target-domain objects)")
    print(f"  shared test set      {len(test_ids):,} (target domain)")

    results = {}
    for label, train_ids in (("shifted", shifted_ids), ("control", control_ids)):
        direct, residual = fit_pair(features.loc[train_ids], y.loc[train_ids])
        for model_name, prediction in predictions_for(
                direct, residual, x_test).items():
            if model_name.startswith("Photo-z") and label == "control":
                continue  # the baseline does not depend on the training set
            key = ("Photo-z as-is (baseline)" if model_name.startswith("Photo-z")
                   else f"{model_name} [{label} training]")
            results[key] = prediction

    table = pd.DataFrame({key: photoz_metrics(y_test.to_numpy(), p)
                          for key, p in results.items()}).T
    table.index.name = f"{name} (test set = target domain)"
    print()
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    report_gap(table, "LightGBM direct")
    report_gap(table, "LightGBM on residual")
    return table, results, y_test.to_numpy(), x_test


def report_gap(table, model_name) -> None:
    """Separate 'the domain is harder' from 'the model did not transfer'."""
    shifted = table.loc[f"{model_name} [shifted training]"]
    control = table.loc[f"{model_name} [control training]"]
    baseline = table.loc["Photo-z as-is (baseline)"]
    print(f"  {model_name}: outlier fraction {control['outlier_frac']:.4f} "
          f"in-domain -> {shifted['outlier_frac']:.4f} shifted "
          f"({shifted['outlier_frac'] - control['outlier_frac']:+.4f} from the "
          f"shift alone; the photo-z baseline sits at "
          f"{baseline['outlier_frac']:.4f} on the same objects).")


def describe_shift(features, source_mask, target_mask, name) -> pd.DataFrame:
    """What actually differs between the two domains, feature by feature."""
    source = features.loc[source_mask, SHIFT_DIAGNOSTIC_FEATURES].median()
    target = features.loc[target_mask, SHIFT_DIAGNOSTIC_FEATURES].median()
    frame = pd.DataFrame({"source_median": source, "target_median": target})
    frame["ratio"] = frame["target_median"] / frame["source_median"]
    frame.index.name = name
    return frame


def explain_redshift_shift(features, y, cut) -> None:
    """Measure the mechanism behind the redshift-shift failure.

    The comparison is the median correction (z_spec - z_photo) that the two
    domains actually require, restricted to objects whose photometric redshift
    is high. If the training domain demands a large negative correction where
    the test domain demands none, then any model that learns the correction
    from the training domain must fail on the test domain, regardless of
    architecture or regularisation.
    """
    photoz = features["hostgal_photoz"]
    high_photoz = photoz > 0.5
    source, target = y < cut, y >= cut
    correction = y - photoz
    print("\nWhy the redshift shift is worse than a plain extrapolation "
          "problem.")
    print(f"  Among objects with photo-z > 0.5, the correction "
          f"(z_spec - z_photo) they need is")
    print(f"    {(source & high_photoz).sum():>5} training objects "
          f"(z_spec < {cut}): median {correction[source & high_photoz].median():+.3f}")
    print(f"    {(target & high_photoz).sum():>5} test objects "
          f"(z_spec >= {cut}): median {correction[target & high_photoz].median():+.3f}")
    print("  Cutting the training set on the target variable means every "
          "high-photo-z\n  training object is a photo-z overestimate. The "
          "model learns to subtract\n  roughly 2 from a high photo-z, and "
          "that is precisely wrong for the test\n  domain, where a high "
          "photo-z is simply correct. This is why the residual\n  "
          "parameterisation rescues the depth shift but not this one.")


def main() -> None:
    features, y, host = build_sample()
    rng = np.random.default_rng(RANDOM_STATE)

    scenarios = {
        "Survey depth: train deep-drilling, test wide-field": (
            (host["ddf"] == 1).to_numpy(), (host["ddf"] == 0).to_numpy()),
        f"Redshift: train z < {REDSHIFT_CUT}, test z >= {REDSHIFT_CUT}": (
            (y < REDSHIFT_CUT).to_numpy(), (y >= REDSHIFT_CUT).to_numpy()),
    }

    tables, payloads, diagnostics = {}, {}, []
    for name, (source_mask, target_mask) in scenarios.items():
        diagnostic = describe_shift(features, source_mask, target_mask, name)
        print(f"\nWhat differs between the domains - "
              f"{name.split(':')[0].lower()}:")
        print(diagnostic.to_string(float_format=lambda v: f"{v:.3f}"))
        diagnostics.append(diagnostic.assign(scenario=name))

        table, results, truth, x_test = run_scenario(
            name, source_mask, target_mask, features, y, rng)
        tables[name] = table
        payloads[name] = (results, truth)
        if name.startswith("Redshift"):
            explain_redshift_shift(features, y, REDSHIFT_CUT)

    combined = pd.concat(
        [t.assign(scenario=name) for name, t in tables.items()])
    combined.to_csv(OUT_DIR / "task4_shift_metrics.csv")
    pd.concat(diagnostics).to_csv(OUT_DIR / "task4_domain_differences.csv")

    plot(payloads, tables, OUT_DIR / "task4_distribution_shift.png")
    print(f"\nWrote task4_distribution_shift.png, task4_shift_metrics.csv, "
          f"task4_domain_differences.csv to {OUT_DIR.name}/")


def plot(payloads, tables, output_path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    limits = (0, 3.6)

    for row, (name, (results, truth)) in enumerate(payloads.items()):
        for column, key in enumerate(("LightGBM direct [control training]",
                                      "LightGBM direct [shifted training]")):
            ax = axes[row, column]
            prediction = results[key]
            outlier = np.abs((prediction - truth) / (1 + truth)) > OUTLIER_CUT
            ax.scatter(truth[~outlier], prediction[~outlier], s=6, alpha=0.3,
                       color=BLUE, linewidths=0, label="Within 0.15")
            ax.scatter(truth[outlier], prediction[outlier], s=12, alpha=0.75,
                       color=ORANGE, linewidths=0,
                       label=f"Catastrophic ({outlier.sum()})")
            ax.plot(limits, limits, color=INK_MUTED, linestyle="--",
                    linewidth=1)
            metrics = tables[name].loc[key]
            ax.set(xlim=limits, ylim=limits)
            ax.set_xlabel("Spectroscopic redshift", fontsize=9.5,
                          color=INK_MUTED)
            ax.set_ylabel("Predicted redshift", fontsize=9.5, color=INK_MUTED)
            ax.set_title(
                f"{'Control training' if 'control' in key else 'Shifted training'}"
                f"\nsigma_NMAD {metrics['sigma_nmad']:.4f}   outliers "
                f"{metrics['outlier_frac']:.2%}   RMSE {metrics['rmse']:.3f}",
                fontsize=10, color=INK, loc="left")
            ax.legend(loc="upper left", frameon=False, fontsize=8,
                      labelcolor=INK)
            style_axes(ax)

        # Metric comparison for this scenario.
        ax = axes[row, 2]
        order = ["Photo-z as-is (baseline)",
                 "LightGBM direct [control training]",
                 "LightGBM direct [shifted training]",
                 "LightGBM on residual [control training]",
                 "LightGBM on residual [shifted training]"]
        values = [tables[name].loc[key, "outlier_frac"] for key in order]
        colors = [INK_MUTED, BLUE, ORANGE, GREEN, PURPLE]
        ax.barh(range(len(order)), values, color=colors)
        for index, value in enumerate(values):
            ax.text(value, index, f" {value:.3f}", va="center", fontsize=8.5,
                    color=INK)
        short = [key.replace("LightGBM ", "").replace(" training]", "]")
                 for key in order]
        ax.set_yticks(range(len(order)), short, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_xlabel("Catastrophic outlier fraction", fontsize=9.5,
                      color=INK_MUTED)
        ax.set_title("Lower is better", fontsize=10, color=INK, loc="left")
        style_axes(ax)

        axes[row, 0].text(-0.28, 0.5, name.split(":")[0], transform=
                          axes[row, 0].transAxes, rotation=90, va="center",
                          ha="center", fontsize=11, color=INK)

    fig.suptitle("Task 4 - the same regression under distribution shift; both "
                 "panels in a row share one test set", fontsize=13, color=INK,
                 y=0.985)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
