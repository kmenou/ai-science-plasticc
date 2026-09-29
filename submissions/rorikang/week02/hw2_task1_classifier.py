"""Task 1 - gradient-boosted classification of the 14 PLAsTiCC classes.

The full ML pipeline, in order:

  features   70 light-curve features (hw2_common) + 5 host/extinction columns
  split      stratified 60 / 20 / 20 train / validation / test, by object
  metric     class-balanced multi-class log loss (the PLAsTiCC metric)
  model      LightGBM, early-stopped on validation
  final      one evaluation on the test set, at the very end

Why these choices are defensible for this dataset:

* The split is by object because every row is one object and objects are
  independent; there is no group structure to leak across. Stratification is
  not optional here - Mira variables have 30 examples in 7,848, so a plain
  random split can easily give a test fold with two of them.
* The metric is class-balanced log loss because the class frequencies in this
  catalogue are an artefact of how the challenge was sampled, not of the sky,
  and because downstream use (triggering spectroscopic follow-up) needs
  calibrated probabilities, not hard labels. Accuracy would be maximised by a
  model that ignores every rare class, which is the opposite of what a
  transient broker wants. Accuracy and macro-F1 are reported alongside as
  secondary, interpretable numbers.
* Validation is used for early stopping and for every look at the results.
  The test set is read once, at the end of `main`, and nothing is tuned after.

Run from the repository root:
    uv run --with lightgbm --with scikit-learn python \\
      submissions/rorikang/week02/hw2_task1_classifier.py
"""

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from hw2_common import (BLUE, INK, INK_MUTED, ORANGE, OUT_DIR,
                        PLASTICC_CLASSES, RANDOM_STATE, class_balanced_weights,
                        load_lightcurve_features, load_metadata, style_axes)

# Host and extinction columns a real survey would have before spectroscopy.
# hostgal_specz is excluded - it is a spectroscopic measurement that exists for
# a small minority of objects and is the target of Task 2. distmod is excluded
# because it is a deterministic function of hostgal_photoz in this catalogue.
METADATA_FEATURES = ["hostgal_photoz", "hostgal_photoz_err", "mwebv", "ddf",
                     "abs_gal_b"]

TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.25  # of the remaining 80%, giving 60 / 20 / 20


def assemble(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and integer label vector, aligned on object_id."""
    features = load_lightcurve_features()
    host = metadata.set_index("object_id").loc[features.index]
    host = host.assign(abs_gal_b=host["gal_b"].abs())
    features = features.join(host[METADATA_FEATURES])
    labels = host["target"].astype(int)
    return features.replace([np.inf, -np.inf], np.nan), labels


def balanced_log_loss(y_true, probability, classes) -> float:
    """Mean over classes of the mean log loss within that class.

    The PLAsTiCC challenge metric with flat per-class weights. Reported instead
    of plain log loss so that 30 Miras count as much as 2,313 SNe Ia.
    """
    per_class = []
    for index, label in enumerate(classes):
        mask = y_true == label
        if mask.sum() == 0:
            continue
        confidence = np.clip(probability[mask, index], 1e-15, 1 - 1e-15)
        per_class.append(-np.mean(np.log(confidence)))
    return float(np.mean(per_class))


def score(y_true, probability, classes) -> dict:
    predicted = classes[np.argmax(probability, axis=1)]
    return {
        "balanced_log_loss": balanced_log_loss(y_true, probability, classes),
        "log_loss": float(log_loss(y_true, probability, labels=list(classes))),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
    }


def main() -> None:
    metadata = load_metadata()
    features, labels = assemble(metadata)
    classes = np.array(sorted(labels.unique()))
    label_index = {label: i for i, label in enumerate(classes)}

    print(f"Features: {features.shape[1]} columns "
          f"({features.shape[1] - len(METADATA_FEATURES)} from photometry, "
          f"{len(METADATA_FEATURES)} from metadata)")
    print(f"Objects:  {len(features):,} across {len(classes)} classes\n")

    # --- split: 60 / 20 / 20, stratified, fixed seed ---
    x_rest, x_test, y_rest, y_test = train_test_split(
        features, labels, test_size=TEST_FRACTION, stratify=labels,
        random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=VALIDATION_FRACTION, stratify=y_rest,
        random_state=RANDOM_STATE)
    print(f"Train {len(x_train):,} | validation {len(x_val):,} | "
          f"test {len(x_test):,} (test held back until the final evaluation)\n")

    y_train_i = y_train.map(label_index).to_numpy()
    y_val_i = y_val.map(label_index).to_numpy()

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(classes), n_estimators=3000,
        learning_rate=0.05, num_leaves=48, min_child_samples=20,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(
        x_train, y_train_i,
        sample_weight=class_balanced_weights(y_train_i),
        eval_X=x_val, eval_y=y_val_i,
        eval_sample_weight=[class_balanced_weights(y_val_i)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(0)])
    print(f"Early stopping chose {model.best_iteration_} trees "
          f"(cap was {model.n_estimators}).\n")

    # --- validation: every comparison and sanity check happens here ---
    validation_probability = model.predict_proba(x_val)
    prior = (pd.Series(y_train).value_counts(normalize=True)
             .reindex(classes).to_numpy())
    baselines = {
        "Training-prior constant": np.tile(prior, (len(y_val), 1)),
        "Uniform constant": np.full((len(y_val), len(classes)), 1 / len(classes)),
        "LightGBM": validation_probability,
    }
    table = pd.DataFrame(
        {name: score(y_val.to_numpy(), p, classes)
         for name, p in baselines.items()}).T
    table.index.name = "model (validation)"
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    photometry_only = [c for c in features.columns if c not in METADATA_FEATURES]
    ablation = lgb.LGBMClassifier(**model.get_params())
    ablation.set_params(n_estimators=model.best_iteration_)
    ablation.fit(x_train[photometry_only], y_train_i,
                 sample_weight=class_balanced_weights(y_train_i))
    ablation_score = score(y_val.to_numpy(),
                           ablation.predict_proba(x_val[photometry_only]), classes)
    print(f"\nPhotometry only, no host metadata: balanced log loss "
          f"{ablation_score['balanced_log_loss']:.4f} versus "
          f"{table.loc['LightGBM', 'balanced_log_loss']:.4f} with it - "
          f"host redshift is doing real work.")

    # --- final: the test set, read once ---
    test_probability = model.predict_proba(x_test)
    final = score(y_test.to_numpy(), test_probability, classes)
    print("\n--- final test-set evaluation (one look, nothing tuned after) ---")
    for key, value in final.items():
        print(f"  {key:>18}: {value:.4f}")
    gap = final["balanced_log_loss"] - table.loc["LightGBM", "balanced_log_loss"]
    print(f"  validation -> test change in balanced log loss: {gap:+.4f}")

    per_class = per_class_table(y_test.to_numpy(), test_probability, classes)
    print("\nPer-class test results:")
    print(per_class.to_string(float_format=lambda v: f"{v:.3f}"))

    importance = pd.Series(model.feature_importances_,
                           index=features.columns).sort_values(ascending=False)
    pd.concat([table, pd.DataFrame({"LightGBM (test)": final}).T]).to_csv(
        OUT_DIR / "task1_scores.csv")
    per_class.to_csv(OUT_DIR / "task1_per_class.csv")
    importance.to_csv(OUT_DIR / "task1_feature_importance.csv",
                      header=["importance"])

    plot(y_test.to_numpy(), test_probability, classes, per_class, importance,
         model, final, OUT_DIR / "task1_classifier.png")
    print(f"\nWrote task1_classifier.png, task1_scores.csv, "
          f"task1_per_class.csv, task1_feature_importance.csv to {OUT_DIR.name}/")


def per_class_table(y_true, probability, classes) -> pd.DataFrame:
    predicted = classes[np.argmax(probability, axis=1)]
    rows = {}
    for index, label in enumerate(classes):
        mask = y_true == label
        confidence = np.clip(probability[mask, index], 1e-15, 1 - 1e-15)
        rows[PLASTICC_CLASSES[label]] = {
            "target": label,
            "n_test": int(mask.sum()),
            "log_loss": float(-np.mean(np.log(confidence))),
            "recall": float(np.mean(predicted[mask] == label)),
            "mean_probability": float(np.mean(probability[mask, index])),
        }
    return pd.DataFrame(rows).T.sort_values("log_loss", ascending=False)


def plot(y_true, probability, classes, per_class, importance, model, final,
         output_path) -> None:
    fig = plt.figure(figsize=(14.5, 11.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.52,
                            wspace=0.28)

    # Confusion matrix, row-normalised: what each true class gets called.
    ax = fig.add_subplot(grid[0, :])
    predicted = classes[np.argmax(probability, axis=1)]
    names = [PLASTICC_CLASSES[c] for c in classes]
    matrix = pd.crosstab(pd.Series(y_true, name="true"),
                         pd.Series(predicted, name="predicted"))
    matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
    normalised = matrix.div(matrix.sum(axis=1), axis=0).to_numpy()
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalised[i, j] > 0.005:
                ax.text(j, i, f"{normalised[i, j]:.2f}", ha="center",
                        va="center", fontsize=7.5,
                        color="white" if normalised[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)), names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(classes)), names, fontsize=8.5)
    ax.set_xlabel("Predicted class", fontsize=10, color=INK_MUTED, labelpad=2)
    ax.set_ylabel("True class", fontsize=10, color=INK_MUTED)
    ax.set_title(
        f"Test-set confusion, row-normalised  -  balanced log loss "
        f"{final['balanced_log_loss']:.3f}, accuracy {final['accuracy']:.3f}, "
        f"macro-F1 {final['macro_f1']:.3f}",
        fontsize=11, color=INK, loc="left")
    ax.tick_params(colors=INK_MUTED)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01,
                 label="Fraction of true class")

    # Where the loss actually sits, class by class.
    ax = fig.add_subplot(grid[1, 0])
    ordered = per_class.sort_values("log_loss")
    colors = [ORANGE if v > final["balanced_log_loss"] else BLUE
              for v in ordered["log_loss"]]
    ax.barh(range(len(ordered)), ordered["log_loss"], color=colors)
    ax.axvline(final["balanced_log_loss"], color=INK_MUTED, linestyle="--",
               linewidth=1)
    ax.set_yticks(range(len(ordered)),
                  [f"{name}  (n={int(n)})"
                   for name, n in zip(ordered.index, ordered["n_test"])],
                  fontsize=8.5)
    ax.set_xlabel("Test log loss within class", fontsize=10, color=INK_MUTED)
    ax.set_title("Dashed line is the class-balanced average",
                 fontsize=11, color=INK, loc="left")
    style_axes(ax)

    # What the model leaned on.
    ax = fig.add_subplot(grid[1, 1])
    top = importance.head(18)[::-1]
    ax.barh(range(len(top)), top.to_numpy(), color=BLUE)
    ax.set_yticks(range(len(top)), top.index, fontsize=8)
    ax.set_xlabel("LightGBM split count", fontsize=10, color=INK_MUTED)
    ax.set_title(f"Top 18 of {len(importance)} features "
                 f"({model.best_iteration_} trees)",
                 fontsize=11, color=INK, loc="left")
    style_axes(ax)

    fig.suptitle("Task 1 - gradient-boosted PLAsTiCC classification",
                 fontsize=13, color=INK, y=0.965)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
