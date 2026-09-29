# Homework 2 report — gradient boosting on the full PLAsTiCC training set

Rori Kang (`rorikang`). All results use the full training set:
7,848 objects and 1,421,705 observations. Every number quoted here is
reproduced in `RESULTS.txt`, which is the captured stdout of the four scripts.

## Contents

| File | What it is |
| --- | --- |
| `hw2_common.py` | Light-curve features, the photo-z metrics, sentinel handling, plot style |
| `hw2_task1_classifier.py` | Task 1 — LightGBM classification of the 14 classes |
| `hw2_task2_redshift.py` | Task 2 — spectroscopic redshift from photo-z + light curves |
| `hw2_task3_outliers.py` | Task 3 — anomalous observations in six light curves |
| `hw2_task4_shift.py` | Task 4 — the Task 2 regression under distribution shift |
| `RESULTS.txt` | Captured output of all four scripts |
| `task*.png`, `task*.csv` | Figures and metric tables |

## Running it

LightGBM and scikit-learn are not in the repository's `pyproject.toml`, so they
are supplied on the command line rather than by editing a shared file:

```bash
uv run --with lightgbm --with scikit-learn python submissions/rorikang/week02/hw2_task1_classifier.py
```

On macOS LightGBM also needs the OpenMP runtime (`brew install libomp`);
without it the import fails with a `libomp.dylib` load error. The same command
form runs tasks 2, 3 and 4. Task 1 builds the light-curve feature table on
first run and caches it in the gitignored `.cache/`; later runs reuse it.

## Two properties of the catalogue that shape everything below

1. **`hostgal_specz == hostgal_photoz == 0` and `distmod == -9` are sentinels**
   meaning "Galactic object, no host galaxy", not measurements. All 2,325 such
   objects are dropped from every redshift task. `extragalactic_mask` checks
   both encodings and raises if they ever disagree, so a change in the
   catalogue shows up as an error instead of silently admitting 2,325 fake
   `z = 0` objects into a regression.
2. **`distmod` is `hostgal_photoz` re-expressed in magnitudes.** Measured, not
   assumed: `spearman(photoz, distmod) = 0.999999`, and the scatter of
   `distmod` inside narrow photo-z bins is 0.0074 mag against 2.263 mag
   overall. It is excluded from every redshift feature set as redundant.

---

## Task 1 — gradient-boosted classification

**Metric.** Class-balanced multi-class log loss: the mean over classes of the
mean log loss within that class. This is the PLAsTiCC challenge metric with
flat per-class weights. Two reasons. The class frequencies here are an artefact
of how the challenge sampled objects, not of the sky, so an unweighted average
would let SN Ia (2,313 objects) drown out Mira variables (30). And the
downstream use of such a model — deciding which transients get spectroscopic
follow-up — needs calibrated probabilities, not hard labels; accuracy would be
maximised by a model that quietly gives up on every rare class. Accuracy and
macro-F1 are reported alongside as interpretable secondary numbers. The model
is trained with inverse-frequency sample weights so it optimises the same
objective it is scored on.

**Split.** Stratified 60 / 20 / 20 train / validation / test by object, seed 42.
By object because each row is one object and objects are independent — there is
no group structure to leak across. Stratified because it is not optional at
these sample sizes: Mira has 30 examples in 7,848, and an unstratified 20% test
fold could easily contain two of them. Validation is used for early stopping,
for the baseline comparison and for the ablation. The test set is read once.

**Features.** 70 from photometry alone plus 5 host/extinction columns
(`hostgal_photoz`, `hostgal_photoz_err`, `mwebv`, `ddf`, `|gal_b|`).
`hostgal_specz` is excluded because spectroscopy exists for a small minority of
real survey objects, and `distmod` for the reason above.

### Results

Validation, with two constant baselines:

| model | balanced log loss | log loss | accuracy | macro-F1 |
| --- | --- | --- | --- | --- |
| Training-prior constant | 3.1754 | 2.1761 | 0.2949 | 0.0325 |
| Uniform constant | 2.6391 | 2.6391 | 0.0191 | 0.0027 |
| LightGBM | **0.7479** | 0.6186 | 0.7873 | 0.7359 |

Final test set, one evaluation: **balanced log loss 0.6678**, log loss 0.5769,
accuracy 0.8210, macro-F1 0.7856. Early stopping kept 74 of a possible 3,000
trees.

The test score is *better* than validation by 0.08. That is the expected sign
of the early-stopping bias — the validation fold chose the tree count, so it is
mildly optimistic about itself and the test fold is the honest number — but the
gap is within what a 1,570-object fold can produce by chance, and I would not
read anything into its direction.

**Ablation.** Photometry only, no host metadata: balanced log loss 0.9627
against 0.7479 with it. Host redshift is doing real work rather than decorating
the feature importance plot.

### What the confusion matrix says

The per-class breakdown in `task1_per_class.csv` splits cleanly into two
regimes, and the split is astrophysical rather than statistical:

- **Periodic and stochastic variables are nearly solved.** Eclipsing binary
  (log loss 0.028, recall 0.995), M-dwarf (0.063), RR Lyrae (0.081), Mira
  (0.097), AGN (0.111). These vary continuously across the whole survey
  baseline, so `detected_span_fraction`, `flux_skew` and the colour features
  separate them from one-shot transients almost perfectly.
- **Supernova subtypes are where the loss lives.** SN Iax (2.666, recall
  0.139), SN Ia-91bg (1.601), SN Ibc (1.274), SN II (1.012). Half the SN Iax
  test objects are called SN Ia and a fifth SN II. This is the right failure:
  SN Iax *are* a faint, low-velocity variety of thermonuclear supernova, and
  91bg-like events are subluminous SNe Ia. With 36 and 42 test objects
  respectively, and with the separating information sitting in spectral
  features the photometry does not resolve, six broad-band light curves may
  simply not contain the distinction.

Rarity is not the problem — Mira (6 test objects) and kilonova (20) are both
essentially perfect. Physical similarity to an abundant class is the problem.

---

## Task 2 — spectroscopic redshift from photo-z plus light curves

**Sample.** The 5,523 extragalactic objects.

**Metric.** All residuals are quoted as `dz = (z_pred - z_spec) / (1 + z_spec)`,
the photo-z convention, because redshift errors grow with redshift and a raw
residual would let the handful of `z > 2` objects dominate any average. The
error distribution is heavy-tailed, so one number will not do:

- **catastrophic outlier fraction** (`|dz| > 0.15`) — *primary*, fixed before
  any model was fitted. A redshift wrong by 0.02 costs a little precision on a
  luminosity; a redshift wrong by 0.5 puts the object in the wrong epoch of the
  universe. At 12% the catastrophic rate is the dominant systematic here, so it
  is what a model should be asked to reduce.
- **σ_NMAD** — carried as a *guard*, not a target. A method that halves the
  tail while doubling the core scatter has not obviously helped. Candidates
  whose σ_NMAD exceeds 1.5× the baseline are disqualified.
- **RMSE** — reported because it is what the squared-error training loss
  optimises, but it is dominated by the tail and says little alone.

Selection rule, fixed in advance: lowest validation outlier fraction subject to
the σ_NMAD guard, with ties inside one binomial standard error broken on
σ_NMAD. The tie-break is not decoration — two candidates landed 0.0000 apart on
the primary metric and would otherwise have been separated by row order.

**Split.** Random 60 / 20 / 20, seed 42. Random rather than stratified because
the target is continuous and a 20% fold of 5,523 objects covers the redshift
range densely; stratifying by redshift bin moves the metrics in the third
decimal. Task 4 deliberately breaks this assumption.

**Baseline.** Not a constant and not the training mean — `hostgal_photoz` used
unchanged, because that is what an astronomer already has for free. A model
only earns its place if it beats "trust the photo-z".

### Results (test set, one evaluation)

| model | σ_NMAD | outlier fraction | RMSE | bias |
| --- | --- | --- | --- | --- |
| Photo-z as-is (baseline) | **0.0278** | 0.1222 | 0.5645 | +0.0052 |
| LightGBM, photo-z only | 0.0506 | 0.0977 | 0.2326 | +0.0034 |
| LightGBM, photo-z + light curves | 0.0347 | **0.0380** | 0.1463 | +0.0013 |
| LightGBM on residual, photo-z + light curves | 0.0330 | 0.0480 | **0.1384** | −0.0025 |
| **Hybrid (chosen)** | 0.0316 | 0.0389 | 0.1461 | +0.0025 |

Against the baseline the chosen model cuts the catastrophic fraction by
**68%** (0.1222 → 0.0389) and RMSE by **74%** (0.5645 → 0.1461), at the cost of
**14% worse** core scatter (0.0278 → 0.0316). Of 135 photo-z catastrophic
failures in the test set it recovers 104, and introduces 12 new failures on
objects the photo-z had right.

### The finding this task actually produced

The honest headline is a trade, not a win. **The photometric redshift's core is
already better than anything the model can produce** — σ_NMAD 0.0278 against
0.0347 for the plain regressor. Where the photo-z fails, it fails
catastrophically: the top-left panel of `task2_redshift.png` shows a dense clump
of objects at `z_spec ≈ 0.2` assigned `z_photo ≈ 2.5`, the classic
colour-degeneracy failure where a low-redshift galaxy's 4000 Å break is
mistaken for the Lyman break of a high-redshift one. The light curves break that
degeneracy because a low-redshift transient is short and a high-redshift one is
stretched by `(1 + z)`, and the feature importances confirm that this is exactly
what the model uses: after `hostgal_photoz` and its error, the top features are
`rise_time`, `flux_weighted_duration`, `fall_time` and `detected_span` — four
independent measurements of the same timescale.

That diagnosis is what motivated the hybrid, and the hybrid is the one model
that captures both effects: **keep the photo-z unless the model disagrees with
it by more than 0.02 in `(1+z)` units, and overwrite it only then.** The
threshold was swept on validation (`task2_hybrid_threshold_sweep.csv`). It
recovers essentially all of the tail improvement while giving back half of the
core degradation.

**Photo-z alone is not enough.** A LightGBM given only `hostgal_photoz` and its
error reaches an outlier fraction of 0.0977 against 0.0380 with light curves
included — it can learn a global recalibration of the photo-z, but with no
independent information it cannot tell which individual objects are the
degenerate ones.

**Where it still fails.** The lower-middle panel shows σ_NMAD by redshift bin.
The hybrid beats the baseline only in the middle, `0.15 < z < 0.5` (0.0250
against 0.0275, and 0.0258 against 0.0298). It is worse in the lowest bin
(0.0330 against 0.0250) and progressively worse above `z = 0.5`: 0.0382 against
0.0196 in `0.5–0.8`, 0.0703 against 0.0314 in `0.8–1.2`, and 0.0979 against
0.0422 above `z = 1.2`.
The median residual turns systematically negative past `z ≈ 1`. Only 72 test
objects sit above `z = 0.8`, and a tree cannot predict outside its training
range, so the high-redshift end is pulled inward. Task 4 turns this from a
footnote into the main result.

---

## Task 3 — anomalous observations in six light curves

**Objects.** One per class, spanning the kinds of variability the survey
contains, each the most-detected deep-drilling object in its class (dense
sampling is what makes both a smooth fit and a visual judgement possible):

| object | class | observations | flagged |
| --- | --- | --- | --- |
| 171194 | SN Ia (z = 0.050) | 352 | 25 (7.1%) |
| 335507 | SLSN-I (z = 0.469) | 350 | 22 (6.3%) |
| 186060 | AGN (z = 0.601) | 352 | 5 (1.4%) |
| 28391 | RR Lyrae (Galactic) | 352 | 26 (7.4%) |
| 151356 | eclipsing binary (Galactic) | 350 | 47 (13.4%) |
| 24903 | M-dwarf (Galactic) | 350 | 15 (4.3%) |

**No metric, no held-out split — deliberately.** There is no ground-truth label
for "bad observation" in this catalogue, so there is nothing to score against
and nothing to hold out. The assessment is visual, which is what the task asks
for, and the honest output of this task is a characterisation of when each
detector works rather than a number.

**Three detectors.**

- **A — residual from a smooth fit.** Leave-one-out local regression per object
  and per passband, kernel bandwidth chosen per band by leave-one-out
  cross-validation on a *median* criterion. Leave-one-out matters twice over: a
  smoother that can see the point it is predicting will chase an outlier and
  hide it, and a bandwidth chosen on in-sample residuals collapses toward zero
  width for the same reason. A median criterion rather than least squares,
  because a squared-error criterion widens the bandwidth to accommodate the very
  outliers the analysis is looking for. Residual is divided by `flux_err` and
  then by the robust spread of residuals among each point's 15 nearest
  neighbours in time; flagged above 5.
- **B — detection flag against measured SNR.** Across the full training set
  17,117 undetected observations exceed |SNR| 5 and 1,048 detected ones sit
  below |SNR| 3. Those are places where the catalogue contradicts itself.
- **C — outsized photometric error.** `flux_err` above 10× the band median. The
  largest `flux_err` in the training set is 2.2 × 10⁶ against a median of 4.7.

### Two iterations, both driven by looking at the figure

**First attempt: one robust scale per passband.** It flagged 82 of 352 points
(23%) on the SN Ia — effectively the entire declining light curve. The cause is
visible in the figure and is not subtle: the pull `(flux − fit) / flux_err`
only measures *noise* where the fit is good. On a supernova decline the fit
error exceeds the quoted photometric error by an order of magnitude, so a single
per-band scale gets set by whichever regime covers most of the light curve —
here the long flat pre-explosion baseline — and the transient then floods with
false flags. On the RR Lyrae the same mechanism ran the other way: the fit was
hopeless everywhere, the scale inflated, and the detector went silent at 1 flag.
Measuring the spread locally fixed both (SN Ia 82 → 21, RR Lyrae 1 → 7).

**Second attempt: the local model.** The first version used a kernel-weighted
*average*. A weighted average is biased wherever the light curve has a slope,
because it mixes in neighbours from the brighter side, and a supernova decline
is nothing but slope. Switching to a local *line* absorbs the slope exactly and
leaves only curvature.

Counting flags does not settle whether that helped — the better-fitting model
has smaller local residuals, so its threshold tightens and it can flag *more*
points (93 → 122 in total, which is what happened). The measurement that does
settle it is *where* the residuals sit. Splitting each light curve into its
steepest decile and its flattest half:

| object | median \|pull\| on steep decile | | median \|pull\| on flat half | |
| --- | --- | --- | --- | --- |
| | kernel average | local line | kernel average | local line |
| SN Ia | 28.10 | **15.33** | 0.65 | 0.66 |
| SLSN-I | 8.52 | **2.78** | 0.83 | 0.80 |
| M-dwarf | 5.71 | **5.00** | 0.65 | 0.66 |
| AGN | 3.28 | 3.60 | 1.93 | 1.51 |
| eclipsing binary | 50.92 | 69.23 | 0.57 | 0.64 |
| RR Lyrae | 68.43 | 70.02 | 113.68 | 61.44 |

This is the predicted signature exactly where it should be and nowhere else:
on the two smooth transients the local line halves the residual on steep
stretches while leaving the flat stretches untouched. It does *not* help the
eclipsing binary — an eclipse is a sharp V, not locally linear, and the line
extrapolates the ingress slope straight through the minimum and overshoots.

### Quality of the results, by object

- **SN Ia 171194 — good.** 8 points are flagged by detector C alone: `z`-band
  measurements near peak with errors ten times the band median. Detector A adds
  three genuine baseline outliers in the flat pre-explosion season. This is the
  case the method is built for.
- **SLSN-I 335507 — good, and detector B earns its place.** 15 of the 22 flags
  are late-time observations that sit above SNR 5 but are marked undetected.
  Whether the pipeline or the SNR is wrong is not decidable from this table, but
  they are exactly the points a follow-up would want to re-reduce.
- **AGN 186060 — good.** 5 flags in 352 points. The smoother tracks stochastic
  variability on the ~6-day bandwidth it selected without over-reacting. Its
  flux is negative throughout (≈ −300), a template-subtraction offset rather
  than an anomaly, and the detector correctly ignores it because it is looking
  at residuals, not at flux.
- **M-dwarf 24903 — a real detection, correctly identified and arguably
  unwanted.** The single largest flag is a `y`-band excursion from −300 to +210
  in one epoch. That is a stellar flare: real astrophysics, the defining feature
  of the class, and the thing a classifier most wants to keep. The detector
  cannot tell it apart from a cosmic ray, and it never will without a model of
  the source.
- **Eclipsing binary 151356 — the clearest false-positive case.** 46 flags,
  13% of the light curve, and on inspection close to none of them are bad data.
  The five most extreme are difference fluxes of −2,287, −2,209, −2,090, −1,176
  and −741 in bands whose baseline sits at +692, +692, +713, +479 and +298.
  My first reading was that fluxes that far below the baseline had to be
  artefacts, but their error bars are entirely normal (5–12, against a band
  median of the same order) and they are all marked `detected`. PLAsTiCC fluxes
  are difference fluxes against a template that contains the star's mean
  brightness, so a deep eclipse *should* go strongly negative. These are real
  eclipses. The period is far shorter than the sampling interval, so each one is
  caught as an isolated point with no neighbour at the same phase, which is
  exactly what detector A is built to flag.
- **RR Lyrae 28391 — failure, and it should be reported as one.** The 25 flags
  are not meaningful. The period is well under a day and the survey samples
  every few days, so the light curve is aliased beyond recovery: the fit is
  noise (median \|pull\| ≈ 61 even on the "flat" half, against ≈ 0.65 for the
  well-behaved objects), and the flags are just wherever the noise happened to
  be largest. The local-line fit makes this worse, not better — visibly
  extrapolating to flux −60,000 across a sampling gap.

### Challenges, stated plainly

1. **"Anomalous" and "astrophysically real but sharp" are the same signal to
   this method.** The M-dwarf flare and the eclipsing-binary eclipses are the
   most interesting features in their light curves and are flagged as hard as
   any artefact. Separating them needs a source model, not a better smoother.
2. **The method assumes the variability is resolved by the sampling.** When the
   period is shorter than the cadence, as for the RR Lyrae, no bandwidth exists
   that works, and the detector produces confident nonsense rather than an
   error. The fix is phase-folding on a period from a Lomb–Scargle
   periodogram — fit in phase rather than in time — which is the obvious next
   step and is not implemented here.
3. **Model error and measurement error are not separable from one light curve.**
   The local-scale rescaling manages the symptom, but it does so by loosening
   the threshold exactly where the fit is worst, which is also where a real
   anomaly would be hardest to see. The detector is least sensitive precisely
   during the transient.
4. **Local linear extrapolation is dangerous across gaps.** Visible in the
   eclipsing binary and RR Lyrae panels. A local line is better *inside* dense
   sampling and worse at the edges of seasonal gaps than a local average.
5. **Only detector C is unambiguous.** An error bar 10× the band median is bad
   data under any interpretation. A and B both flag things that may be entirely
   correct.

---

## Task 4 — distribution shift

The Task 2 regression, rerun with the same features, model and metrics under
two shifts:

- **survey depth** — train on deep-drilling-field objects, test on wide-field;
- **redshift** — train on `z_spec < 0.4`, test on `z_spec ≥ 0.4`.

**The measurement problem, and how it is handled.** A shifted test set can score
worse for two different reasons, and only one is a transfer failure: the target
domain may simply be harder, or the model may fail to carry over. Reporting the
shifted score alone cannot separate them. Each scenario therefore runs a
**matched control** — the same number of training objects, drawn at random from
everything outside the test set, evaluated on exactly the same test objects. The
control absorbs "the domain is harder", so the shifted-minus-control gap is the
transfer failure. The photo-z baseline is carried through as a third reference,
since it learns nothing and so measures the domain's intrinsic difficulty.

The validation fold is always carved from the **training** distribution. Using
target-domain objects to early-stop would assume away the problem: in the real
setting this task models — train on objects that have spectra, apply to the ones
that do not — target-domain labels are precisely what is missing.

### Shift 1 — deep-drilling → wide-field

Test set 1,974 wide-field objects, shared by both models.

| model | σ_NMAD | outlier fraction | RMSE | bias |
| --- | --- | --- | --- | --- |
| Photo-z as-is (baseline) | 0.0270 | 0.1185 | 0.5805 | +0.0048 |
| Direct, control training | 0.0374 | **0.0461** | 0.1396 | +0.0062 |
| Direct, shifted training | 0.0684 | 0.2503 | 0.2287 | +0.0885 |
| On residual, control training | 0.0386 | 0.0694 | 0.1605 | +0.0006 |
| On residual, shifted training | 0.0534 | 0.1484 | 0.2198 | +0.0519 |

The catastrophic fraction goes **0.046 → 0.250**, a 5.4× degradation
attributable to the shift alone. The model trained on deep-drilling data is
**worse than doing nothing**: the photo-z baseline sits at 0.119 on the same
objects.

**Why.** The domains differ in exactly the features the model relies on. Median
wide-field object against median deep-drilling object: 130 observations against
350, 8 detections against 39, and `flux_err_median` 9.70 against 2.04 — the
wide-field photometry is **4.7× less precise**. The model learned to read
timescales (`rise_time`, `flux_weighted_duration`, `detected_span`) off densely
sampled, high-precision light curves. Measured on 8 detections instead of 39,
those same features take values the model never saw in training, and it has no
way to know that the feature has become unreliable rather than different. The
bias of +0.089 shows it is not just noisier but **systematically wrong**: a
sparsely sampled light curve looks short, a short light curve reads as
low-redshift, and the correction is applied in a consistent direction. The class
mix compounds it — TDEs are 12.3% of wide-field extragalactic objects and 0.5%
of deep-drilling ones.

The residual parameterisation halves the damage (0.250 → 0.148), because
anchoring on the photo-z limits how far a bad feature reading can drag the
prediction. It still does not beat the baseline.

### Shift 2 — low redshift → high redshift

Test set 775 objects at `z_spec ≥ 0.4`.

| model | σ_NMAD | outlier fraction | RMSE | bias |
| --- | --- | --- | --- | --- |
| Photo-z as-is (baseline) | 0.0292 | **0.0465** | 0.2551 | −0.0019 |
| Direct, control training | 0.0519 | 0.1110 | 0.2619 | −0.0296 |
| Direct, shifted training | 0.1505 | 0.5574 | 0.6770 | −0.1696 |
| On residual, control training | 0.0460 | 0.0890 | 0.2725 | −0.0234 |
| On residual, shifted training | 0.1655 | 0.5626 | 0.6689 | −0.1683 |

The catastrophic fraction goes **0.111 → 0.557**. More than half the test set is
catastrophically wrong, against 4.7% for the photo-z baseline on the same
objects. The middle-bottom panel of `task4_distribution_shift.png` is the whole
story in one picture: every prediction is pinned between 0.2 and 0.45 while the
true redshift runs out to 3.4.

**Why, part one — the extrapolation ceiling.** A gradient-boosted tree ensemble
predicts a weighted average of training leaf values. It cannot output a number
outside its training range however clear the evidence, and the training range
here stops at `z = 0.399`.

**Why, part two — and this is the part I did not anticipate.** The residual
parameterisation should have escaped the ceiling: it predicts a *correction* to
be added to the photo-z, and the photo-z is not capped. It does escape under
shift 1. Under shift 2 it fails just as badly (0.563), which sent me back to the
data:

> Among objects with `hostgal_photoz > 0.5`, the correction `z_spec − z_photo`
> they actually require is
> **−1.949** for the 525 such objects in the training set (`z_spec < 0.4`), and
> **−0.006** for the 1,005 such objects in the test set (`z_spec ≥ 0.4`).

Cutting the training set on the target variable means that *every* training
object with a high photo-z is, by construction, a photo-z overestimate — a
genuinely high-redshift object with a correct high photo-z cannot be in the
training set, because its `z_spec` would exceed the cut. So the model learns,
correctly for its training distribution and disastrously for the test one, that
a high photo-z must be corrected sharply downward. It then applies that lesson
to objects whose high photo-z is simply right.

This is worse than ordinary covariate shift. Selecting on the target induces a
spurious feature-label relationship, and no amount of regularisation,
reparameterisation or model capacity fixes it — the training data genuinely
support the wrong conclusion. It is also the more realistic of the two
scenarios: spectroscopic training sets are built from objects bright enough to
get a spectrum, which is a selection on redshift in all but name.

### What this means for Task 2's headline number

The 68% reduction in catastrophic outliers from Task 2 is real but is a
statement about a random split, in which training and test objects are drawn
from the same population. Under either realistic shift the same model gives back
that gain and then some, and lands behind the photo-z it was built to improve.
The photo-z baseline degrades only mildly in both scenarios (0.122 → 0.119 and
0.122 → 0.047) — being a fixed, physics-based estimator with no fitted
parameters is a real advantage when the population moves.

---

## Reflection

**One statement supported directly by the data.** Light-curve timescale features
break the photometric-redshift colour degeneracy. On a random split the model
recovers 104 of 135 catastrophic photo-z failures while introducing 12, and the
four highest-ranked features after the photo-z itself (`rise_time`,
`flux_weighted_duration`, `fall_time`, `detected_span`) are four independent
measurements of the one quantity that `(1 + z)` time dilation acts on.

**One statement requiring further analysis.** That a model trained on
deep-drilling data could be made to transfer to the wide-field survey by
degrading the deep-drilling light curves to wide-field cadence and depth before
extracting features, so the model learns from feature values it will actually
encounter. This is plausible and standard practice, but nothing here tests it.
The available check would be to resample each deep-drilling light curve onto a
wide-field observing pattern, rebuild the features, and rerun shift 1.

**One thing I would do differently.** Tasks 2 and 4 should have been written in
that order from the start. I selected the Task 2 hybrid on a random split and
only then discovered that the ranking of the direct and residual models reverses
under shift — the residual model is worse in-domain and much better under the
depth shift. A model chosen against a shifted validation split would have been
the better model to ship, and the random-split choice is only defensible because
the homework asked for a random-split evaluation first.

---

## Agent interaction

**Setup.** macOS 15 (Darwin 24.0.0), Python 3.12, `uv` and Homebrew, Claude
Code as the coding agent.

**Main task-level requests.** Inspect the schemas and the sentinel encodings
before writing anything; build one light-curve feature table shared by all four
tasks so no metadata column can leak into a model that should not see it; for
each task state the metric and the split before fitting; run each script and
show me the figure.

**One useful suggestion.** The agent proposed the matched control in Task 4 —
same training-set size, same test objects, random rather than shifted training
sample — rather than reporting the shifted score against the Task 2 number. That
is what makes "how much performance drops" answerable: without it the depth
shift's 0.046 → 0.250 would have been confounded with the fact that a
1,576-object training set is smaller than Task 2's 3,313.

**Errors, weak assumptions and corrections.** Three worth recording, all caught
by looking at output rather than at code:

1. The first Task 3 smoother flagged 23% of the SN Ia light curve. Diagnosed
   from the figure as a per-band robust scale being set by the wrong regime;
   fixed with a local scale.
2. The agent asserted in a docstring that switching from a kernel average to a
   local line would *reduce* the flag count. It increased it, 93 → 122. The
   claim was wrong because a better fit tightens the threshold; the assertion
   was replaced with a direct measurement of residuals on steep versus flat
   stretches, which does support the underlying bias argument.
3. The agent asserted that the residual parameterisation would survive the
   redshift shift because the photo-z is not capped. It did not (0.563 against
   0.557 for the direct model). Chasing that down produced the
   −1.949-against-−0.006 comparison, which turned out to be the most
   interesting result in the homework.

Two of the three were the agent confidently predicting a result and the run
disagreeing. The claims that survived are the ones with a measurement attached.
