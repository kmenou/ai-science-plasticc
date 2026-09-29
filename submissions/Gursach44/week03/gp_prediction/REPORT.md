# Week 3 report

## Setup
- Coding agent: Claude
- Added dependency: `celerite2` 0.3.3 (`uv add celerite2`)
- Data: `plasticc_course.data.load_object(..., dataset="full")` — object 713 is
  not in the tiny300 subset
- Passband mapping assumed (standard PLAsTiCC/LSST order, not documented in the
  repo): 0–5 = u, g, r, i, z, y, so i = 3 and r = 2

## Work products
- `analysis.py` → `lightcurves.png`: single-band light curves of SN Ia 252646
  (i band) and AGN 713 (r band), detected vs non-detected points distinguished
- `gp_forecast.py` → `gp_forecast.png`: celerite2 GP fit to the first 80% of
  each light curve with a forecast of the last 20%

## Light curves
- SN Ia 252646 (i): flat at zero flux through the first two seasons, then a
  sharp rise to ~370 near MJD 60580 and a smooth decline to ~40 over ~90 days.
- AGN 713 (r): no discrete event; slow, stochastic variation of order ±10 flux
  across all three seasons, trending from ~+5–10 early to ~−5–10 in the last
  season. Negative fluxes are expected because PLAsTiCC fluxes are
  difference-image fluxes relative to a template. Many points are
  non-detections because the signal is weak.

## GP forecast
Method:
- Observations in the chosen band are sorted by MJD; the first 80% of points
  (44 of 56) are used for training and the last 20% (12) are held out. The
  split is by number of points, not by time span: 20% of the time span would
  hold out the entire final season, including the whole supernova.
- Model: celerite2 `Matern32Term` (amplitude σ, timescale ρ) + constant mean +
  a white-noise jitter added in quadrature to `flux_err`. Hyperparameters
  maximise the marginal likelihood (L-BFGS-B, three starting values of ρ, best
  kept).
- The plot shows the GP mean and ±1σ band conditioned on the training data
  (blue over the training range, orange over the forecast range), the training
  data (black) and the held-out data (red).
- Held-out statistics use a predictive σ that combines GP variance, jitter and
  `flux_err`.

| Object | Band | mean | σ | ρ (days) | jitter | held-out RMSE | χ²/point | within 1σ |
|---|---|---|---|---|---|---|---|---|
| 252646 (SN Ia) | i | 33.8 | 96.3 | 40.8 | 0.00 | 18.8 | 0.09 | 12/12 |
| 713 (AGN) | r | −0.86 | 6.18 | 66.3 | 0.45 | 2.41 | 0.60 | 10/12 |

Interpretation:
- **SN Ia:** the forecast starts at the last training point and decays toward
  the GP mean (~34) on the ~41-day timescale, so it consistently
  **over-predicts** the actual, faster decline (held-out points lie below the
  forecast mean). All points fall within 1σ only because the band is very
  wide: χ²/point ≈ 0.09 means the uncertainty is heavily overestimated. A
  stationary GP is a poor model for a one-off transient. The single large
  event forces a large σ, which is also why the between-season bands balloon
  to ±70 even though the baseline is flat at zero. A parametric SN template
  (or a GP with a transient mean function) would forecast the tail better.
- **AGN:** the stationary GP is a natural fit for stochastic variability (a
  Matérn/DRW-like process). The forecast mean slowly reverts from ~−9 toward
  the mean, the held-out data scatter around it with RMSE ≈ 2.4, and 10 of 12
  points fall within 1σ, which is roughly consistent with calibrated
  uncertainties (χ²/point 0.6).

## Verification
- I (Claude) viewed both PNGs: correct objects and bands in the titles, error
  bars present, the 80/20 split line sits between the last training point and
  the first held-out point, and the forecast band widens away from the data as
  expected.
- One check I performed myself: _TODO_

## Reflection
- One statement supported directly by the data: _TODO_
- One statement requiring further analysis: _TODO_
