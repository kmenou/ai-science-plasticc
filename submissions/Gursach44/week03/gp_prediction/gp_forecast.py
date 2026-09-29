"""Week 3: celerite2 GP fits on the first 80% of each light curve, forecasting the last 20%."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from celerite2 import GaussianProcess, terms
from scipy.optimize import minimize

from analysis import PASSBAND_INDEX, TARGETS
from plasticc_course import data as course_data


TRAIN_FRACTION = 0.8
# Starting guesses for the kernel timescale (days); the best optimum is kept.
RHO_STARTS = (5.0, 20.0, 100.0)


def load_band(object_id: int, band_name: str):
    """Return time-sorted MJD, flux and flux_err arrays for one passband."""

    observations = course_data.load_object(object_id, dataset="full")
    band = observations[observations["passband"] == PASSBAND_INDEX[band_name]]
    band = band.sort_values("mjd")
    return band["mjd"].to_numpy(), band["flux"].to_numpy(), band["flux_err"].to_numpy()


def build_gp(params, t, yerr):
    """Matern-3/2 GP with a constant mean and a jitter term added to flux_err."""

    mean, log_sigma, log_rho, log_jitter = params
    kernel = terms.Matern32Term(sigma=np.exp(log_sigma), rho=np.exp(log_rho))
    gp = GaussianProcess(kernel, mean=mean)
    gp.compute(t, diag=yerr**2 + np.exp(2 * log_jitter), quiet=True)
    return gp


def fit_gp(t, y, yerr):
    """Maximise the GP marginal likelihood; return the best parameter vector."""

    def neg_log_like(params):
        value = -build_gp(params, t, yerr).log_likelihood(y)
        return value if np.isfinite(value) else 1e25

    best = None
    for rho in RHO_STARTS:
        start = [np.median(y), np.log(np.std(y)), np.log(rho), np.log(np.median(yerr))]
        bounds = [(None, None), (-5, 10), (np.log(0.5), np.log(2000)), (-10, 10)]
        result = minimize(neg_log_like, start, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    return best.x


def forecast_panel(ax, object_id: int, label: str, band_name: str) -> dict:
    """Fit on the first 80% of points, draw the GP and data, return held-out stats."""

    t, y, yerr = load_band(object_id, band_name)
    n_train = int(np.floor(TRAIN_FRACTION * len(t)))
    t_train, y_train, e_train = t[:n_train], y[:n_train], yerr[:n_train]
    t_test, y_test, e_test = t[n_train:], y[n_train:], yerr[n_train:]
    split = 0.5 * (t_train[-1] + t_test[0])

    params = fit_gp(t_train, y_train, e_train)
    gp = build_gp(params, t_train, e_train)

    t_grid = np.linspace(t[0] - 10, t[-1] + 10, 2000)
    mu, var = gp.predict(y_train, t=t_grid, return_var=True)
    sd = np.sqrt(var)
    fit_region = t_grid <= split
    for region, color, name in ((fit_region, "C0", "GP fit (first 80%)"),
                                (~fit_region, "C1", "GP forecast (last 20%)")):
        ax.plot(t_grid[region], mu[region], color=color, label=f"{name} mean")
        ax.fill_between(t_grid[region], (mu - sd)[region], (mu + sd)[region],
                        color=color, alpha=0.25, linewidth=0, label=f"{name} ±1σ")

    ax.errorbar(t_train, y_train, yerr=e_train, fmt="o", color="k", markersize=4,
                capsize=2, label="training data")
    ax.errorbar(t_test, y_test, yerr=e_test, fmt="s", color="C3", markersize=4,
                capsize=2, label="held-out data")
    ax.axvline(split, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux")
    ax.set_title(f"{label} — object {object_id}, {band_name} band")
    ax.legend(fontsize=8, ncol=2)

    # Held-out comparison: predictive sigma includes GP variance, jitter and flux_err.
    mu_test, var_test = gp.predict(y_train, t=t_test, return_var=True)
    total_sd = np.sqrt(var_test + np.exp(2 * params[3]) + e_test**2)
    residual = y_test - mu_test
    return {
        "object_id": object_id,
        "band": band_name,
        "n_train": n_train,
        "n_test": len(t_test),
        "split_mjd": round(float(split), 1),
        "mean": round(float(params[0]), 2),
        "sigma": round(float(np.exp(params[1])), 2),
        "rho_days": round(float(np.exp(params[2])), 1),
        "jitter": round(float(np.exp(params[3])), 3),
        "test_rmse": round(float(np.sqrt(np.mean(residual**2))), 2),
        "test_chi2_per_point": round(float(np.mean((residual / total_sd) ** 2)), 2),
        "test_within_1sigma": round(float(np.mean(np.abs(residual) < total_sd)), 2),
    }


def plot_forecasts(output_path: str | Path) -> list[dict]:
    """Save the stacked GP forecast panels and return per-object statistics."""

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(10, 9))
    stats = [forecast_panel(ax, *target) for ax, target in zip(axes, TARGETS)]
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return stats


if __name__ == "__main__":
    for row in plot_forecasts(Path(__file__).with_name("gp_forecast.png")):
        print(row)
