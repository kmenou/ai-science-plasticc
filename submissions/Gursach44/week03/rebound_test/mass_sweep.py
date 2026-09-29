"""Week 3: REBOUND sweep for the largest asteroid mass that keeps Sun + Mars + Jupiter stable.

Each simulation contains only the Sun, Mars, Jupiter and one asteroid placed between
the orbits of Mars and Jupiter. For a grid of asteroid semi-major axes and masses
(and several random orbital phases per grid point) we integrate for 1000 years and
flag the run as unstable if any two adjacent orbits cross or any body is ejected.
A stricter second criterion additionally requires Mars's eccentricity to change by
less than MARS_DE_LIMIT, i.e. Mars's orbit is left essentially undisturbed.
"""

from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rebound


OUTPUT_DIR = Path(__file__).resolve().parent
T_END = 1000.0          # years
CHECK_INTERVAL = 0.5    # years between stability checks
N_PHASES = 4            # random orbital phases per (a, mass) grid point
MARS_DE_LIMIT = 0.01    # stricter criterion: Mars eccentricity may change by less than this

# Units: AU, yr, Msun. Approximate J2000 heliocentric elements.
M_MARS = 3.227e-7
M_JUPITER = 9.548e-4
M_EARTH = 3.003e-6
MARS = dict(a=1.5237, e=0.0934, inc=np.radians(1.85))
JUPITER = dict(a=5.2034, e=0.0484, inc=np.radians(1.30))
ASTEROID_E = 0.05
ASTEROID_INC = np.radians(2.0)

A_GRID = np.round(np.arange(2.1, 3.31, 0.1), 2)            # main belt, AU
LOG_M_GRID = np.round(np.arange(-9.0, -0.99, 0.1), 2)      # log10(M / Msun)


def build_simulation(a_ast: float, m_ast: float, seed: int) -> rebound.Simulation:
    """Sun + Mars + asteroid + Jupiter with random angles drawn from seed."""

    rng = np.random.default_rng(seed)
    angles = lambda: dict(Omega=rng.uniform(0, 2 * np.pi),
                          omega=rng.uniform(0, 2 * np.pi),
                          M=rng.uniform(0, 2 * np.pi))
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0)
    sim.add(m=M_MARS, **MARS, **angles())
    sim.add(m=m_ast, a=a_ast, e=ASTEROID_E, inc=ASTEROID_INC, **angles())
    sim.add(m=M_JUPITER, **JUPITER, **angles())
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = MARS["a"] ** 1.5 / 25  # Mars period (yr, since G = 4 pi^2) / 25
    return sim


def orbits_unstable(sim: rebound.Simulation) -> str | None:
    """Return a reason string if any orbit is unbound or adjacent orbits cross."""

    orbits = sim.orbits()
    if any(o.e >= 1 or o.a <= 0 for o in orbits):
        return "ejected"
    orbits = sorted(orbits, key=lambda o: o.a)
    for inner, outer in zip(orbits, orbits[1:]):
        if outer.a * (1 - outer.e) <= inner.a * (1 + inner.e):
            return "orbit crossing"
    return None


def run_one(args) -> dict:
    """Integrate one configuration; report stability, when it failed, and Mars's peak e."""

    a_ast, log_m, seed = args
    sim = build_simulation(a_ast, 10**log_m, seed)
    mars_e0, mars_e_max, reason = sim.particles[1].e, sim.particles[1].e, None
    for t in np.arange(CHECK_INTERVAL, T_END + 1e-9, CHECK_INTERVAL):
        sim.integrate(t, exact_finish_time=0)
        reason = orbits_unstable(sim)
        mars_e_max = max(mars_e_max, sim.particles[1].e)
        if reason:
            break
    return {
        "a_ast": a_ast,
        "log10_m_ast": log_m,
        "seed": seed,
        "stable": reason is None,
        "reason": reason or "",
        "t_end": round(float(sim.t), 1),
        "mars_de_max": mars_e_max - mars_e0,
    }


def critical_masses(results: pd.DataFrame, column: str) -> pd.Series:
    """Per a: largest log mass where every phase passes `column`, as do all smaller masses."""

    crit = {}
    for a_ast, group in results.groupby("a_ast"):
        passes = group.groupby("log10_m_ast")[column].all().sort_index()
        failing = passes.index[~passes.values]
        cutoff = failing.min() if len(failing) else np.inf
        crit[a_ast] = passes.index[passes.index < cutoff].max()
    return pd.Series(crit, name=column)


def plot_sweep(results: pd.DataFrame, crit: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of stable fraction vs (a, mass) with both critical-mass curves overlaid."""

    grid = (results.groupby(["log10_m_ast", "a_ast"])["stable"].mean()
            .unstack("a_ast").sort_index())
    da, dm = np.diff(A_GRID).mean(), np.diff(LOG_M_GRID).mean()

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="Blues",
                      vmin=0, vmax=1,
                      extent=[A_GRID[0] - da / 2, A_GRID[-1] + da / 2,
                              LOG_M_GRID[0] - dm / 2, LOG_M_GRID[-1] + dm / 2])
    fig.colorbar(image, ax=ax,
                 label=f"fraction of {N_PHASES} phases with no crossing/ejection in {T_END:.0f} yr")
    ax.plot(crit.index, crit["stable"], "o-", color="black", linewidth=2, markersize=6,
            label="largest mass: no orbit crossing or ejection")
    ax.plot(crit.index, crit["mars_quiet"], "s--", color="tab:orange", linewidth=2,
            markersize=6, label=f"largest mass: also Mars |Δe| < {MARS_DE_LIMIT}")
    for name, log_m in (("Earth mass", np.log10(M_EARTH)), ("Jupiter mass", np.log10(M_JUPITER))):
        ax.axhline(log_m, color="white", linestyle=":", linewidth=1.2)
        ax.text(A_GRID[0] - da / 2 + 0.02, log_m + 0.08, name, fontsize=9, color="black",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.set_xlabel("Asteroid semi-major axis (AU)")
    ax.set_ylabel(r"$\log_{10}$(asteroid mass / $M_\odot$)")
    ax.set_title("Sun + Mars + Jupiter + asteroid: stability over 1000 yr (REBOUND WHFast)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    jobs = [(a, m, seed) for a in A_GRID for m in LOG_M_GRID for seed in range(N_PHASES)]
    with Pool() as pool:
        results = pd.DataFrame(pool.map(run_one, jobs, chunksize=16))
    results["mars_quiet"] = results["stable"] & (results["mars_de_max"] < MARS_DE_LIMIT)
    results.to_csv(OUTPUT_DIR / "mass_sweep_results.csv", index=False)

    crit = pd.concat([critical_masses(results, "stable"),
                      critical_masses(results, "mars_quiet")], axis=1)
    table = pd.DataFrame(index=crit.index.rename("a_ast"))
    for column, label in (("stable", "no_crossing"), ("mars_quiet", "mars_quiet")):
        table[f"{label}_log10_msun"] = crit[column]
        table[f"{label}_mearth"] = 10 ** crit[column] / M_EARTH
        table[f"{label}_mjup"] = 10 ** crit[column] / M_JUPITER
    table.to_csv(OUTPUT_DIR / "critical_mass.csv")
    print(table.round(3).to_string())
    plot_sweep(results, crit, OUTPUT_DIR / "mass_sweep.png")
