# REBOUND test: largest asteroid mass between Mars and Jupiter

## Question
What is the largest mass an asteroid between the orbits of Mars and Jupiter can
have? Tested with a REBOUND parameter sweep containing only the Sun, Mars,
Jupiter and the asteroid.

## Setup
- Dependency added: `rebound` 5.2.0 (`uv add rebound`)
- `mass_sweep.py` → `mass_sweep_results.csv` (every run), `critical_mass.csv`
  (threshold per semi-major axis), `mass_sweep.png`
- Units AU, yr, M☉. WHFast integrator, dt = Mars period / 25 ≈ 0.075 yr,
  **integration time 1000 yr** (~530 Mars orbits, ~84 Jupiter orbits)
- Mars and Jupiter: approximate J2000 a, e, i and real masses
  (Mars 3.227e-7 M☉, Jupiter 9.548e-4 M☉)
- Asteroid: e = 0.05, i = 2°
- Grid: a = 2.1–3.3 AU in 0.1 AU steps (13 values) × log10(m/M☉) = −9 to −1 in
  0.1 dex steps (81 values) × 4 random sets of orbital angles (Ω, ω, M for all
  three bodies) = 4212 runs, run in about 90 s on 8 cores
- Stability is checked every 0.5 yr using Jacobi orbital elements. Two criteria:
  1. **No crossing:** no body is unbound (e ≥ 1) and no two adjacent orbits
     cross (the outer body's perihelion must stay outside the inner body's
     aphelion).
  2. **Mars undisturbed:** criterion 1, and Mars's eccentricity changes by
     less than 0.01 (its actual e is 0.093).
- A mass counts as allowed at a given a only if all 4 phase sets pass at that
  mass and at every smaller mass.

## Results

| a (AU) | no crossing: max mass | Mars undisturbed: max mass |
|---|---|---|
| 2.1 | 2.1 M_J | 13 M⊕ (0.04 M_J) |
| 2.2 | 4.2 M_J | 21 M⊕ |
| 2.3 | 3.3 M_J | 26 M⊕ |
| 2.4 | 10.5 M_J | 13 M⊕ |
| 2.5 | 13.2 M_J | 66 M⊕ |
| 2.6 | 16.6 M_J | 66 M⊕ |
| 2.7 | 20.9 M_J | 84 M⊕ |
| 2.8 | 26.3 M_J | 105 M⊕ |
| 2.9 | 16.6 M_J | 133 M⊕ |
| 3.0 | 20.9 M_J | 167 M⊕ |
| 3.1 | 13.2 M_J | 167 M⊕ |
| 3.2 | 13.2 M_J | 210 M⊕ |
| 3.3 | 10.5 M_J | 528 M⊕ (1.7 M_J) |

(M_J = Jupiter masses, M⊕ = Earth masses; the grid resolution is 0.1 dex,
i.e. a factor of 1.26. Full values are in `critical_mass.csv`.)

Answer, under the 1000-yr run length:
- **No orbit crossing or ejection:** the asteroid can be as massive as about
  **2 M_J (0.002 M☉)** anywhere in the belt; the most favourable location
  (~2.8 AU) tolerates ~26 M_J. The first failures are Mars being scattered
  onto a crossing orbit or ejected.
- **Leaving Mars's orbit essentially intact:** the limit drops by about two
  orders of magnitude, to about **13 M⊕ (~4e-5 M☉)** in the inner belt, rising
  to ~100–500 M⊕ in the outer belt, further from Mars.
- For comparison, Ceres is ~4.7e-10 M☉ (log10 ≈ −9.3), about five orders of
  magnitude below even the stricter limit.

The two criteria differ because, near the crossing threshold, Mars's
eccentricity is pumped by up to ~0.5 while its orbit has not yet crossed
another. Such a system is technically "not crossed yet" after 1000 yr, but it
is clearly being disrupted. The stricter criterion is the more physically
meaningful answer.

## Caveats
- **1000 yr is very short.** Secular timescales for these orbits are ~10⁴–10⁵
  yr, and planetary-system instabilities can take Myr–Gyr. Longer runs would
  lower both thresholds, so these values are upper limits.
- Only 4 random phase sets per grid point, one asteroid eccentricity (0.05) and
  one inclination. Jagged features, such as the dip at 2.4 AU, are within
  phase-to-phase scatter and not established resonance effects.
- By construction, the other planets (Earth, Saturn, etc.) are excluded. Bodies
  above ~13 M_J are brown dwarfs rather than asteroids, so the upper end of the
  "no crossing" curve is a dynamical limit, not a realistic object.
- Instability is only checked every 0.5 yr from orbital elements. There is no
  explicit close-encounter detection, but orbit crossing precedes any
  encounter.

## Verification
- I (Claude) viewed `mass_sweep.png`. Every low-mass run is stable (4212 runs:
  3773 no-crossing, 3057 also Mars-undisturbed). The transition band sits
  just above the black curve, and the orange curve lies below the black curve
  at every a, as it must.
- One check I performed myself: _TODO_

## Reflection
- One statement supported directly by the data: _TODO_
- One statement requiring further analysis: _TODO_
