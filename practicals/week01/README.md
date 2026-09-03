# Week 1 practical

Week 1 is an onboarding practical: install the environment, access PLAsTiCC in
two ways, make small task-level requests to a coding agent, create and inspect a
scientific figure, verify the work locally, and submit one pull request.

Suggested timing:

- 0–15 min: clone, install, and run preflight;
- 15–20 min: create a branch and submission folder;
- 20–38 min: Exercise 1;
- 38–50 min: Exercise 3 and visual inspection;
- 50–60 min: checker, diff, commit, push, and pull request.

All commands below run from the repository root.

## Setup

Install and verify the environment:

```bash
uv sync
uv run python scripts/preflight.py
```

Create one branch and copy the starter files:

```bash
git switch main
git pull --ff-only
git switch -c week01/<github-handle>
mkdir -p submissions/<github-handle>/week01
cp practicals/week01/starter.py \
  submissions/<github-handle>/week01/analysis.py
cp practicals/week01/REPORT_TEMPLATE.md \
  submissions/<github-handle>/week01/REPORT.md
```

If `cp` is unavailable, use a file manager. `starter.py` is a template and is
not intended to be run in place.

Your submission must contain exactly the required work products:

```text
analysis.py
lightcurve.png
REPORT.md
```

## The data

The metadata table has one row per astronomical object. The observations table
has many flux measurements per object, time, and passband. Inspect both schemas
before editing your copied starter.

For direct pandas access, read `TINY_METADATA_PATH` and
`TINY_OBSERVATIONS_PATH` from the starter with `pandas.read_parquet`. For course
access, use `plasticc_course.data`.

## Exercise 1 — load and summarise

Implement:

```python
def dataset_summary(method: str = "pandas") -> dict:
    ...
```

The accepted methods are:

- `"pandas"`: read the two supplied path constants directly with
  `pandas.read_parquet`;
- `"course"`: use `plasticc_course.data.load_metadata` and
  `load_observations`.

Any other value must raise a short `ValueError` naming both accepted values.
Return exactly these keys:

```text
metadata_rows
observation_rows
unique_objects
passbands
target_classes
detected_fraction
negative_flux_fraction
```

Use these exact definitions:

- `metadata_rows` — `len(metadata)` for the tiny metadata table.
- `observation_rows` — `len(observations)` for the tiny observations table.
- `unique_objects` — `metadata["object_id"].nunique()` from the tiny metadata
  table. Do not derive it from observations.
- `passbands` — sorted Python list of unique integer values in
  `observations["passband"]`.
- `target_classes` — sorted Python list of unique integer values in
  `metadata["target"]`.
- `detected_fraction` — observation rows where `detected == 1` divided by all
  observation rows. Do not drop rows.
- `negative_flux_fraction` — observation rows where `flux < 0`, using strict
  less-than, divided by all observation rows. Zero is not negative; do not drop
  rows.

The checker uses exact comparisons for counts and lists and absolute tolerance
`1e-6` for both fractions.

## Exercise 3 — plot a multiband light curve

Implement:

```python
def plot_lightcurve(object_id: int, output_path: str | Path) -> None:
    ...
```

Plot object `252646`, the selected target-90 Type Ia supernova. Create
`lightcurve.png` in your submission folder with:

- flux versus MJD;
- all six passbands visibly distinguished;
- flux-error bars;
- detected and non-detected observations visibly distinguished;
- labelled axes;
- the object ID in the title.

Your file must call `matplotlib.use("Agg")` before importing
`matplotlib.pyplot`, and the function must honor the caller's `output_path`.
Inspect the saved figure yourself and record that inspection in `REPORT.md`.
Figure readability is reviewed by a human; the checker does not score
aesthetics.

## Coding-agent prompts

You may use any coding agent. These task-sized prompts are suggested:

1. “Read `AGENTS.md` and the Week 1 README. Inspect the two tiny Parquet schemas
   and my copied starter. Do not edit yet. Explain what each table represents
   and restate the exact summary definitions.”
2. “Work only in my submission folder. Implement Exercise 1 for both access
   methods using the supplied paths. Run the checker and explain any failure
   before changing more code.”
3. “Implement Exercise 3 with the smallest reasonable change. Honor the
   supplied output path. Run the checker, help me compare the figure against the
   README, and summarize the final diff.”

Do not submit private chain-of-thought transcripts. Summarise useful requests,
suggestions, and errors in the report instead.

## Verify and submit

Run the checker against your folder:

```bash
uv run python practicals/week01/check_submission.py \
  submissions/<github-handle>/week01
```

Before pushing, run:

```bash
uv run python scripts/preflight.py
uv run python practicals/week01/check_submission.py \
  submissions/<github-handle>/week01
git status
git diff
```

Then commit and push:

```bash
git add submissions/<github-handle>/week01
git commit -m "Complete Week 1 PLAsTiCC practical"
git push -u origin week01/<github-handle>
```

Open one pull request to `main` containing both exercises. Keep any corrections
in the same pull request. If collaborator access is unavailable, fork the public
repository and open the pull request from your fork to upstream `main`.

