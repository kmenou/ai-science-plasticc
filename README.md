# AI Science: PLAsTiCC

Repository: <https://github.com/kmenou/ai-science-plasticc>

This small public course repository uses the PLAsTiCC astronomical time-series
data to practise scientific Python, coding-agent collaboration, visual
inspection, local verification, and pull-request review.

Week 1 contains two exercises: loading and summarising the supplied teaching
subset, and plotting a multiband light curve. Weeks 2–6 will be designed after
reviewing earlier student work.

## Setup

Use Python 3.12. From the repository root, run:

```bash
uv sync
uv run python scripts/preflight.py
```

If `uv` is unavailable, use the fallback installation:

```bash
python -m venv .venv
# activate the environment
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/preflight.py
```

Start with [the Week 1 practical](practicals/week01/README.md). All student work
belongs under `submissions/<student-or-pair-slug>/week01/`.

## Data and licenses

The original code and teaching materials are licensed under the MIT License.
The PLAsTiCC data retain their Creative Commons Attribution 4.0 International
license; see [data/README.md](data/README.md) for attribution and provenance.

No GitHub Actions or other CI are used. Verification is deliberately local.
