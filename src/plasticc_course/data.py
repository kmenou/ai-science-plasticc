"""Small, file-relative data-loading helpers for the course datasets."""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
_DATASETS = {
    "tiny": {
        "metadata": DATA_DIR / "plasticc_tiny300_metadata.parquet",
        "observations": DATA_DIR / "plasticc_tiny300_observations.parquet",
    },
    "full": {
        "metadata": DATA_DIR / "plasticc_train_metadata.parquet",
        "observations": DATA_DIR / "plasticc_train_observations.parquet",
    },
}


def _path(dataset: str, table: str) -> Path:
    if dataset not in _DATASETS:
        raise ValueError("dataset must be 'tiny' or 'full'")
    return _DATASETS[dataset][table]


def load_metadata(dataset: str = "tiny") -> pd.DataFrame:
    """Return the selected one-row-per-object metadata table."""

    return pd.read_parquet(_path(dataset, "metadata"))


def load_observations(dataset: str = "tiny") -> pd.DataFrame:
    """Return the selected time-series observation table."""

    return pd.read_parquet(_path(dataset, "observations"))


def load_object(object_id: int, dataset: str = "tiny") -> pd.DataFrame:
    """Return all observations for one object in the selected dataset."""

    metadata = load_metadata(dataset)
    if object_id not in set(metadata["object_id"]):
        raise KeyError(f"object_id {object_id} is not present in the {dataset} dataset")
    observations = load_observations(dataset)
    return observations.loc[observations["object_id"] == object_id].reset_index(drop=True)

