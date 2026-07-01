"""Data loading and merging for the Kaggle provider fraud dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    configure_logging,
    ensure_directories,
)


LOGGER = configure_logging(__name__)


@dataclass(frozen=True)
class DatasetFiles:
    """Resolved CSV file paths for one split."""

    beneficiary: Path
    inpatient: Path
    outpatient: Path
    provider_labels: Path | None = None


def _find_csv(patterns: list[str], search_dirs: list[Path]) -> Path:
    """Find the first CSV matching any of the provided patterns."""
    for directory in search_dirs:
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    joined = ", ".join(patterns)
    raise FileNotFoundError(f"Could not find CSV matching: {joined}")


def resolve_dataset_files(split: str = "train") -> DatasetFiles:
    """Resolve raw dataset files for train or test data.

    The Kaggle files are often downloaded with timestamp suffixes, so this
    function accepts both canonical names and timestamped names.
    """
    split_lower = split.lower()
    split_title = split_lower.capitalize()
    search_dirs = [RAW_DATA_DIR, PROJECT_ROOT]

    beneficiary = _find_csv(
        [
            f"{split_title}_Beneficiarydata.csv",
            f"{split_title}_Beneficiarydata*.csv",
        ],
        search_dirs,
    )
    inpatient = _find_csv(
        [
            f"{split_title}_Inpatientdata.csv",
            f"{split_title}_Inpatientdata*.csv",
        ],
        search_dirs,
    )
    outpatient = _find_csv(
        [
            f"{split_title}_Outpatientdata.csv",
            f"{split_title}_Outpatientdata*.csv",
        ],
        search_dirs,
    )

    labels = None
    if split_lower == "train":
        labels = _find_csv(
            [f"{split_title}.csv", f"{split_title}-*.csv"],
            search_dirs,
        )
    else:
        try:
            labels = _find_csv([f"{split_title}.csv", f"{split_title}-*.csv"], search_dirs)
        except FileNotFoundError:
            labels = None

    return DatasetFiles(
        beneficiary=beneficiary,
        inpatient=inpatient,
        outpatient=outpatient,
        provider_labels=labels,
    )


def load_raw_data(split: str = "train", nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """Load raw CSV files for the selected split."""
    files = resolve_dataset_files(split)
    LOGGER.info("Loading %s beneficiary data from %s", split, files.beneficiary.name)
    beneficiary = pd.read_csv(files.beneficiary, nrows=nrows)
    LOGGER.info("Loading %s inpatient claims from %s", split, files.inpatient.name)
    inpatient = pd.read_csv(files.inpatient, nrows=nrows)
    LOGGER.info("Loading %s outpatient claims from %s", split, files.outpatient.name)
    outpatient = pd.read_csv(files.outpatient, nrows=nrows)

    data = {
        "beneficiary": beneficiary,
        "inpatient": inpatient,
        "outpatient": outpatient,
    }
    if files.provider_labels:
        LOGGER.info("Loading provider labels from %s", files.provider_labels.name)
        data["provider_labels"] = pd.read_csv(files.provider_labels)
    return data


def combine_claims(inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> pd.DataFrame:
    """Combine inpatient and outpatient claims with a claim type indicator."""
    inpatient_claims = inpatient.copy()
    outpatient_claims = outpatient.copy()
    inpatient_claims["ClaimType"] = "Inpatient"
    outpatient_claims["ClaimType"] = "Outpatient"

    combined = pd.concat([inpatient_claims, outpatient_claims], ignore_index=True, sort=False)
    return combined


def merge_claims_with_beneficiaries(
    beneficiary: pd.DataFrame,
    inpatient: pd.DataFrame,
    outpatient: pd.DataFrame,
    provider_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge claim, beneficiary, and provider label datasets."""
    claims = combine_claims(inpatient, outpatient)
    merged = claims.merge(beneficiary, on="BeneID", how="left", validate="many_to_one")

    if provider_labels is not None and "PotentialFraud" in provider_labels.columns:
        merged = merged.merge(provider_labels, on="Provider", how="left", validate="many_to_one")
        merged["PotentialFraud"] = merged["PotentialFraud"].fillna("Unknown")

    return merged


def describe_schema(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a compact schema report for loaded dataframes."""
    rows = []
    for name, dataframe in dataframes.items():
        for column in dataframe.columns:
            rows.append(
                {
                    "dataset": name,
                    "column": column,
                    "dtype": str(dataframe[column].dtype),
                    "missing_count": int(dataframe[column].isna().sum()),
                    "unique_count": int(dataframe[column].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def load_and_merge(
    split: str = "train",
    nrows: int | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Load raw data, merge it, and optionally persist the merged dataframe."""
    ensure_directories()
    raw = load_raw_data(split=split, nrows=nrows)
    schema = describe_schema(raw)
    schema_path = PROCESSED_DATA_DIR / f"{split}_schema.csv"
    schema.to_csv(schema_path, index=False)
    LOGGER.info("Saved schema report to %s", schema_path)

    merged = merge_claims_with_beneficiaries(
        beneficiary=raw["beneficiary"],
        inpatient=raw["inpatient"],
        outpatient=raw["outpatient"],
        provider_labels=raw.get("provider_labels"),
    )

    if save:
        output_path = PROCESSED_DATA_DIR / f"{split}_merged_claims.csv"
        merged.to_csv(output_path, index=False)
        LOGGER.info("Saved merged data with shape %s to %s", merged.shape, output_path)

    return merged


if __name__ == "__main__":
    load_and_merge("train")

