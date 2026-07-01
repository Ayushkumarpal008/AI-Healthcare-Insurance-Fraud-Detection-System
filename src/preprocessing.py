"""Reusable preprocessing utilities for healthcare claim data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATE_COLUMNS = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD"]
AMOUNT_COLUMNS = [
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt",
]
CHRONIC_COLUMNS = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]


class HealthcareClaimsCleaner:
    """Clean merged healthcare claim records before feature engineering."""

    def fit(self, dataframe: pd.DataFrame) -> "HealthcareClaimsCleaner":
        """Fit the cleaner. Kept for pipeline symmetry."""
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the input dataframe."""
        df = dataframe.copy()
        df = df.drop_duplicates()

        for column in DATE_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")

        for column in AMOUNT_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
                df.loc[df[column] < 0, column] = np.nan

        if "DeductibleAmtPaid" in df.columns:
            df["DeductibleAmtPaid"] = df["DeductibleAmtPaid"].fillna(0)

        for column in CHRONIC_COLUMNS:
            if column in df.columns:
                numeric = pd.to_numeric(df[column], errors="coerce")
                df[column] = np.where(numeric == 1, 1, 0)

        if "RenalDiseaseIndicator" in df.columns:
            renal = df["RenalDiseaseIndicator"].astype(str).str.strip().str.upper()
            df["RenalDiseaseIndicator"] = np.where(renal.isin(["1", "Y", "YES", "TRUE"]), 1, 0)

        if {"ClaimStartDt", "ClaimEndDt"}.issubset(df.columns):
            invalid = df["ClaimEndDt"].notna() & df["ClaimStartDt"].notna()
            invalid &= df["ClaimEndDt"] < df["ClaimStartDt"]
            df.loc[invalid, "ClaimEndDt"] = df.loc[invalid, "ClaimStartDt"]

        if {"AdmissionDt", "DischargeDt"}.issubset(df.columns):
            invalid = df["DischargeDt"].notna() & df["AdmissionDt"].notna()
            invalid &= df["DischargeDt"] < df["AdmissionDt"]
            df.loc[invalid, "DischargeDt"] = df.loc[invalid, "AdmissionDt"]

        for column in ["Gender", "Race", "State", "County", "ClaimType"]:
            if column in df.columns:
                df[column] = df[column].astype("category")

        if "PotentialFraud" in df.columns:
            df["PotentialFraud"] = df["PotentialFraud"].astype(str).str.strip()

        all_null_columns = [column for column in df.columns if df[column].isna().all()]
        protected_prefixes = ("ClmDiagnosisCode_", "ClmProcedureCode_")
        removable = [
            column
            for column in all_null_columns
            if not column.startswith(protected_prefixes)
        ]
        if removable:
            df = df.drop(columns=removable)

        return df

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the dataframe."""
        return self.fit(dataframe).transform(dataframe)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_model_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a scikit-learn preprocessing transformer for model training."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

