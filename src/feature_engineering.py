"""Feature engineering for healthcare fraud detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import CHRONIC_COLUMNS, HealthcareClaimsCleaner
from src.utils import PROCESSED_DATA_DIR, configure_logging, safe_divide


LOGGER = configure_logging(__name__)
DIAGNOSIS_PREFIX = "ClmDiagnosisCode_"
PROCEDURE_PREFIX = "ClmProcedureCode_"


class HealthcareFeatureEngineer:
    """Create claim, beneficiary, and provider behavior features."""

    def __init__(self) -> None:
        self.claim_amount_p95_: float = 0.0
        self.provider_claim_count_p95_: float = 0.0
        self.diagnosis_frequency_: dict[str, int] = {}
        self.procedure_frequency_: dict[str, int] = {}

    def fit(self, dataframe: pd.DataFrame) -> "HealthcareFeatureEngineer":
        """Fit thresholds and frequency maps used by feature engineering."""
        df = dataframe.copy()
        amount = pd.to_numeric(df.get("InscClaimAmtReimbursed", pd.Series(dtype=float)), errors="coerce")
        self.claim_amount_p95_ = float(amount.quantile(0.95)) if not amount.empty else 0.0

        if "Provider" in df.columns:
            provider_counts = df.groupby("Provider").size()
            self.provider_claim_count_p95_ = float(provider_counts.quantile(0.95))

        diagnosis_cols = [col for col in df.columns if col.startswith(DIAGNOSIS_PREFIX)]
        procedure_cols = [col for col in df.columns if col.startswith(PROCEDURE_PREFIX)]
        self.diagnosis_frequency_ = self._value_frequency(df, diagnosis_cols)
        self.procedure_frequency_ = self._value_frequency(df, procedure_cols)
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from cleaned claims."""
        df = dataframe.copy()

        if "ClaimStartDt" in df.columns:
            claim_start = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
        else:
            claim_start = pd.Series(pd.NaT, index=df.index)

        if "ClaimEndDt" in df.columns:
            claim_end = pd.to_datetime(df["ClaimEndDt"], errors="coerce")
        else:
            claim_end = claim_start

        if "DOB" in df.columns:
            dob = pd.to_datetime(df["DOB"], errors="coerce")
            df["PatientAge"] = safe_divide((claim_start - dob).dt.days, 365.25)
            df["PatientAge"] = df["PatientAge"].clip(lower=0, upper=115)
        else:
            df["PatientAge"] = np.nan

        if "DOD" in df.columns:
            dod = pd.to_datetime(df["DOD"], errors="coerce")
            df["IsDeceased"] = dod.notna().astype(int)
        else:
            df["IsDeceased"] = 0

        df["ClaimDurationDays"] = ((claim_end - claim_start).dt.days + 1).clip(lower=1)
        df["ClaimMonth"] = claim_start.dt.month
        df["ClaimDayOfWeek"] = claim_start.dt.dayofweek
        df["ClaimQuarter"] = claim_start.dt.quarter

        if {"AdmissionDt", "DischargeDt"}.issubset(df.columns):
            admission = pd.to_datetime(df["AdmissionDt"], errors="coerce")
            discharge = pd.to_datetime(df["DischargeDt"], errors="coerce")
            df["AdmissionLengthDays"] = ((discharge - admission).dt.days + 1).clip(lower=0)
            df["HasAdmission"] = admission.notna().astype(int)
        else:
            df["AdmissionLengthDays"] = 0
            df["HasAdmission"] = 0

        amount = pd.to_numeric(df.get("InscClaimAmtReimbursed", 0), errors="coerce").fillna(0)
        deductible = pd.to_numeric(df.get("DeductibleAmtPaid", 0), errors="coerce").fillna(0)
        df["ReimbursementToDeductibleRatio"] = safe_divide(amount, deductible + 1).replace([np.inf, -np.inf], np.nan)
        df["HighClaimAmountFlag"] = (amount > self.claim_amount_p95_).astype(int)

        annual_cols = [
            "IPAnnualReimbursementAmt",
            "OPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualDeductibleAmt",
        ]
        for column in annual_cols:
            if column not in df.columns:
                df[column] = 0
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        df["TotalAnnualReimbursementAmt"] = (
            df["IPAnnualReimbursementAmt"] + df["OPAnnualReimbursementAmt"]
        )
        df["TotalAnnualDeductibleAmt"] = (
            df["IPAnnualDeductibleAmt"] + df["OPAnnualDeductibleAmt"]
        )

        chronic_present = [column for column in CHRONIC_COLUMNS if column in df.columns]
        df["ChronicConditionCount"] = df[chronic_present].sum(axis=1) if chronic_present else 0

        diagnosis_cols = [col for col in df.columns if col.startswith(DIAGNOSIS_PREFIX)]
        procedure_cols = [col for col in df.columns if col.startswith(PROCEDURE_PREFIX)]
        df["DiagnosisCodeCount"] = df[diagnosis_cols].notna().sum(axis=1) if diagnosis_cols else 0
        df["ProcedureCodeCount"] = df[procedure_cols].notna().sum(axis=1) if procedure_cols else 0
        df["PrimaryDiagnosisFrequency"] = self._map_frequency(df, "ClmDiagnosisCode_1", self.diagnosis_frequency_)
        df["PrimaryProcedureFrequency"] = self._map_frequency(df, "ClmProcedureCode_1", self.procedure_frequency_)

        physician_cols = [
            col for col in ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"] if col in df.columns
        ]
        df["PhysicianCount"] = df[physician_cols].notna().sum(axis=1) if physician_cols else 0

        df = self._add_provider_features(df)
        df = self._add_beneficiary_features(df)

        for column in df.select_dtypes(include=["float64", "float32"]).columns:
            df[column] = df[column].replace([np.inf, -np.inf], np.nan)

        return df

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the dataframe."""
        return self.fit(dataframe).transform(dataframe)

    @staticmethod
    def _value_frequency(dataframe: pd.DataFrame, columns: list[str]) -> dict[str, int]:
        values = pd.Series(dtype=object)
        for column in columns:
            values = pd.concat([values, dataframe[column].dropna().astype(str)], ignore_index=True)
        return values.value_counts().to_dict()

    @staticmethod
    def _map_frequency(dataframe: pd.DataFrame, column: str, frequency: dict[str, int]) -> pd.Series:
        if column not in dataframe.columns:
            return pd.Series(0, index=dataframe.index)
        return dataframe[column].astype(str).map(frequency).fillna(0).astype(float)

    def _add_provider_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if "Provider" not in dataframe.columns:
            return dataframe
        df = dataframe.copy()
        provider_group = df.groupby("Provider", dropna=False)
        provider_stats = provider_group.agg(
            ProviderClaimCount=("ClaimID", "nunique") if "ClaimID" in df.columns else ("Provider", "size"),
            ProviderTotalReimbursement=("InscClaimAmtReimbursed", "sum"),
            ProviderAvgReimbursement=("InscClaimAmtReimbursed", "mean"),
            ProviderMaxReimbursement=("InscClaimAmtReimbursed", "max"),
            ProviderUniqueBeneficiaries=("BeneID", "nunique") if "BeneID" in df.columns else ("Provider", "size"),
            ProviderAvgClaimDuration=("ClaimDurationDays", "mean"),
            ProviderAvgDiagnosisCount=("DiagnosisCodeCount", "mean"),
            ProviderAvgProcedureCount=("ProcedureCodeCount", "mean"),
            ProviderInpatientClaims=("HasAdmission", "sum"),
        )
        provider_stats["ProviderInpatientRatio"] = safe_divide(
            provider_stats["ProviderInpatientClaims"],
            provider_stats["ProviderClaimCount"],
        ).fillna(0)
        provider_stats["ProviderHighVolumeFlag"] = (
            provider_stats["ProviderClaimCount"] > self.provider_claim_count_p95_
        ).astype(int)
        provider_stats = provider_stats.reset_index()
        return df.merge(provider_stats, on="Provider", how="left")

    @staticmethod
    def _add_beneficiary_features(dataframe: pd.DataFrame) -> pd.DataFrame:
        if "BeneID" not in dataframe.columns:
            return dataframe
        df = dataframe.copy()
        beneficiary_group = df.groupby("BeneID", dropna=False)
        beneficiary_stats = beneficiary_group.agg(
            BeneficiaryClaimCount=("ClaimID", "nunique") if "ClaimID" in df.columns else ("BeneID", "size"),
            BeneficiaryTotalReimbursement=("InscClaimAmtReimbursed", "sum"),
            BeneficiaryAvgReimbursement=("InscClaimAmtReimbursed", "mean"),
            BeneficiaryProviderCount=("Provider", "nunique") if "Provider" in df.columns else ("BeneID", "size"),
        ).reset_index()
        df = df.merge(beneficiary_stats, on="BeneID", how="left")

        if "ClaimStartDt" in df.columns:
            df = df.sort_values(["BeneID", "ClaimStartDt"])
            df["DaysSincePreviousClaim"] = (
                df.groupby("BeneID")["ClaimStartDt"].diff().dt.days.fillna(999)
            )
            df = df.sort_index()
        else:
            df["DaysSincePreviousClaim"] = 999
        return df


def build_features(dataframe: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Clean and engineer features, then optionally persist the result."""
    cleaner = HealthcareClaimsCleaner()
    cleaned = cleaner.fit_transform(dataframe)
    engineer = HealthcareFeatureEngineer()
    engineered = engineer.fit_transform(cleaned)
    if save:
        output_path = PROCESSED_DATA_DIR / "train_engineered_claims.csv"
        engineered.to_csv(output_path, index=False)
        LOGGER.info("Saved engineered data with shape %s to %s", engineered.shape, output_path)
    return engineered

