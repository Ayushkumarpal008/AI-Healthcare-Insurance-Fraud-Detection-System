"""Explainability helpers for model predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DOMAIN_REASON_RULES = [
    (
        "InscClaimAmtReimbursed",
        "High claim reimbursement amount compared with typical claims.",
    ),
    (
        "ProviderClaimCount",
        "Provider has unusually high claim volume.",
    ),
    (
        "ProviderAvgReimbursement",
        "Provider average reimbursement is elevated.",
    ),
    (
        "ProviderInpatientRatio",
        "Provider has a high inpatient claim ratio.",
    ),
    (
        "BeneficiaryClaimCount",
        "Beneficiary has frequent claim activity.",
    ),
    (
        "DiagnosisCodeCount",
        "Claim contains many diagnosis codes.",
    ),
    (
        "ProcedureCodeCount",
        "Claim contains many procedure codes.",
    ),
    (
        "ReimbursementToDeductibleRatio",
        "Reimbursement is unusually high relative to deductible paid.",
    ),
]


def risk_level(probability: float) -> str:
    """Convert fraud probability to a readable risk level."""
    if probability >= 0.75:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


class ExplanationEngine:
    """Create global and local explanations without requiring SHAP."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle

    def global_feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        """Return global feature importance when available."""
        model = self.bundle["model"]
        names = self.bundle.get("transformed_feature_names", [])
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_).ravel()
        else:
            values = np.zeros(len(names))

        if len(names) != len(values):
            names = [f"feature_{idx}" for idx in range(len(values))]

        return (
            pd.DataFrame({"feature": names, "importance": values})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def explain_record(
        self,
        engineered_record: pd.Series,
        fraud_probability: float,
        max_reasons: int = 5,
    ) -> list[str]:
        """Return human-readable reasons for a single prediction."""
        reasons: list[str] = []
        profile = self.bundle.get("feature_profile", {})

        for feature, reason in DOMAIN_REASON_RULES:
            if feature not in engineered_record.index:
                continue
            value = engineered_record.get(feature)
            high_cutoff = profile.get(feature, {}).get("p90")
            if pd.notna(value) and high_cutoff is not None and value >= high_cutoff:
                reasons.append(reason)

        if fraud_probability >= 0.75 and not reasons:
            reasons.append("The model identified a high-risk combination of provider, claim, and beneficiary behavior.")
        elif fraud_probability >= 0.4 and not reasons:
            reasons.append("The claim has a moderate-risk feature profile compared with historical claims.")
        elif not reasons:
            reasons.append("No strong fraud indicators were detected in the available claim features.")

        return reasons[:max_reasons]


def build_feature_profile(dataframe: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    """Store high-percentile feature cutoffs for explanations."""
    profile: dict[str, dict[str, float]] = {}
    for feature in features:
        if feature in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature]):
            series = dataframe[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if not series.empty:
                profile[feature] = {
                    "p50": float(series.quantile(0.5)),
                    "p90": float(series.quantile(0.9)),
                    "p95": float(series.quantile(0.95)),
                }
    return profile

