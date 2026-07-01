"""Prediction pipeline for new healthcare claim data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pyexpat import features
from typing import Any
from unittest import result
import numpy as np
import joblib
import pandas as pd
from streamlit import dataframe

from src.explainability import ExplanationEngine, risk_level
from src.train import MODEL_PATH
from src.utils import PROCESSED_DATA_DIR, configure_logging


LOGGER = configure_logging(__name__)


class FraudPredictor:
    """Reusable prediction interface for single or batch claims."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.bundle: dict[str, Any] = joblib.load(model_path)
        self.explainer = ExplanationEngine(self.bundle)

    def prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Clean, engineer, and align incoming data with training features."""

        cleaner = self.bundle["cleaner"]
        engineer = self.bundle["feature_engineer"]

    # STEP 1: cleaning + feature engineering
        cleaned = cleaner.transform(dataframe)
        engineered = engineer.transform(cleaned)

    # STEP 2: ensure all required columns exist
        for column in self.bundle["feature_columns"]:
            if column not in engineered.columns:
                engineered[column] = np.nan   

    # STEP 3: FIX categorical handling (IMPORTANT)
        for column in self.bundle["categorical_features"]:
            if column in engineered.columns:
                engineered[column] = engineered[column].astype(object)
                engineered[column] = engineered[column].replace("nan", np.nan)

    # STEP 4: FINAL SAFE CONVERSION (MOST IMPORTANT FIX)
        engineered = engineered.replace({pd.NA: np.nan})

        return engineered

    def predict_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud risk for a dataframe of raw or merged claim records."""

        engineered = self.prepare(dataframe)

        features = engineered[self.bundle["feature_columns"]].copy()

        # Convert pandas nullable values to normal numpy NaN
        features = features.astype(object)
        features = features.replace(
            [pd.NA, "nan", "NaN", "None", "none", "", "NA", "N/A"],
            np.nan
        )
        features = features.where(pd.notna(features), np.nan)

        categorical_cols = self.bundle.get("categorical_features", [])
        numeric_cols = [
            col for col in self.bundle["feature_columns"]
            if col not in categorical_cols
        ]

        # Only numeric columns should be converted to number
        for col in numeric_cols:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors="coerce")

        # Keep categorical columns as normal object, not pandas string dtype
        for col in categorical_cols:
            if col in features.columns:
                features[col] = features[col].astype(object)
                features[col] = features[col].where(pd.notna(features[col]), np.nan)

        transformed = self.bundle["preprocessor"].transform(features)

        probability = self.bundle["model"].predict_proba(transformed)[:, 1]
        probability = np.nan_to_num(probability.astype(float), nan=0.0)

        threshold = self.bundle.get("threshold", 0.5)

        result = dataframe.copy()
        result["fraud_probability"] = probability
        result["fraud_prediction"] = [
            "Fraud" if float(value) >= threshold else "Not Fraud"
            for value in probability
        ]
        result["risk_level"] = [risk_level(float(value)) for value in probability]

        explanations = []
        for idx, prob in enumerate(probability):
            try:
                explanations.append(
                self.explainer.explain_record(engineered.iloc[idx], float(prob))
                )
            except Exception:
                explanations.append("Explanation unavailable")

        result["explanation"] = explanations

        return result

    def predict_single(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Predict one claim and return a JSON-serializable response."""
        result = self.predict_dataframe(pd.DataFrame([claim])).iloc[0]
        probability = float(result["fraud_probability"])
        return {
            "fraud_prediction": result["fraud_prediction"],
            "fraud_probability": f"{probability:.1%}",
            "risk_level": result["risk_level"],
            "explanation": result["explanation"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict healthcare fraud risk.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV with claims.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "predictions.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = FraudPredictor()
    input_df = pd.read_csv(args.input)
    predictions = predictor.predict_dataframe(input_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    LOGGER.info("Saved predictions to %s", args.output)
    print(json.dumps(predictions.head(3).to_dict(orient="records"), default=str, indent=2))

