"""Train and evaluate fraud detection models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from src.data_loader import load_and_merge
from src.evaluate import (
    choose_recall_focused_threshold,
    classification_metrics,
    generate_eda,
    generate_markdown_report,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    save_model_comparison,
)
from src.explainability import build_feature_profile
from src.feature_engineering import HealthcareFeatureEngineer
from src.preprocessing import HealthcareClaimsCleaner, build_model_preprocessor
from src.utils import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, configure_logging, ensure_directories, save_json, set_seed


LOGGER = configure_logging(__name__)
TARGET_COLUMN = "PotentialFraud"
MODEL_PATH = MODELS_DIR / "fraud_detection_model.joblib"


EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "FraudFlag",
    "Provider",
    "BeneID",
    "ClaimID",
    "DOB",
    "DOD",
    "ClaimStartDt",
    "ClaimEndDt",
    "AdmissionDt",
    "DischargeDt",
    "AttendingPhysician",
    "OperatingPhysician",
    "OtherPhysician",
    "DiagnosisGroupCode",
    "ClmAdmitDiagnosisCode",
}


def _optional_model_classes() -> dict[str, Any]:
    """Return optional model classes for installed gradient boosting libraries."""
    classes: dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier

        classes["XGBoost"] = XGBClassifier
    except ImportError:
        LOGGER.info("XGBoost is not installed; skipping XGBoost.")
    try:
        from lightgbm import LGBMClassifier

        classes["LightGBM"] = LGBMClassifier
    except ImportError:
        LOGGER.info("LightGBM is not installed; skipping LightGBM.")
    try:
        from catboost import CatBoostClassifier

        classes["CatBoost"] = CatBoostClassifier
    except ImportError:
        LOGGER.info("CatBoost is not installed; skipping CatBoost.")
    return classes


def select_model_features(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select numeric and categorical columns suitable for model training."""
    feature_columns = []
    for column in dataframe.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if column.startswith("ClmDiagnosisCode_") or column.startswith("ClmProcedureCode_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            continue
        feature_columns.append(column)

    categorical_candidates = ["ClaimType", "Gender", "Race", "RenalDiseaseIndicator", "State", "County"]
    categorical_features = [column for column in categorical_candidates if column in feature_columns]
    numeric_features = [
        column
        for column in feature_columns
        if column not in categorical_features and pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    return numeric_features, categorical_features


def make_models(random_state: int = 42) -> dict[str, Any]:
    """Create model candidates."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=90,
            learning_rate=0.06,
            max_depth=3,
            random_state=random_state,
        ),
    }
    optional = _optional_model_classes()
    if "XGBoost" in optional:
        models["XGBoost"] = optional["XGBoost"](
            n_estimators=160,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    if "LightGBM" in optional:
        models["LightGBM"] = optional["LightGBM"](
            n_estimators=180,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
        )
    if "CatBoost" in optional:
        models["CatBoost"] = optional["CatBoost"](
            iterations=180,
            learning_rate=0.05,
            depth=5,
            loss_function="Logloss",
            verbose=False,
            random_seed=random_state,
        )
    return models


def create_splits(
    dataframe: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a provider-aware train/test split when Provider exists."""
    if "Provider" in dataframe.columns and dataframe["Provider"].nunique() > 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = dataframe["Provider"]
        train_idx, test_idx = next(splitter.split(dataframe[features], target, groups=groups))
        return (
            dataframe.iloc[train_idx][features],
            dataframe.iloc[test_idx][features],
            target.iloc[train_idx],
            target.iloc[test_idx],
        )

    return train_test_split(
        dataframe[features],
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )


def compare_imbalance_methods(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: Any,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare class weighting, oversampling, undersampling, and optional SMOTE."""
    rows = []

    transformed_train = preprocessor.fit_transform(x_train)
    transformed_test = preprocessor.transform(x_test)

    methods = {
        "class_weight": (transformed_train, y_train),
    }

    train_frame = pd.DataFrame(transformed_train)
    train_frame["target"] = y_train.to_numpy()
    majority = train_frame[train_frame["target"] == 0]
    minority = train_frame[train_frame["target"] == 1]
    if len(minority) > 0 and len(majority) > 0:
        oversampled_minority = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42,
        )
        oversampled = pd.concat([majority, oversampled_minority], axis=0)
        methods["random_oversampling"] = (
            oversampled.drop(columns="target").to_numpy(),
            oversampled["target"].astype(int),
        )
        undersampled_majority = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=42,
        )
        undersampled = pd.concat([undersampled_majority, minority], axis=0)
        methods["random_undersampling"] = (
            undersampled.drop(columns="target").to_numpy(),
            undersampled["target"].astype(int),
        )

    try:
        from imblearn.over_sampling import SMOTE

        smote_x, smote_y = SMOTE(random_state=42).fit_resample(transformed_train, y_train)
        methods["smote"] = (smote_x, smote_y)
    except ImportError:
        LOGGER.info("imbalanced-learn is not installed; SMOTE comparison skipped.")

    for method, (x_resampled, y_resampled) in methods.items():
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(x_resampled, y_resampled)
        probability = model.predict_proba(transformed_test)[:, 1]
        rows.append(
            {
                "method": method,
                "pr_auc": average_precision_score(y_test, probability),
                **classification_metrics(y_test, probability, threshold=0.5),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("recall", ascending=False)
    comparison.to_csv(output_dir / "imbalance_comparison.csv", index=False)
    return comparison


def tune_random_forest(
    x_train_transformed: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Run a compact hyperparameter search for Random Forest."""
    max_rows = min(30000, len(y_train))
    indices = np.random.default_rng(random_state).choice(len(y_train), size=max_rows, replace=False)
    estimator = RandomForestClassifier(
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    grid = {
        "n_estimators": [80, 140],
        "max_depth": [10, 16],
        "min_samples_leaf": [20, 50],
    }
    search = GridSearchCV(
        estimator,
        param_grid=grid,
        scoring="average_precision",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train_transformed[indices], y_train.iloc[indices])
    LOGGER.info("Best Random Forest params: %s", search.best_params_)
    return search.best_estimator_


def train_pipeline(max_rows: int | None = 120000, random_state: int = 42) -> dict[str, Any]:
    """Run the full training pipeline and save outputs."""
    ensure_directories()
    set_seed(random_state)

    LOGGER.info("Starting data load and merge.")
    merged = load_and_merge("train", save=True)
    cleaner = HealthcareClaimsCleaner()
    cleaned = cleaner.fit_transform(merged)
    feature_engineer = HealthcareFeatureEngineer()
    engineered = feature_engineer.fit_transform(cleaned)
    engineered_path = PROCESSED_DATA_DIR / "train_engineered_claims.csv"
    engineered.to_csv(engineered_path, index=False)
    LOGGER.info("Saved engineered data to %s", engineered_path)

    generate_eda(engineered, REPORTS_DIR)

    trainable = engineered[engineered[TARGET_COLUMN].isin(["Yes", "No"])].copy()
    trainable["FraudFlag"] = (trainable[TARGET_COLUMN] == "Yes").astype(int)
    if max_rows and max_rows > 0 and len(trainable) > max_rows:
        trainable, _ = train_test_split(
            trainable,
            train_size=max_rows,
            stratify=trainable["FraudFlag"],
            random_state=random_state,
        )
        trainable = trainable.reset_index(drop=True)
        LOGGER.info("Training on a stratified sample of %s rows for local runtime.", len(trainable))

    numeric_features, categorical_features = select_model_features(trainable)
    feature_columns = numeric_features + categorical_features
    for column in categorical_features:
        trainable[column] = trainable[column].astype(str)

    target = trainable["FraudFlag"]
    x_train, x_test, y_train, y_test = create_splits(trainable, feature_columns, target, random_state=random_state)
    preprocessor = build_model_preprocessor(numeric_features, categorical_features)

    imbalance = compare_imbalance_methods(x_train, y_train, x_test, y_test, preprocessor, REPORTS_DIR)
    LOGGER.info("Imbalance comparison complete: %s", imbalance[["method", "recall", "precision"]].to_dict("records"))

    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    try:
        transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        transformed_feature_names = [f"feature_{idx}" for idx in range(x_train_transformed.shape[1])]

    models = make_models(random_state=random_state)
    models["Tuned Random Forest"] = tune_random_forest(x_train_transformed, y_train, random_state=random_state)

    results = []
    fitted_models = {}
    best_name = None
    best_score = -np.inf
    best_probability = None
    best_threshold = 0.5

    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(x_train_transformed, y_train)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(x_test_transformed)[:, 1]
        else:
            raw_score = model.decision_function(x_test_transformed)
            probability = 1 / (1 + np.exp(-raw_score))
        threshold = choose_recall_focused_threshold(y_test, probability)
        metrics = classification_metrics(y_test, probability, threshold=threshold)
        metrics["model"] = name
        results.append(metrics)
        fitted_models[name] = model
        score = (2.0 * metrics["recall"]) + metrics["pr_auc"] + (0.25 * metrics["precision"])
        if score > best_score:
            best_score = score
            best_name = name
            best_probability = probability
            best_threshold = threshold

    comparison = save_model_comparison(results, REPORTS_DIR)
    assert best_name is not None and best_probability is not None
    best_model = fitted_models[best_name]
    best_metrics = classification_metrics(y_test, best_probability, threshold=best_threshold)
    best_metrics["model"] = best_name

    plot_confusion_matrix(y_test, best_probability, best_threshold, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_pr_curves(y_test, best_probability, REPORTS_DIR)

    profile = build_feature_profile(trainable, feature_columns)
    bundle = {
        "model": best_model,
        "preprocessor": preprocessor,
        "cleaner": cleaner,
        "feature_engineer": feature_engineer,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "transformed_feature_names": transformed_feature_names,
        "threshold": best_threshold,
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "feature_profile": profile,
    }
    joblib.dump(bundle, MODEL_PATH)
    LOGGER.info("Saved model bundle to %s", MODEL_PATH)

    run_summary = {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "class_distribution": target.value_counts().to_dict(),
        "feature_count": len(feature_columns),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "training_rows": int(len(trainable)),
        "model_path": str(MODEL_PATH),
    }
    save_json(run_summary, REPORTS_DIR / "training_summary.json")
    generate_markdown_report(run_summary, comparison, REPORTS_DIR / "project_report.md")
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train healthcare fraud detection models.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=120000,
        help="Maximum stratified rows for local training. Use 0 for full data.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_rows = None if args.max_rows == 0 else args.max_rows
    train_pipeline(max_rows=max_rows, random_state=args.random_state)
