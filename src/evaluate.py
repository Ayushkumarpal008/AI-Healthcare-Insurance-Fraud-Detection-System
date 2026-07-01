"""Evaluation, visualization, and reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import REPORTS_DIR, configure_logging, save_json


LOGGER = configure_logging(__name__)
sns.set_theme(style="whitegrid")


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate fraud detection metrics for a probability threshold."""
    y_pred = (y_probability >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "fraud_detection_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def choose_recall_focused_threshold(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray,
) -> float:
    """Choose a threshold with a recall-heavy F2 score objective."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probability)
    if len(thresholds) == 0:
        return 0.5
    beta_squared = 4.0
    f2_scores = (1 + beta_squared) * precision[:-1] * recall[:-1]
    denominator = beta_squared * precision[:-1] + recall[:-1]
    f2_scores = np.divide(f2_scores, denominator, out=np.zeros_like(f2_scores), where=denominator != 0)
    best_index = int(np.nanargmax(f2_scores))
    return float(thresholds[best_index])


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    y_pred = (y_probability >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_pr_curves(
    y_true: np.ndarray | pd.Series,
    y_probability: np.ndarray,
    output_dir: Path,
) -> None:
    """Save ROC and precision-recall curve visualizations."""
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    precision, recall, _ = precision_recall_curve(y_true, y_probability)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6d28d9", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#7c3aed", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)


def save_model_comparison(results: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    """Save model comparison metrics."""
    comparison = pd.DataFrame(results).sort_values(
        ["recall", "pr_auc", "precision"],
        ascending=[False, False, False],
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    return comparison


def generate_eda(dataframe: pd.DataFrame, output_dir: Path | None = None) -> None:
    """Generate exploratory analysis charts and summary tables."""
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df = dataframe.copy()
    target = "PotentialFraud"

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    }
    if target in df.columns:
        summary["class_distribution"] = df[target].value_counts(dropna=False).to_dict()
    save_json(summary, output_dir / "eda_summary.json")

    if target in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=target, palette=["#7e22ce", "#ef4444"], ax=ax)
        ax.set_title("Fraud Class Distribution")
        fig.tight_layout()
        fig.savefig(output_dir / "class_distribution.png", dpi=160)
        plt.close(fig)

    if "InscClaimAmtReimbursed" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["InscClaimAmtReimbursed"].dropna(), bins=60, color="#7c3aed", ax=ax)
        ax.set_title("Claim Reimbursement Distribution")
        fig.tight_layout()
        fig.savefig(output_dir / "claim_amount_histogram.png", dpi=160)
        plt.close(fig)

        if target in df.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=df, x=target, y="InscClaimAmtReimbursed", palette="magma", ax=ax)
            ax.set_yscale("symlog")
            ax.set_title("Claim Amount by Fraud Label")
            fig.tight_layout()
            fig.savefig(output_dir / "claim_amount_by_fraud_boxplot.png", dpi=160)
            plt.close(fig)

    if "PatientAge" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        hue = target if target in df.columns else None
        sns.histplot(data=df, x="PatientAge", hue=hue, bins=40, multiple="stack", ax=ax)
        ax.set_title("Patient Age Distribution")
        fig.tight_layout()
        fig.savefig(output_dir / "age_distribution.png", dpi=160)
        plt.close(fig)

    if {"Provider", target}.issubset(df.columns):
        provider = (
            df.groupby("Provider")
            .agg(
                claims=("ClaimID", "nunique"),
                fraud=(target, lambda x: int((x == "Yes").max())),
                total_reimbursement=("InscClaimAmtReimbursed", "sum"),
            )
            .sort_values("claims", ascending=False)
            .head(20)
            .reset_index()
        )
        provider.to_csv(output_dir / "top_provider_patterns.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=provider, y="Provider", x="claims", hue="fraud", dodge=False, palette="rocket", ax=ax)
        ax.set_title("Top Providers by Claim Volume")
        fig.tight_layout()
        fig.savefig(output_dir / "top_providers_by_claims.png", dpi=160)
        plt.close(fig)

    if {"ClaimMonth", target}.issubset(df.columns):
        trend = (
            df.assign(FraudFlag=(df[target] == "Yes").astype(int))
            .groupby("ClaimMonth")
            .agg(claims=("ClaimID", "nunique"), fraud_rate=("FraudFlag", "mean"))
            .reset_index()
        )
        trend.to_csv(output_dir / "monthly_fraud_trends.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=trend, x="ClaimMonth", y="claims", marker="o", ax=ax1, color="#6d28d9")
        ax2 = ax1.twinx()
        sns.lineplot(data=trend, x="ClaimMonth", y="fraud_rate", marker="s", ax=ax2, color="#ef4444")
        ax1.set_title("Monthly Claim Volume and Fraud Rate")
        ax1.set_ylabel("Claims")
        ax2.set_ylabel("Fraud Rate")
        fig.tight_layout()
        fig.savefig(output_dir / "monthly_fraud_trends.png", dpi=160)
        plt.close(fig)

    numeric = df.select_dtypes(include=[np.number])
    preferred = [
        "InscClaimAmtReimbursed",
        "DeductibleAmtPaid",
        "PatientAge",
        "ChronicConditionCount",
        "ProviderClaimCount",
        "ProviderAvgReimbursement",
        "BeneficiaryClaimCount",
        "DiagnosisCodeCount",
        "ProcedureCodeCount",
    ]
    columns = [col for col in preferred if col in numeric.columns]
    if len(columns) >= 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(numeric[columns].corr(), cmap="magma", annot=False, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(output_dir / "correlation_heatmap.png", dpi=160)
        plt.close(fig)


def generate_markdown_report(
    metrics: dict[str, Any],
    model_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a concise project report in Markdown."""
    best_model = metrics.get("best_model", "Unknown")
    table_columns = [
        column
        for column in [
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "fraud_detection_rate",
            "false_positive_rate",
        ]
        if column in model_comparison.columns
    ]
    comparison_table = model_comparison[table_columns].copy()
    for column in comparison_table.select_dtypes(include=[np.number]).columns:
        comparison_table[column] = comparison_table[column].map(lambda value: f"{value:.4f}")
    markdown_table = [
        "| " + " | ".join(comparison_table.columns) + " |",
        "| " + " | ".join(["---"] * len(comparison_table.columns)) + " |",
    ]
    for _, row in comparison_table.iterrows():
        markdown_table.append("| " + " | ".join(str(row[column]) for column in comparison_table.columns) + " |")
    lines = [
        "# AI System for Detecting Financial Fraud in Healthcare Insurance Claims",
        "",
        "## Problem Statement",
        "Detect suspicious healthcare insurance claims and provider billing behavior using supervised machine learning.",
        "",
        "## Dataset",
        "Kaggle Healthcare Provider Fraud Detection data merged from beneficiary, inpatient, outpatient, and provider label files.",
        "",
        "## Methodology",
        "Claims are cleaned, date fields are normalized, inpatient and outpatient claims are combined, beneficiary attributes are joined, and provider labels are attached.",
        "Feature engineering creates patient age, claim duration, chronic-condition counts, claim amount ratios, provider behavior metrics, beneficiary activity metrics, and code-frequency features.",
        "",
        "## Model Results",
        f"Selected model: **{best_model}**",
        "",
        "\n".join(markdown_table),
        "",
        "## Final Model Metrics",
    ]
    for key, value in metrics.get("best_metrics", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Explainability",
            "The prediction layer returns risk level, fraud probability, global feature importance, and domain-specific local reasons such as abnormal reimbursement amount, high provider claim volume, high inpatient ratio, and unusual beneficiary activity.",
            "",
            "## Future Improvements",
            "- Deep learning over sequential claim histories",
            "- Unsupervised anomaly detection for unknown fraud patterns",
            "- Graph-based provider-beneficiary-physician fraud rings",
            "- Real-time prediction API",
            "- Cloud deployment and monitoring",
            "- Scheduled model retraining with drift detection",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Saved report to %s", output_path)
