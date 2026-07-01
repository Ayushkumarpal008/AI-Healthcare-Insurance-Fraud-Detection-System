"""Shared utilities for the healthcare fraud detection project."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARDS_DIR = PROJECT_ROOT / "dashboards"
APP_DIR = PROJECT_ROOT / "app"


def ensure_directories() -> None:
    """Create the expected project directories if they are missing."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        DASHBOARDS_DIR,
        APP_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(name: str = "fraud_detection") -> logging.Logger:
    """Return a configured project logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds used by numpy and Python."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a JSON file with readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def optional_import(module_name: str) -> Any | None:
    """Import an optional dependency and return None when unavailable."""
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return module


def safe_divide(numerator: Any, denominator: Any) -> Any:
    """Divide while replacing zero denominators with NaN."""
    denominator = np.where(np.asarray(denominator) == 0, np.nan, denominator)
    return numerator / denominator

