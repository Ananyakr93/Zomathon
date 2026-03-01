"""
ranker.py
=========
LightGBM LambdaRank model for Cart Super Add-On ranking.

Why LambdaRank?
---------------
The task is *ranking* add-on candidates by relevance to the current cart,
not binary classification.  LambdaRank directly optimises NDCG, which
aligns with the evaluation metric.  LightGBM's implementation is
extremely fast on CPU and handles the group (query = cart session)
structure natively.

Sequential Cart Updates
-----------------------
When a user adds an item, we re-compute the cart embedding & features and
re-rank remaining candidates.  Because LightGBM inference is <1 ms per
query group, we can do this on every cart mutation while staying well
under the 200 ms latency budget.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# ── Default hyper-parameters (tuned for CPU speed + quality) ───────────
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5, 10],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,       # use all cores
    "seed": 42,
}


def train(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> lgb.Booster:
    """
    Train a LightGBM LambdaRank model.

    Parameters
    ----------
    X_train, y_train : training features & relevance labels
    groups_train : array of group sizes (one per cart session)
    X_val, y_val, groups_val : optional validation set
    params : override default hyper-parameters
    num_boost_round : max boosting iterations
    early_stopping_rounds : early-stop patience

    Returns
    -------
    lgb.Booster  — trained model
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train)

    callbacks = [lgb.log_evaluation(period=50)]
    valid_sets = [train_set]
    valid_names = ["train"]

    if X_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_set)
        valid_sets.append(val_set)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(early_stopping_rounds))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    logger.info("Training complete — best iteration: %d", booster.best_iteration)
    return booster


def save_model(booster: lgb.Booster, name: str = "cart_ranker") -> Path:
    """Save trained booster to models/saved/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.txt"
    booster.save_model(str(path))
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str = "cart_ranker") -> lgb.Booster:
    """Load a saved booster."""
    path = MODEL_DIR / f"{name}.txt"
    return lgb.Booster(model_file=str(path))


def predict(booster: lgb.Booster, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return raw ranking scores for candidate add-ons."""
    return booster.predict(X)
