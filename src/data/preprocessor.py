"""
preprocessor.py
===============
Data cleaning, feature engineering, and train / val / test splitting.

Key features engineered
-----------------------
* Cart-level aggregates  (total price, item count, cuisine mix entropy)
* Candidate add-on features  (category, avg price, popularity rank)
* Cross features  (addon-cuisine affinity, price ratio addon/cart)
* Temporal features  (hour bucket, is_weekend)
* Embedding similarity  (cosine sim between cart embedding & addon embedding)

The output is a single Parquet with columns ready for LightGBM ranking.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the synthetic (or real) cart dataset."""
    path = path or PROCESSED_DIR / "synthetic_carts.parquet"
    return pd.read_parquet(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    raise NotImplementedError("Step 2: implement feature engineering")


def split_data(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Group-aware split (by session_id) so no session leaks across sets.
    """
    raise NotImplementedError("Step 2: implement group-aware splits")
