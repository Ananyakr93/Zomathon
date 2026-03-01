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
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ═════════════════════════════════════════════════════════════════════════════
#  LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the synthetic (or real) cart dataset."""
    path = path or PROCESSED_DIR / "synthetic_carts.parquet"
    return pd.read_parquet(path)


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Input DataFrame should have at minimum:
      session_id, cart_items (list), candidate_name, candidate_category,
      candidate_price, candidate_is_veg, label (0/1), hour, day_of_week,
      cuisine_type, user_segment

    Returns DataFrame with engineered features ready for LightGBM.
    """
    df = df.copy()

    # ── 1. Cart-level aggregates ────────────────────────────────────────
    if "cart_total_price" not in df.columns and "cart_items" in df.columns:
        df["cart_total_price"] = df["cart_items"].apply(
            lambda items: sum(i.get("price", 0) for i in items) if isinstance(items, list) else 0
        )
        df["cart_item_count"] = df["cart_items"].apply(
            lambda items: len(items) if isinstance(items, list) else 0
        )
        df["cart_avg_price"] = df["cart_total_price"] / df["cart_item_count"].clip(lower=1)

        # Unique categories in cart
        df["cart_n_categories"] = df["cart_items"].apply(
            lambda items: len({i.get("category", "") for i in items}) if isinstance(items, list) else 0
        )

        # Cuisine entropy (diversity measure)
        df["cart_cuisine_entropy"] = df["cart_items"].apply(_cuisine_entropy)

        # Has main dish, has beverage, has dessert
        df["cart_has_main"] = df["cart_items"].apply(
            lambda items: any(i.get("category", "") == "main" for i in items) if isinstance(items, list) else False
        ).astype(int)
        df["cart_has_beverage"] = df["cart_items"].apply(
            lambda items: any(i.get("category", "") in ("beverage", "drink") for i in items) if isinstance(items, list) else False
        ).astype(int)
        df["cart_has_dessert"] = df["cart_items"].apply(
            lambda items: any(i.get("category", "") == "dessert" for i in items) if isinstance(items, list) else False
        ).astype(int)
        df["cart_has_bread"] = df["cart_items"].apply(
            lambda items: any(i.get("category", "") in ("bread", "roti") for i in items) if isinstance(items, list) else False
        ).astype(int)

        # All veg cart
        df["cart_all_veg"] = df["cart_items"].apply(
            lambda items: all(i.get("is_veg", True) for i in items) if isinstance(items, list) else True
        ).astype(int)

    # ── 2. Candidate add-on features ────────────────────────────────────
    if "candidate_price" in df.columns:
        df["candidate_price_log"] = np.log1p(df["candidate_price"])

    # Price ratio (candidate vs cart average)
    if "candidate_price" in df.columns and "cart_avg_price" in df.columns:
        df["price_ratio"] = df["candidate_price"] / df["cart_avg_price"].clip(lower=1)

    # Same category already in cart
    if "cart_items" in df.columns and "candidate_category" in df.columns:
        df["candidate_category_in_cart"] = df.apply(
            lambda row: int(
                row["candidate_category"] in {i.get("category", "") for i in row["cart_items"]}
                if isinstance(row["cart_items"], list) else False
            ),
            axis=1,
        )

    # ── 3. Temporal features ────────────────────────────────────────────
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["is_lunch"] = df["hour"].between(11, 14).astype(int)
        df["is_dinner"] = df["hour"].between(18, 22).astype(int)
        df["is_late_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # ── 4. Categorical encoding ─────────────────────────────────────────
    for cat_col in ["cuisine_type", "user_segment", "candidate_category"]:
        if cat_col in df.columns:
            df[f"{cat_col}_code"] = df[cat_col].astype("category").cat.codes

    # ── 5. Cross features ───────────────────────────────────────────────
    if "candidate_is_veg" in df.columns and "cart_all_veg" in df.columns:
        df["veg_match"] = (df["candidate_is_veg"].astype(int) == df["cart_all_veg"]).astype(int)

    # ── 6. Drop non-numeric columns for model input ─────────────────────
    # (Keep them in the DF but mark which are features)
    df.attrs["feature_columns"] = _get_feature_columns(df)

    logger.info("Engineered %d features for %d rows", len(df.attrs.get("feature_columns", [])), len(df))
    return df


def _cuisine_entropy(items: list) -> float:
    """Shannon entropy over cuisine categories in the cart."""
    if not isinstance(items, list) or len(items) == 0:
        return 0.0
    cats = [i.get("category", "other") for i in items]
    n = len(cats)
    counts: dict[str, int] = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identify numeric feature columns suitable for LightGBM."""
    exclude = {
        "session_id", "label", "cart_items", "candidate_name",
        "cuisine_type", "user_segment", "candidate_category",
        "user_id", "restaurant_id",
    }
    return [
        col for col in df.columns
        if col not in exclude and df[col].dtype in ("int64", "float64", "int32", "float32", "bool")
    ]


# ═════════════════════════════════════════════════════════════════════════════
#  TRAIN / VAL / TEST SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def split_data(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Group-aware split (by session_id) so no session leaks across sets.

    Uses temporal ordering if a timestamp column exists, otherwise
    random group-level shuffling.

    Parameters
    ----------
    df : DataFrame with 'session_id' column and engineered features
    val_frac : fraction of sessions for validation
    test_frac : fraction of sessions for test
    seed : random seed for reproducibility

    Returns
    -------
    (train_df, val_df, test_df)
    """
    rng = np.random.RandomState(seed)

    # Get unique sessions
    all_sessions = df["session_id"].unique()

    # If temporal column exists, sort by it for honest temporal split
    if "timestamp" in df.columns:
        session_times = df.groupby("session_id")["timestamp"].min().sort_values()
        all_sessions = session_times.index.values
    else:
        rng.shuffle(all_sessions)

    n = len(all_sessions)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_test - n_val

    train_sessions = set(all_sessions[:n_train])
    val_sessions = set(all_sessions[n_train:n_train + n_val])
    test_sessions = set(all_sessions[n_train + n_val:])

    train_df = df[df["session_id"].isin(train_sessions)].reset_index(drop=True)
    val_df = df[df["session_id"].isin(val_sessions)].reset_index(drop=True)
    test_df = df[df["session_id"].isin(test_sessions)].reset_index(drop=True)

    logger.info(
        "Split: train=%d sessions (%d rows), val=%d sessions (%d rows), test=%d sessions (%d rows)",
        len(train_sessions), len(train_df),
        len(val_sessions), len(val_df),
        len(test_sessions), len(test_df),
    )

    return train_df, val_df, test_df
