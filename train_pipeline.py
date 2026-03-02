"""
train_pipeline.py
=================
End-to-end training runner for the CSAO recommendation model.

Usage:
    python train_pipeline.py                   # generate synthetic + train
    python train_pipeline.py --skip-generate   # train on existing data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import engineer_features, split_data, _get_feature_columns
from src.models.ranker import train, save_model, predict, DEFAULT_PARAMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("train_pipeline")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "models" / "results"


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — GENERATE SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_scenarios: int = 1000) -> pd.DataFrame:
    """Generate synthetic cart scenarios — self-contained, no external files needed."""
    logger.info("Generating %d synthetic cart scenarios …", n_scenarios)
    import random

    rng = random.Random(42)

    # ── Inline cuisine menus (no external file dependency) ──────────────
    CUISINES = {
        "North Indian": {
            "mains": [
                {"name": "Butter Chicken", "price": 320, "is_veg": False},
                {"name": "Paneer Tikka Masala", "price": 280, "is_veg": True},
                {"name": "Dal Makhani", "price": 220, "is_veg": True},
                {"name": "Chole Bhature", "price": 180, "is_veg": True},
                {"name": "Kadai Chicken", "price": 300, "is_veg": False},
                {"name": "Shahi Paneer", "price": 260, "is_veg": True},
            ],
            "sides": [
                {"name": "Raita", "price": 50, "is_veg": True},
                {"name": "Onion Salad", "price": 30, "is_veg": True},
                {"name": "Papad Masala", "price": 40, "is_veg": True},
            ],
            "breads": [
                {"name": "Butter Naan", "price": 50, "is_veg": True},
                {"name": "Garlic Naan", "price": 60, "is_veg": True},
                {"name": "Tandoori Roti", "price": 30, "is_veg": True},
            ],
            "beverages": [
                {"name": "Lassi (Sweet)", "price": 60, "is_veg": True},
                {"name": "Masala Chaas", "price": 40, "is_veg": True},
                {"name": "Coke 500ml", "price": 50, "is_veg": True},
            ],
            "desserts": [
                {"name": "Gulab Jamun (2pc)", "price": 80, "is_veg": True},
                {"name": "Rasmalai", "price": 120, "is_veg": True},
                {"name": "Gajar Ka Halwa", "price": 100, "is_veg": True},
            ],
        },
        "Biryani Specialist": {
            "mains": [
                {"name": "Chicken Dum Biryani", "price": 349, "is_veg": False},
                {"name": "Mutton Biryani", "price": 449, "is_veg": False},
                {"name": "Veg Biryani", "price": 249, "is_veg": True},
                {"name": "Hyderabadi Biryani", "price": 399, "is_veg": False},
            ],
            "sides": [
                {"name": "Raita", "price": 50, "is_veg": True},
                {"name": "Mirchi Ka Salan", "price": 120, "is_veg": True},
                {"name": "Masala Papad", "price": 40, "is_veg": True},
            ],
            "beverages": [
                {"name": "Coke 500ml", "price": 50, "is_veg": True},
                {"name": "Lassi (Sweet)", "price": 60, "is_veg": True},
            ],
            "desserts": [
                {"name": "Phirni", "price": 100, "is_veg": True},
                {"name": "Double Ka Meetha", "price": 120, "is_veg": True},
                {"name": "Gulab Jamun (2pc)", "price": 80, "is_veg": True},
            ],
        },
        "South Indian": {
            "mains": [
                {"name": "Masala Dosa", "price": 120, "is_veg": True},
                {"name": "Idli Sambhar", "price": 80, "is_veg": True},
                {"name": "Uttapam", "price": 100, "is_veg": True},
                {"name": "Rava Dosa", "price": 130, "is_veg": True},
            ],
            "sides": [
                {"name": "Sambhar", "price": 50, "is_veg": True},
                {"name": "Coconut Chutney", "price": 30, "is_veg": True},
            ],
            "beverages": [
                {"name": "Filter Coffee", "price": 50, "is_veg": True},
                {"name": "Buttermilk", "price": 40, "is_veg": True},
                {"name": "Coke 500ml", "price": 50, "is_veg": True},
            ],
            "desserts": [
                {"name": "Payasam", "price": 90, "is_veg": True},
                {"name": "Mysore Pak", "price": 60, "is_veg": True},
            ],
        },
        "Chinese": {
            "mains": [
                {"name": "Hakka Noodles", "price": 220, "is_veg": True},
                {"name": "Schezwan Fried Rice", "price": 200, "is_veg": True},
                {"name": "Chilli Chicken", "price": 280, "is_veg": False},
                {"name": "Manchurian", "price": 200, "is_veg": True},
            ],
            "sides": [
                {"name": "Spring Rolls", "price": 150, "is_veg": True},
                {"name": "Honey Chilli Potato", "price": 180, "is_veg": True},
            ],
            "beverages": [
                {"name": "Coke 500ml", "price": 50, "is_veg": True},
                {"name": "Iced Tea", "price": 60, "is_veg": True},
            ],
            "desserts": [
                {"name": "Brownie with Ice Cream", "price": 150, "is_veg": True},
                {"name": "Ice Cream Sundae", "price": 130, "is_veg": True},
            ],
            "starters": [
                {"name": "Hot and Sour Soup", "price": 120, "is_veg": True},
                {"name": "Manchow Soup", "price": 130, "is_veg": True},
                {"name": "Crispy Corn", "price": 150, "is_veg": True},
            ],
        },
        "Continental": {
            "mains": [
                {"name": "Margherita Pizza", "price": 299, "is_veg": True},
                {"name": "Pasta Alfredo", "price": 280, "is_veg": True},
                {"name": "Grilled Chicken", "price": 350, "is_veg": False},
            ],
            "sides": [
                {"name": "Garlic Bread", "price": 120, "is_veg": True},
                {"name": "French Fries", "price": 100, "is_veg": True},
                {"name": "Caesar Salad", "price": 180, "is_veg": True},
            ],
            "beverages": [
                {"name": "Cold Coffee", "price": 120, "is_veg": True},
                {"name": "Coke 500ml", "price": 50, "is_veg": True},
                {"name": "Iced Tea", "price": 80, "is_veg": True},
            ],
            "desserts": [
                {"name": "Brownie", "price": 130, "is_veg": True},
                {"name": "Tiramisu", "price": 180, "is_veg": True},
            ],
        },
    }

    SEGMENTS = ["budget", "mid_tier", "premium", "student", "frequent"]
    HOURS = list(range(6, 24))

    rows = []
    session_id = 0

    for i in range(n_scenarios):
        cuisine_key = rng.choice(list(CUISINES.keys()))
        menu = CUISINES[cuisine_key]

        # Build random cart (1-3 items from mains)
        n_cart = rng.randint(1, min(3, len(menu["mains"])))
        cart_items = rng.sample(menu["mains"], n_cart)
        for item in cart_items:
            item["category"] = "main"

        # Sometimes add a side/bread already
        if rng.random() < 0.3 and "sides" in menu:
            side = rng.choice(menu["sides"])
            cart_items.append({**side, "category": "side"})

        hour = rng.choice(HOURS)
        day_of_week = rng.randint(0, 6)
        user_segment = rng.choice(SEGMENTS)
        cart_names = {item["name"].lower() for item in cart_items}
        cart_cats = {item.get("category", "") for item in cart_items}

        # Generate candidates from all non-main categories
        candidates = []
        for cat_key, items in menu.items():
            if cat_key == "mains":
                continue
            cat_label = cat_key.rstrip("s")  # "sides" -> "side"
            for item in items:
                if item["name"].lower() in cart_names:
                    continue
                candidates.append({**item, "category": cat_label})

        for cand in candidates:
            is_missing_cat = cand["category"] not in cart_cats
            accept_prob = 0.35 if is_missing_cat else 0.08
            if cand.get("price", 100) < 100:
                accept_prob += 0.05
            label = 1 if rng.random() < accept_prob else 0

            rows.append({
                "session_id": f"s_{session_id:06d}",
                "user_id": f"u_{i % 200}",
                "user_segment": user_segment,
                "cart_items": cart_items,
                "candidate_name": cand["name"],
                "candidate_category": cand["category"],
                "candidate_price": cand.get("price", 100),
                "candidate_is_veg": cand.get("is_veg", True),
                "cuisine_type": cuisine_key,
                "hour": hour,
                "day_of_week": day_of_week,
                "label": label,
            })

        session_id += 1

    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "synthetic_carts.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows (%d sessions) -> %s", len(df), session_id, path)
    return df



# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — FEATURE ENGINEERING + SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data(df: pd.DataFrame):
    """Engineer features and split data."""
    logger.info("Engineering features …")
    df = engineer_features(df)

    feature_cols = df.attrs.get("feature_columns") or _get_feature_columns(df)
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)

    logger.info("Splitting data …")
    train_df, val_df, test_df = split_data(df)

    return train_df, val_df, test_df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — COMPUTE GROUP SIZES
# ═══════════════════════════════════════════════════════════════════════════

def compute_groups(df: pd.DataFrame) -> np.ndarray:
    """Compute group sizes (rows per session) for LambdaRank."""
    return df.groupby("session_id").size().values


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN + EVALUATE
# ═══════════════════════════════════════════════════════════════════════════

def run_training(train_df, val_df, test_df, feature_cols):
    """Train LightGBM LambdaRank, evaluate on test set, save model."""
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    groups_train = compute_groups(train_df)

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    groups_val = compute_groups(val_df)

    logger.info(
        "Training LightGBM LambdaRank: %d train rows, %d val rows, %d features",
        len(X_train), len(X_val), len(feature_cols),
    )

    t0 = time.perf_counter()
    booster = train(
        X_train, y_train, groups_train,
        X_val, y_val, groups_val,
        num_boost_round=300,
        early_stopping_rounds=20,
    )
    train_time = time.perf_counter() - t0
    logger.info("Training took %.1f seconds", train_time)

    # Save model
    model_path = save_model(booster)
    logger.info("Model saved → %s", model_path)

    # ── Evaluate on test set ──────────────────────────────────────────
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    groups_test = compute_groups(test_df)

    scores = predict(booster, X_test)

    # Compute NDCG@K per group
    ndcg_scores = []
    offset = 0
    for g in groups_test:
        group_labels = y_test[offset:offset + g]
        group_scores = scores[offset:offset + g]
        offset += g

        # Sort by predicted score
        order = np.argsort(-group_scores)
        sorted_labels = group_labels[order]

        # NDCG@10
        k = min(10, len(sorted_labels))
        dcg = sum(
            sorted_labels[i] / np.log2(i + 2) for i in range(k)
        )
        ideal_labels = np.sort(group_labels)[::-1]
        idcg = sum(
            ideal_labels[i] / np.log2(i + 2) for i in range(min(k, len(ideal_labels)))
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    results = {
        "model": "LightGBM_LambdaRank",
        "n_train_rows": len(X_train),
        "n_val_rows": len(X_val),
        "n_test_rows": len(X_test),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "best_iteration": booster.best_iteration,
        "train_time_seconds": round(train_time, 2),
        "test_ndcg_at_10": round(avg_ndcg, 4),
        "n_test_groups": len(groups_test),
        "model_path": str(model_path),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results → %s", results_path)
    logger.info("Test NDCG@10 = %.4f", avg_ndcg)

    return booster, results


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CSAO Model Training Pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip data generation, use existing parquet")
    parser.add_argument("--n-scenarios", type=int, default=1000,
                        help="Number of synthetic scenarios to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("  CSAO MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Data
    if args.skip_generate:
        path = PROCESSED_DIR / "synthetic_carts.parquet"
        if not path.exists():
            logger.error("No data found at %s, run without --skip-generate first", path)
            sys.exit(1)
        df = pd.read_parquet(path)
        logger.info("Loaded existing data: %d rows", len(df))
    else:
        df = generate_synthetic_data(n_scenarios=args.n_scenarios)

    # Step 2: Features + Split
    train_df, val_df, test_df, feature_cols = prepare_data(df)

    # Step 3: Train + Evaluate
    booster, results = run_training(train_df, val_df, test_df, feature_cols)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  NDCG@10: {results['test_ndcg_at_10']:.4f}")
    print(f"  Model:   {results['model_path']}")
    print(f"  Time:    {results['train_time_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
