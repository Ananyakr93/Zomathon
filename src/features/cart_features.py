"""
cart_features.py
================
Tier 1: Real-Time Cart Feature Engineering.

Computed on every request (<10ms budget). Captures the immediate cart
state through composition features, meal completeness scoring, category
distribution, and price positioning.

Key Features
------------
* Cart aggregates : total_value, item_count, avg_price, price_variance, veg_ratio
* Meal completeness : rule-engine score (main, side, beverage, dessert)
* Category distribution : counts of each dish category
* Price positioning : budget/premium counts, price diversity score
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ── Category mapping ──────────────────────────────────────────────────────────
# Maps dish categories to meal-role buckets for completeness scoring.
CATEGORY_ROLES = {
    # Mains
    "main": "main", "biryani": "main", "rice": "main", "curry": "main",
    "noodles": "main", "pizza": "main", "burger": "main", "pasta": "main",
    "thali": "main", "wrap": "main", "paratha": "main", "dosa": "main",
    "sandwich": "main", "meal": "main",
    # Sides
    "side": "side", "bread": "side", "naan": "side", "roti": "side",
    "raita": "side", "chutney": "side", "salad": "side", "papad": "side",
    "fries": "side", "garlic_bread": "side", "accompaniment": "side",
    # Starters
    "starter": "starter", "appetizer": "starter", "soup": "starter",
    "snack": "starter", "tikka": "starter", "kebab": "starter",
    "spring_roll": "starter", "momos": "starter",
    # Beverages
    "beverage": "beverage", "drink": "beverage", "lassi": "beverage",
    "juice": "beverage", "shake": "beverage", "tea": "beverage",
    "coffee": "beverage", "soda": "beverage", "water": "beverage",
    "cold_drink": "beverage", "mocktail": "beverage",
    # Desserts
    "dessert": "dessert", "sweet": "dessert", "ice_cream": "dessert",
    "gulab_jamun": "dessert", "cake": "dessert", "pastry": "dessert",
    "kulfi": "dessert", "halwa": "dessert",
}

# Meal completeness weights by role
COMPLETENESS_WEIGHTS = {
    "main": 1.0,
    "side": 0.5,
    "beverage": 0.25,
    "dessert": 0.25,
}

# Price buckets
BUDGET_THRESHOLD = 150      # items below ₹150 = budget
PREMIUM_THRESHOLD = 400     # items above ₹400 = premium


# ═══════════════════════════════════════════════════════════════════════════════
#  CART FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CartFeatures:
    """Container for all Tier-1 cart-level features."""

    # ── Composition ───────────────────────────────────────────
    total_value: float = 0.0
    item_count: int = 0
    avg_price: float = 0.0
    price_variance: float = 0.0
    veg_ratio: float = 0.0

    # ── Meal completeness ─────────────────────────────────────
    has_main: bool = False
    has_side: bool = False
    has_beverage: bool = False
    has_dessert: bool = False
    has_starter: bool = False
    meal_completeness_score: float = 0.0

    # ── Category distribution ─────────────────────────────────
    starter_count: int = 0
    main_count: int = 0
    side_count: int = 0
    beverage_count: int = 0
    dessert_count: int = 0
    other_count: int = 0

    # ── Price positioning ─────────────────────────────────────
    budget_items_count: int = 0
    premium_items_count: int = 0
    mid_items_count: int = 0
    price_diversity_score: float = 0.0

    # ── Cuisine ───────────────────────────────────────────────
    cuisine_types: list[str] = field(default_factory=list)
    cuisine_entropy: float = 0.0
    dominant_cuisine: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a feature dict suitable for ML models."""
        return {
            # Composition
            "cart_total_value": self.total_value,
            "cart_item_count": self.item_count,
            "cart_avg_price": round(self.avg_price, 2),
            "cart_price_variance": round(self.price_variance, 2),
            "cart_veg_ratio": round(self.veg_ratio, 2),

            # Meal completeness
            "has_main": int(self.has_main),
            "has_side": int(self.has_side),
            "has_beverage": int(self.has_beverage),
            "has_dessert": int(self.has_dessert),
            "has_starter": int(self.has_starter),
            "meal_completeness_score": round(self.meal_completeness_score, 4),

            # Category distribution
            "starter_count": self.starter_count,
            "main_count": self.main_count,
            "side_count": self.side_count,
            "beverage_count": self.beverage_count,
            "dessert_count": self.dessert_count,
            "other_count": self.other_count,

            # Price positioning
            "budget_items_count": self.budget_items_count,
            "premium_items_count": self.premium_items_count,
            "mid_items_count": self.mid_items_count,
            "price_diversity_score": round(self.price_diversity_score, 4),

            # Cuisine
            "cuisine_entropy": round(self.cuisine_entropy, 4),
            "dominant_cuisine": self.dominant_cuisine,
            "n_cuisine_types": len(self.cuisine_types),
        }


def _classify_role(category: str) -> str:
    """Map a dish category to a meal role (main/side/starter/beverage/dessert/other)."""
    cat = category.lower().strip().replace(" ", "_")
    return CATEGORY_ROLES.get(cat, "other")


def _price_bucket(price: float) -> str:
    """Classify an item price into budget / mid / premium."""
    if price < BUDGET_THRESHOLD:
        return "budget"
    elif price > PREMIUM_THRESHOLD:
        return "premium"
    return "mid"


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Shannon entropy over a category distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _price_diversity(prices: list[float]) -> float:
    """
    Price diversity = coefficient of variation (std / mean).
    0 means all same price, higher = more diverse.
    """
    if len(prices) < 2:
        return 0.0
    mean_p = sum(prices) / len(prices)
    if mean_p == 0:
        return 0.0
    variance = sum((p - mean_p) ** 2 for p in prices) / len(prices)
    return math.sqrt(variance) / mean_p


def compute_cart_features(cart_items: list[dict[str, Any]]) -> CartFeatures:
    """
    Compute all Tier-1 cart features from the raw cart state.

    Parameters
    ----------
    cart_items : list[dict]
        Each dict should have at minimum:
          name, category, price, is_veg (bool)
        Optional: cuisine_type, spice_level, portion_size

    Returns
    -------
    CartFeatures dataclass with all computed features.

    Performance: <5ms for typical carts (1–10 items).
    """
    feats = CartFeatures()

    if not cart_items:
        return feats

    n = len(cart_items)
    feats.item_count = n

    prices: list[float] = []
    veg_count = 0
    role_counts: dict[str, int] = {
        "starter": 0, "main": 0, "side": 0,
        "beverage": 0, "dessert": 0, "other": 0,
    }
    cuisine_counts: dict[str, int] = {}
    price_buckets: dict[str, int] = {"budget": 0, "mid": 0, "premium": 0}

    for item in cart_items:
        # Price
        price = float(item.get("price", 0))
        prices.append(price)

        # Veg
        is_veg = item.get("is_veg", item.get("veg_flag", True))
        if is_veg:
            veg_count += 1

        # Category role
        cat = item.get("category", "other")
        role = _classify_role(cat)
        role_counts[role] = role_counts.get(role, 0) + 1

        # Price bucket
        bucket = _price_bucket(price)
        price_buckets[bucket] += 1

        # Cuisine
        cuisine = item.get("cuisine_type", item.get("cuisine", "unknown"))
        cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

    # ── Composition ───────────────────────────────────────────
    feats.total_value = sum(prices)
    feats.avg_price = feats.total_value / n
    feats.veg_ratio = veg_count / n

    if n > 1:
        mean_p = feats.avg_price
        feats.price_variance = sum((p - mean_p) ** 2 for p in prices) / n
    else:
        feats.price_variance = 0.0

    # ── Meal completeness ─────────────────────────────────────
    feats.has_main = role_counts.get("main", 0) > 0
    feats.has_side = role_counts.get("side", 0) > 0
    feats.has_beverage = role_counts.get("beverage", 0) > 0
    feats.has_dessert = role_counts.get("dessert", 0) > 0
    feats.has_starter = role_counts.get("starter", 0) > 0

    total_weight = sum(COMPLETENESS_WEIGHTS.values())
    score = 0.0
    if feats.has_main:
        score += COMPLETENESS_WEIGHTS["main"]
    if feats.has_side:
        score += COMPLETENESS_WEIGHTS["side"]
    if feats.has_beverage:
        score += COMPLETENESS_WEIGHTS["beverage"]
    if feats.has_dessert:
        score += COMPLETENESS_WEIGHTS["dessert"]
    feats.meal_completeness_score = (score / total_weight) * 100.0

    # ── Category distribution ─────────────────────────────────
    feats.starter_count = role_counts.get("starter", 0)
    feats.main_count = role_counts.get("main", 0)
    feats.side_count = role_counts.get("side", 0)
    feats.beverage_count = role_counts.get("beverage", 0)
    feats.dessert_count = role_counts.get("dessert", 0)
    feats.other_count = role_counts.get("other", 0)

    # ── Price positioning ─────────────────────────────────────
    feats.budget_items_count = price_buckets["budget"]
    feats.premium_items_count = price_buckets["premium"]
    feats.mid_items_count = price_buckets["mid"]
    feats.price_diversity_score = _price_diversity(prices)

    # ── Cuisine ───────────────────────────────────────────────
    feats.cuisine_types = list(cuisine_counts.keys())
    feats.cuisine_entropy = _shannon_entropy(cuisine_counts)
    if cuisine_counts:
        feats.dominant_cuisine = max(cuisine_counts, key=cuisine_counts.get)

    return feats
