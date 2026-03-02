"""
context_features.py
===================
Tier 3: Contextual & Restaurant Feature Engineering.

Pre-computed and updated daily. Provides temporal context, geographic
context, restaurant features, and candidate item features.

Key Features
------------
* Temporal : hour bins, day of week, weekend flag, meal time encoding, holiday flags
* Geographic : city encoding, delivery zone characteristics, regional preferences
* Restaurant : type, cuisine specialization, price point, rating, menu diversity
* Candidate item : price, category, margin, popularity, restaurant fit, acceptance rate
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── 10 cities (matching synthetic_generator.py) ──────────────────────────────
CITY_LIST = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Lucknow", "Jaipur",
]

# Meal time boundaries
MEAL_TIME_RANGES = {
    "breakfast":   (6, 10),
    "lunch":       (12, 15),
    "evening":     (16, 19),
    "dinner":      (19, 23),
    "late_night":  (23, 6),   # wraps around midnight
}

# Regional preference vectors (cuisine → preference score per city)
REGIONAL_PREFERENCES: dict[str, dict[str, float]] = {
    "Mumbai":    {"pav_preference": 0.8, "south_preference": 0.3, "chinese_preference": 0.5,
                  "street_food_preference": 0.9, "mughlai_preference": 0.3},
    "Delhi":     {"pav_preference": 0.2, "south_preference": 0.2, "chinese_preference": 0.4,
                  "street_food_preference": 0.7, "mughlai_preference": 0.8},
    "Bangalore": {"pav_preference": 0.2, "south_preference": 0.7, "chinese_preference": 0.5,
                  "street_food_preference": 0.4, "mughlai_preference": 0.3},
    "Hyderabad": {"pav_preference": 0.1, "south_preference": 0.5, "chinese_preference": 0.3,
                  "street_food_preference": 0.4, "mughlai_preference": 0.7},
    "Chennai":   {"pav_preference": 0.1, "south_preference": 0.9, "chinese_preference": 0.3,
                  "street_food_preference": 0.5, "mughlai_preference": 0.2},
    "Kolkata":   {"pav_preference": 0.1, "south_preference": 0.2, "chinese_preference": 0.7,
                  "street_food_preference": 0.6, "mughlai_preference": 0.5},
    "Pune":      {"pav_preference": 0.6, "south_preference": 0.3, "chinese_preference": 0.4,
                  "street_food_preference": 0.7, "mughlai_preference": 0.3},
    "Ahmedabad": {"pav_preference": 0.5, "south_preference": 0.3, "chinese_preference": 0.4,
                  "street_food_preference": 0.8, "mughlai_preference": 0.2},
    "Lucknow":   {"pav_preference": 0.1, "south_preference": 0.1, "chinese_preference": 0.3,
                  "street_food_preference": 0.5, "mughlai_preference": 0.9},
    "Jaipur":    {"pav_preference": 0.2, "south_preference": 0.2, "chinese_preference": 0.3,
                  "street_food_preference": 0.6, "mughlai_preference": 0.5},
}

# Delivery zone characteristics per city
DELIVERY_ZONES: dict[str, dict[str, float]] = {
    "Mumbai":    {"affluence_score": 0.75, "restaurant_density": 0.90, "avg_delivery_min": 32},
    "Delhi":     {"affluence_score": 0.70, "restaurant_density": 0.85, "avg_delivery_min": 35},
    "Bangalore": {"affluence_score": 0.80, "restaurant_density": 0.80, "avg_delivery_min": 30},
    "Hyderabad": {"affluence_score": 0.65, "restaurant_density": 0.70, "avg_delivery_min": 28},
    "Chennai":   {"affluence_score": 0.65, "restaurant_density": 0.75, "avg_delivery_min": 30},
    "Kolkata":   {"affluence_score": 0.55, "restaurant_density": 0.65, "avg_delivery_min": 33},
    "Pune":      {"affluence_score": 0.70, "restaurant_density": 0.75, "avg_delivery_min": 28},
    "Ahmedabad": {"affluence_score": 0.60, "restaurant_density": 0.60, "avg_delivery_min": 25},
    "Lucknow":   {"affluence_score": 0.50, "restaurant_density": 0.55, "avg_delivery_min": 30},
    "Jaipur":    {"affluence_score": 0.55, "restaurant_density": 0.55, "avg_delivery_min": 28},
}

# Known Indian holidays/events (month, day) → name
HOLIDAYS = {
    (1, 26): "republic_day",
    (3, 8):  "holi",
    (8, 15): "independence_day",
    (10, 2): "gandhi_jayanti",
    (10, 24): "diwali",         # approximate
    (11, 1):  "diwali",         # approximate
    (12, 25): "christmas",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalFeatures:
    """Temporal context features."""
    hour: int = 12
    day_of_week: int = 0          # 0=Mon … 6=Sun
    is_weekend: bool = False
    meal_time: str = "lunch"
    is_holiday: bool = False
    holiday_name: str = ""

    # Binned encodings
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "is_weekend": int(self.is_weekend),
            "meal_time": self.meal_time,
            "is_holiday": int(self.is_holiday),
            "holiday_name": self.holiday_name,
            "hour_sin": round(self.hour_sin, 4),
            "hour_cos": round(self.hour_cos, 4),
            "dow_sin": round(self.dow_sin, 4),
            "dow_cos": round(self.dow_cos, 4),
        }


def compute_temporal_features(
    dt: datetime | None = None,
) -> TemporalFeatures:
    """
    Compute temporal context from a datetime.

    Uses cyclical encoding (sin/cos) for hour and day-of-week.
    """
    if dt is None:
        dt = datetime.now()

    feats = TemporalFeatures()
    feats.hour = dt.hour
    feats.day_of_week = dt.weekday()
    feats.is_weekend = dt.weekday() >= 5

    # Meal time
    h = dt.hour
    if 6 <= h < 10:
        feats.meal_time = "breakfast"
    elif 12 <= h < 15:
        feats.meal_time = "lunch"
    elif 16 <= h < 19:
        feats.meal_time = "evening"
    elif 19 <= h < 23:
        feats.meal_time = "dinner"
    else:
        feats.meal_time = "late_night"

    # Holiday check
    key = (dt.month, dt.day)
    if key in HOLIDAYS:
        feats.is_holiday = True
        feats.holiday_name = HOLIDAYS[key]

    # Cyclical encoding
    feats.hour_sin = math.sin(2 * math.pi * dt.hour / 24)
    feats.hour_cos = math.cos(2 * math.pi * dt.hour / 24)
    feats.dow_sin = math.sin(2 * math.pi * dt.weekday() / 7)
    feats.dow_cos = math.cos(2 * math.pi * dt.weekday() / 7)

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOGRAPHIC CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeographicFeatures:
    """Geographic context features."""

    city: str = "Mumbai"
    city_encoding: dict[str, int] = field(default_factory=dict)

    # Delivery zone
    affluence_score: float = 0.5
    restaurant_density: float = 0.5
    avg_delivery_time: float = 30.0

    # Regional preferences
    regional_preferences: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}

        # One-hot city encoding
        for c in CITY_LIST:
            d[f"city_{c.lower()}"] = self.city_encoding.get(c, 0)

        d["affluence_score"] = round(self.affluence_score, 2)
        d["restaurant_density"] = round(self.restaurant_density, 2)
        d["avg_delivery_time"] = round(self.avg_delivery_time, 1)

        # Regional preference vectors
        for k, v in self.regional_preferences.items():
            d[f"regional_{k}"] = round(v, 2)

        return d


def compute_geographic_features(city: str) -> GeographicFeatures:
    """Compute geographic context from the city name."""
    feats = GeographicFeatures()
    feats.city = city

    # One-hot encoding
    feats.city_encoding = {c: (1 if c == city else 0) for c in CITY_LIST}

    # Delivery zone
    zone = DELIVERY_ZONES.get(city, DELIVERY_ZONES.get("Mumbai", {}))
    feats.affluence_score = zone.get("affluence_score", 0.5)
    feats.restaurant_density = zone.get("restaurant_density", 0.5)
    feats.avg_delivery_time = zone.get("avg_delivery_min", 30.0)

    # Regional preferences
    feats.regional_preferences = REGIONAL_PREFERENCES.get(
        city, REGIONAL_PREFERENCES.get("Mumbai", {})
    )

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  RESTAURANT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RestaurantFeatures:
    """Restaurant-level features."""

    restaurant_type: str = "local"           # cloud_kitchen / chain / local / premium
    cuisine_specialization: str = "multi"
    avg_price_point: float = 250.0
    rating: float = 4.0
    delivery_performance: float = 0.85       # on-time delivery ratio

    menu_diversity_score: float = 0.5        # 0-1 how many categories covered
    veg_menu_ratio: float = 0.5
    total_menu_items: int = 50

    historical_addon_acceptance_rate: float = 0.15  # restaurant-level benchmark

    def to_dict(self) -> dict[str, Any]:
        type_map = {"cloud_kitchen": 0, "chain": 1, "local": 2, "premium": 3}
        return {
            "restaurant_type_code": type_map.get(self.restaurant_type, 2),
            "restaurant_type": self.restaurant_type,
            "cuisine_specialization": self.cuisine_specialization,
            "restaurant_avg_price": round(self.avg_price_point, 2),
            "restaurant_rating": round(self.rating, 2),
            "delivery_performance": round(self.delivery_performance, 2),
            "menu_diversity_score": round(self.menu_diversity_score, 2),
            "veg_menu_ratio": round(self.veg_menu_ratio, 2),
            "total_menu_items": self.total_menu_items,
            "restaurant_addon_acceptance_rate": round(self.historical_addon_acceptance_rate, 4),
        }


def compute_restaurant_features(restaurant: dict[str, Any]) -> RestaurantFeatures:
    """
    Compute restaurant features from the restaurant metadata.

    Parameters
    ----------
    restaurant : dict
        Keys: restaurant_type, cuisine_type, avg_price, rating,
              delivery_performance, menu_items (list), veg_menu_ratio, ...
    """
    feats = RestaurantFeatures()

    feats.restaurant_type = restaurant.get("restaurant_type", "local")
    feats.cuisine_specialization = restaurant.get("cuisine_type", "multi")
    feats.avg_price_point = float(restaurant.get("avg_price",
                                   restaurant.get("price_tier_value", 250)))
    feats.rating = float(restaurant.get("rating", 4.0))
    feats.delivery_performance = float(restaurant.get("delivery_performance", 0.85))

    # Menu diversity: count distinct categories
    menu_items = restaurant.get("menu_items", [])
    if menu_items:
        categories = set()
        veg_count = 0
        for item in menu_items:
            categories.add(item.get("category", "other"))
            if item.get("is_veg", False):
                veg_count += 1
        feats.menu_diversity_score = min(len(categories) / 8.0, 1.0)
        feats.veg_menu_ratio = veg_count / max(len(menu_items), 1)
        feats.total_menu_items = len(menu_items)
    else:
        feats.veg_menu_ratio = float(restaurant.get("veg_menu_ratio", 0.5))
        feats.total_menu_items = int(restaurant.get("total_menu_items", 50))
        feats.menu_diversity_score = float(restaurant.get("menu_diversity_score", 0.5))

    feats.historical_addon_acceptance_rate = float(
        restaurant.get("addon_acceptance_rate",
                       restaurant.get("historical_addon_acceptance_rate", 0.15))
    )

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  CANDIDATE ITEM FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateItemFeatures:
    """Features for a single candidate add-on item."""

    name: str = ""
    price: float = 0.0
    category: str = "other"
    margin_level: str = "medium"          # low / medium / high
    popularity_score: float = 0.5
    restaurant_fit_score: float = 0.5     # how well item matches restaurant cuisine
    historical_acceptance_rate: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        margin_map = {"low": 0, "medium": 1, "high": 2}
        return {
            "candidate_price": round(self.price, 2),
            "candidate_category": self.category,
            "candidate_margin_code": margin_map.get(self.margin_level, 1),
            "candidate_popularity": round(self.popularity_score, 4),
            "candidate_restaurant_fit": round(self.restaurant_fit_score, 4),
            "candidate_acceptance_rate": round(self.historical_acceptance_rate, 4),
        }


def compute_candidate_features(
    candidate: dict[str, Any],
    restaurant: dict[str, Any],
    cart_items: list[dict[str, Any]] | None = None,
) -> CandidateItemFeatures:
    """
    Compute features for a single candidate item.

    Parameters
    ----------
    candidate : dict
        Must have 'name'. Optional: price, category, margin, popularity_score.
    restaurant : dict
        Restaurant metadata for computing fit score.
    cart_items : list[dict], optional
        Current cart for acceptance rate adjustment.
    """
    feats = CandidateItemFeatures()

    feats.name = candidate.get("name", "")
    feats.price = float(candidate.get("price", 0))
    feats.category = candidate.get("category", "other")
    feats.margin_level = candidate.get("margin", candidate.get("margin_level", "medium"))
    feats.popularity_score = float(candidate.get("popularity_score",
                                    candidate.get("confidence", 0.5)))

    # Restaurant fit score: does the candidate's cuisine match the restaurant?
    restaurant_cuisine = restaurant.get("cuisine_type", "").lower().replace(" ", "_")
    candidate_cuisine = candidate.get("cuisine_type",
                         candidate.get("cuisine", "")).lower().replace(" ", "_")
    if candidate_cuisine and restaurant_cuisine:
        feats.restaurant_fit_score = 1.0 if candidate_cuisine == restaurant_cuisine else 0.3
    else:
        feats.restaurant_fit_score = 0.5

    # Historical acceptance rate (simulation — in production from analytics DB)
    base_rate = float(candidate.get("acceptance_rate", 0.15))
    # Adjust by category: beverages/desserts have higher acceptance on average
    category_boost = {
        "beverage": 1.3, "dessert": 1.2, "side": 1.1,
        "bread": 1.15, "starter": 1.0, "main": 0.8,
    }
    cat_key = feats.category.lower()
    boost = category_boost.get(cat_key, 1.0)
    feats.historical_acceptance_rate = min(base_rate * boost, 1.0)

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_context_features(
    restaurant: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute all Tier-3 context features in one call.

    Parameters
    ----------
    restaurant : dict
        Restaurant metadata.
    context : dict, optional
        Keys: city, hour, is_weekend, dt (datetime object)

    Returns
    -------
    dict of all context features merged together.
    """
    ctx = context or {}

    # Temporal
    dt = ctx.get("dt", None)
    if dt is None:
        dt = datetime.now()
    temporal = compute_temporal_features(dt)

    # Geographic
    city = ctx.get("city", restaurant.get("city", "Mumbai"))
    geo = compute_geographic_features(city)

    # Restaurant
    rest_feats = compute_restaurant_features(restaurant)

    # Merge all
    merged = {}
    merged.update(temporal.to_dict())
    merged.update(geo.to_dict())
    merged.update(rest_feats.to_dict())

    return merged
