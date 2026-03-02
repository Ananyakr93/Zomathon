"""
user_features.py
================
Tier 2: User-Level Feature Engineering.

Pre-computed and updated hourly. Stores user behavioral embeddings in
ChromaDB for fast retrieval, with cold-start handling for new users.

Key Features
------------
* User behavioral embeddings (384-dim via all-MiniLM-L6-v2) stored in ChromaDB
* Cuisine affinity scores (8 cuisines)
* Price sensitivity segment (budget / mid / premium)
* Temporal ordering preferences (breakfast / lunch / dinner patterns)
* Dietary preferences (veg_ratio, spice_tolerance, health_conscious_flag)
* Recency-weighted: last 30 days weighted 3x, 30–90 days 1x
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMADB_DIR = PROJECT_ROOT / "data" / "chromadb"

# 8 cuisine categories for affinity scoring
CUISINES = [
    "north_indian", "south_indian", "chinese", "continental",
    "fast_food", "mughlai", "street_food", "desserts",
]

# Recency weighting constants
RECENCY_RECENT_DAYS = 30
RECENCY_OLD_DAYS = 90
RECENCY_RECENT_WEIGHT = 3.0
RECENCY_OLD_WEIGHT = 1.0

# City-level aggregate profiles for cold-start
CITY_PROFILES: dict[str, dict[str, Any]] = {
    "Mumbai": {
        "top_cuisines": ["street_food", "north_indian", "chinese"],
        "avg_order_value": 350, "veg_ratio": 0.45,
        "popular_items": ["pav_bhaji", "vada_pav", "butter_chicken", "biryani"],
    },
    "Delhi": {
        "top_cuisines": ["north_indian", "mughlai", "street_food"],
        "avg_order_value": 400, "veg_ratio": 0.55,
        "popular_items": ["chole_bhature", "butter_chicken", "biryani", "paratha"],
    },
    "Bangalore": {
        "top_cuisines": ["south_indian", "north_indian", "continental"],
        "avg_order_value": 380, "veg_ratio": 0.50,
        "popular_items": ["dosa", "biryani", "pizza", "idli"],
    },
    "Hyderabad": {
        "top_cuisines": ["mughlai", "south_indian", "north_indian"],
        "avg_order_value": 320, "veg_ratio": 0.35,
        "popular_items": ["biryani", "haleem", "dosa", "kebab"],
    },
    "Chennai": {
        "top_cuisines": ["south_indian", "north_indian", "chinese"],
        "avg_order_value": 300, "veg_ratio": 0.55,
        "popular_items": ["dosa", "idli", "biryani", "filter_coffee"],
    },
    "Kolkata": {
        "top_cuisines": ["north_indian", "chinese", "street_food"],
        "avg_order_value": 280, "veg_ratio": 0.40,
        "popular_items": ["biryani", "fish_curry", "momos", "roll"],
    },
    "Pune": {
        "top_cuisines": ["north_indian", "street_food", "continental"],
        "avg_order_value": 340, "veg_ratio": 0.50,
        "popular_items": ["misal_pav", "vada_pav", "pizza", "biryani"],
    },
    "Ahmedabad": {
        "top_cuisines": ["street_food", "north_indian", "south_indian"],
        "avg_order_value": 250, "veg_ratio": 0.80,
        "popular_items": ["dhokla", "thali", "pizza", "pav_bhaji"],
    },
    "Lucknow": {
        "top_cuisines": ["mughlai", "north_indian", "street_food"],
        "avg_order_value": 300, "veg_ratio": 0.40,
        "popular_items": ["biryani", "kebab", "korma", "kulfi"],
    },
    "Jaipur": {
        "top_cuisines": ["north_indian", "street_food", "mughlai"],
        "avg_order_value": 280, "veg_ratio": 0.65,
        "popular_items": ["dal_bati", "thali", "kachori", "lassi"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  USER PREFERENCE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserPreferences:
    """Computed user preference features."""

    # Cuisine affinity (0-1 scores for 8 cuisines)
    cuisine_affinities: dict[str, float] = field(default_factory=dict)

    # Price
    price_sensitivity: str = "mid"          # budget / mid / premium
    avg_order_value: float = 0.0

    # Temporal
    preferred_meal_times: list[str] = field(default_factory=list)
    weekday_order_ratio: float = 0.5

    # Dietary
    veg_ratio: float = 0.5
    spice_tolerance: str = "medium"         # low / medium / high
    health_conscious_flag: bool = False

    # Recency
    orders_last_30d: int = 0
    orders_30_90d: int = 0
    total_orders: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Flatten to feature dict."""
        d: dict[str, Any] = {}

        # Cuisine affinities
        for cuisine in CUISINES:
            d[f"cuisine_affinity_{cuisine}"] = self.cuisine_affinities.get(cuisine, 0.0)

        d["price_sensitivity"] = self.price_sensitivity
        d["user_avg_order_value"] = round(self.avg_order_value, 2)
        d["user_veg_ratio"] = round(self.veg_ratio, 2)
        d["spice_tolerance"] = self.spice_tolerance
        d["health_conscious"] = int(self.health_conscious_flag)
        d["weekday_order_ratio"] = round(self.weekday_order_ratio, 2)
        d["orders_last_30d"] = self.orders_last_30d
        d["orders_30_90d"] = self.orders_30_90d
        d["total_orders"] = self.total_orders
        d["preferred_meal_times"] = ",".join(self.preferred_meal_times) if self.preferred_meal_times else "any"

        return d


def compute_user_preferences(
    order_history: list[dict[str, Any]],
    current_time: float | None = None,
) -> UserPreferences:
    """
    Compute user preference features from order history.

    Parameters
    ----------
    order_history : list[dict]
        Each dict: {timestamp, items: [...], total_value, cuisine, meal_type,
                     is_veg, city, ...}
    current_time : float
        Unix timestamp for recency calculation. Defaults to now.

    Returns
    -------
    UserPreferences dataclass.
    """
    prefs = UserPreferences()

    if not order_history:
        # Cold-start: return defaults
        prefs.cuisine_affinities = {c: 1.0 / len(CUISINES) for c in CUISINES}
        return prefs

    now = current_time or time.time()
    day_30_ago = now - 30 * 86400
    day_90_ago = now - 90 * 86400

    prefs.total_orders = len(order_history)

    # Recency-weighted accumulators
    cuisine_scores: dict[str, float] = {c: 0.0 for c in CUISINES}
    total_value = 0.0
    veg_count = 0.0
    total_weight = 0.0
    meal_time_counts: dict[str, float] = {}
    spice_scores: list[float] = []
    weekday_count = 0
    recent_30d = 0
    period_30_90d = 0

    for order in order_history:
        ts = order.get("timestamp", 0)

        # Recency weight
        if ts >= day_30_ago:
            weight = RECENCY_RECENT_WEIGHT
            recent_30d += 1
        elif ts >= day_90_ago:
            weight = RECENCY_OLD_WEIGHT
            period_30_90d += 1
        else:
            weight = RECENCY_OLD_WEIGHT * 0.5  # older orders contribute less

        total_weight += weight

        # Cuisine
        cuisine = order.get("cuisine", "").lower().replace(" ", "_")
        if cuisine in cuisine_scores:
            cuisine_scores[cuisine] += weight

        # Price
        total_value += order.get("total_value", 0) * weight

        # Veg
        if order.get("is_veg", False):
            veg_count += weight

        # Meal time
        mt = order.get("meal_type", "dinner")
        meal_time_counts[mt] = meal_time_counts.get(mt, 0) + weight

        # Spice
        spice_map = {"low": 1, "medium": 2, "high": 3}
        spice = order.get("spice_level", "medium")
        spice_scores.append(spice_map.get(spice, 2))

        # Weekday
        if order.get("is_weekday", True):
            weekday_count += 1

    # Normalise cuisine affinities
    total_cuisine = sum(cuisine_scores.values())
    if total_cuisine > 0:
        prefs.cuisine_affinities = {
            c: round(s / total_cuisine, 4) for c, s in cuisine_scores.items()
        }
    else:
        prefs.cuisine_affinities = {c: 1.0 / len(CUISINES) for c in CUISINES}

    # Price sensitivity
    if total_weight > 0:
        prefs.avg_order_value = total_value / total_weight
    if prefs.avg_order_value < 250:
        prefs.price_sensitivity = "budget"
    elif prefs.avg_order_value > 500:
        prefs.price_sensitivity = "premium"
    else:
        prefs.price_sensitivity = "mid"

    # Veg ratio
    if total_weight > 0:
        prefs.veg_ratio = veg_count / total_weight

    # Meal times (top 2)
    sorted_meals = sorted(meal_time_counts.items(), key=lambda x: -x[1])
    prefs.preferred_meal_times = [m for m, _ in sorted_meals[:2]]

    # Spice tolerance
    if spice_scores:
        avg_spice = sum(spice_scores) / len(spice_scores)
        if avg_spice < 1.5:
            prefs.spice_tolerance = "low"
        elif avg_spice > 2.5:
            prefs.spice_tolerance = "high"
        else:
            prefs.spice_tolerance = "medium"

    # Weekday ratio
    prefs.weekday_order_ratio = weekday_count / max(len(order_history), 1)
    prefs.orders_last_30d = recent_30d
    prefs.orders_30_90d = period_30_90d

    # Health-conscious heuristic
    prefs.health_conscious_flag = (prefs.veg_ratio > 0.7 and prefs.spice_tolerance == "low")

    return prefs


# ═══════════════════════════════════════════════════════════════════════════════
#  USER EMBEDDING STORE (ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════════

class UserEmbeddingStore:
    """
    ChromaDB-backed store for user behavioral embeddings.

    Each user's order history is summarised as a text description, embedded
    via all-MiniLM-L6-v2 (384-dim), and stored for fast cosine-similarity
    retrieval.

    Cold-start: new users use city-level aggregate embeddings.
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str = "user_embeddings",
    ) -> None:
        self._persist_dir = str(persist_dir or CHROMADB_DIR)
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._model = None

    def _init_client(self) -> None:
        """Lazy-initialise ChromaDB client and collection."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            persist_path = Path(self._persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self._persist_dir,
                anonymized_telemetry=False,
            ))
        except Exception:
            # Fallback: ephemeral in-memory client
            import chromadb
            logger.warning("ChromaDB disk persistence failed — using in-memory mode")
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready (%d entries)",
                     self._collection_name, self._collection.count())

    def _get_model(self):
        """Lazy-load sentence-transformer."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._model

    def _history_to_text(self, user_id: str, prefs: UserPreferences,
                          order_history: list[dict] | None = None) -> str:
        """Convert user history into a natural-language summary for embedding."""
        parts = [f"User {user_id}"]

        if prefs.preferred_meal_times:
            parts.append(f"prefers {', '.join(prefs.preferred_meal_times)} meals")

        # Top cuisines
        top_cuisines = sorted(prefs.cuisine_affinities.items(), key=lambda x: -x[1])[:3]
        cuisine_str = ", ".join(c.replace("_", " ") for c, _ in top_cuisines if _ > 0.05)
        if cuisine_str:
            parts.append(f"frequently orders {cuisine_str}")

        # Dietary
        if prefs.veg_ratio > 0.8:
            parts.append("vegetarian")
        elif prefs.veg_ratio < 0.2:
            parts.append("non-vegetarian")

        parts.append(f"price segment {prefs.price_sensitivity}")
        parts.append(f"average order value ₹{prefs.avg_order_value:.0f}")

        if prefs.spice_tolerance == "high":
            parts.append("likes spicy food")
        elif prefs.spice_tolerance == "low":
            parts.append("prefers mild food")

        parts.append(f"ordered {prefs.total_orders} times total")

        return ", ".join(parts)

    def upsert_user(
        self,
        user_id: str,
        order_history: list[dict[str, Any]],
        prefs: UserPreferences | None = None,
    ) -> None:
        """Embed and store/update a user's behavioral profile."""
        self._init_client()

        if prefs is None:
            prefs = compute_user_preferences(order_history)

        text = self._history_to_text(user_id, prefs, order_history)
        model = self._get_model()
        embedding = model.encode([text], normalize_embeddings=True)[0].tolist()

        self._collection.upsert(
            ids=[user_id],
            embeddings=[embedding],
            metadatas=[{
                "user_id": user_id,
                "text": text[:500],
                "price_sensitivity": prefs.price_sensitivity,
                "veg_ratio": str(round(prefs.veg_ratio, 2)),
                "total_orders": str(prefs.total_orders),
            }],
            documents=[text],
        )
        logger.debug("Upserted user embedding for %s", user_id)

    def get_similar_users(
        self,
        user_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find top-K most similar users by embedding cosine similarity.

        Used for collaborative filtering and cold-start warm-up.
        """
        self._init_client()

        # Get target user's embedding
        result = self._collection.get(ids=[user_id], include=["embeddings"])
        if not result["embeddings"]:
            return []

        query_embedding = result["embeddings"][0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k + 1,  # +1 to exclude self
            include=["metadatas", "distances", "documents"],
        )

        similar = []
        for i, uid in enumerate(results["ids"][0]):
            if uid == user_id:
                continue
            similar.append({
                "user_id": uid,
                "similarity": round(1.0 - (results["distances"][0][i] if results["distances"] else 0), 4),
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })

        return similar[:top_k]

    def get_user_embedding(self, user_id: str) -> np.ndarray | None:
        """Retrieve a user's embedding vector (384-dim)."""
        self._init_client()

        result = self._collection.get(ids=[user_id], include=["embeddings"])
        if result["embeddings"]:
            return np.array(result["embeddings"][0], dtype=np.float32)
        return None

    def get_cold_start_embedding(self, city: str, meal_type: str = "dinner") -> np.ndarray:
        """
        Generate a cold-start embedding from city-level aggregates.

        For new users, we create a synthetic profile text based on the city's
        popular preferences and embed it.
        """
        profile = CITY_PROFILES.get(city, CITY_PROFILES.get("Mumbai", {}))
        text = (
            f"New user in {city}, "
            f"popular cuisines: {', '.join(profile.get('top_cuisines', []))}, "
            f"popular items: {', '.join(profile.get('popular_items', []))}, "
            f"avg order value ₹{profile.get('avg_order_value', 300)}, "
            f"meal time {meal_type}, "
            f"veg ratio {profile.get('veg_ratio', 0.5)}"
        )

        model = self._get_model()
        embedding = model.encode([text], normalize_embeddings=True)[0]
        return embedding.astype(np.float32)

    @property
    def count(self) -> int:
        """Number of users in the store."""
        self._init_client()
        return self._collection.count()
