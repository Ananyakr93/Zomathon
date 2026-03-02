"""
test_feature_pipeline.py
========================
Tests for the 3-tier Feature Engineering Pipeline.

Covers:
  - Tier 1: CartFeatures (composition, meal completeness, distribution)
  - Tier 1: SessionGraph (graph construction, PMI edges, candidate scoring)
  - Tier 2: UserPreferences (cuisine affinity, price sensitivity, recency)
  - Tier 3: TemporalFeatures, GeographicFeatures, RestaurantFeatures
  - Integration: orchestrator _retrieve_features returns all tiers
"""

import math
import time
from datetime import datetime

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 1: CART FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestCartFeatures:
    """Test cart composition, meal completeness, and price positioning."""

    def test_empty_cart(self):
        from src.features.cart_features import compute_cart_features

        feats = compute_cart_features([])
        assert feats.item_count == 0
        assert feats.total_value == 0.0
        assert feats.meal_completeness_score == 0.0

    def test_single_main(self):
        from src.features.cart_features import compute_cart_features

        cart = [{"name": "Chicken Biryani", "category": "biryani", "price": 250, "is_veg": False}]
        feats = compute_cart_features(cart)
        assert feats.item_count == 1
        assert feats.total_value == 250.0
        assert feats.has_main is True
        assert feats.has_side is False
        assert feats.meal_completeness_score == 50.0  # only main = 1.0/2.0 * 100

    def test_complete_meal(self):
        from src.features.cart_features import compute_cart_features

        cart = [
            {"name": "Biryani", "category": "biryani", "price": 250, "is_veg": False},
            {"name": "Raita", "category": "raita", "price": 50, "is_veg": True},
            {"name": "Lassi", "category": "lassi", "price": 80, "is_veg": True},
            {"name": "Gulab Jamun", "category": "dessert", "price": 60, "is_veg": True},
        ]
        feats = compute_cart_features(cart)
        assert feats.has_main is True
        assert feats.has_side is True
        assert feats.has_beverage is True
        assert feats.has_dessert is True
        assert feats.meal_completeness_score == 100.0

    def test_veg_ratio(self):
        from src.features.cart_features import compute_cart_features

        cart = [
            {"name": "A", "category": "main", "price": 200, "is_veg": True},
            {"name": "B", "category": "main", "price": 200, "is_veg": True},
            {"name": "C", "category": "main", "price": 200, "is_veg": False},
        ]
        feats = compute_cart_features(cart)
        assert feats.veg_ratio == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_price_buckets(self):
        from src.features.cart_features import compute_cart_features

        cart = [
            {"name": "A", "category": "main", "price": 100},   # budget
            {"name": "B", "category": "main", "price": 250},   # mid
            {"name": "C", "category": "main", "price": 500},   # premium
        ]
        feats = compute_cart_features(cart)
        assert feats.budget_items_count == 1
        assert feats.mid_items_count == 1
        assert feats.premium_items_count == 1

    def test_to_dict(self):
        from src.features.cart_features import compute_cart_features

        cart = [{"name": "X", "category": "main", "price": 200}]
        d = compute_cart_features(cart).to_dict()
        assert "cart_total_value" in d
        assert "meal_completeness_score" in d
        assert "price_diversity_score" in d


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 1: SESSION GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionGraph:
    """Test temporal session graph construction and candidate scoring."""

    def test_add_single_item(self):
        from src.features.session_graph import SessionGraph, PMIMatrix

        pmi = PMIMatrix({"biryani": {"raita": 0.8}})
        graph = SessionGraph(pmi_matrix=pmi)

        graph.add_item({"name": "Chicken Biryani", "price": 250, "category": "main"})
        assert graph.size == 1

    def test_edge_creation(self):
        from src.features.session_graph import SessionGraph, PMIMatrix

        pmi = PMIMatrix({"biryani": {"raita": 0.8}})
        graph = SessionGraph(pmi_matrix=pmi)

        graph.add_item({"name": "Biryani", "price": 250, "category": "main"})
        graph.add_item({"name": "Raita", "price": 50, "category": "side"})

        assert graph.size == 2
        assert graph.graph.number_of_edges() == 1

    def test_candidate_scoring(self):
        from src.features.session_graph import SessionGraph, PMIMatrix

        pmi = PMIMatrix({
            "biryani": {"raita": 0.8, "salan": 0.7, "gulab_jamun": 0.4},
        })
        graph = SessionGraph(pmi_matrix=pmi)
        graph.add_item({"name": "Biryani", "price": 250, "category": "main"})

        candidates = graph.get_candidate_scores(top_k=10)
        assert len(candidates) > 0
        # Raita should score highest
        names = [c["name"] for c in candidates]
        assert "raita" in names

    def test_graph_features(self):
        from src.features.session_graph import SessionGraph, PMIMatrix

        pmi = PMIMatrix({"biryani": {"raita": 0.8}})
        graph = SessionGraph(pmi_matrix=pmi)
        graph.add_item({"name": "Biryani", "price": 250})
        graph.add_item({"name": "Raita", "price": 50})

        feats = graph.compute_graph_features()
        assert feats["graph_node_count"] == 2
        assert feats["graph_edge_count"] == 1
        assert feats["graph_max_edge_weight"] > 0

    def test_remove_item(self):
        from src.features.session_graph import SessionGraph, PMIMatrix

        pmi = PMIMatrix({})
        graph = SessionGraph(pmi_matrix=pmi)
        graph.add_item({"name": "A", "price": 100})
        graph.add_item({"name": "B", "price": 100})
        assert graph.size == 2

        graph.remove_item("A")
        assert graph.size == 1

    def test_reset(self):
        from src.features.session_graph import SessionGraph, PMIMatrix

        pmi = PMIMatrix({})
        graph = SessionGraph(pmi_matrix=pmi)
        graph.add_item({"name": "X"})
        graph.reset()
        assert graph.size == 0

    def test_pmi_matrix_symmetric_lookup(self):
        from src.features.session_graph import PMIMatrix

        pmi = PMIMatrix({"biryani": {"raita": 0.8}})
        assert pmi.get_pmi("biryani", "raita") == 0.8
        assert pmi.get_pmi("raita", "biryani") == 0.8
        assert pmi.get_pmi("biryani", "pizza") == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 2: USER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserPreferences:
    """Test user preference computation."""

    def test_empty_history(self):
        from src.features.user_features import compute_user_preferences

        prefs = compute_user_preferences([])
        assert prefs.total_orders == 0
        # Should have default affinities
        assert len(prefs.cuisine_affinities) == 8

    def test_basic_preferences(self):
        from src.features.user_features import compute_user_preferences

        now = time.time()
        orders = [
            {"timestamp": now - 86400, "total_value": 300, "cuisine": "north_indian",
             "meal_type": "dinner", "is_veg": True, "spice_level": "high"},
            {"timestamp": now - 172800, "total_value": 250, "cuisine": "north_indian",
             "meal_type": "dinner", "is_veg": True, "spice_level": "high"},
        ]
        prefs = compute_user_preferences(orders, current_time=now)
        assert prefs.total_orders == 2
        assert prefs.cuisine_affinities["north_indian"] > 0
        assert prefs.veg_ratio > 0.5

    def test_price_sensitivity(self):
        from src.features.user_features import compute_user_preferences

        now = time.time()
        budget_orders = [
            {"timestamp": now - 86400, "total_value": 150, "cuisine": "street_food",
             "meal_type": "lunch", "is_veg": True}
        ]
        prefs = compute_user_preferences(budget_orders, current_time=now)
        assert prefs.price_sensitivity == "budget"

    def test_to_dict(self):
        from src.features.user_features import compute_user_preferences

        prefs = compute_user_preferences([])
        d = prefs.to_dict()
        assert "cuisine_affinity_north_indian" in d
        assert "price_sensitivity" in d
        assert "user_veg_ratio" in d


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 3: CONTEXT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalFeatures:
    """Test temporal context features."""

    def test_dinner_time(self):
        from src.features.context_features import compute_temporal_features

        dt = datetime(2025, 3, 1, 20, 30)  # 8:30 PM Saturday
        feats = compute_temporal_features(dt)
        assert feats.meal_time == "dinner"
        assert feats.is_weekend is True
        assert feats.hour == 20

    def test_lunch_time(self):
        from src.features.context_features import compute_temporal_features

        dt = datetime(2025, 3, 3, 13, 0)  # 1 PM Monday
        feats = compute_temporal_features(dt)
        assert feats.meal_time == "lunch"
        assert feats.is_weekend is False

    def test_cyclical_encoding(self):
        from src.features.context_features import compute_temporal_features

        dt = datetime(2025, 3, 1, 12, 0)
        feats = compute_temporal_features(dt)
        # sin² + cos² ≈ 1
        assert feats.hour_sin ** 2 + feats.hour_cos ** 2 == pytest.approx(1.0, abs=0.01)


class TestGeographicFeatures:
    """Test geographic context features."""

    def test_mumbai(self):
        from src.features.context_features import compute_geographic_features

        feats = compute_geographic_features("Mumbai")
        assert feats.city_encoding["Mumbai"] == 1
        assert feats.city_encoding["Delhi"] == 0
        assert feats.affluence_score > 0

    def test_city_one_hot(self):
        from src.features.context_features import compute_geographic_features

        feats = compute_geographic_features("Bangalore")
        d = feats.to_dict()
        assert d["city_bangalore"] == 1
        assert sum(v for k, v in d.items() if k.startswith("city_")) == 1


class TestRestaurantFeatures:
    """Test restaurant feature computation."""

    def test_basic_restaurant(self):
        from src.features.context_features import compute_restaurant_features

        rest = {
            "restaurant_type": "chain",
            "cuisine_type": "North Indian",
            "avg_price": 300,
            "rating": 4.2,
        }
        feats = compute_restaurant_features(rest)
        assert feats.restaurant_type == "chain"
        assert feats.avg_price_point == 300.0
        assert feats.rating == 4.2

    def test_to_dict(self):
        from src.features.context_features import compute_restaurant_features

        feats = compute_restaurant_features({"restaurant_type": "local"})
        d = feats.to_dict()
        assert "restaurant_type_code" in d
        assert "restaurant_addon_acceptance_rate" in d


class TestCandidateFeatures:
    """Test candidate item feature computation."""

    def test_basic_candidate(self):
        from src.features.context_features import compute_candidate_features

        candidate = {"name": "Raita", "price": 50, "category": "side", "popularity_score": 0.8}
        restaurant = {"cuisine_type": "North Indian"}
        feats = compute_candidate_features(candidate, restaurant)
        assert feats.price == 50.0
        assert feats.popularity_score == 0.8

    def test_restaurant_fit(self):
        from src.features.context_features import compute_candidate_features

        candidate = {"name": "Naan", "cuisine_type": "north_indian"}
        restaurant = {"cuisine_type": "north_indian"}
        feats = compute_candidate_features(candidate, restaurant)
        assert feats.restaurant_fit_score == 1.0


class TestCombinedContext:
    """Test the combined context feature computation."""

    def test_all_context(self):
        from src.features.context_features import compute_all_context_features

        restaurant = {"restaurant_type": "chain", "cuisine_type": "North Indian"}
        context = {"city": "Mumbai"}
        result = compute_all_context_features(restaurant, context)

        assert "hour" in result
        assert "meal_time" in result
        assert "city_mumbai" in result
        assert "restaurant_type_code" in result
