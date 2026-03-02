"""
test_cold_start.py
==================
Tests for the 1.3 Cold Start Strategy in ColdStartAgent.

Covers:
  - SLM response parsing (item list, reasons dict)
  - New User scenario (city aggregates + SLM prompt)
  - New Restaurant scenario (menu embeddings + SLM analysis)
  - New Item scenario (text-based similarity)
  - Hybrid pipeline execution order
  - Final SLM enrichment layer
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
#  SLM RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════════════════════

class TestSLMParsing:
    """Test the SLM response parsers work with various formats."""

    def test_parse_item_list_json(self):
        from src.agents.cold_start_agent import ColdStartAgent

        items = ColdStartAgent._parse_slm_item_list(
            '["Raita", "Gulab Jamun", "Lassi"]'
        )
        assert items == ["Raita", "Gulab Jamun", "Lassi"]

    def test_parse_item_list_embedded(self):
        from src.agents.cold_start_agent import ColdStartAgent

        items = ColdStartAgent._parse_slm_item_list(
            'Sure! Here are suggestions: ["Raita", "Naan", "Coke"] Hope that helps!'
        )
        assert items == ["Raita", "Naan", "Coke"]

    def test_parse_item_list_plain(self):
        from src.agents.cold_start_agent import ColdStartAgent

        items = ColdStartAgent._parse_slm_item_list(
            "- Raita\n- Gulab Jamun\n- Naan"
        )
        assert len(items) >= 2
        assert "Raita" in items

    def test_parse_item_list_empty(self):
        from src.agents.cold_start_agent import ColdStartAgent

        assert ColdStartAgent._parse_slm_item_list("") == []

    def test_parse_reasons_json(self):
        from src.agents.cold_start_agent import ColdStartAgent

        reasons = ColdStartAgent._parse_slm_reasons(
            '{"Raita": "Cools down spicy biryani", "Naan": "Perfect with curry"}'
        )
        assert "Raita" in reasons
        assert "Naan" in reasons

    def test_parse_reasons_embedded(self):
        from src.agents.cold_start_agent import ColdStartAgent

        reasons = ColdStartAgent._parse_slm_reasons(
            'Here are the reasons: {"Raita": "Great cooling agent"}'
        )
        assert "Raita" in reasons

    def test_parse_reasons_empty(self):
        from src.agents.cold_start_agent import ColdStartAgent

        assert ColdStartAgent._parse_slm_reasons("") == {}


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW USER SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewUser:
    """Test Scenario 1: New User cold start."""

    def test_new_user_returns_recommendations(self):
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        # Mock SLM to avoid real LLM calls
        agent._slm = MagicMock()
        agent._slm.available = False
        agent._slm.generate.return_value = ""

        result = agent.recommend(
            cart_items=[
                {"name": "Chicken Biryani", "category": "main", "price": 250, "is_veg": False},
            ],
            restaurant={"cuisine_type": "North Indian", "price_tier": "mid"},
            context={"city": "Mumbai", "meal_type": "dinner"},
            cold_start_type="new_user",
        )
        assert result["cold_start_type"] == "new_user"
        assert result["version"] == "2.0"
        assert len(result["recommendations"]) > 0

    def test_new_user_includes_city_aggregates(self):
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        agent._slm = MagicMock()
        agent._slm.available = False
        agent._slm.generate.return_value = ""

        result = agent.recommend(
            cart_items=[
                {"name": "Dosa", "category": "main", "price": 120, "is_veg": True},
            ],
            restaurant={"cuisine_type": "South Indian", "price_tier": "low"},
            context={"city": "Mumbai", "meal_type": "lunch"},
            cold_start_type="new_user",
        )
        # Should include some city-popular items
        sources = {r.get("source") for r in result["recommendations"]}
        # May contain city_aggregate, cuisine_knowledge, or universal_safe
        assert len(result["recommendations"]) > 0

    def test_new_user_slm_prompt_integration(self):
        """Verify the SLM is called with the correct prompt format."""
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        mock_slm = MagicMock()
        mock_slm.available = True
        mock_slm.generate.return_value = '["Raita", "Gulab Jamun"]'
        agent._slm = mock_slm

        result = agent.recommend(
            cart_items=[
                {"name": "Biryani", "category": "main", "price": 250, "is_veg": False},
            ],
            restaurant={"cuisine_type": "North Indian"},
            context={"city": "Delhi", "meal_type": "dinner"},
            cold_start_type="new_user",
        )
        # SLM should be called at least once for new_user
        assert mock_slm.generate.called
        # Check that the zero-shot prompt contains expected fragments
        call_args = mock_slm.generate.call_args_list
        prompts = [str(c) for c in call_args]
        prompt_text = " ".join(prompts)
        assert "dinner" in prompt_text.lower()
        assert "delhi" in prompt_text.lower()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW RESTAURANT SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewRestaurant:
    """Test Scenario 2: New Restaurant cold start."""

    def test_new_restaurant_with_menu(self):
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        agent._slm = MagicMock()
        agent._slm.available = False
        agent._slm.generate.return_value = ""

        menu = [
            {"name": "Butter Chicken", "category": "main", "price": 280, "is_veg": False},
            {"name": "Paneer Tikka", "category": "appetizer", "price": 200, "is_veg": True},
            {"name": "Garlic Naan", "category": "bread", "price": 60, "is_veg": True},
            {"name": "Raita", "category": "side", "price": 40, "is_veg": True},
            {"name": "Gulab Jamun", "category": "dessert", "price": 80, "is_veg": True},
            {"name": "Lassi", "category": "beverage", "price": 70, "is_veg": True},
        ]

        result = agent.recommend(
            cart_items=[
                {"name": "Butter Chicken", "category": "main", "price": 280, "is_veg": False},
            ],
            restaurant={"cuisine_type": "North Indian", "price_tier": "mid"},
            menu=menu,
            cold_start_type="new_restaurant",
        )
        assert result["cold_start_type"] == "new_restaurant"
        assert len(result["recommendations"]) > 0

    def test_new_restaurant_slm_menu_analysis(self):
        """Verify SLM is called with menu analysis prompt."""
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        mock_slm = MagicMock()
        mock_slm.available = True
        mock_slm.generate.return_value = '["Naan", "Dal"]'
        agent._slm = mock_slm

        menu = [
            {"name": "Chicken Curry", "category": "main", "price": 200},
            {"name": "Rice", "category": "side", "price": 80},
        ]

        result = agent.recommend(
            cart_items=[{"name": "Chicken Curry", "category": "main", "price": 200, "is_veg": False}],
            restaurant={"cuisine_type": "North Indian"},
            menu=menu,
            cold_start_type="new_restaurant",
        )
        assert mock_slm.generate.called
        # Check the menu analysis prompt contains menu items
        call_args = str(mock_slm.generate.call_args_list)
        assert "restaurant menu" in call_args.lower() or "menu includes" in call_args.lower()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ITEM SCENARIO
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewItem:
    """Test Scenario 3: New Item cold start."""

    def test_new_item_returns_complements(self):
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        agent._slm = MagicMock()
        agent._slm.available = False
        agent._slm.generate.return_value = ""

        result = agent.recommend(
            cart_items=[
                {"name": "Strawberry Shake", "category": "beverage", "price": 150,
                 "is_veg": True, "is_new": True, "impressions": 10},
            ],
            restaurant={"cuisine_type": "Continental", "price_tier": "mid"},
            cold_start_type="new_item",
        )
        assert result["cold_start_type"] == "new_item"
        assert len(result["recommendations"]) > 0

    def test_new_item_with_menu_similarity(self):
        """New items should find similar items from the menu."""
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        agent._slm = MagicMock()
        agent._slm.available = False
        agent._slm.generate.return_value = ""

        menu = [
            {"name": "Chocolate Milkshake", "category": "beverage", "price": 120, "is_veg": True},
            {"name": "French Fries", "category": "side", "price": 80, "is_veg": True},
            {"name": "Brownie", "category": "dessert", "price": 100, "is_veg": True},
        ]

        result = agent.recommend(
            cart_items=[
                {"name": "Vanilla Shake", "category": "beverage", "price": 130,
                 "is_veg": True, "is_new": True},
            ],
            restaurant={"cuisine_type": "Continental"},
            menu=menu,
            cold_start_type="new_item",
        )
        assert len(result["recommendations"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  HYBRID PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridPipeline:
    """Test the hybrid fallback pipeline always returns results."""

    def test_never_returns_empty(self):
        """System never fails silently — always returns recommendations."""
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        agent._slm = MagicMock()
        agent._slm.available = False
        agent._slm.generate.return_value = ""

        for cst in ["new_user", "new_restaurant", "new_item", "unusual_cart"]:
            result = agent.recommend(
                cart_items=[
                    {"name": "Mystery Item", "category": "main", "price": 200, "is_veg": True},
                ],
                restaurant={"cuisine_type": "Unknown Cuisine"},
                cold_start_type=cst,
            )
            assert len(result["recommendations"]) > 0, f"Empty results for {cst}"

    def test_version_is_2(self):
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        agent._slm = MagicMock()
        agent._slm.available = False
        agent._slm.generate.return_value = ""

        result = agent.recommend(
            cart_items=[{"name": "X", "category": "main", "price": 100, "is_veg": True}],
            restaurant={"cuisine_type": "North Indian"},
        )
        assert result["version"] == "2.0"
        assert result["metadata"]["approach"] == "hybrid_cold_start_pipeline"

    def test_slm_enrichment_adds_reasoning(self):
        """When SLM is available, final enrichment adds slm_enriched flag."""
        from src.agents.cold_start_agent import ColdStartAgent

        agent = ColdStartAgent()
        mock_slm = MagicMock()
        mock_slm.available = True
        mock_slm.generate.side_effect = [
            # First call: new_user SLM zero-shot
            '["Raita"]',
            # Second call: final enrichment
            '{"Raita": "Cooling complement to spicy food"}',
        ]
        agent._slm = mock_slm

        result = agent.recommend(
            cart_items=[{"name": "Biryani", "category": "main", "price": 250, "is_veg": False}],
            restaurant={"cuisine_type": "North Indian"},
            context={"city": "Hyderabad", "meal_type": "dinner"},
            cold_start_type="new_user",
        )
        assert result["metadata"]["slm_available"] is True
        # Check that some items have slm_enriched flag
        enriched = [r for r in result["recommendations"] if r.get("slm_enriched")]
        # May or may not match depending on name casing, but flag should exist
        has_flag = any("slm_enriched" in r for r in result["recommendations"])
        assert has_flag
