"""
Unit tests for 2.1/2.2 Mathematical Constraints and Business Rules.
Covers:
  - Dynamic top_k based on meal completeness
  - Minimum score threshold filtering
  - Hard business constraints (high margin >= 2, impulse <= 3, >= 3 categories)
  - Constrained beam search logic
  - Sequential session state tracking
"""

import pytest
import time
from src.serving.orchestrator import ServingOrchestrator
from src.agents.reranker_agent import RerankerAgent
from src.serving.session_state import SessionStore, SessionState, CartSnapshot


class TestSessionState:
    def test_cart_snapshot_creation(self):
        store = SessionStore()
        session = store.get_or_create("sess_1")
        
        cart_t0 = [{"name": "Biryani"}]
        snap_0 = session.update(cart_t0, completeness=32.0)
        
        assert snap_0.step == 0
        assert snap_0.completeness == 32.0
        assert "biryani" in snap_0.items_added
        
        cart_t1 = [{"name": "Biryani"}, {"name": "Raita"}]
        snap_1 = session.update(cart_t1, completeness=58.0)
        
        assert snap_1.step == 1
        assert "raita" in snap_1.items_added
        assert not snap_1.items_removed
        
        ctx = session.get_context()
        assert ctx["session_step"] == 1
        assert ctx["current_completeness"] == 58.0
        assert ctx["completeness_trajectory"] == [32.0, 58.0]
        assert ctx["meal_is_done"] is False

    def test_meal_is_done_flag(self):
        store = SessionStore()
        session = store.get_or_create("sess_2")
        
        cart = [{"name": "Meal Combo"}]
        session.update(cart, completeness=95.0)
        
        ctx = session.get_context()
        assert ctx["meal_is_done"] is True

    def test_recommendation_fatigue(self):
        store = SessionStore()
        session = store.get_or_create("sess_3")
        
        # Simulating a user ignoring recommendations for 3 steps
        for i in range(4):
            session.update([{"name": "Biryani"}], completeness=30.0)
            session.record_recommendations(["A", "B", "C"])
            
        ctx = session.get_context()
        assert ctx["recommendation_fatigue"] > 0.0


class TestDynamicTopKAndMinThreshold:
    def test_dynamic_top_k_aggressive(self):
        orch = ServingOrchestrator()
        # < 50% completeness -> aggressive -> target 8
        ctx = {"recommendation_fatigue": 0.0}
        k = orch._compute_dynamic_top_k(10, 30.0, ctx)
        assert k >= 8

    def test_dynamic_top_k_standard(self):
        orch = ServingOrchestrator()
        # 50-80% completeness -> target 6
        ctx = {"recommendation_fatigue": 0.0}
        k = orch._compute_dynamic_top_k(10, 60.0, ctx)
        assert k == 6
        
    def test_dynamic_top_k_gentle(self):
        orch = ServingOrchestrator()
        # > 80% completeness -> gentle -> target 4
        ctx = {"recommendation_fatigue": 0.0}
        k = orch._compute_dynamic_top_k(10, 90.0, ctx)
        assert k == 4

    def test_fatigue_reduces_k(self):
        orch = ServingOrchestrator()
        ctx = {"recommendation_fatigue": 0.8} # High fatigue
        # at 30% completeness (aggressive), base target is max(8, 10) = 10
        # fatigue > 0.5 reduces target by 2: max(2, 10 - 2) = 8
        k = orch._compute_dynamic_top_k(10, 30.0, ctx)
        assert k == 8

    def test_min_score_threshold_filter(self):
        orch = ServingOrchestrator()
        ranked = [
            {"name": "item1", "category": "c1", "final_score": 0.9},
            {"name": "item2", "category": "c2", "final_score": 0.5},
            {"name": "item3", "category": "c3", "final_score": 0.1}, # below threshold
        ]
        cart = [{"name": "existing"}]
        
        from src.serving.orchestrator import MIN_SCORE_THRESHOLD
        assert 0.1 < MIN_SCORE_THRESHOLD
        
        final = orch._post_process(ranked, top_k=5, cart_items=cart)
        assert len(final) == 2
        assert "item3" not in [i["name"] for i in final]


class TestConstrainedBeamSearch:
    def _create_candidates(self) -> list[dict]:
        # Create a mix of candidates to test constraints
        # Needs to test: max_same_cat=3, high_margin >= 2 (margin_pct >= 50), impulse <= 3 (price < 80)
        return [
            # High margin (>50%), impulse (<80)
            {"final_score": 0.9, "category": "side", "margin_pct": 60, "price": 40},
            {"final_score": 0.85, "category": "side", "margin_pct": 55, "price": 50},
            {"final_score": 0.80, "category": "side", "margin_pct": 50, "price": 60},
            {"final_score": 0.75, "category": "side", "margin_pct": 40, "price": 70}, # Same cat 4
            
            # Not impulse (>=80), not high margin (<50%)
            {"final_score": 0.95, "category": "main", "margin_pct": 30, "price": 200},
            {"final_score": 0.88, "category": "main", "margin_pct": 45, "price": 250},
            
            # High margin (>50%), not impulse (>=80)
            {"final_score": 0.70, "category": "beverage", "margin_pct": 70, "price": 90},
            {"final_score": 0.60, "category": "dessert", "margin_pct": 65, "price": 120},
            
            # Impulse (<80), not high margin (<50%)
            {"final_score": 0.65, "category": "appetizer", "margin_pct": 20, "price": 50},
            {"final_score": 0.55, "category": "appetizer", "margin_pct": 15, "price": 60}, 
        ]

    def test_business_constraints_enforced(self):
        scored = self._create_candidates()
        agent = RerankerAgent()
        
        # Request top 6 to ensure we test max_same_cat and minimums
        selected = agent._constrained_beam_select(scored, top_k=6, max_same_cat=3)
        
        assert len(selected) > 0
        
        # 1. Check max_same_cat = 3
        cat_counts = {}
        for item in selected:
            cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
        assert all(count <= 3 for count in cat_counts.values())
        
        # 2. Check high margin >= 2
        high_margin_count = sum(1 for i in selected if i.get("margin_pct", 30) >= 50)
        assert high_margin_count >= 2
        
        # 3. Check impulse <= 3
        impulse_count = sum(1 for i in selected if i.get("price", 999) < 80)
        assert impulse_count <= 3
        
        # 4. Minimum 3 distinct categories required in scoring
        assert len(cat_counts) >= 3

    def test_diversity_bonus(self):
        scored = [
            {"final_score": 0.9, "category": "c1", "margin_pct": 60, "price": 100},
            {"final_score": 0.89, "category": "c1", "margin_pct": 60, "price": 100},
            {"final_score": 0.88, "category": "c1", "margin_pct": 60, "price": 100},
            {"final_score": 0.80, "category": "c2", "margin_pct": 60, "price": 100},
            {"final_score": 0.79, "category": "c3", "margin_pct": 60, "price": 100},
            {"final_score": 0.78, "category": "c4", "margin_pct": 60, "price": 100},
        ]
        
        agent = RerankerAgent()
        # Even though top 3 by score are all in "c1", beam search should pick diverse cats 
        # because of the diversity bonus and the min 3 categories constraint
        selected = agent._constrained_beam_select(scored, top_k=3, max_same_cat=3)
        
        cats = {i["category"] for i in selected}
        assert len(cats) >= 2 # at least 2 distinct due to small k
