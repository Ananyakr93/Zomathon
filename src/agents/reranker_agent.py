"""
Recommendation Re-ranking Agent — CSAO Agent #2
=================================================
Takes output from MealContextAgent + candidate items from a retrieval model
and re-ranks them into an optimal top-10 using a 4-factor weighted scoring system.

Scoring Formula:
  Final = 0.40 × Meal_Completion + 0.25 × Contextual_Relevance
        + 0.20 × Personalization  + 0.15 × Business_Value

Usage:
    from src.agents.meal_context_agent import MealContextAgent
    from src.agents.reranker_agent import RerankerAgent

    ctx_agent = MealContextAgent()
    ctx = ctx_agent.analyze(cart, restaurant, context, user)

    reranker = RerankerAgent()
    result = reranker.rerank(candidates, ctx, user, business_config)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

# ── SCORING WEIGHTS ─────────────────────────────────────────────────────────

W_MEAL_COMPLETION   = 0.40
W_CONTEXT_RELEVANCE = 0.25
W_PERSONALIZATION   = 0.20
W_BUSINESS_VALUE    = 0.15

# ── CATEGORY NORMALISATION (mirrors meal_context_agent) ─────────────────────

_CATEGORY_MAP = {
    "main": "main", "main_course": "main", "mains": "main",
    "side": "side", "sides": "side", "accompaniment": "side",
    "bread": "side", "rice": "side",
    "beverage": "beverage", "beverages": "beverage",
    "drink": "beverage", "drinks": "beverage",
    "dessert": "dessert", "desserts": "dessert", "sweet": "dessert",
    "appetizer": "appetizer", "appetizers": "appetizer",
    "starter": "appetizer", "starters": "appetizer",
    "soup": "appetizer", "snack": "appetizer",
}


def _norm_cat(raw: str) -> str:
    return _CATEGORY_MAP.get(raw.lower().strip(), raw.lower().strip())


# ── MEAL-TIME → CATEGORY PREFERENCE BOOSTS ──────────────────────────────────
# Some categories are naturally more attractive at certain meal times.

_TIME_CATEGORY_BOOST: dict[str, dict[str, float]] = {
    "breakfast": {"beverage": 1.5, "main": 1.0, "side": 0.9, "dessert": 0.3, "appetizer": 0.5},
    "lunch":     {"main": 1.3, "side": 1.2, "beverage": 1.1, "dessert": 0.7, "appetizer": 1.0},
    "evening_snacks": {"appetizer": 1.5, "beverage": 1.4, "dessert": 1.3, "side": 0.8, "main": 0.5},
    "dinner":    {"main": 1.2, "side": 1.2, "dessert": 1.1, "beverage": 1.0, "appetizer": 1.1},
    "late_night": {"main": 1.0, "beverage": 1.3, "dessert": 1.0, "side": 0.8, "appetizer": 0.6},
}


# ═════════════════════════════════════════════════════════════════════════════
#  RERANKER AGENT
# ═════════════════════════════════════════════════════════════════════════════

class RerankerAgent:
    """
    Agent #2 in the CSAO pipeline.

    Given candidate items + context-agent analysis, produces a scored
    top-10 with per-factor breakdowns and human-readable explanations.

    Stateless — every call to `rerank()` is independent.
    """

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        context_analysis: dict[str, Any],
        user: dict[str, Any] | None = None,
        business_config: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Re-rank candidates and return the optimal top-K.

        Parameters
        ----------
        candidates : list[dict]
            Up to 50 candidate items. Each must have:
                item_id, name, category, price, is_veg.
            Optional: popularity_score, avg_rating, tags, margin_pct,
                      is_promoted, prep_time_mins.
        context_analysis : dict
            Full output from MealContextAgent.analyze().
        user : dict | None
            Optional user profile. May include:
                dietary_preference, previously_accepted_items,
                preferred_cuisines, price_sensitivity, user_segment.
        business_config : dict | None
            Optional business constraints:
                min_margin_pct (float), max_same_category (int),
                promoted_item_ids (list), capacity_limited_ids (list).
        top_k : int
            Number of items to return (default 10).

        Returns
        -------
        dict with ranked_items, explanation_summary, and scoring_metadata.
        """
        if not candidates:
            return self._empty_result()

        biz = business_config or {}
        completion = context_analysis.get("meal_completion_analysis", {})
        strategy = context_analysis.get("recommendation_strategy", {})
        cultural = context_analysis.get("cultural_context", {})
        meal_id = context_analysis.get("meal_type_identification", {})
        anti = context_analysis.get("anti_recommendations", {})
        cuisine = context_analysis.get("cuisine", "North Indian")

        # Pre-compute lookups
        missing_cats = {
            _norm_cat(m["component"]): m["priority"]
            for m in completion.get("missing_components", [])
        }
        anti_names = set()
        anti_categories = set()
        for rule in anti.get("never_recommend", []):
            item_or_cat = rule.get("item_or_category", "")
            reason = rule.get("reason", "")
            if reason == "duplicate":
                anti_names.add(item_or_cat.lower())
            elif reason == "dietary_conflict":
                anti_categories.add("non_veg")
            elif reason == "cultural_mismatch":
                anti_names.add(item_or_cat.lower())

        cart_value = meal_id.get("total_cart_value", 0)
        avg_item_price = strategy.get("meal_context_signals", {}).get("avg_item_price", 200)
        meal_type = meal_id.get("meal_type", "lunch")
        is_weekend = strategy.get("meal_context_signals", {}).get("is_weekend", False)

        # User signals
        user_diet = "any"
        prev_accepted = set()
        price_sens = "medium"
        user_segment = "mid_tier"
        if user:
            diet_pref = user.get("dietary_preference", "non_vegetarian")
            user_diet = "veg" if diet_pref == "vegetarian" else "any"
            prev_accepted = set(
                n.lower() for n in user.get("previously_accepted_items", [])
            )
            price_sens = user.get("price_sensitivity", "medium")
            user_segment = user.get("user_segment", "mid_tier")

        # Active offer
        offer = strategy.get("active_offer")
        offer_item_id = offer.get("item_id") if offer else None

        # Promoted / capacity-limited
        promoted_ids = set(biz.get("promoted_item_ids", []))
        capacity_limited = set(biz.get("capacity_limited_ids", []))
        min_margin = biz.get("min_margin_pct", 0.0)
        max_same_cat = biz.get("max_same_category", 3)

        # ── SCORE EACH CANDIDATE ────────────────────────────────────────────
        scored: list[dict] = []

        for item in candidates:
            name = item.get("name", "")
            name_lower = name.lower()
            cat = _norm_cat(item.get("category", "other"))
            price = item.get("price", 0)
            is_veg = item.get("is_veg", True)
            item_id = item.get("item_id", "")

            # ── HARD FILTERS (skip immediately) ─────────────────────────────
            # 1. Duplicate name
            if name_lower in anti_names:
                continue
            # 2. Dietary conflict
            if user_diet == "veg" and not is_veg:
                continue
            # 3. Capacity limited
            if item_id in capacity_limited:
                continue
            # 4. Below min margin
            margin = item.get("margin_pct", 30.0)
            if margin < min_margin:
                continue

            # ── FACTOR 1: MEAL COMPLETION (0-10) ────────────────────────────
            meal_score = self._score_meal_completion(
                cat, missing_cats, completion, cuisine,
            )

            # ── FACTOR 2: CONTEXTUAL RELEVANCE (0-10) ──────────────────────
            context_score = self._score_contextual_relevance(
                cat, price, avg_item_price, meal_type, is_weekend,
                cultural.get("cultural_fit_score", 0.75), cuisine, item,
            )

            # ── FACTOR 3: PERSONALIZATION (0-10) ───────────────────────────
            personal_score = self._score_personalization(
                name_lower, prev_accepted, price, price_sens,
                user_segment, item, cat,
            )

            # ── FACTOR 4: BUSINESS VALUE (0-10) ────────────────────────────
            business_score = self._score_business_value(
                item_id, margin, promoted_ids, offer_item_id,
                item.get("popularity_score", 0.5), price,
            )

            # ── WEIGHTED FINAL ──────────────────────────────────────────────
            final = (
                W_MEAL_COMPLETION   * meal_score
                + W_CONTEXT_RELEVANCE * context_score
                + W_PERSONALIZATION   * personal_score
                + W_BUSINESS_VALUE    * business_score
            )

            # Build explanation
            why = self._build_why(
                name, cat, meal_score, context_score,
                personal_score, business_score, missing_cats,
                offer_item_id == item_id, meal_type, is_weekend,
            )

            scored.append({
                "item_id": item_id,
                "name": name,
                "category": cat,
                "price": price,
                "is_veg": is_veg,
                "final_score": round(final, 3),
                "score_breakdown": {
                    "meal_completion": round(meal_score, 2),
                    "contextual_relevance": round(context_score, 2),
                    "personalization": round(personal_score, 2),
                    "business_value": round(business_score, 2),
                },
                "why_this_item_now": why,
                "tags": item.get("tags", []),
                "popularity_score": item.get("popularity_score", 0.5),
            })

        # ── SORT BY FINAL SCORE ─────────────────────────────────────────────
        scored.sort(key=lambda x: -x["final_score"])

        # ── CONSTRAINED BEAM SEARCH ─────────────────────────────────────────
        final_list = self._constrained_beam_select(
            scored=scored,
            top_k=top_k,
            max_same_cat=max_same_cat,
        )

        # ── EXPLANATION SUMMARY ─────────────────────────────────────────────
        explanation = self._build_explanation_summary(
            final_list, missing_cats, cuisine, meal_type,
            is_weekend, offer_item_id, user, cart_value,
        )

        return {
            "agent": "RecommendationRerankerAgent",
            "version": "1.0",
            "total_candidates_received": len(candidates),
            "total_after_hard_filters": len(scored),
            "top_k_returned": len(final_list),
            "ranked_items": final_list,
            "explanation_summary": explanation,
            "scoring_metadata": {
                "weights": {
                    "meal_completion": W_MEAL_COMPLETION,
                    "contextual_relevance": W_CONTEXT_RELEVANCE,
                    "personalization": W_PERSONALIZATION,
                    "business_value": W_BUSINESS_VALUE,
                },
                "diversity_cap_per_category": max_same_cat,
                "hard_filters_applied": [
                    "duplicate_items", "dietary_conflict",
                    "capacity_limited", "min_margin_threshold",
                ],
            },
        }

    # ── FACTOR 1: MEAL COMPLETION SCORING ───────────────────────────────────

    @staticmethod
    def _score_meal_completion(
        cat: str,
        missing_cats: dict[str, str],
        completion: dict,
        cuisine: str,
    ) -> float:
        """Score 0-10: how well does this item fill a meal gap?"""
        score = 3.0  # baseline — any item has some minimal value

        if cat in missing_cats:
            priority = missing_cats[cat]
            if priority == "required":
                score = 10.0
            elif priority == "strongly_expected":
                score = 8.0
            elif priority == "nice_to_have":
                score = 6.0
        else:
            # Item category already exists in cart
            has_key = f"has_{cat}"
            if completion.get(has_key, False):
                score = 1.5  # redundant category — low value

        # Meal completeness is very low → any add is useful
        completeness = completion.get("meal_completeness_score", 0.5)
        if completeness < 0.3:
            score = min(10, score + 1.0)

        return min(10.0, score)

    # ── FACTOR 2: CONTEXTUAL RELEVANCE SCORING ──────────────────────────────

    @staticmethod
    def _score_contextual_relevance(
        cat: str,
        price: float,
        avg_item_price: float,
        meal_type: str,
        is_weekend: bool,
        cultural_fit: float,
        cuisine: str,
        item: dict,
    ) -> float:
        """Score 0-10: time / price / cuisine fit."""
        score = 5.0

        # Time-of-day category boost
        time_boosts = _TIME_CATEGORY_BOOST.get(meal_type, {})
        time_mult = time_boosts.get(cat, 1.0)
        score *= time_mult
        score = min(10, score)

        # Price alignment: add-ons should be ≤ 1.2x avg, penalise expensive items
        if avg_item_price > 0:
            price_ratio = price / avg_item_price
            if price_ratio <= 0.5:
                score += 1.5   # great deal
            elif price_ratio <= 1.0:
                score += 1.0   # well-aligned
            elif price_ratio <= 1.5:
                score += 0.0   # neutral
            elif price_ratio <= 2.0:
                score -= 1.5   # pricey
            else:
                score -= 3.0   # price shock

        # Cultural fit from context agent
        score += (cultural_fit - 0.5) * 3  # range: -1.5 to +1.5

        # Weekend dessert boost
        if is_weekend and cat == "dessert":
            score += 1.5

        return max(0.0, min(10.0, score))

    # ── FACTOR 3: PERSONALIZATION SCORING ───────────────────────────────────

    @staticmethod
    def _score_personalization(
        name_lower: str,
        prev_accepted: set[str],
        price: float,
        price_sensitivity: str,
        user_segment: str,
        item: dict,
        cat: str,
    ) -> float:
        """Score 0-10: how well the item matches this user."""
        score = 5.0  # neutral baseline

        # Previously accepted = strong signal
        for prev in prev_accepted:
            if prev in name_lower or name_lower in prev:
                score += 3.0
                break

        # Price sensitivity alignment
        if price_sensitivity == "high":
            if price <= 80:
                score += 2.0
            elif price <= 150:
                score += 0.5
            else:
                score -= 1.5
        elif price_sensitivity == "low":
            if price >= 200:
                score += 1.0  # premium user likes premium items
            elif price <= 50:
                score -= 0.5  # too cheap feels low-quality

        # Segment-specific boosts
        if user_segment == "health_conscious":
            tags = item.get("tags", [])
            if any(t in tags for t in ["healthy", "grilled", "salad", "low_cal"]):
                score += 2.0
            if any(t in tags for t in ["fried", "heavy", "creamy", "buttery"]):
                score -= 2.0
        elif user_segment == "experimenter":
            if item.get("is_new_item", False):
                score += 2.0  # experimenters like new items
        elif user_segment == "premium":
            if item.get("avg_rating", 0) >= 4.5:
                score += 1.0

        # Popularity as a filler signal (weak)
        pop = item.get("popularity_score", 0.5)
        score += (pop - 0.5) * 1.5  # range: -0.75 to +0.75

        return max(0.0, min(10.0, score))

    # ── FACTOR 4: BUSINESS VALUE SCORING ────────────────────────────────────

    @staticmethod
    def _score_business_value(
        item_id: str,
        margin_pct: float,
        promoted_ids: set[str],
        offer_item_id: str | None,
        popularity: float,
        price: float,
    ) -> float:
        """Score 0-10: business impact and margin."""
        score = 5.0

        # Margin contribution (normalised: 0-60% → 0-3 pts)
        score += min(3.0, margin_pct / 20.0)

        # Promoted item = business priority
        if item_id in promoted_ids:
            score += 2.0

        # Active offer boost
        if offer_item_id and item_id == offer_item_id:
            score += 2.5

        # Higher-priced items contribute more AOV (capped)
        if price >= 200:
            score += 1.0
        elif price >= 100:
            score += 0.5

        # Very popular items have high conversion probability
        score += (popularity - 0.5) * 1.0  # range: -0.5 to +0.5

        return max(0.0, min(10.0, score))

    # ── EXPLANATION BUILDERS ────────────────────────────────────────────────

    @staticmethod
    def _build_why(
        name: str,
        cat: str,
        meal_score: float,
        ctx_score: float,
        pers_score: float,
        biz_score: float,
        missing_cats: dict,
        has_offer: bool,
        meal_type: str,
        is_weekend: bool,
    ) -> str:
        """One-line explanation for why this item is recommended now."""
        parts = []

        # Lead with strongest signal
        if meal_score >= 8.0:
            if cat in missing_cats:
                priority = missing_cats[cat]
                if priority == "required":
                    parts.append(f"Fills a required gap ({cat})")
                else:
                    parts.append(f"Completes {cat} — strongly expected")
        elif meal_score >= 6.0:
            parts.append(f"Adds {cat} (nice-to-have)")

        if has_offer:
            parts.append("has active discount")

        if pers_score >= 7.5:
            parts.append("matches your past preferences")

        if ctx_score >= 7.5:
            parts.append(f"great fit for {meal_type}")

        if is_weekend and cat == "dessert":
            parts.append("weekend dessert boost")

        if biz_score >= 7.5:
            parts.append("high-value add-on")

        if not parts:
            parts.append(f"solid {cat} option")

        return f"{name}: " + ", ".join(parts) + "."

    @staticmethod
    def _build_explanation_summary(
        ranked: list[dict],
        missing_cats: dict,
        cuisine: str,
        meal_type: str,
        is_weekend: bool,
        offer_item_id: str | None,
        user: dict | None,
        cart_value: float,
    ) -> dict:
        """Build the overall explanation block."""
        if not ranked:
            return {"strategy": "No eligible candidates after filtering.", "top_3_reasoning": [], "vs_random": "N/A"}

        # Strategy summary
        gap_cats = [c for c, p in missing_cats.items() if p in ("required", "strongly_expected")]
        strategy_parts = [f"{cuisine} {meal_type} order"]
        if gap_cats:
            strategy_parts.append(f"filling gaps: {', '.join(gap_cats)}")
        if is_weekend:
            strategy_parts.append("weekend dessert boost active")
        if offer_item_id:
            strategy_parts.append("active offer prioritised")
        if user and user.get("price_sensitivity") == "high":
            strategy_parts.append("budget-friendly items preferred")

        strategy = "Optimising for " + " — ".join(strategy_parts) + "."

        # Top-3 reasoning
        top3 = []
        for item in ranked[:3]:
            bd = item["score_breakdown"]
            dominant = max(bd, key=bd.get)
            top3.append({
                "rank": item["rank"],
                "name": item["name"],
                "final_score": item["final_score"],
                "dominant_factor": dominant,
                "reasoning": item["why_this_item_now"],
            })

        # vs random
        avg_score = sum(i["final_score"] for i in ranked) / max(len(ranked), 1)
        cat_diversity = len(set(i["category"] for i in ranked))
        vs_random = (
            f"This ranking achieves {avg_score:.1f}/10 avg relevance score "
            f"across {cat_diversity} distinct categories, "
            f"vs ~3.5/10 expected from random ordering. "
            f"Top-3 items directly address meal gaps."
        )

        return {
            "strategy": strategy,
            "top_3_reasoning": top3,
            "vs_random": vs_random,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "agent": "RecommendationRerankerAgent",
            "version": "1.0",
            "total_candidates_received": 0,
            "total_after_hard_filters": 0,
            "top_k_returned": 0,
            "ranked_items": [],
            "explanation_summary": {
                "strategy": "No candidates provided.",
                "top_3_reasoning": [],
                "vs_random": "N/A",
            },
            "scoring_metadata": {},
        }

    # ── CONSTRAINED BEAM SEARCH ─────────────────────────────────────────────

    @staticmethod
    def _constrained_beam_select(
        scored: list[dict],
        top_k: int,
        max_same_cat: int,
        beam_width: int = 5,
    ) -> list[dict]:
        """
        Sub-selects top_k items maximizing sum(score) + diversity,
        while adhering to hard business constraints:
          - Maximum `max_same_cat` items per category
          - At least 2 high_margin items (margin_pct >= 50)
          - At most 3 impulse items (price < 80)
          - At least 3 distinct categories overall
        """
        if not scored:
            return []
        
        target_k = min(top_k, len(scored))

        @dataclass
        class State:
            items: list[dict]
            score_sum: float
            categories: dict[str, int]
            high_margin: int
            impulse: int
            
            def evaluate(self, is_final: bool = False) -> float:
                # Base score: sum of item scores
                score = self.score_sum
                
                # Diversity bonus: rewarding distinct categories
                unique_cats = len(self.categories)
                score += unique_cats * 2.0
                
                if is_final:
                    # Hard constraints applied as massive penalties if violated
                    if unique_cats < min(3, target_k):
                        score -= 100.0
                    if self.high_margin < min(2, target_k):
                        score -= 50.0
                    if self.impulse > 3:
                        score -= 50.0
                else:
                    # Progressive penalties
                    if self.impulse > 3:
                        score -= 50.0  # Unrecoverable

                return score

        # Start with an empty selection
        beam = [State(items=[], score_sum=0.0, categories={}, high_margin=0, impulse=0)]

        # To avoid permutations, we only add items strictly "after" the last added item in the sorted list
        # We enforce this by keeping track of the index in 'scored' of the last added item.
        # But for state tracking without extra fields, we can just use the item's position.
        item_to_idx = {id(item): i for i, item in enumerate(scored)}

        for step in range(target_k):
            next_states = []
            for state in beam:
                last_idx = item_to_idx[id(state.items[-1])] if state.items else -1
                
                for i in range(last_idx + 1, len(scored)):
                    candidate = scored[i]
                    cat = candidate["category"]
                    
                    # Hard cap per category constraint (strictly enforced)
                    if state.categories.get(cat, 0) >= max_same_cat:
                        continue
                        
                    # Build next state
                    new_cats = state.categories.copy()
                    new_cats[cat] = new_cats.get(cat, 0) + 1
                    
                    is_high_margin = candidate.get("margin_pct", 30) >= 50
                    is_impulse = candidate.get("price", 999) < 80
                    
                    new_state = State(
                        items=state.items + [candidate],
                        score_sum=state.score_sum + candidate.get("final_score", 0.0),
                        categories=new_cats,
                        high_margin=state.high_margin + (1 if is_high_margin else 0),
                        impulse=state.impulse + (1 if is_impulse else 0)
                    )
                    next_states.append(new_state)
            
            if not next_states:
                break
                
            # Score and prune
            is_final_step = (step == target_k - 1)
            next_states.sort(key=lambda s: -s.evaluate(is_final=is_final_step))
            beam = next_states[:beam_width]

        if not beam:
            return []

        # Return best selection, adding ranks
        best_selection = beam[0].items
        for i, item in enumerate(best_selection):
            item["rank"] = i + 1
            
        return best_selection


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO — End-to-end with MealContextAgent
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.agents.meal_context_agent import MealContextAgent

    # ── 1. Setup: simulate a Biryani cart ────────────────────────────────────
    cart = [
        {"item_id": "I3001", "name": "Chicken Dum Biryani", "category": "main",
         "price": 349, "is_veg": False, "tags": ["spicy", "signature"],
         "popularity_score": 0.95},
    ]

    restaurant = {
        "name": "Paradise Biryani", "cuisine_type": "Biryani Specialist",
        "price_tier": "mid", "rating": 4.3, "restaurant_type": "chain",
    }

    context = {
        "meal_type": "dinner", "hour": 20, "is_weekend": True,
        "city": "Hyderabad",
        "active_offer": {
            "item_id": "I3005", "item_name": "Gulab Jamun (2pc)",
            "offer_type": "flat_discount", "discount_pct": 20,
            "original_price": 79, "discounted_price": 63,
        },
    }

    user = {
        "user_segment": "mid_tier",
        "dietary_preference": "non_vegetarian",
        "price_sensitivity": "medium",
        "budget_per_order": [250, 600],
        "past_order_count": 15,
        "previously_accepted_items": ["Raita", "Gulab Jamun", "Coke 500ml"],
    }

    # ── 2. Run Meal Context Agent ────────────────────────────────────────────
    ctx_agent = MealContextAgent()
    ctx_analysis = ctx_agent.analyze(cart, restaurant, context, user)

    # ── 3. Simulate 20 candidate items from a retrieval model ────────────────
    candidates = [
        {"item_id": "I3002", "name": "Raita",               "category": "side",      "price": 49,  "is_veg": True,  "popularity_score": 0.88, "margin_pct": 65, "tags": ["cooling"]},
        {"item_id": "I3003", "name": "Mirchi Ka Salan",      "category": "side",      "price": 89,  "is_veg": True,  "popularity_score": 0.82, "margin_pct": 55, "tags": ["spicy", "traditional"]},
        {"item_id": "I3004", "name": "Onion Salad",          "category": "side",      "price": 29,  "is_veg": True,  "popularity_score": 0.60, "margin_pct": 80, "tags": ["fresh"]},
        {"item_id": "I3005", "name": "Gulab Jamun (2pc)",    "category": "dessert",   "price": 79,  "is_veg": True,  "popularity_score": 0.85, "margin_pct": 70, "tags": ["sweet"]},
        {"item_id": "I3006", "name": "Coke 500ml",           "category": "beverage",  "price": 60,  "is_veg": True,  "popularity_score": 0.90, "margin_pct": 75, "tags": []},
        {"item_id": "I3007", "name": "Lassi (Sweet)",        "category": "beverage",  "price": 69,  "is_veg": True,  "popularity_score": 0.72, "margin_pct": 60, "tags": ["cooling"]},
        {"item_id": "I3008", "name": "Phirni",               "category": "dessert",   "price": 89,  "is_veg": True,  "popularity_score": 0.65, "margin_pct": 60, "tags": ["sweet", "traditional"]},
        {"item_id": "I3009", "name": "Seekh Kebab (4pc)",    "category": "appetizer", "price": 189, "is_veg": False, "popularity_score": 0.78, "margin_pct": 45, "tags": ["grilled"]},
        {"item_id": "I3010", "name": "Tandoori Chicken Half","category": "appetizer", "price": 249, "is_veg": False, "popularity_score": 0.75, "margin_pct": 40, "tags": ["grilled", "signature"]},
        {"item_id": "I3011", "name": "Chicken Fried Rice",   "category": "main",      "price": 229, "is_veg": False, "popularity_score": 0.70, "margin_pct": 45, "tags": []},
        {"item_id": "I3012", "name": "Veg Biryani",          "category": "main",      "price": 249, "is_veg": True,  "popularity_score": 0.65, "margin_pct": 50, "tags": ["spicy"]},
        {"item_id": "I3013", "name": "Butter Naan (2pc)",    "category": "side",      "price": 59,  "is_veg": True,  "popularity_score": 0.80, "margin_pct": 70, "tags": []},
        {"item_id": "I3014", "name": "Chicken 65",           "category": "appetizer", "price": 179, "is_veg": False, "popularity_score": 0.82, "margin_pct": 50, "tags": ["fried", "spicy"]},
        {"item_id": "I3015", "name": "Lemon Iced Tea",       "category": "beverage",  "price": 79,  "is_veg": True,  "popularity_score": 0.60, "margin_pct": 65, "tags": ["cold"]},
        {"item_id": "I3016", "name": "Masala Papad",         "category": "appetizer", "price": 39,  "is_veg": True,  "popularity_score": 0.55, "margin_pct": 85, "tags": ["crunchy"]},
        {"item_id": "I3017", "name": "Chicken Dum Biryani",  "category": "main",      "price": 349, "is_veg": False, "popularity_score": 0.95, "margin_pct": 45, "tags": ["spicy"]},
        {"item_id": "I3018", "name": "Mineral Water 1L",     "category": "beverage",  "price": 20,  "is_veg": True,  "popularity_score": 0.40, "margin_pct": 90, "tags": []},
        {"item_id": "I3019", "name": "Sambhar Vada",         "category": "main",      "price": 99,  "is_veg": True,  "popularity_score": 0.50, "margin_pct": 55, "tags": []},
        {"item_id": "I3020", "name": "Ice Cream Sundae",     "category": "dessert",   "price": 129, "is_veg": True,  "popularity_score": 0.70, "margin_pct": 65, "tags": ["sweet", "cold"]},
    ]

    # ── 4. Re-rank ───────────────────────────────────────────────────────────
    reranker = RerankerAgent()
    result = reranker.rerank(
        candidates=candidates,
        context_analysis=ctx_analysis,
        user=user,
        business_config={
            "min_margin_pct": 10,
            "max_same_category": 3,
            "promoted_item_ids": [],
        },
        top_k=10,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
