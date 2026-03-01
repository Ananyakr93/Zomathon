"""
Meal Context Understanding Agent — CSAO Agent #1
=================================================
Analyzes cart composition with cultural intelligence and outputs
structured JSON for downstream recommendation agents.

Responsibilities:
  1. Meal Type Identification (breakfast / lunch / dinner / snack)
  2. Meal Completion Analysis (what's present, what's missing)
  3. Cultural Context (cuisine-specific meal patterns)
  4. Recommendation Strategy (top-3 categories to suggest next)
  5. Anti-Recommendations (what NOT to suggest)

Usage:
    agent = MealContextAgent()
    result = agent.analyze(cart_items, restaurant, context, user)
    # result is a dict ready for json.dumps()
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

# ── CUISINE MEAL BLUEPRINTS ─────────────────────────────────────────────────
# Each blueprint defines what a "complete meal" looks like in that cuisine.
# Keys: required components, optional extras, typical_combos, and red_flags.

MEAL_BLUEPRINTS: dict[str, dict] = {
    "North Indian": {
        "complete_meal": {
            "required": ["main", "bread/rice", "beverage"],
            "strongly_expected": ["side"],   # raita / dal / salad
            "nice_to_have": ["dessert", "appetizer"],
        },
        "typical_combos": [
            ("Paneer Butter Masala", "Garlic Naan"),
            ("Butter Chicken", "Jeera Rice"),
            ("Dal Makhani", "Tandoori Roti"),
            ("Chole", "Bhature"),
        ],
        "red_flags": [
            "Sambhar with North Indian main",
            "Coconut Chutney with North Indian main",
            "Sushi with curry",
        ],
        "default_side": "Raita",
        "default_beverage": "Lassi",
        "default_bread": "Garlic Naan",
    },
    "South Indian": {
        "complete_meal": {
            "required": ["main", "accompaniment"],  # sambhar / chutney
            "strongly_expected": ["beverage"],
            "nice_to_have": ["dessert"],
        },
        "typical_combos": [
            ("Masala Dosa", "Sambhar", "Coconut Chutney"),
            ("Idli", "Sambhar", "Chutney"),
            ("Uttapam", "Sambhar"),
            ("Rava Idli", "Filter Coffee"),
        ],
        "red_flags": [
            "Naan with Dosa",
            "Raita with South Indian breakfast items",
            "Butter Chicken with Dosa",
        ],
        "default_side": "Sambhar",
        "default_beverage": "Filter Coffee",
    },
    "Biryani Specialist": {
        "complete_meal": {
            "required": ["main"],          # biryani IS the main + rice
            "strongly_expected": ["side"],  # raita / salan
            "nice_to_have": ["beverage", "dessert"],
        },
        "typical_combos": [
            ("Chicken Dum Biryani", "Raita", "Mirchi Ka Salan"),
            ("Veg Biryani", "Raita", "Onion Salad"),
            ("Mutton Biryani", "Salan", "Gulab Jamun"),
        ],
        "red_flags": [
            "Naan with Biryani (redundant carbs unless explicitly ordered)",
            "Fried Rice with Biryani",
            "Sambhar with Biryani",
        ],
        "default_side": "Raita",
        "default_beverage": "Coke / Soft Drink",
        "default_dessert": "Gulab Jamun",
    },
    "Chinese": {
        "complete_meal": {
            "required": ["main"],  # noodles / fried rice / manchurian
            "strongly_expected": ["appetizer"],  # soup or starter
            "nice_to_have": ["beverage", "dessert"],
        },
        "typical_combos": [
            ("Veg Hakka Noodles", "Hot and Sour Soup"),
            ("Chicken Fried Rice", "Manchow Soup", "Spring Rolls"),
            ("Schezwan Noodles", "Gobi Manchurian"),
        ],
        "red_flags": [
            "Raita with Chinese",
            "Naan or Roti with Chinese",
            "Sambhar with Chinese",
            "Dal with Chinese",
        ],
        "default_side": "Spring Rolls",
        "default_beverage": "Coke / Iced Tea",
        "default_appetizer": "Hot and Sour Soup",
    },
    "Continental": {
        "complete_meal": {
            "required": ["main"],
            "strongly_expected": ["side", "beverage"],
            "nice_to_have": ["dessert", "appetizer"],
        },
        "typical_combos": [
            ("Margherita Pizza", "Garlic Bread", "Coke"),
            ("Pasta Arrabiata", "Caesar Salad", "Tiramisu"),
            ("Pasta Alfredo", "Garlic Bread", "Iced Tea"),
        ],
        "red_flags": [
            "Raita with Pizza",
            "Naan with Pasta",
            "Sambhar with Pizza",
            "Dal with Continental",
        ],
        "default_side": "Garlic Bread",
        "default_beverage": "Coke / Cold Coffee",
        "default_dessert": "Tiramisu / Brownie",
    },
    "Fast Food": {
        "complete_meal": {
            "required": ["main"],
            "strongly_expected": ["side", "beverage"],
            "nice_to_have": ["dessert"],
        },
        "typical_combos": [
            ("McAloo Tikki Burger", "Medium Fries", "Coke"),
            ("Zinger Burger", "French Fries", "Coke"),
            ("Whopper", "King Fries", "Pepsi"),
        ],
        "red_flags": [
            "Rice with Burger (unless it's a rice bowl)",
            "Raita with Fast Food",
            "Naan with Burger",
        ],
        "default_side": "French Fries",
        "default_beverage": "Coke (M)",
        "default_dessert": "McFlurry / Sundae",
    },
    "Street Food": {
        "complete_meal": {
            "required": ["main"],
            "strongly_expected": [],
            "nice_to_have": ["beverage", "dessert"],
        },
        "typical_combos": [
            ("Pav Bhaji", "Masala Soda"),
            ("Vada Pav", "Cutting Chai"),
            ("Pani Puri", "Bhel Puri"),
            ("Seekh Kebab", "Pav", "Green Chutney"),
        ],
        "red_flags": [
            "Formal appetizer with street food",
            "Tiramisu with Pav Bhaji",
            "Sushi with Chaat",
        ],
        "default_beverage": "Masala Soda / Chai",
    },
    "Dessert": {
        "complete_meal": {
            "required": ["dessert"],
            "strongly_expected": [],
            "nice_to_have": ["beverage"],
        },
        "typical_combos": [
            ("Chocolate Shake", "Brownie"),
            ("Waffle", "Ice Cream"),
            ("Cheesecake", "Cold Coffee"),
        ],
        "red_flags": [
            "Main course from a dessert-only order",
            "Rice / Naan with dessert restaurant",
        ],
        "default_beverage": "Cold Coffee / Shake",
    },
}

# ── MEAL TIME WINDOWS ───────────────────────────────────────────────────────

MEAL_TIME_WINDOWS = {
    "breakfast":       (6, 11),    # 6 AM – 11 AM
    "lunch":           (11, 15),   # 11 AM – 3 PM
    "evening_snacks":  (15, 19),   # 3 PM – 7 PM
    "dinner":          (19, 23),   # 7 PM – 11 PM
    "late_night":      (23, 6),    # 11 PM – 6 AM (wraps midnight)
}

# ── CATEGORY ALIASES ────────────────────────────────────────────────────────
# Normalise the various ways categories appear in the dataset.

_CATEGORY_MAP = {
    "main": "main",
    "main_course": "main",
    "mains": "main",
    "side": "side",
    "sides": "side",
    "accompaniment": "side",
    "bread": "side",
    "rice": "side",
    "beverage": "beverage",
    "beverages": "beverage",
    "drink": "beverage",
    "drinks": "beverage",
    "dessert": "dessert",
    "desserts": "dessert",
    "sweet": "dessert",
    "appetizer": "appetizer",
    "appetizers": "appetizer",
    "starter": "appetizer",
    "starters": "appetizer",
    "soup": "appetizer",
    "snack": "appetizer",
}


def _norm_cat(raw: str) -> str:
    """Normalise a menu category string to one of: main, side, beverage, dessert, appetizer."""
    return _CATEGORY_MAP.get(raw.lower().strip(), raw.lower().strip())


# ═════════════════════════════════════════════════════════════════════════════
#  MEAL CONTEXT AGENT
# ═════════════════════════════════════════════════════════════════════════════

class MealContextAgent:
    """
    Agent #1 in the CSAO pipeline.

    Given a cart snapshot + context, produces a structured analysis dict
    covering meal type, completion, cultural fit, recommendation strategy,
    and anti-recommendations.

    Stateless — every call to `analyze()` is independent.
    """

    # ── public API ──────────────────────────────────────────────────────────

    def analyze(
        self,
        cart_items: list[dict[str, Any]],
        restaurant: dict[str, Any],
        context: dict[str, Any],
        user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run full meal-context analysis.

        Parameters
        ----------
        cart_items : list[dict]
            Each dict must have at minimum: name, category, price, is_veg.
            Optional: subcategory, tags, popularity_score, avg_rating.
        restaurant : dict
            Must include: cuisine_type, price_tier.
            Optional: name, restaurant_type, rating, menu (full menu for gap analysis).
        context : dict
            Must include: meal_type (or timestamp/hour for inference).
            Optional: is_weekend, city, active_offer.
        user : dict | None
            Optional. May include: user_segment, dietary_preference, price_sensitivity,
            budget_per_order, previously_accepted_items, past_order_count.

        Returns
        -------
        dict  — structured analysis JSON with 5 sections.
        """
        cuisine = restaurant.get("cuisine_type", "North Indian")
        blueprint = MEAL_BLUEPRINTS.get(cuisine, MEAL_BLUEPRINTS["North Indian"])

        # ── Step 1: Meal Type Identification ────────────────────────────────
        meal_type_result = self._identify_meal_type(cart_items, context, cuisine)

        # ── Step 2: Meal Completion Analysis ────────────────────────────────
        completion_result = self._analyze_completion(cart_items, cuisine, blueprint)

        # ── Step 3: Cultural Context ────────────────────────────────────────
        cultural_result = self._assess_cultural_context(cart_items, cuisine, blueprint)

        # ── Step 4: Recommendation Strategy ─────────────────────────────────
        strategy_result = self._build_recommendation_strategy(
            cart_items, cuisine, blueprint, completion_result,
            restaurant, context, user,
        )

        # ── Step 5: Anti-Recommendations ────────────────────────────────────
        anti_result = self._build_anti_recommendations(
            cart_items, cuisine, blueprint, completion_result,
        )

        return {
            "agent": "MealContextUnderstandingAgent",
            "version": "1.0",
            "cuisine": cuisine,
            "meal_type_identification": meal_type_result,
            "meal_completion_analysis": completion_result,
            "cultural_context": cultural_result,
            "recommendation_strategy": strategy_result,
            "anti_recommendations": anti_result,
        }

    # ── SECTION 1: MEAL TYPE IDENTIFICATION ─────────────────────────────────

    def _identify_meal_type(
        self,
        cart_items: list[dict],
        context: dict,
        cuisine: str,
    ) -> dict:
        # Determine meal type from explicit context or hour
        meal_type = context.get("meal_type")
        hour = context.get("hour")
        confidence = 1.0

        if not meal_type and hour is not None:
            meal_type, confidence = self._infer_meal_from_hour(hour)
        elif not meal_type:
            meal_type = "lunch"
            confidence = 0.5

        # Full-meal vs quick bite heuristic
        total_value = sum(item.get("price", 0) for item in cart_items)
        num_items = len(cart_items)
        has_main = any(_norm_cat(i.get("category", "")) == "main" for i in cart_items)

        if num_items >= 3 and has_main and total_value >= 250:
            eating_mode = "full_meal"
            mode_confidence = 0.90
        elif num_items == 1 and total_value < 150:
            eating_mode = "quick_bite"
            mode_confidence = 0.85
        elif num_items <= 2 and not has_main:
            eating_mode = "snacking"
            mode_confidence = 0.75
        else:
            eating_mode = "building_meal"
            mode_confidence = 0.70

        return {
            "meal_type": meal_type,
            "meal_type_confidence": round(confidence, 2),
            "eating_mode": eating_mode,
            "eating_mode_confidence": round(mode_confidence, 2),
            "is_breakfast_cuisine_match": meal_type == "breakfast" and cuisine in ("South Indian", "Street Food"),
            "total_cart_value": total_value,
            "item_count": num_items,
        }

    @staticmethod
    def _infer_meal_from_hour(hour: int) -> tuple[str, float]:
        for meal, (start, end) in MEAL_TIME_WINDOWS.items():
            if start < end:
                if start <= hour < end:
                    return meal, 0.85
            else:  # wraps midnight (late_night)
                if hour >= start or hour < end:
                    return meal, 0.80
        return "lunch", 0.50

    # ── SECTION 2: MEAL COMPLETION ANALYSIS ─────────────────────────────────

    def _analyze_completion(
        self,
        cart_items: list[dict],
        cuisine: str,
        blueprint: dict,
    ) -> dict:
        # Categorise what's in the cart
        present_categories: dict[str, list[str]] = {}
        for item in cart_items:
            cat = _norm_cat(item.get("category", "other"))
            present_categories.setdefault(cat, []).append(item.get("name", "Unknown"))

        has = {
            "main": "main" in present_categories,
            "side": "side" in present_categories,
            "beverage": "beverage" in present_categories,
            "dessert": "dessert" in present_categories,
            "appetizer": "appetizer" in present_categories,
        }

        # Score completion against the blueprint
        bp = blueprint["complete_meal"]
        total_weight = 0.0
        earned_weight = 0.0

        # Required components — heavy weight
        for comp in bp["required"]:
            total_weight += 0.30
            norm = _norm_cat(comp)
            if has.get(norm, False):
                earned_weight += 0.30
            # Special: "bread/rice" counts as side
            if "/" in comp:
                for sub in comp.split("/"):
                    if has.get(_norm_cat(sub), False):
                        earned_weight += 0.30
                        break

        # Strongly expected — medium weight
        for comp in bp.get("strongly_expected", []):
            total_weight += 0.20
            if has.get(_norm_cat(comp), False):
                earned_weight += 0.20

        # Nice-to-have — light weight
        for comp in bp.get("nice_to_have", []):
            total_weight += 0.10
            if has.get(_norm_cat(comp), False):
                earned_weight += 0.10

        completeness = round(earned_weight / max(total_weight, 0.01), 2)

        # Build missing-components list
        missing = []
        for comp in bp["required"]:
            norm = _norm_cat(comp)
            if not has.get(norm, False):
                # check for slash-alternatives
                if "/" in comp:
                    if not any(has.get(_norm_cat(s), False) for s in comp.split("/")):
                        missing.append({"component": comp, "priority": "required"})
                else:
                    missing.append({"component": comp, "priority": "required"})

        for comp in bp.get("strongly_expected", []):
            if not has.get(_norm_cat(comp), False):
                missing.append({"component": comp, "priority": "strongly_expected"})

        for comp in bp.get("nice_to_have", []):
            if not has.get(_norm_cat(comp), False):
                missing.append({"component": comp, "priority": "nice_to_have"})

        # Natural-language reasoning
        reasoning = self._build_completion_reasoning(
            cuisine, has, missing, present_categories, completeness,
        )

        return {
            "components_present": present_categories,
            "has_main": has["main"],
            "has_side": has["side"],
            "has_beverage": has["beverage"],
            "has_dessert": has["dessert"],
            "has_appetizer": has["appetizer"],
            "meal_completeness_score": completeness,
            "missing_components": missing,
            "reasoning": reasoning,
        }

    @staticmethod
    def _build_completion_reasoning(
        cuisine: str,
        has: dict[str, bool],
        missing: list[dict],
        present: dict[str, list],
        score: float,
    ) -> str:
        parts = []

        if score >= 0.90:
            parts.append(f"This is a well-rounded {cuisine} meal.")
        elif score >= 0.60:
            parts.append(f"This {cuisine} order is partially complete.")
        else:
            parts.append(f"This {cuisine} order is missing key components.")

        # Mention what's present
        found = [f"{cat} ({', '.join(items)})" for cat, items in present.items()]
        if found:
            parts.append("Cart has: " + "; ".join(found) + ".")

        # Mention critical gaps
        required_gaps = [m["component"] for m in missing if m["priority"] == "required"]
        if required_gaps:
            parts.append(f"Missing required: {', '.join(required_gaps)}.")

        expected_gaps = [m["component"] for m in missing if m["priority"] == "strongly_expected"]
        if expected_gaps:
            parts.append(f"Strongly expected: {', '.join(expected_gaps)}.")

        return " ".join(parts)

    # ── SECTION 3: CULTURAL CONTEXT ─────────────────────────────────────────

    def _assess_cultural_context(
        self,
        cart_items: list[dict],
        cuisine: str,
        blueprint: dict,
    ) -> dict:
        item_names_lower = [i.get("name", "").lower() for i in cart_items]
        all_text = " ".join(item_names_lower)

        # Check for matching typical combos
        matched_combos = []
        for combo in blueprint.get("typical_combos", []):
            combo_lower = [c.lower() for c in combo]
            if any(cn in all_text for cn in combo_lower):
                matched_combos.append(list(combo))

        # Check for cultural mismatches (red flags)
        detected_flags = []
        for flag in blueprint.get("red_flags", []):
            # Flag format: "X with Y" — check if both keywords are present
            parts = flag.lower().split(" with ")
            if len(parts) == 2:
                if any(parts[0].strip() in n for n in item_names_lower):
                    detected_flags.append({
                        "flag": flag,
                        "severity": "warning",
                        "suggestion": f"Consider if '{parts[0].strip()}' fits a {cuisine} order.",
                    })
            else:
                # Generic flag check
                if any(flag.lower() in n for n in item_names_lower):
                    detected_flags.append({"flag": flag, "severity": "info"})

        # Cultural fit score
        if detected_flags:
            cultural_fit = max(0.4, 1.0 - 0.15 * len(detected_flags))
        elif matched_combos:
            cultural_fit = min(1.0, 0.85 + 0.05 * len(matched_combos))
        else:
            cultural_fit = 0.75  # neutral

        return {
            "cuisine": cuisine,
            "cultural_fit_score": round(cultural_fit, 2),
            "matched_typical_combos": matched_combos,
            "cultural_mismatches": detected_flags,
            "expected_meal_pattern": blueprint["complete_meal"],
        }

    # ── SECTION 4: RECOMMENDATION STRATEGY ──────────────────────────────────

    def _build_recommendation_strategy(
        self,
        cart_items: list[dict],
        cuisine: str,
        blueprint: dict,
        completion: dict,
        restaurant: dict,
        context: dict,
        user: dict | None,
    ) -> dict:
        missing = completion["missing_components"]
        has = {k: completion[f"has_{k}"] for k in ("main", "side", "beverage", "dessert", "appetizer")}

        # Determine user constraints
        is_veg = True  # default safe
        if user:
            is_veg = user.get("dietary_preference", "vegetarian") == "vegetarian"
        elif cart_items:
            is_veg = all(i.get("is_veg", True) for i in cart_items)

        price_tier = restaurant.get("price_tier", "mid")
        price_sensitivity = "medium"
        if user:
            price_sensitivity = user.get("price_sensitivity", "medium")

        avg_item_price = (
            sum(i.get("price", 0) for i in cart_items) / max(len(cart_items), 1)
        )
        target_price_range = self._get_target_price_range(avg_item_price, price_tier, price_sensitivity)

        is_weekend = context.get("is_weekend", False)
        meal_type = context.get("meal_type", "lunch")

        # Build ranked recommendation slots
        recs: list[dict] = []

        # Priority 1: Fill required gaps
        for m in missing:
            if m["priority"] == "required" and len(recs) < 3:
                rec = self._make_rec_slot(
                    category=_norm_cat(m["component"]),
                    cuisine=cuisine,
                    blueprint=blueprint,
                    priority="high",
                    rationale=f"{m['component']} is required for a complete {cuisine} meal.",
                    is_veg=is_veg,
                    target_price_range=target_price_range,
                )
                recs.append(rec)

        # Priority 2: Fill strongly expected
        for m in missing:
            if m["priority"] == "strongly_expected" and len(recs) < 3:
                rec = self._make_rec_slot(
                    category=_norm_cat(m["component"]),
                    cuisine=cuisine,
                    blueprint=blueprint,
                    priority="medium",
                    rationale=f"{m['component']} is strongly expected with {cuisine} orders.",
                    is_veg=is_veg,
                    target_price_range=target_price_range,
                )
                recs.append(rec)

        # Priority 3: Weekend dessert boost
        if is_weekend and not has["dessert"] and len(recs) < 3:
            recs.append(self._make_rec_slot(
                category="dessert",
                cuisine=cuisine,
                blueprint=blueprint,
                priority="medium",
                rationale="Weekend orders have 60%+ higher dessert acceptance rate.",
                is_veg=is_veg,
                target_price_range=target_price_range,
            ))

        # Priority 4: Beverage if missing (universally useful)
        if not has["beverage"] and len(recs) < 3:
            recs.append(self._make_rec_slot(
                category="beverage",
                cuisine=cuisine,
                blueprint=blueprint,
                priority="low",
                rationale="A beverage pairs well with any meal and has high acceptance.",
                is_veg=is_veg,
                target_price_range=target_price_range,
            ))

        # Ensure we have at least 3 recs by filling with nice_to_have
        for m in missing:
            if m["priority"] == "nice_to_have" and len(recs) < 3:
                recs.append(self._make_rec_slot(
                    category=_norm_cat(m["component"]),
                    cuisine=cuisine,
                    blueprint=blueprint,
                    priority="low",
                    rationale=f"{m['component']} would enhance this {cuisine} experience.",
                    is_veg=is_veg,
                    target_price_range=target_price_range,
                ))

        # Active offer awareness
        active_offer = context.get("active_offer")
        offer_info = None
        if active_offer:
            offer_info = {
                "item_id": active_offer.get("item_id"),
                "item_name": active_offer.get("item_name"),
                "offer_type": active_offer.get("offer_type"),
                "discount_pct": active_offer.get("discount_pct"),
                "should_boost": True,
                "reason": "Active offer increases acceptance probability; prioritise if category matches a gap.",
            }

        return {
            "top_recommendations": recs[:3],
            "user_constraints": {
                "dietary": "vegetarian" if is_veg else "non_vegetarian",
                "price_sensitivity": price_sensitivity,
                "target_price_range": target_price_range,
            },
            "active_offer": offer_info,
            "meal_context_signals": {
                "is_weekend": is_weekend,
                "meal_type": meal_type,
                "cart_value_so_far": sum(i.get("price", 0) for i in cart_items),
                "avg_item_price": round(avg_item_price),
            },
        }

    def _make_rec_slot(
        self,
        category: str,
        cuisine: str,
        blueprint: dict,
        priority: str,
        rationale: str,
        is_veg: bool,
        target_price_range: list[int],
    ) -> dict:
        """Build a single recommendation slot (category + characteristics)."""
        # Try to get a cuisine-specific default for this category
        specific_item = blueprint.get(f"default_{category}", None)

        # Temperature / spice hints based on cuisine
        temp_hint = "any"
        spice_hint = "any"
        if category == "beverage":
            temp_hint = "cold" if cuisine in ("Fast Food", "Continental", "Chinese") else "any"
        if cuisine in ("Biryani Specialist", "Street Food"):
            spice_hint = "mild_or_cooling" if category == "side" else "any"

        return {
            "rank": None,  # filled by caller based on position
            "category": category,
            "priority": priority,
            "specific_suggestion": specific_item,
            "characteristics": {
                "dietary": "veg" if is_veg else "any",
                "temperature": temp_hint,
                "spice_level": spice_hint,
            },
            "price_range": target_price_range,
            "rationale": rationale,
        }

    @staticmethod
    def _get_target_price_range(
        avg_price: float,
        price_tier: str,
        sensitivity: str,
    ) -> list[int]:
        """Suggest a price range for add-on items aligned to cart context."""
        # Add-ons are typically cheaper than the primary item
        if sensitivity == "high":
            return [20, int(avg_price * 0.5)]
        elif sensitivity == "low":
            return [40, int(avg_price * 1.2)]
        else:  # medium
            return [30, int(avg_price * 0.8)]

    # ── SECTION 5: ANTI-RECOMMENDATIONS ─────────────────────────────────────

    def _build_anti_recommendations(
        self,
        cart_items: list[dict],
        cuisine: str,
        blueprint: dict,
        completion: dict,
    ) -> dict:
        item_names = [i.get("name", "") for i in cart_items]
        item_names_lower = [n.lower() for n in item_names]
        cart_categories = list(completion["components_present"].keys())

        never_recommend: list[dict] = []

        # Rule 1: Don't recommend items already in the cart
        for name in item_names:
            never_recommend.append({
                "item_or_category": name,
                "reason": "duplicate",
                "explanation": f"'{name}' is already in the cart.",
            })

        # Rule 2: Cultural mismatches from red flags
        for flag in blueprint.get("red_flags", []):
            parts = flag.lower().split(" with ")
            if len(parts) == 2:
                offending = parts[0].strip()
                # If the second part matches something in cart, block the first
                if any(parts[1].strip() in n for n in item_names_lower):
                    never_recommend.append({
                        "item_or_category": offending.title(),
                        "reason": "cultural_mismatch",
                        "explanation": flag,
                    })

        # Rule 3: Don't double-up on mains (unless explicitly a "combo" order)
        if completion["has_main"] and len(completion["components_present"].get("main", [])) >= 2:
            never_recommend.append({
                "item_or_category": "main",
                "reason": "redundant_category",
                "explanation": "Cart already has 2+ main items; adding more may overwhelm the order.",
            })

        # Rule 4: Price guardrails — don't recommend items > 2x average
        avg_price = sum(i.get("price", 0) for i in cart_items) / max(len(cart_items), 1)
        never_recommend.append({
            "item_or_category": f"items above ₹{int(avg_price * 2)}",
            "reason": "price_shock",
            "explanation": f"Add-ons should stay under 2x the avg item price (₹{int(avg_price)}) to avoid rejection.",
        })

        # Rule 5: Dietary — if user is veg, never recommend non-veg
        all_veg = all(i.get("is_veg", True) for i in cart_items)
        if all_veg:
            never_recommend.append({
                "item_or_category": "non_vegetarian items",
                "reason": "dietary_conflict",
                "explanation": "All cart items are vegetarian; recommending non-veg risks offending the user.",
            })

        # Rule 6: Cuisine-specific redundancies
        redundancy_rules = {
            "Biryani Specialist": [("main", "Fried Rice", "Biryani already provides rice; don't suggest more rice dishes.")],
            "South Indian": [("main", "Naan", "Naan doesn't pair with South Indian items.")],
            "Fast Food": [("main", "Rice", "Rice-based items don't fit a burger meal.")],
        }
        for cat, keyword, reason in redundancy_rules.get(cuisine, []):
            never_recommend.append({
                "item_or_category": keyword,
                "reason": "redundant_cuisine_clash",
                "explanation": reason,
            })

        return {
            "never_recommend": never_recommend,
            "total_anti_rules": len(never_recommend),
        }


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick demo with a sample Biryani cart
    agent = MealContextAgent()

    sample_cart = [
        {
            "item_id": "I3001",
            "name": "Chicken Dum Biryani",
            "category": "main",
            "price": 349,
            "is_veg": False,
            "tags": ["spicy", "signature"],
            "popularity_score": 0.95,
        },
    ]

    sample_restaurant = {
        "name": "Paradise Biryani",
        "cuisine_type": "Biryani Specialist",
        "price_tier": "mid",
        "rating": 4.3,
        "restaurant_type": "chain",
    }

    sample_context = {
        "meal_type": "dinner",
        "hour": 20,
        "is_weekend": True,
        "city": "Hyderabad",
        "active_offer": {
            "item_id": "I3005",
            "item_name": "Gulab Jamun (2pc)",
            "offer_type": "flat_discount",
            "discount_pct": 20,
            "original_price": 79,
            "discounted_price": 63,
        },
    }

    sample_user = {
        "user_segment": "mid_tier",
        "dietary_preference": "non_vegetarian",
        "price_sensitivity": "medium",
        "budget_per_order": [250, 600],
        "past_order_count": 15,
        "previously_accepted_items": ["Raita", "Gulab Jamun"],
    }

    result = agent.analyze(sample_cart, sample_restaurant, sample_context, sample_user)
    print(json.dumps(result, indent=2, ensure_ascii=False))
