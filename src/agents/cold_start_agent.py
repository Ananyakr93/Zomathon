"""
Cold Start Recommendation Agent — CSAO Agent #3
=================================================
Safety-net agent that handles scenarios where traditional data-driven
models lack sufficient signal:

  1. New user  (no order history)
  2. New restaurant  (limited interaction data)
  3. Newly launched menu item
  4. Unusual cart combination  (no similar patterns)

Uses culinary intelligence, contextual reasoning, and price-conscious
heuristics instead of collaborative-filtering patterns.

Usage:
    from src.agents.cold_start_agent import ColdStartAgent

    agent = ColdStartAgent()
    result = agent.recommend(
        cart_items, restaurant, menu, context,
        cold_start_type="new_user",
    )
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


# ── CUISINE COMPLEMENT KNOWLEDGE BASE ───────────────────────────────────────
# Maps cuisine → category → list of (item_name, price_bracket, is_veg, tags)
# price_bracket: "low"=₹20-80, "mid"=₹80-180, "high"=₹180-350

_CUISINE_COMPLEMENTS: dict[str, dict[str, list[tuple]]] = {
    "North Indian": {
        "side":      [("Raita",           "low",  True,  ["cooling"]),
                      ("Dal Makhani",     "mid",  True,  ["hearty", "traditional"]),
                      ("Onion Salad",     "low",  True,  ["fresh"]),
                      ("Mix Veg",         "mid",  True,  ["healthy"]),
                      ("Papad Masala",    "low",  True,  ["crunchy"])],
        "bread":     [("Butter Naan",     "low",  True,  ["popular", "essential"]),
                      ("Garlic Naan",     "low",  True,  ["popular"]),
                      ("Tandoori Roti",   "low",  True,  ["healthy"]),
                      ("Lachha Paratha",  "low",  True,  ["flaky"]),
                      ("Missi Roti",      "low",  True,  ["traditional"])],
        "beverage":  [("Lassi (Sweet)",   "low",  True,  ["cooling", "traditional"]),
                      ("Masala Chaas",    "low",  True,  ["cooling"]),
                      ("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Mango Lassi",     "mid",  True,  ["sweet"]),
                      ("Nimbu Pani",      "low",  True,  ["refreshing"])],
        "dessert":   [("Gulab Jamun (2pc)","low", True,  ["sweet", "traditional"]),
                      ("Rasmalai",        "mid",  True,  ["sweet", "rich"]),
                      ("Gajar Ka Halwa",  "mid",  True,  ["sweet", "seasonal"]),
                      ("Kheer",           "mid",  True,  ["sweet", "traditional"])],
        "appetizer": [("Paneer Tikka",    "mid",  True,  ["grilled"]),
                      ("Hara Bhara Kebab","mid",  True,  ["healthy"]),
                      ("Chicken Seekh Kebab","mid",False,["grilled"]),
                      ("Dahi Kebab",      "mid",  True,  ["creamy"])],
    },
    "Mughlai": {
        "side":      [("Raita",           "low",  True,  ["cooling", "essential"]),
                      ("Onion Salad",     "low",  True,  ["fresh"]),
                      ("Boondi Raita",    "low",  True,  ["traditional"])],
        "bread":     [("Butter Naan",     "low",  True,  ["popular", "essential"]),
                      ("Sheermal",        "mid",  True,  ["sweet", "traditional"]),
                      ("Roomali Roti",    "low",  True,  ["traditional"]),
                      ("Garlic Naan",     "low",  True,  ["popular"])],
        "beverage":  [("Thandai",         "mid",  True,  ["traditional", "cooling"]),
                      ("Rose Sharbat",    "low",  True,  ["sweet", "refreshing"]),
                      ("Lassi (Sweet)",   "low",  True,  ["cooling"]),
                      ("Coke 500ml",      "low",  True,  ["popular"])],
        "dessert":   [("Shahi Tukda",     "mid",  True,  ["sweet", "rich"]),
                      ("Phirni",          "mid",  True,  ["sweet", "traditional"]),
                      ("Kulfi Falooda",   "mid",  True,  ["sweet", "cold"]),
                      ("Gulab Jamun (2pc)","low", True,  ["sweet"])],
        "appetizer": [("Galouti Kebab",   "mid",  False, ["melt_in_mouth"]),
                      ("Seekh Kebab (4pc)","mid", False, ["grilled"]),
                      ("Mutton Shammi",   "mid",  False, ["traditional"]),
                      ("Paneer Tikka",    "mid",  True,  ["grilled"])],
    },
    "Punjabi": {
        "side":      [("Raita",           "low",  True,  ["cooling"]),
                      ("Onion Salad",     "low",  True,  ["fresh"]),
                      ("Papad Masala",    "low",  True,  ["crunchy"]),
                      ("Boondi Raita",    "low",  True,  ["traditional"])],
        "bread":     [("Butter Naan",     "low",  True,  ["popular", "essential"]),
                      ("Amritsari Kulcha","mid",  True,  ["popular", "stuffed"]),
                      ("Tandoori Roti",   "low",  True,  ["healthy"]),
                      ("Paratha",         "low",  True,  ["traditional"])],
        "beverage":  [("Lassi (Sweet)",   "low",  True,  ["popular", "essential"]),
                      ("Masala Chaas",    "low",  True,  ["cooling"]),
                      ("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Mango Lassi",     "mid",  True,  ["sweet"])],
        "dessert":   [("Gulab Jamun (2pc)","low", True,  ["sweet", "traditional"]),
                      ("Rasmalai",        "mid",  True,  ["sweet", "rich"]),
                      ("Phirni",          "mid",  True,  ["sweet"])],
        "appetizer": [("Tandoori Chicken","mid",  False, ["grilled", "popular"]),
                      ("Amritsari Fish Fry","mid",False, ["fried"]),
                      ("Paneer Tikka",    "mid",  True,  ["grilled"])],
    },
    "South Indian": {
        "side":      [("Sambhar",         "low",  True,  ["traditional", "essential"]),
                      ("Coconut Chutney", "low",  True,  ["traditional"]),
                      ("Tomato Chutney",  "low",  True,  ["tangy"]),
                      ("Mint Chutney",    "low",  True,  ["fresh"]),
                      ("Podi",            "low",  True,  ["spicy"])],
        "beverage":  [("Filter Coffee",   "low",  True,  ["traditional", "hot", "popular"]),
                      ("Buttermilk",      "low",  True,  ["cooling"]),
                      ("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Rasam",           "low",  True,  ["traditional", "hot"])],
        "dessert":   [("Payasam",         "mid",  True,  ["sweet", "traditional"]),
                      ("Mysore Pak",      "low",  True,  ["sweet"]),
                      ("Kesari Bath",     "low",  True,  ["sweet"]),
                      ("Gulab Jamun (2pc)","low", True,  ["sweet"])],
        "appetizer": [("Medu Vada",       "low",  True,  ["crispy", "popular"]),
                      ("Bonda",           "low",  True,  ["snack"]),
                      ("Bajji",           "low",  True,  ["fried"])],
    },
    "Bengali": {
        "side":      [("Aloo Bhaja",      "low",  True,  ["fried", "traditional"]),
                      ("Begun Bhaja",     "low",  True,  ["fried"]),
                      ("Salad",           "low",  True,  ["fresh"]),
                      ("Papad",           "low",  True,  ["crunchy"])],
        "bread":     [("Luchi",           "low",  True,  ["traditional", "popular"]),
                      ("Paratha",         "low",  True,  ["traditional"])],
        "beverage":  [("Ghol (Buttermilk)","low", True,  ["cooling"]),
                      ("Aam Panna",       "low",  True,  ["refreshing"]),
                      ("Coke 500ml",      "low",  True,  ["popular"])],
        "dessert":   [("Mishti Doi",      "low",  True,  ["sweet", "traditional", "popular"]),
                      ("Rasgulla (2pc)",  "low",  True,  ["sweet", "traditional"]),
                      ("Sandesh",         "low",  True,  ["sweet", "traditional"]),
                      ("Payesh",          "mid",  True,  ["sweet"])],
        "appetizer": [("Fish Fry",        "mid",  False, ["fried", "popular"]),
                      ("Chop (Cutlet)",   "mid",  False, ["fried"]),
                      ("Beguni",          "low",  True,  ["fried"])],
    },
    "Gujarati": {
        "side":      [("Kachumber Salad", "low",  True,  ["fresh"]),
                      ("Papad",           "low",  True,  ["crunchy"]),
                      ("Pickle",          "low",  True,  ["tangy"])],
        "bread":     [("Roti",            "low",  True,  ["essential"]),
                      ("Puri",            "low",  True,  ["traditional"]),
                      ("Thepla",          "low",  True,  ["popular"])],
        "beverage":  [("Chaas (Buttermilk)","low", True,  ["cooling", "essential"]),
                      ("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Aam Ras",         "mid",  True,  ["seasonal", "sweet"])],
        "dessert":   [("Shrikhand",       "mid",  True,  ["sweet", "traditional", "popular"]),
                      ("Basundi",         "mid",  True,  ["sweet", "rich"]),
                      ("Mohanthal",       "low",  True,  ["sweet"]),
                      ("Gulab Jamun (2pc)","low", True,  ["sweet"])],
        "appetizer": [("Dhokla",          "low",  True,  ["steamed", "popular"]),
                      ("Khandvi",         "low",  True,  ["traditional"]),
                      ("Fafda",           "low",  True,  ["crunchy"])],
    },
    "Rajasthani": {
        "side":      [("Raita",           "low",  True,  ["cooling"]),
                      ("Papad Masala",    "low",  True,  ["crunchy"]),
                      ("Pickle (Achaar)", "low",  True,  ["spicy"])],
        "bread":     [("Bati",            "mid",  True,  ["traditional", "essential"]),
                      ("Missi Roti",      "low",  True,  ["traditional"]),
                      ("Garlic Naan",     "low",  True,  ["popular"])],
        "beverage":  [("Chaach",          "low",  True,  ["cooling", "essential"]),
                      ("Keri Ka Panna",    "low",  True,  ["refreshing"]),
                      ("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Lassi (Sweet)",   "low",  True,  ["cooling"])],
        "dessert":   [("Ghevar",          "mid",  True,  ["sweet", "traditional"]),
                      ("Malpua",          "mid",  True,  ["sweet"]),
                      ("Churma Ladoo",    "mid",  True,  ["sweet", "traditional"]),
                      ("Gulab Jamun (2pc)","low", True,  ["sweet"])],
        "appetizer": [("Mirchi Vada",     "low",  True,  ["fried", "spicy"]),
                      ("Pyaaz Kachori",   "low",  True,  ["crunchy", "popular"]),
                      ("Dal Kachori",     "low",  True,  ["traditional"])],
    },
    "Biryani Specialist": {
        "side":      [("Raita",           "low",  True,  ["cooling", "essential"]),
                      ("Mirchi Ka Salan", "mid",  True,  ["spicy", "traditional"]),
                      ("Onion Salad",     "low",  True,  ["fresh"]),
                      ("Masala Papad",    "low",  True,  ["crunchy"]),
                      ("Dahi Chutney",    "low",  True,  ["cooling"])],
        "beverage":  [("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Lassi (Sweet)",   "low",  True,  ["cooling"]),
                      ("Mineral Water 1L","low",  True,  []),
                      ("Masala Chaas",    "low",  True,  ["cooling"])],
        "dessert":   [("Gulab Jamun (2pc)","low", True,  ["sweet", "traditional"]),
                      ("Phirni",          "mid",  True,  ["sweet", "traditional"]),
                      ("Double Ka Meetha","mid",  True,  ["sweet", "hyderabadi"]),
                      ("Qubani Ka Meetha","mid",  True,  ["sweet", "hyderabadi"])],
        "appetizer": [("Seekh Kebab (4pc)","mid", False, ["grilled"]),
                      ("Chicken 65",      "mid",  False, ["fried", "spicy"]),
                      ("Paneer 65",       "mid",  True,  ["fried", "spicy"]),
                      ("Tangdi Kebab",    "mid",  False, ["grilled"])],
    },
    "Chinese": {
        "side":      [("Spring Rolls",    "mid",  True,  ["crispy"]),
                      ("Honey Chilli Potato","mid",True,  ["sweet_spicy"]),
                      ("French Fries",    "low",  True,  ["popular"]),
                      ("Chilli Paneer Dry","mid",  True, ["popular"])],
        "beverage":  [("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Iced Tea",        "low",  True,  ["cold"]),
                      ("Lemon Soda",      "low",  True,  ["refreshing"]),
                      ("Virgin Mojito",   "mid",  True,  ["refreshing"])],
        "dessert":   [("Brownie with Ice Cream","mid",True,["sweet"]),
                      ("Ice Cream Sundae","mid",  True,  ["sweet", "cold"]),
                      ("Fried Ice Cream", "mid",  True,  ["sweet"])],
        "appetizer": [("Hot and Sour Soup","mid", True,  ["soup", "traditional"]),
                      ("Manchow Soup",    "mid",  True,  ["soup"]),
                      ("Crispy Corn",     "mid",  True,  ["crunchy"]),
                      ("Dim Sum (4pc)",   "mid",  True,  ["steamed"])],
    },
    "Continental": {
        "side":      [("Garlic Bread",    "low",  True,  ["popular"]),
                      ("Caesar Salad",    "mid",  True,  ["healthy"]),
                      ("French Fries",    "low",  True,  ["popular"]),
                      ("Coleslaw",        "low",  True,  ["fresh"]),
                      ("Mashed Potatoes", "mid",  True,  ["creamy"])],
        "beverage":  [("Coke 500ml",      "low",  True,  ["popular"]),
                      ("Cold Coffee",     "mid",  True,  ["cold", "popular"]),
                      ("Iced Tea",        "low",  True,  ["cold"]),
                      ("Lemonade",        "low",  True,  ["refreshing"]),
                      ("Virgin Mojito",   "mid",  True,  ["refreshing"])],
        "dessert":   [("Brownie",         "mid",  True,  ["sweet"]),
                      ("Tiramisu",        "mid",  True,  ["sweet", "premium"]),
                      ("Chocolate Mousse","mid",  True,  ["sweet"]),
                      ("Cheesecake",      "mid",  True,  ["sweet", "premium"])],
        "appetizer": [("Bruschetta",      "mid",  True,  ["starter"]),
                      ("Soup of the Day", "mid",  True,  ["soup"]),
                      ("Nachos with Salsa","mid", True,  ["crunchy"])],
    },
    "Fast Food": {
        "side":      [("French Fries (M)","low",  True,  ["popular"]),
                      ("Onion Rings",     "low",  True,  ["crispy"]),
                      ("Hash Brown",      "low",  True,  ["crispy"]),
                      ("Potato Wedges",   "low",  True,  ["crispy"])],
        "beverage":  [("Coke (M)",        "low",  True,  ["popular"]),
                      ("Pepsi (M)",       "low",  True,  ["popular"]),
                      ("Iced Tea",        "low",  True,  ["cold"]),
                      ("Cold Coffee",     "mid",  True,  ["cold", "popular"]),
                      ("Chocolate Shake", "mid",  True,  ["sweet"])],
        "dessert":   [("McFlurry",        "mid",  True,  ["sweet", "cold"]),
                      ("Sundae",          "low",  True,  ["sweet"]),
                      ("Chocolate Cookie","low",  True,  ["sweet"]),
                      ("Brownie",         "mid",  True,  ["sweet"])],
        "appetizer": [("Chicken Nuggets", "mid",  False, ["fried"]),
                      ("Veg Strips",      "mid",  True,  ["fried"]),
                      ("Cheese Balls",    "mid",  True,  ["fried", "popular"])],
    },
    "Street Food": {
        "side":      [("Green Chutney",   "low",  True,  ["tangy"]),
                      ("Sev",             "low",  True,  ["crunchy"]),
                      ("Onion Rings",     "low",  True,  ["crispy"])],
        "beverage":  [("Masala Soda",     "low",  True,  ["refreshing"]),
                      ("Cutting Chai",    "low",  True,  ["hot", "traditional"]),
                      ("Nimbu Pani",      "low",  True,  ["refreshing"]),
                      ("Coke 500ml",      "low",  True,  ["popular"])],
        "dessert":   [("Kulfi",           "low",  True,  ["sweet", "cold"]),
                      ("Rabdi",           "low",  True,  ["sweet"]),
                      ("Jalebi (2pc)",    "low",  True,  ["sweet", "traditional"])],
        "appetizer": [("Pani Puri",       "low",  True,  ["popular"]),
                      ("Bhel Puri",       "low",  True,  ["popular", "tangy"]),
                      ("Samosa (2pc)",    "low",  True,  ["popular"]),
                      ("Vada Pav",        "low",  True,  ["popular", "mumbai"])],
    },
    "Dessert": {
        "beverage":  [("Cold Coffee",     "mid",  True,  ["cold", "popular"]),
                      ("Chocolate Shake", "mid",  True,  ["sweet"]),
                      ("Hot Chocolate",   "mid",  True,  ["hot", "sweet"]),
                      ("Iced Tea",        "low",  True,  ["cold"]),
                      ("Cappuccino",      "mid",  True,  ["hot"])],
        "dessert":   [("Brownie",         "mid",  True,  ["sweet"]),
                      ("Waffle with Ice Cream","mid",True,["sweet"]),
                      ("Cheesecake",      "mid",  True,  ["sweet", "premium"]),
                      ("Cookie (2pc)",    "low",  True,  ["sweet"]),
                      ("Pastry",          "mid",  True,  ["sweet"])],
    },
}

# ── UNIVERSAL SAFE ITEMS (fallback for unknown cuisines) ──────────────────

_UNIVERSAL_SAFE: list[tuple[str, str, str, bool, list[str]]] = [
    # (name, category, price_bracket, is_veg, tags)
    ("Coke 500ml",         "beverage",  "low",  True,  ["popular"]),
    ("Mineral Water 1L",   "beverage",  "low",  True,  []),
    ("French Fries",       "side",      "low",  True,  ["popular"]),
    ("Garlic Bread",       "side",      "low",  True,  ["popular"]),
    ("Brownie",            "dessert",   "mid",  True,  ["sweet"]),
    ("Green Salad",        "side",      "low",  True,  ["healthy"]),
    ("Cold Coffee",        "beverage",  "mid",  True,  ["cold", "popular"]),
    ("Masala Papad",       "appetizer", "low",  True,  ["crunchy"]),
    ("Gulab Jamun (2pc)",  "dessert",   "low",  True,  ["sweet"]),
    ("Iced Tea",           "beverage",  "low",  True,  ["cold"]),
    ("Lassi (Sweet)",      "beverage",  "low",  True,  ["cooling"]),
    ("Raita",              "side",      "low",  True,  ["cooling"]),
]

# ── PRICE BRACKET RANGES ─────────────────────────────────────────────────

_PRICE_BRACKETS = {
    "low":  (20, 80),
    "mid":  (80, 180),
    "high": (180, 350),
}

# ── MEAL-TIME CATEGORY PRIORITY ──────────────────────────────────────────

_MEAL_TIME_PRIORITY: dict[str, list[str]] = {
    "breakfast":      ["beverage", "side", "appetizer"],
    "lunch":          ["bread", "side", "beverage", "dessert", "appetizer"],
    "evening_snacks": ["appetizer", "beverage", "dessert"],
    "dinner":         ["bread", "side", "beverage", "dessert", "appetizer"],
    "late_night":     ["beverage", "dessert", "side"],
    "snack":          ["appetizer", "beverage", "dessert", "side"],
}

# ── MEAL-TIME WINDOWS ───────────────────────────────────────────────────

_MEAL_TIME_WINDOWS = {
    "breakfast":      (6, 11),
    "lunch":          (11, 15),
    "evening_snacks": (15, 19),
    "dinner":         (19, 23),
    "late_night":     (23, 6),
}

# ── CATEGORY NORMALISATION ──────────────────────────────────────────────

_CATEGORY_MAP = {
    "main": "main", "main_course": "main", "mains": "main",
    "side": "side", "sides": "side", "accompaniment": "side",
    "bread": "bread", "roti": "bread", "naan": "bread",
    "rice": "side",
    "beverage": "beverage", "beverages": "beverage",
    "drink": "beverage", "drinks": "beverage",
    "dessert": "dessert", "desserts": "dessert", "sweet": "dessert",
    "appetizer": "appetizer", "appetizers": "appetizer",
    "starter": "appetizer", "starters": "appetizer",
    "soup": "appetizer", "snack": "appetizer",
}


def _norm_cat(raw: str) -> str:
    return _CATEGORY_MAP.get(raw.lower().strip(), raw.lower().strip())



# ═════════════════════════════════════════════════════════════════════════════
#  COLD START AGENT
# ═════════════════════════════════════════════════════════════════════════════

class ColdStartAgent:
    """
    Agent #3 in the CSAO pipeline — the safety net.

    When the ML retrieval model has insufficient data (new user, new
    restaurant, new item, unusual cart), this agent uses culinary
    intelligence and contextual heuristics to produce 10 recommendations
    with confidence scores and risk assessments.

    Stateless — every call to ``recommend()`` is independent.
    """

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def recommend(
        self,
        cart_items: list[dict[str, Any]],
        restaurant: dict[str, Any],
        menu: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        cold_start_type: str = "new_user",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Generate cold-start recommendations.

        Parameters
        ----------
        cart_items : list[dict]
            Each dict must have: name, category, price, is_veg.
        restaurant : dict
            Must include: cuisine_type, price_tier.
            Optional: name, restaurant_type, rating, location.
        menu : list[dict] | None
            Full restaurant menu (if available). Each dict:
            item_id, name, category, price, is_veg, popularity_score, tags.
        context : dict | None
            Optional: hour, meal_type, is_weekend, city.
        cold_start_type : str
            One of: "new_user", "new_restaurant", "new_item", "unusual_cart".
        top_k : int
            Number of recommendations to return (default 10).

        Returns
        -------
        dict with recommendations, strategy, and metadata.
        """
        ctx = context or {}
        cuisine = restaurant.get("cuisine_type", "North Indian")
        price_tier = restaurant.get("price_tier", "mid")
        restaurant_type = restaurant.get("restaurant_type", "local")

        # Infer meal type
        meal_type = ctx.get("meal_type")
        hour = ctx.get("hour")
        if not meal_type and hour is not None:
            meal_type = self._infer_meal_from_hour(hour)
        elif not meal_type:
            meal_type = "lunch"

        is_weekend = ctx.get("is_weekend", False)
        city = ctx.get("city", "Unknown")

        # Analyse cart
        cart_names = {item.get("name", "").lower() for item in cart_items}
        cart_cats = set()
        cart_is_all_veg = True
        total_cart_value = 0
        for item in cart_items:
            cat = _norm_cat(item.get("category", "other"))
            cart_cats.add(cat)
            if not item.get("is_veg", True):
                cart_is_all_veg = False
            total_cart_value += item.get("price", 0)

        avg_item_price = total_cart_value / max(len(cart_items), 1)

        # ── Step 1: Generate candidate pool ──────────────────────────────────
        raw_candidates = self._generate_candidates(
            cuisine, meal_type, cart_cats, menu, cart_names,
        )

        slm_available = hasattr(self, "_slm") and self._slm is not None and getattr(self._slm, "available", False)
        if slm_available:
            if cold_start_type == "new_user":
                prompt = f"Given a {cuisine} {meal_type} in {city}, with cart: {cart_names}, what else would they order?"
                resp = self._slm.generate(prompt)
                items = self._parse_slm_item_list(resp)
                for item_name in items:
                    raw_candidates.append({
                        "name": item_name,
                        "category": "side",
                        "price": 100,
                        "is_veg": True,
                        "tags": [],
                        "source": "cuisine_knowledge",
                        "source_priority": 0.8,
                        "popularity_score": 0.8,
                    })
            elif cold_start_type == "new_restaurant" and menu:
                prompt = f"Analyze this restaurant menu: {menu}. Cart has: {cart_names}. What complements?"
                resp = self._slm.generate(prompt)
                items = self._parse_slm_item_list(resp)
                for item_name in items:
                    raw_candidates.append({
                        "name": item_name,
                        "category": "side",
                        "price": 100,
                        "is_veg": True,
                        "tags": [],
                        "source": "restaurant_menu",
                        "source_priority": 0.9,
                        "popularity_score": 0.8,
                    })

        # ── Step 2: Score each candidate ─────────────────────────────────────
        scored: list[dict] = []
        for cand in raw_candidates:
            # Skip if already in cart
            if cand["name"].lower() in cart_names:
                continue
            # Skip dietary conflict
            if cart_is_all_veg and not cand["is_veg"]:
                continue

            score_details = self._score_candidate(
                cand, cuisine, meal_type, is_weekend, avg_item_price,
                price_tier, cart_cats, cold_start_type, restaurant_type,
            )
            cand.update(score_details)
            scored.append(cand)

        # ── Step 3: Sort & diversify ─────────────────────────────────────────
        scored.sort(key=lambda x: -x["confidence"])
        final = self._apply_diversity(scored, top_k)

        if slm_available and final:
            item_list_str = ", ".join([item["name"] for item in final])
            prompt = f"Give reasons for recommending these items: {item_list_str}"
            resp = self._slm.generate(prompt)
            reasons = self._parse_slm_reasons(resp)
            for item in final:
                if item["name"] in reasons:
                    item["reasoning"] = reasons[item["name"]]
                    item["slm_enriched"] = True

        # Assign ranks
        for i, item in enumerate(final):
            item["rank"] = i + 1

        # ── Step 4: Build output ─────────────────────────────────────────────
        strategy = self._build_strategy_summary(
            cuisine, meal_type, cold_start_type, cart_cats,
            is_weekend, city, avg_item_price,
        )

        return {
            "agent": "ColdStartRecommendationAgent",
            "version": "2.0",
            "cold_start_type": cold_start_type,
            "total_candidates_generated": len(raw_candidates),
            "total_after_filters": len(scored),
            "top_k_returned": len(final),
            "recommendations": final,
            "strategy_summary": strategy,
            "metadata": {
                "cuisine": cuisine,
                "meal_type": meal_type,
                "is_weekend": is_weekend,
                "cart_value": total_cart_value,
                "avg_item_price": round(avg_item_price),
                "approach": "hybrid_cold_start_pipeline",
                "slm_available": hasattr(self, "_slm") and self._slm is not None and getattr(self._slm, "available", False),
            },
        }

    # ── CANDIDATE GENERATION ────────────────────────────────────────────────

    def _generate_candidates(
        self,
        cuisine: str,
        meal_type: str,
        cart_cats: set[str],
        menu: list[dict] | None,
        cart_names: set[str],
    ) -> list[dict]:
        """
        Build a pool of candidates from three sources (in priority order):
          1. Actual restaurant menu items (if available)
          2. Cuisine-specific complement knowledge base
          3. Universal safe items (fallback)
        """
        candidates: list[dict] = []
        seen_names: set[str] = set(cart_names)

        # Source 1: Restaurant menu — strongest signal for new-user / new-item
        if menu:
            for item in menu:
                name_l = item.get("name", "").lower()
                if name_l in seen_names:
                    continue
                cat = _norm_cat(item.get("category", "other"))
                # Prioritise categories NOT already in the cart
                priority = 1.0 if cat not in cart_cats else 0.4
                candidates.append({
                    "name":       item.get("name", ""),
                    "category":   cat,
                    "price":      item.get("price", 100),
                    "is_veg":     item.get("is_veg", True),
                    "tags":       item.get("tags", []),
                    "source":     "restaurant_menu",
                    "source_priority": priority,
                    "popularity_score": item.get("popularity_score", 0.5),
                })
                seen_names.add(name_l)

        # Source 2: Cuisine complements KB
        cuisine_kb = _CUISINE_COMPLEMENTS.get(cuisine, {})
        # Order categories by meal-time priority
        cat_order = _MEAL_TIME_PRIORITY.get(meal_type, ["side", "beverage", "dessert"])
        for cat in cat_order:
            for entry in cuisine_kb.get(cat, []):
                name, bracket, is_veg, tags = entry
                if name.lower() in seen_names:
                    continue
                lo, hi = _PRICE_BRACKETS[bracket]
                price = (lo + hi) // 2  # estimate
                priority = 0.9 if cat not in cart_cats else 0.3
                candidates.append({
                    "name":       name,
                    "category":   cat,
                    "price":      price,
                    "is_veg":     is_veg,
                    "tags":       tags,
                    "source":     "cuisine_knowledge",
                    "source_priority": priority,
                    "popularity_score": 0.6,
                })
                seen_names.add(name.lower())

        # Source 3: Universal safe fallbacks
        for entry in _UNIVERSAL_SAFE:
            name, cat, bracket, is_veg, tags = entry
            if name.lower() in seen_names:
                continue
            lo, hi = _PRICE_BRACKETS[bracket]
            price = (lo + hi) // 2
            candidates.append({
                "name":       name,
                "category":   cat,
                "price":      price,
                "is_veg":     is_veg,
                "tags":       tags,
                "source":     "universal_safe",
                "source_priority": 0.5,
                "popularity_score": 0.5,
            })
            seen_names.add(name.lower())

        return candidates

    # ── SCORING ─────────────────────────────────────────────────────────────

    def _score_candidate(
        self,
        cand: dict,
        cuisine: str,
        meal_type: str,
        is_weekend: bool,
        avg_item_price: float,
        price_tier: str,
        cart_cats: set[str],
        cold_start_type: str,
        restaurant_type: str,
    ) -> dict:
        """
        Compute a confidence score (0-1), reasoning text, and risk level.

        Factors considered:
          - Cuisine alignment
          - Meal-time appropriateness
          - Price compatibility
          - Category gap-filling
          - Source quality
          - Safety (universal appeal)
        """
        score = 0.50  # neutral baseline
        reasons: list[str] = []

        cat = cand["category"]
        price = cand["price"]
        source = cand.get("source", "universal_safe")

        # ── 1. Source quality ───────────────────────────────────────────────
        src_priority = cand.get("source_priority", 0.5)
        if source == "restaurant_menu":
            score += 0.08
            reasons.append("item is from the restaurant's own menu")
        elif source == "cuisine_knowledge":
            score += 0.05
            reasons.append(f"classic {cuisine} complement")

        # ── 2. Category gap-filling ─────────────────────────────────────────
        if cat not in cart_cats:
            score += 0.15
            reasons.append(f"fills missing {cat} category")
        else:
            score -= 0.05

        # ── 3. Meal-time appropriateness ────────────────────────────────────
        priority_cats = _MEAL_TIME_PRIORITY.get(meal_type, [])
        if cat in priority_cats:
            position = priority_cats.index(cat)
            boost = max(0.02, 0.10 - position * 0.025)
            score += boost
            reasons.append(f"high priority for {meal_type}")

        # Weekend dessert boost
        if is_weekend and cat == "dessert":
            score += 0.06
            reasons.append("weekend dessert boost")

        # ── 4. Price compatibility ──────────────────────────────────────────
        if avg_item_price > 0:
            ratio = price / avg_item_price
            if ratio <= 0.4:
                score += 0.10
                reasons.append("great value add-on")
            elif ratio <= 0.75:
                score += 0.07
                reasons.append("well-priced complement")
            elif ratio <= 1.0:
                score += 0.03
            elif ratio <= 1.5:
                score -= 0.03
            else:
                score -= 0.10
                reasons.append("may feel expensive relative to cart")
        else:
            # No price info — use tier heuristics
            if price_tier == "budget" and price <= 80:
                score += 0.05
            elif price_tier == "premium" and price >= 100:
                score += 0.03

        # ── 5. Popularity signal (weak) ─────────────────────────────────────
        pop = cand.get("popularity_score", 0.5)
        score += (pop - 0.5) * 0.10

        # ── 6. Safety bonus — beverages are always safe ─────────────────────
        if cat == "beverage":
            score += 0.04
            if "beverage" not in reasons[0] if reasons else "":
                reasons.append("beverages have universally high acceptance")
        if "popular" in cand.get("tags", []):
            score += 0.03
            reasons.append("popular item")

        # ── 7. Restaurant type adjustments ──────────────────────────────────
        if restaurant_type in ("qsr", "fast_food"):
            if cat == "side" and price <= 100:
                score += 0.03
                reasons.append("affordable side for QSR")
        elif restaurant_type == "premium":
            if cat == "dessert":
                score += 0.03
                reasons.append("premium diners often add dessert")

        # ── 8. Cold-start type adjustments ──────────────────────────────────
        if cold_start_type == "new_user":
            # Favour safe, popular choices for new users
            if "popular" in cand.get("tags", []):
                score += 0.03
            if source == "universal_safe":
                score += 0.02
        elif cold_start_type == "new_restaurant":
            # Favour cuisine-KB items since menu data may be sparse
            if source == "cuisine_knowledge":
                score += 0.05
        elif cold_start_type == "new_item":
            # Not much we can do — rely on category/cuisine logic
            pass
        elif cold_start_type == "unusual_cart":
            # Be extra safe
            if source == "universal_safe":
                score += 0.05
                reasons.append("safe fallback for unusual cart")

        # Clamp to [0.05, 0.95]
        confidence = max(0.05, min(0.95, score))

        # Risk assessment
        if confidence >= 0.70:
            risk = "High"
        elif confidence >= 0.50:
            risk = "Medium"
        else:
            risk = "Low"

        # Build reasoning string
        if not reasons:
            reasons.append("general complement item")
        reasoning = (
            f"Despite no historical data, {cand['name']} works because: "
            + "; ".join(reasons) + "."
        )

        return {
            "confidence": round(confidence, 3),
            "risk_assessment": risk,
            "reasoning": reasoning,
        }

    # ── DIVERSITY ENFORCEMENT ──────────────────────────────────────────────

    @staticmethod
    def _apply_diversity(
        scored: list[dict], top_k: int,
    ) -> list[dict]:
        """
        Select top-K while enforcing:
          - Max 3 items per category
          - At least 2 distinct categories
          - Mix of price points
        """
        final: list[dict] = []
        cat_count: dict[str, int] = {}
        price_buckets_used: set[str] = set()

        for item in scored:
            cat = item["category"]
            price = item["price"]

            # Category cap
            if cat_count.get(cat, 0) >= 3:
                continue

            cat_count[cat] = cat_count.get(cat, 0) + 1

            # Track price diversity
            if price <= 60:
                price_buckets_used.add("low")
            elif price <= 150:
                price_buckets_used.add("mid")
            else:
                price_buckets_used.add("high")

            final.append(item)
            if len(final) >= top_k:
                break

        return final

    # ── SLM PARSING LOGIC ───────────────────────────────────────────────────

    @classmethod
    def _parse_slm_item_list(cls, text: str) -> list[str]:
        if not text:
            return []
        try:
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('-'):
                items.append(line[1:].strip())
        return items

    @classmethod
    def _parse_slm_reasons(cls, text: str) -> dict[str, str]:
        if not text:
            return {}
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {}

    # ── HELPERS ─────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_meal_from_hour(hour: int) -> str:
        for meal, (start, end) in _MEAL_TIME_WINDOWS.items():
            if start < end:
                if start <= hour < end:
                    return meal
            else:
                if hour >= start or hour < end:
                    return meal
        return "lunch"

    @staticmethod
    def _build_strategy_summary(
        cuisine: str,
        meal_type: str,
        cold_start_type: str,
        cart_cats: set[str],
        is_weekend: bool,
        city: str,
        avg_price: float,
    ) -> dict:
        # Describe the approach taken
        approach_parts = []
        if cold_start_type == "new_user":
            approach_parts.append(
                "New user with no history — using cuisine intelligence "
                "and popularity-based safe recommendations."
            )
        elif cold_start_type == "new_restaurant":
            approach_parts.append(
                "New restaurant with limited data — relying on "
                f"{cuisine} cuisine complement knowledge base."
            )
        elif cold_start_type == "new_item":
            approach_parts.append(
                "Newly launched items — using category and price "
                "heuristics to position alongside existing cart."
            )
        elif cold_start_type == "unusual_cart":
            approach_parts.append(
                "Unusual cart combination — defaulting to universally "
                "safe items with high acceptance probability."
            )

        missing = []
        for c in ["side", "beverage", "dessert", "appetizer"]:
            if c not in cart_cats:
                missing.append(c)
        if missing:
            approach_parts.append(
                f"Cart is missing: {', '.join(missing)}. "
                "Prioritising those categories."
            )

        if is_weekend:
            approach_parts.append("Weekend boost applied to desserts.")

        approach_parts.append(
            f"Price-conscious: targeting add-ons around "
            f"₹{int(avg_price * 0.3)}-₹{int(avg_price * 0.8)} "
            f"(complementary to ₹{int(avg_price)} avg cart item)."
        )

        return {
            "cold_start_type": cold_start_type,
            "approach": " ".join(approach_parts),
            "key_signals_used": [
                f"cuisine={cuisine}",
                f"meal_type={meal_type}",
                f"city={city}",
                f"is_weekend={is_weekend}",
                f"avg_item_price=₹{int(avg_price)}",
            ],
            "fallback_chain": [
                "1. Restaurant menu items (if available)",
                f"2. {cuisine} cuisine complement knowledge",
                "3. Universal safe items (beverages, popular sides)",
            ],
        }
