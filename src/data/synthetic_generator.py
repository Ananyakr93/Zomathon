"""
synthetic_generator.py
======================
LLM-augmented synthetic data generation for Cart Super Add-On training.

Generates 1,000 hyper-realistic cart scenarios with:
  - User personas (age, location, income, dietary preferences)
  - Restaurant profiles (cuisine, price range, ratings, city)
  - Cart items (2-4 items with prices)
  - Meal context (time, day, occasion)
  - Recommended add-ons (with reasoning)
  - Anti-recommendations (with reasoning)
  - Meal completion score

Pipeline
--------
1. Load expanded cuisine ontology (8 cuisines, 100+ dishes, 100+ add-ons).
2. Generate diverse user personas with realistic Indian demographics.
3. Build cart sessions using meal-pattern templates per cuisine.
4. Compute meal-completion scores and generate recommendations.
5. Apply LLM-style co-purchase reasoning for each recommendation.
6. Create anti-recommendations with exclusion reasoning.
7. Inject edge cases: cold-start, budget constraints, incomplete meals.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Indian demographics for realistic personas ───────────────

CITIES = [
    {"name": "Mumbai", "localities": ["Andheri", "Bandra", "Powai", "Juhu", "Lower Parel", "Malad", "Goregaon", "Dadar"]},
    {"name": "Delhi", "localities": ["Connaught Place", "Hauz Khas", "Saket", "Rajouri Garden", "Dwarka", "Lajpat Nagar", "Karol Bagh", "Rohini"]},
    {"name": "Bangalore", "localities": ["Koramangala", "Indiranagar", "HSR Layout", "Whitefield", "JP Nagar", "MG Road", "Electronic City", "Marathahalli"]},
    {"name": "Hyderabad", "localities": ["Banjara Hills", "Jubilee Hills", "Madhapur", "Gachibowli", "Kukatpally", "Secunderabad", "Ameerpet", "Kondapur"]},
    {"name": "Chennai", "localities": ["T. Nagar", "Anna Nagar", "Velachery", "Adyar", "Mylapore", "Nungambakkam", "OMR", "Tambaram"]},
    {"name": "Kolkata", "localities": ["Park Street", "Salt Lake", "New Town", "Gariahat", "Ballygunge", "Howrah", "Dumdum", "Jadavpur"]},
    {"name": "Pune", "localities": ["Koregaon Park", "Viman Nagar", "Hinjewadi", "Kothrud", "Baner", "Wakad", "Aundh", "Hadapsar"]},
    {"name": "Gurgaon", "localities": ["DLF Phase 1", "Sohna Road", "Golf Course Road", "Sector 29", "MG Road", "Sector 56", "Cyber Hub", "Udyog Vihar"]},
    {"name": "Lucknow", "localities": ["Hazratganj", "Gomti Nagar", "Aminabad", "Aliganj", "Indira Nagar", "Mahanagar"]},
    {"name": "Jaipur", "localities": ["MI Road", "Vaishali Nagar", "Malviya Nagar", "C-Scheme", "Tonk Road", "Mansarovar"]},
]

INCOME_BRACKETS = [
    {"label": "student", "range": [0, 15000], "budget_per_order": [100, 300], "weight": 0.15},
    {"label": "entry_level", "range": [15000, 35000], "budget_per_order": [200, 500], "weight": 0.25},
    {"label": "mid_level", "range": [35000, 75000], "budget_per_order": [300, 800], "weight": 0.30},
    {"label": "senior", "range": [75000, 150000], "budget_per_order": [400, 1500], "weight": 0.20},
    {"label": "premium", "range": [150000, 500000], "budget_per_order": [500, 3000], "weight": 0.10},
]

DIETARY_PREFS = [
    {"label": "vegetarian", "weight": 0.35},
    {"label": "non_vegetarian", "weight": 0.45},
    {"label": "eggetarian", "weight": 0.10},
    {"label": "vegan", "weight": 0.05},
    {"label": "jain", "weight": 0.03},
    {"label": "no_preference", "weight": 0.02},
]

OCCASIONS = [
    {"label": "regular", "weight": 0.55},
    {"label": "office_lunch", "weight": 0.15},
    {"label": "family_dinner", "weight": 0.10},
    {"label": "date_night", "weight": 0.05},
    {"label": "party", "weight": 0.05},
    {"label": "late_night_craving", "weight": 0.05},
    {"label": "weekend_brunch", "weight": 0.03},
    {"label": "post_workout", "weight": 0.02},
]

MEAL_TYPES_BY_HOUR = {
    range(6, 11): "breakfast",
    range(11, 15): "lunch",
    range(15, 18): "snack",
    range(18, 22): "dinner",
    range(22, 24): "late_night",
    range(0, 6): "late_night",
}

# ── Reasoning templates ───────────────────────────────────────

RECOMMEND_REASONS = {
    "bread": [
        "Indian gravy dishes are almost always eaten with bread — {dish} pairs naturally with {addon}",
        "Cart has a curry but no bread — {addon} completes the meal",
        "{dish} without {addon} is an incomplete North Indian dinner",
        "Most customers ordering {dish} add {addon} (85% co-purchase rate)",
    ],
    "side": [
        "{addon} is the classic side pairing for {dish}",
        "{addon} balances the spice level of {dish}",
        "Light side dish to complement the heavy main — {addon}",
        "{addon} is ordered by 70%+ customers with {dish}",
    ],
    "beverage": [
        "No beverage in cart — {addon} is the most popular with {dish}",
        "Meal feels incomplete without a drink — {addon} pairs well",
        "{addon} is a classic Indian meal finisher",
        "Spicy {dish} pairs well with a cooling {addon}",
    ],
    "dessert": [
        "No dessert in cart — {addon} is a popular impulse add for {dish} diners",
        "Sweet ending to the meal — {addon} at just Rs.{price} is a low-commitment add-on",
        "{addon} is the top-ordered dessert at this restaurant",
        "Customers who order {dish} add {addon} 40% of the time",
    ],
    "starter": [
        "Cart has mains but no starter — {addon} is a great appetiser",
        "{addon} pairs well as a sharing starter before {dish}",
        "Popular combo: {addon} + {dish}",
    ],
    "soup": [
        "Chinese meals typically start with soup — {addon} is the classic opener",
        "No soup in cart — {addon} complements {dish} perfectly",
        "Warming {addon} before the main course is a common pattern",
    ],
    "condiment": [
        "{addon} enhances the flavour of {dish}",
        "Small add-on at Rs.{price} — almost always ordered with {dish}",
        "{addon} is included free at dine-in but a small add-on for delivery",
    ],
    "rice": [
        "Gravy dish without rice — {addon} is the natural pairing",
        "{dish} is best enjoyed with {addon}",
    ],
    "topping": [
        "Enhance your {dish} with {addon} for just Rs.{price}",
        "{addon} makes {dish} even better — popular upgrade",
    ],
}

ANTI_RECOMMEND_REASONS = {
    "dietary_conflict": "User is {diet} — {item} contains {conflict}",
    "already_in_cart": "{item} is already in the cart or too similar to {existing}",
    "price_too_high": "{item} at Rs.{price} exceeds the add-on budget for this order",
    "cuisine_mismatch": "{item} is from {wrong_cuisine} — doesn't pair with {cart_cuisine}",
    "category_saturation": "Cart already has {count} {category} items — adding more would be redundant",
    "time_mismatch": "{item} is typically a {meal_type} item — not suitable for {current_meal}",
    "portion_mismatch": "{item} is too large for the current order size",
    "duplicate_category": "Already have {existing} in {category} — {item} would be redundant",
}


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_ontology(path: Path | None = None) -> dict[str, Any]:
    """Load the expanded cuisine ontology."""
    path = path or RAW_DIR / "cuisine_ontology.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# User persona generation
# ═══════════════════════════════════════════════════════════════

def _weighted_choice(items: list[dict], key: str = "weight") -> dict:
    weights = [it[key] for it in items]
    return random.choices(items, weights=weights, k=1)[0]


def generate_user_persona(rng: random.Random, is_new_user: bool = False) -> dict[str, Any]:
    """Generate a realistic Indian user persona."""
    city_info = rng.choice(CITIES)
    income = _weighted_choice(INCOME_BRACKETS)
    diet = _weighted_choice(DIETARY_PREFS)
    age = rng.randint(18, 55)

    # age-income correlation
    if age < 22:
        income = INCOME_BRACKETS[0]  # student
    elif age > 45 and rng.random() < 0.4:
        income = rng.choice(INCOME_BRACKETS[3:])  # senior / premium

    return {
        "user_id": f"U{rng.randint(10000, 99999)}",
        "age": age,
        "gender": rng.choice(["male", "female", "other"]),
        "city": city_info["name"],
        "locality": rng.choice(city_info["localities"]),
        "income_bracket": income["label"],
        "monthly_income_range": income["range"],
        "budget_per_order": income["budget_per_order"],
        "dietary_preference": diet["label"],
        "is_new_user": is_new_user,
        "past_order_count": 0 if is_new_user else rng.randint(1, 200),
        "avg_rating_given": round(rng.uniform(3.0, 5.0), 1) if not is_new_user else None,
        "preferred_cuisines": [],  # filled later based on cart
    }


# ═══════════════════════════════════════════════════════════════
# Restaurant selection
# ═══════════════════════════════════════════════════════════════

def select_restaurant(
    ontology: dict, cuisine_key: str, rng: random.Random, is_new_restaurant: bool = False
) -> dict[str, Any]:
    """Pick a restaurant from the ontology (or generate a new one for cold-start)."""
    cuisine_data = ontology[cuisine_key]
    restaurants = cuisine_data.get("restaurants", [])

    if is_new_restaurant or not restaurants:
        return {
            "restaurant_id": f"R{rng.randint(10000, 99999)}",
            "name": f"New {cuisine_key} Kitchen",
            "cuisine_type": cuisine_key,
            "price_tier": rng.choice(["budget", "mid", "premium"]),
            "rating": round(rng.uniform(3.0, 4.0), 1),
            "city": rng.choice(CITIES)["name"],
            "is_new_restaurant": True,
            "total_orders": rng.randint(0, 50),
            "delivery_time_mins": rng.randint(25, 55),
        }

    rest = rng.choice(restaurants)
    return {
        "restaurant_id": f"R{rng.randint(10000, 99999)}",
        "name": rest["name"],
        "cuisine_type": cuisine_key,
        "price_tier": rest["price_tier"],
        "rating": rest["rating"],
        "city": rest["city"],
        "is_new_restaurant": False,
        "total_orders": rng.randint(500, 50000),
        "delivery_time_mins": rng.randint(20, 45),
    }


# ═══════════════════════════════════════════════════════════════
# Cart construction
# ═══════════════════════════════════════════════════════════════

def _price_in_range(price_range: list, tier: str, rng: random.Random) -> int:
    """Generate a price within range, adjusted by restaurant tier."""
    lo, hi = price_range
    if tier == "budget":
        return rng.randint(lo, lo + (hi - lo) // 3)
    elif tier == "premium":
        return rng.randint(lo + 2 * (hi - lo) // 3, hi)
    else:
        return rng.randint(lo, hi)


def build_cart(
    ontology: dict,
    cuisine_key: str,
    restaurant: dict,
    user: dict,
    meal_type: str,
    rng: random.Random,
    force_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """
    Build a realistic cart of 1-4 items from the cuisine's dish list.

    Respects:
    - Dietary preference (veg users don't get non-veg items)
    - Budget constraints
    - Meal patterns (breakfast vs dinner vs snack)
    - Incomplete carts (for edge cases)
    """
    cuisine_data = ontology[cuisine_key]
    dishes = cuisine_data.get("dishes", {})
    tier = restaurant["price_tier"]
    diet = user["dietary_preference"]
    budget = rng.randint(*user["budget_per_order"])

    # Filter dishes by dietary preference
    eligible_dishes = {}
    for name, info in dishes.items():
        is_veg = info.get("is_veg", True)
        if diet in ("vegetarian", "vegan", "jain") and not is_veg:
            continue
        if diet == "eggetarian" and not is_veg and "egg" not in info.get("tags", []):
            continue
        eligible_dishes[name] = info

    if not eligible_dishes:
        # fallback: use all veg dishes
        eligible_dishes = {n: i for n, i in dishes.items() if i.get("is_veg", True)}

    if not eligible_dishes:
        eligible_dishes = dishes  # absolute fallback

    dish_names = list(eligible_dishes.keys())

    # Decide cart size (bias toward 2-3 for realistic meal orders)
    if force_incomplete:
        n_items = 1
    else:
        n_items = rng.choices([1, 2, 3, 4], weights=[0.05, 0.30, 0.40, 0.25], k=1)[0]

    n_items = min(n_items, len(dish_names))

    # Select dishes (biased toward mains)
    selected_names = []
    remaining = list(dish_names)
    rng.shuffle(remaining)

    # Always pick at least one main if available
    mains = [d for d in remaining if eligible_dishes[d].get("category") == "main"]
    if mains:
        selected_names.append(mains[0])
        remaining.remove(mains[0])

    # Fill remaining slots
    while len(selected_names) < n_items and remaining:
        selected_names.append(remaining.pop(0))

    # Build cart items
    cart = []
    total_price = 0
    for name in selected_names:
        info = eligible_dishes[name]
        price = _price_in_range(info["price_range"], tier, rng)

        # Budget check — only skip if price is > 2x remaining budget
        # (people stretch budgets slightly for food delivery)
        remaining_budget = budget - total_price
        if remaining_budget < 0 and cart:
            break
        if price > remaining_budget * 2 and cart:
            break

        item = {
            "item_id": f"I{rng.randint(10000, 99999)}",
            "name": name,
            "category": info.get("category", "main"),
            "price": price,
            "qty": rng.choices([1, 2], weights=[0.85, 0.15], k=1)[0],
            "is_veg": info.get("is_veg", True),
            "tags": info.get("tags", []),
        }
        cart.append(item)
        total_price += price * item["qty"]

    return cart


# ═══════════════════════════════════════════════════════════════
# Meal context generation
# ═══════════════════════════════════════════════════════════════

def generate_context(rng: random.Random, meal_type_override: str | None = None) -> dict[str, Any]:
    """Generate meal context (time, day, occasion)."""
    # Generate a realistic timestamp
    base_date = datetime(2026, 2, 1)
    day_offset = rng.randint(0, 27)
    date = base_date + timedelta(days=day_offset)

    if meal_type_override:
        meal_type = meal_type_override
        hour_ranges = {
            "breakfast": (7, 10), "lunch": (11, 14), "snack": (15, 17),
            "dinner": (18, 21), "late_night": (22, 23),
        }
        h_lo, h_hi = hour_ranges.get(meal_type, (11, 21))
        hour = rng.randint(h_lo, h_hi)
    else:
        hour = rng.choices(
            list(range(7, 24)),
            weights=[2, 5, 8, 3,  # 7-10 breakfast
                     8, 10, 8, 3,  # 11-14 lunch
                     4, 5, 6,      # 15-17 snack
                     8, 10, 8, 5,  # 18-21 dinner
                     3, 2],        # 22-23 late night
            k=1
        )[0]
        for hr_range, mt in MEAL_TYPES_BY_HOUR.items():
            if hour in hr_range:
                meal_type = mt
                break
        else:
            meal_type = "dinner"

    minute = rng.randint(0, 59)
    is_weekend = date.weekday() >= 5
    occasion = _weighted_choice(OCCASIONS)["label"]

    # Adjust occasion based on context
    if not is_weekend and hour >= 11 and hour <= 14:
        if rng.random() < 0.3:
            occasion = "office_lunch"
    if is_weekend and hour >= 9 and hour <= 12:
        if rng.random() < 0.2:
            occasion = "weekend_brunch"
    if hour >= 22:
        if rng.random() < 0.5:
            occasion = "late_night_craving"

    return {
        "timestamp": f"{date.strftime('%Y-%m-%d')}T{hour:02d}:{minute:02d}:00",
        "hour": hour,
        "day_of_week": date.strftime("%A"),
        "is_weekend": is_weekend,
        "meal_type": meal_type,
        "occasion": occasion,
        "weather": rng.choices(
            ["clear", "cloudy", "rainy", "hot", "cold"],
            weights=[0.35, 0.25, 0.15, 0.15, 0.10],
            k=1
        )[0],
        "is_festival_season": rng.random() < 0.08,
    }


# ═══════════════════════════════════════════════════════════════
# Recommendation generation
# ═══════════════════════════════════════════════════════════════

def _get_categories_in_cart(cart: list[dict]) -> set[str]:
    return {item.get("category", "main") for item in cart}


def _get_cart_names(cart: list[dict]) -> set[str]:
    return {item["name"] for item in cart}


def generate_recommendations(
    ontology: dict,
    cuisine_key: str,
    cart: list[dict],
    user: dict,
    restaurant: dict,
    context: dict,
    rng: random.Random,
) -> tuple[list[dict], list[dict], float]:
    """
    Generate recommended add-ons and anti-recommendations.

    Returns
    -------
    (recommended_items, anti_recommendations, meal_completion_score)
    """
    cuisine_data = ontology[cuisine_key]
    addons = cuisine_data.get("addons", {})
    diet = user["dietary_preference"]
    cart_categories = _get_categories_in_cart(cart)
    cart_names = _get_cart_names(cart)
    budget_remaining = rng.randint(*user["budget_per_order"]) - sum(
        i["price"] * i["qty"] for i in cart
    )

    # Get meal pattern for this cuisine + context
    patterns = cuisine_data.get("meal_patterns", {})
    pattern_key = list(patterns.keys())[0] if patterns else None
    ideal_categories = set()
    if pattern_key:
        pattern = patterns[pattern_key]
        ideal_categories = set(pattern["sequence"])

    # ── Positive recommendations ──────────────────────────────
    recommended = []
    for addon_name, addon_info in addons.items():
        if addon_name in cart_names:
            continue

        # Dietary filter
        addon_is_veg = addon_info.get("is_veg", True)
        if diet in ("vegetarian", "vegan", "jain") and not addon_is_veg:
            continue

        addon_category = addon_info.get("category", "side")
        addon_price = addon_info.get("price", 50)

        # Check affinity
        affinity = addon_info.get("affinity", [])
        has_affinity = "*" in affinity or any(n in affinity for n in cart_names)

        # Score the addon
        score = 0.0

        # Affinity bonus
        if has_affinity:
            if "*" in affinity:
                score += 0.3
            else:
                score += 0.6  # specific affinity is stronger

        # Missing category bonus (meal completion)
        if addon_category not in cart_categories and addon_category in ideal_categories:
            score += 0.4

        # Price reasonableness (impulse buy zone)
        if addon_price <= 100:
            score += 0.2
        elif addon_price <= 50:
            score += 0.3

        # Budget check
        if addon_price > budget_remaining:
            score *= 0.2  # heavily penalise over-budget

        # Time-of-day adjustments
        if context["meal_type"] == "breakfast" and addon_category == "beverage":
            score += 0.15  # beverages popular at breakfast
        if context["meal_type"] == "dinner" and addon_category == "dessert":
            score += 0.1

        # Noise
        score += rng.uniform(-0.05, 0.1)
        score = max(0, min(1, score))

        if score > 0.2:
            # Generate reasoning
            main_dish = cart[0]["name"] if cart else "the order"
            reason_templates = RECOMMEND_REASONS.get(addon_category, [
                "{addon} complements {dish} well"
            ])
            reason = rng.choice(reason_templates).format(
                dish=main_dish, addon=addon_name, price=addon_price
            )

            recommended.append({
                "item_id": f"A{rng.randint(10000, 99999)}",
                "name": addon_name,
                "category": addon_category,
                "price": addon_price,
                "is_veg": addon_is_veg,
                "relevance_score": round(score, 3),
                "reasoning": reason,
            })

    # Sort by score, take top 5
    recommended.sort(key=lambda x: x["relevance_score"], reverse=True)
    recommended = recommended[:5]

    # ── Anti-recommendations ─────────────────────────────────
    anti_recs = []

    # 1. Dietary conflicts
    for addon_name, addon_info in addons.items():
        if not addon_info.get("is_veg", True) and diet in ("vegetarian", "vegan", "jain"):
            anti_recs.append({
                "item_id": f"X{rng.randint(10000, 99999)}",
                "name": addon_name,
                "category": addon_info.get("category", "unknown"),
                "price": addon_info.get("price", 0),
                "reasoning": ANTI_RECOMMEND_REASONS["dietary_conflict"].format(
                    diet=diet, item=addon_name,
                    conflict="meat/egg" if not addon_info.get("is_veg") else "animal products"
                ),
            })
            if len(anti_recs) >= 2:
                break

    # 2. Already in cart / similar
    for addon_name in list(cart_names)[:2]:
        anti_recs.append({
            "item_id": f"X{rng.randint(10000, 99999)}",
            "name": addon_name,
            "category": "duplicate",
            "price": 0,
            "reasoning": ANTI_RECOMMEND_REASONS["already_in_cart"].format(
                item=addon_name, existing=addon_name
            ),
        })

    # 3. Cross-cuisine mismatch — pick from a different cuisine
    other_cuisines = [c for c in ontology.keys() if c != cuisine_key]
    if other_cuisines:
        other_cuisine = rng.choice(other_cuisines)
        other_addons = ontology[other_cuisine].get("addons", {})
        if other_addons:
            mismatch_name = rng.choice(list(other_addons.keys()))
            mismatch_info = other_addons[mismatch_name]
            anti_recs.append({
                "item_id": f"X{rng.randint(10000, 99999)}",
                "name": mismatch_name,
                "category": mismatch_info.get("category", "unknown"),
                "price": mismatch_info.get("price", 0),
                "reasoning": ANTI_RECOMMEND_REASONS["cuisine_mismatch"].format(
                    item=mismatch_name, wrong_cuisine=other_cuisine,
                    cart_cuisine=cuisine_key
                ),
            })

    # 4. Price too high
    expensive_addons = [
        (n, a) for n, a in addons.items()
        if a.get("price", 0) > budget_remaining and n not in cart_names
    ]
    if expensive_addons:
        exp_name, exp_info = rng.choice(expensive_addons)
        anti_recs.append({
            "item_id": f"X{rng.randint(10000, 99999)}",
            "name": exp_name,
            "category": exp_info.get("category", "unknown"),
            "price": exp_info.get("price", 0),
            "reasoning": ANTI_RECOMMEND_REASONS["price_too_high"].format(
                item=exp_name, price=exp_info.get("price", 0)
            ),
        })

    anti_recs = anti_recs[:5]

    # ── Meal completion score ────────────────────────────────
    if ideal_categories:
        covered = cart_categories.intersection(ideal_categories)
        # Add categories from recommended items (simulating what WOULD complete the meal)
        completion_score = len(covered) / len(ideal_categories)
    else:
        # Heuristic: at least main + beverage + side = complete
        basic_categories = {"main", "beverage", "side"}
        covered = cart_categories.intersection(basic_categories)
        completion_score = len(covered) / len(basic_categories)

    completion_score = round(min(1.0, completion_score), 2)

    return recommended, anti_recs, completion_score


# ═══════════════════════════════════════════════════════════════
# Cold-start scenario injection
# ═══════════════════════════════════════════════════════════════

def inject_cold_start_flags(
    scenario: dict, rng: random.Random, scenario_idx: int
) -> dict:
    """
    Mark cold-start scenarios:
    - New users (no history)        — ~15% of scenarios
    - New restaurants (limited data) — ~10% of scenarios
    - Newly launched items           — ~8% of scenarios
    """
    # New user
    if scenario_idx % 7 == 0 or rng.random() < 0.15:
        scenario["user_profile"]["is_new_user"] = True
        scenario["user_profile"]["past_order_count"] = 0
        scenario["user_profile"]["avg_rating_given"] = None
        scenario["cold_start_type"] = scenario.get("cold_start_type", [])
        scenario["cold_start_type"].append("new_user")

    # New restaurant
    if scenario_idx % 10 == 0 or rng.random() < 0.10:
        scenario["restaurant_profile"]["is_new_restaurant"] = True
        scenario["restaurant_profile"]["total_orders"] = rng.randint(0, 50)
        scenario["cold_start_type"] = scenario.get("cold_start_type", [])
        scenario["cold_start_type"].append("new_restaurant")

    # Newly launched items
    if scenario_idx % 12 == 0 or rng.random() < 0.08:
        if scenario["cart_items"]:
            new_item_idx = rng.randint(0, len(scenario["cart_items"]) - 1)
            scenario["cart_items"][new_item_idx]["is_new_item"] = True
            scenario["cart_items"][new_item_idx]["days_since_launch"] = rng.randint(0, 14)
            scenario["cold_start_type"] = scenario.get("cold_start_type", [])
            scenario["cold_start_type"].append("new_item")

    if "cold_start_type" not in scenario:
        scenario["cold_start_type"] = []

    return scenario


# ═══════════════════════════════════════════════════════════════
# Edge-case injection
# ═══════════════════════════════════════════════════════════════

def inject_edge_cases(
    scenario: dict, rng: random.Random, scenario_idx: int
) -> dict:
    """
    Add edge-case flags and adjustments.
    ~20% of scenarios get an edge-case treatment.
    """
    edge_cases = []

    # Budget-constrained order (~8% of scenarios)
    if scenario_idx % 12 == 0:
        user = scenario["user_profile"]
        user["budget_per_order"] = [100, 250]
        user["income_bracket"] = "student"
        edge_cases.append("tight_budget")

    # Single-item cart (incomplete meal, ~5% of scenarios)
    if scenario_idx % 20 == 0 and len(scenario["cart_items"]) > 1:
        scenario["cart_items"] = [scenario["cart_items"][0]]
        edge_cases.append("single_item_cart")

    # Late-night order
    if scenario_idx % 9 == 0:
        scenario["context"]["hour"] = rng.randint(22, 23)
        scenario["context"]["meal_type"] = "late_night"
        scenario["context"]["occasion"] = "late_night_craving"
        edge_cases.append("late_night_order")

    # Very large order (party)
    if scenario_idx % 25 == 0:
        scenario["context"]["occasion"] = "party"
        # Duplicate some items with higher qty
        for item in scenario["cart_items"]:
            item["qty"] = rng.randint(2, 4)
        edge_cases.append("party_order")

    # Repeat customer (high loyalty)
    if scenario_idx % 15 == 0 and not scenario["user_profile"].get("is_new_user"):
        scenario["user_profile"]["past_order_count"] = rng.randint(50, 200)
        scenario["user_profile"]["is_loyal_customer"] = True
        edge_cases.append("loyal_customer")

    scenario["edge_cases"] = edge_cases
    return scenario


# ═══════════════════════════════════════════════════════════════
# Main generation pipeline
# ═══════════════════════════════════════════════════════════════

def generate_scenarios(
    n_scenarios: int = 1000,
    seed: int = 42,
    ontology_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Generate n_scenarios hyper-realistic cart scenarios.

    Returns
    -------
    list[dict]  — each scenario has:
        scenario_id, user_profile, restaurant_profile, cart_items,
        context, recommended_items, anti_recommendations,
        meal_completion_score, cold_start_type, edge_cases
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    ontology = load_ontology(ontology_path)
    cuisine_keys = list(ontology.keys())

    # Cuisine distribution (weighted toward popular cuisines)
    cuisine_weights = {
        "North Indian": 0.25,
        "South Indian": 0.12,
        "Chinese / Indo-Chinese": 0.18,
        "Italian / Pizza": 0.18,
        "Biryani Specialist": 0.10,
        "Burger / Fast Food": 0.10,
        "Street Food / Chaat": 0.04,
        "Desserts / Bakery": 0.03,
    }

    # Normalise weights for available cuisines
    available_weights = {k: v for k, v in cuisine_weights.items() if k in cuisine_keys}
    if not available_weights:
        available_weights = {k: 1.0 / len(cuisine_keys) for k in cuisine_keys}
    total_w = sum(available_weights.values())
    available_weights = {k: v / total_w for k, v in available_weights.items()}

    scenarios = []

    for idx in range(n_scenarios):
        # Pick cuisine
        cuisine_key = rng.choices(
            list(available_weights.keys()),
            weights=list(available_weights.values()),
            k=1,
        )[0]

        # Cold-start flags
        is_new_user = (idx % 7 == 0)
        is_new_restaurant = (idx % 10 == 0)
        force_incomplete = (idx % 8 == 0)

        # Determine meal type based on cuisine
        meal_type = None
        if cuisine_key == "South Indian" and rng.random() < 0.5:
            meal_type = "breakfast"
        elif cuisine_key == "Street Food / Chaat":
            meal_type = "snack"
        elif cuisine_key == "Desserts / Bakery":
            meal_type = rng.choice(["snack", "dinner"])

        # Generate components
        user = generate_user_persona(rng, is_new_user=is_new_user)
        restaurant = select_restaurant(ontology, cuisine_key, rng, is_new_restaurant)
        context = generate_context(rng, meal_type_override=meal_type)
        cart = build_cart(
            ontology, cuisine_key, restaurant, user,
            context["meal_type"], rng,
            force_incomplete=force_incomplete,
        )

        if not cart:
            # Ensure at least one item
            cuisine_data = ontology[cuisine_key]
            dishes = cuisine_data.get("dishes", {})
            if dishes:
                fallback_name = rng.choice(list(dishes.keys()))
                fallback_info = dishes[fallback_name]
                cart = [{
                    "item_id": f"I{rng.randint(10000, 99999)}",
                    "name": fallback_name,
                    "category": fallback_info.get("category", "main"),
                    "price": rng.randint(*fallback_info["price_range"]),
                    "qty": 1,
                    "is_veg": fallback_info.get("is_veg", True),
                    "tags": fallback_info.get("tags", []),
                }]

        user["preferred_cuisines"] = [cuisine_key]

        # Generate recommendations
        recommended, anti_recs, completion_score = generate_recommendations(
            ontology, cuisine_key, cart, user, restaurant, context, rng
        )

        scenario = {
            "scenario_id": f"SC{idx + 1:04d}",
            "user_profile": user,
            "restaurant_profile": restaurant,
            "cart_items": cart,
            "context": context,
            "recommended_items": recommended,
            "anti_recommendations": anti_recs,
            "meal_completion_score": completion_score,
            "cuisine": cuisine_key,
        }

        # Inject cold-start and edge cases
        scenario = inject_cold_start_flags(scenario, rng, idx)
        scenario = inject_edge_cases(scenario, rng, idx)

        scenarios.append(scenario)

        if (idx + 1) % 200 == 0:
            logger.info("Generated %d / %d scenarios", idx + 1, n_scenarios)

    logger.info("Generation complete: %d scenarios", len(scenarios))
    return scenarios


# ═══════════════════════════════════════════════════════════════
# Save / export
# ═══════════════════════════════════════════════════════════════

def save_scenarios_json(scenarios: list[dict], name: str = "cart_scenarios") -> Path:
    """Save scenarios as a JSON file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved %d scenarios → %s (%.1f MB)", len(scenarios), out, out.stat().st_size / 1e6)
    return out


def save_dataset_parquet(scenarios: list[dict], name: str = "synthetic_carts") -> Path:
    """
    Flatten scenarios into a training-ready Parquet file.

    Each row = one (cart, candidate_addon) pair with a label.
    """
    import pandas as pd

    rows = []
    for sc in scenarios:
        base = {
            "scenario_id": sc["scenario_id"],
            "user_id": sc["user_profile"]["user_id"],
            "user_age": sc["user_profile"]["age"],
            "user_city": sc["user_profile"]["city"],
            "user_diet": sc["user_profile"]["dietary_preference"],
            "user_income": sc["user_profile"]["income_bracket"],
            "is_new_user": sc["user_profile"]["is_new_user"],
            "restaurant_id": sc["restaurant_profile"]["restaurant_id"],
            "restaurant_name": sc["restaurant_profile"]["name"],
            "cuisine": sc["cuisine"],
            "price_tier": sc["restaurant_profile"]["price_tier"],
            "restaurant_rating": sc["restaurant_profile"]["rating"],
            "is_new_restaurant": sc["restaurant_profile"]["is_new_restaurant"],
            "hour": sc["context"]["hour"],
            "day_of_week": sc["context"]["day_of_week"],
            "is_weekend": sc["context"]["is_weekend"],
            "meal_type": sc["context"]["meal_type"],
            "occasion": sc["context"]["occasion"],
            "cart_size": len(sc["cart_items"]),
            "cart_total": sum(i["price"] * i["qty"] for i in sc["cart_items"]),
            "cart_categories": "|".join(sorted({i["category"] for i in sc["cart_items"]})),
            "meal_completion_score": sc["meal_completion_score"],
        }

        # Positive examples (recommended)
        for rec in sc["recommended_items"]:
            row = {**base}
            row["candidate_id"] = rec["item_id"]
            row["candidate_name"] = rec["name"]
            row["candidate_category"] = rec["category"]
            row["candidate_price"] = rec["price"]
            row["candidate_is_veg"] = rec["is_veg"]
            row["relevance_score"] = rec["relevance_score"]
            row["label"] = 1
            rows.append(row)

        # Negative examples (anti-recommendations)
        for anti in sc["anti_recommendations"]:
            row = {**base}
            row["candidate_id"] = anti["item_id"]
            row["candidate_name"] = anti["name"]
            row["candidate_category"] = anti["category"]
            row["candidate_price"] = anti["price"]
            row["candidate_is_veg"] = True  # unknown, default
            row["relevance_score"] = 0.0
            row["label"] = 0
            rows.append(row)

    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Saved %d rows → %s", len(df), out)
    return out


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    scenarios = generate_scenarios(n_scenarios=1000, seed=42)

    # Save full scenarios as JSON (for inspection & debugging)
    json_path = save_scenarios_json(scenarios, "cart_scenarios_1000")

    # Save flattened training data as Parquet
    parquet_path = save_dataset_parquet(scenarios, "synthetic_carts_1000")

    # Print summary stats
    import collections
    cuisine_dist = collections.Counter(s["cuisine"] for s in scenarios)
    cold_start_count = sum(1 for s in scenarios if s.get("cold_start_type"))
    edge_case_count = sum(1 for s in scenarios if s.get("edge_cases"))
    avg_cart_size = np.mean([len(s["cart_items"]) for s in scenarios])
    avg_recs = np.mean([len(s["recommended_items"]) for s in scenarios])
    avg_anti = np.mean([len(s["anti_recommendations"]) for s in scenarios])
    avg_completion = np.mean([s["meal_completion_score"] for s in scenarios])

    print("\n" + "=" * 60)
    print("CART SCENARIO GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total scenarios     : {len(scenarios)}")
    print(f"Cuisine distribution:")
    for cuisine, count in sorted(cuisine_dist.items(), key=lambda x: -x[1]):
        print(f"  {cuisine:30s} : {count:4d} ({count/len(scenarios)*100:.1f}%)")
    print(f"Cold-start scenarios: {cold_start_count}")
    print(f"Edge-case scenarios : {edge_case_count}")
    print(f"Avg cart size       : {avg_cart_size:.1f} items")
    print(f"Avg recommendations : {avg_recs:.1f}")
    print(f"Avg anti-recs       : {avg_anti:.1f}")
    print(f"Avg meal completion : {avg_completion:.2f}")
    print(f"\nJSON output  : {json_path}")
    print(f"Parquet output: {parquet_path}")
    print("=" * 60)
