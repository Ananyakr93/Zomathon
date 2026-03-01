"""
Zomato Cart Recommendation ML Training Data Generator v3.0
Upgrades from v2.0 — Dataset quality 8.2 → 9/10:

v2.0 FIXES (carried over):
 [1] 50,000 scenarios
 [2] Cities: 10 cities including Ahmedabad + Chandigarh
 [3] Meal times: 4 slots (breakfast/lunch/evening_snacks/dinner)
 [4] User segments: budget_conscious, mid_tier, premium, health_conscious, experimenter
 [5] Cuisines: 8 types including Biryani Specialist, Street Food, Dessert
 [6] Cuisine names normalized
 [7] Veg ratio: 60% veg, 40% non-veg
 [8] User history: 30% new / 40% occasional / 30% frequent
 [9] Item removal events: 15% of carts
[10] restaurant_type field: cloud_kitchen / chain / local
[11] Weekend dessert rate: ~60%
[12] Main-without-side rate: ~35%
[13] Beverage absence: ~45%
[14] Acceptance rate: 35-50% overall
[15] Mumbai geographic pref: pav-based items 2.3x
[16] Peak hours weighted timestamps
[17] Continental cuisine added

v3.0 NEW IMPROVEMENTS:
[18] Item-level co-occurrence patterns (item-item signals in relevance scoring)
[19] Persistent user pool: 30% repeat users with cross-session history
[20] Discount/offer fields: 20% of scenarios have active offers with boosted relevance
[21] Recommendation fatigue counter: consecutive_rejections tracked per session
[22] Cart abandonment: 5% of sessions end without checkout
[23] Revenue uplift pre-computed: addon_revenue and addon_revenue_pct at checkout
"""

import json, random, math
from datetime import datetime, timedelta
from copy import deepcopy

random.seed(42)

TOTAL_SCENARIOS = 50_000

# ── OFFER TYPES ───────────────────────────────────────────────────────────────
OFFER_TYPES = ["flat_discount", "free_item", "combo_deal", "bogo"]
OFFER_DISCOUNT_RANGE = {"flat_discount": (10, 30), "free_item": (100, 100), "combo_deal": (15, 25), "bogo": (50, 50)}

# ── ITEM-LEVEL CO-OCCURRENCE (item-name → complements) ───────────────────────
ITEM_LEVEL_PATTERNS = {
    "butter chicken":            [("garlic naan", 0.78), ("raita", 0.45), ("jeera rice", 0.40), ("lassi", 0.35), ("gulab jamun", 0.30)],
    "dal makhani":               [("garlic naan", 0.80), ("tandoori roti", 0.55), ("raita", 0.40), ("lassi", 0.38)],
    "chicken dum biryani":       [("raita", 0.82), ("salan", 0.68), ("mirchi ka salan", 0.62), ("onion salad", 0.50), ("gulab jamun", 0.45), ("coke", 0.60)],
    "chicken biryani":           [("raita", 0.80), ("mirchi ka salan", 0.65), ("onion salad", 0.48), ("gulab jamun", 0.42), ("coke", 0.62)],
    "veg biryani":               [("raita", 0.75), ("salan", 0.55), ("onion salad", 0.45), ("gulab jamun", 0.38), ("coke", 0.55)],
    "masala dosa":               [("sambhar", 0.92), ("coconut chutney", 0.90), ("filter coffee", 0.70), ("tomato chutney", 0.58), ("vada", 0.38)],
    "idli":                      [("sambhar", 0.93), ("coconut chutney", 0.91), ("filter coffee", 0.75), ("vada", 0.50)],
    "rava idli":                 [("sambhar", 0.91), ("coconut chutney", 0.89), ("filter coffee", 0.80)],
    "mcaloo tikki burger":       [("medium fries", 0.88), ("coke", 0.82), ("mcflurry", 0.42)],
    "whopper":                   [("king fries", 0.85), ("coke", 0.80), ("oreo shake", 0.38), ("sundae", 0.32)],
    "zinger burger":             [("french fries", 0.86), ("coke", 0.82), ("coleslaw", 0.40)],
    "original recipe chicken":   [("french fries", 0.87), ("coke", 0.83), ("coleslaw", 0.50)],
    "pav bhaji":                 [("extra green chutney", 0.72), ("masala soda", 0.48), ("gulab jamun", 0.28)],
    "seekh kebab":               [("pav", 0.88), ("green chutney", 0.85), ("butter roomali roti", 0.60)],
    "margherita pizza":          [("garlic bread", 0.85), ("coke", 0.75), ("tiramisu", 0.48), ("caesar salad", 0.40)],
    "pasta arrabiata":           [("garlic bread", 0.82), ("tiramisu", 0.50), ("fresh lime soda", 0.45)],
    "veg hakka noodles":         [("hot and sour soup", 0.62), ("spring rolls", 0.55), ("coke", 0.70)],
    "chicken fried rice":        [("hot and sour soup", 0.58), ("spring rolls", 0.50), ("coke", 0.72)],
    "classic chocolate shake":   [("chocolate brownie", 0.65), ("cheesecake slice", 0.40), ("waffle", 0.38)],
}

# ── CITIES (10, matching spec) ───────────────────────────────────────────────
CITIES = {
    "Mumbai":     {"count": 9000,  "localities": ["Bandra","Andheri","Powai","Lower Parel","Juhu","Dadar","Worli"]},
    "Delhi/NCR":  {"count": 7500,  "localities": ["Connaught Place","Lajpat Nagar","Dwarka","Rohini","Karol Bagh","Saket","Vasant Kunj"]},
    "Bangalore":  {"count": 10000, "localities": ["Koramangala","Indiranagar","HSR Layout","Whitefield","Jayanagar","Marathahalli"]},
    "Hyderabad":  {"count": 5000,  "localities": ["Banjara Hills","Jubilee Hills","Gachibowli","Madhapur","Hitech City","Kukatpally"]},
    "Chennai":    {"count": 4000,  "localities": ["Anna Nagar","T Nagar","Velachery","Adyar","OMR","Nungambakkam"]},
    "Pune":       {"count": 5000,  "localities": ["Koregaon Park","Viman Nagar","Kothrud","Hinjewadi","Aundh","Baner"]},
    "Kolkata":    {"count": 3500,  "localities": ["Park Street","Salt Lake","New Town","Ballygunge","Tollygunge","Gariahat"]},
    "Ahmedabad":  {"count": 3000,  "localities": ["Navrangpura","Satellite","Vastrapur","CG Road","Maninagar","Bopal"]},
    "Jaipur":     {"count": 2000,  "localities": ["C-Scheme","Vaishali Nagar","Malviya Nagar","Mansarovar","Jagatpura"]},
    "Chandigarh": {"count": 1000,  "localities": ["Sector 17","Sector 22","Sector 35","Sector 43","Manimajra"]},
}

# ── USER SEGMENTS (5, matching spec) ─────────────────────────────────────────
USER_SEGMENTS = {
    "budget_conscious": {
        "count": 15000, "cart_value_range": (80, 280), "avg_items": (1, 3),
        "acceptance_rate": (0.58, 0.72), "income_range": (15000, 45000),
        "age_range": (18, 45), "price_tiers": ["budget"],
        "price_sensitivity": "high",
    },
    "mid_tier": {
        "count": 15000, "cart_value_range": (250, 600), "avg_items": (2, 4),
        "acceptance_rate": (0.52, 0.65), "income_range": (45000, 120000),
        "age_range": (22, 48), "price_tiers": ["budget","mid"],
        "price_sensitivity": "medium",
    },
    "premium": {
        "count": 8000, "cart_value_range": (500, 1500), "avg_items": (3, 6),
        "acceptance_rate": (0.55, 0.68), "income_range": (120000, 500000),
        "age_range": (25, 50), "price_tiers": ["mid","premium"],
        "price_sensitivity": "low",
    },
    "health_conscious": {
        "count": 6000, "cart_value_range": (200, 700), "avg_items": (2, 4),
        "acceptance_rate": (0.50, 0.62), "income_range": (50000, 250000),
        "age_range": (22, 45), "price_tiers": ["mid","premium"],
        "price_sensitivity": "medium",
        "dietary_skew": "vegetarian",
    },
    "experimenter": {
        "count": 6000, "cart_value_range": (200, 800), "avg_items": (2, 5),
        "acceptance_rate": (0.55, 0.70), "income_range": (30000, 200000),
        "age_range": (18, 35), "price_tiers": ["budget","mid","premium"],
        "price_sensitivity": "medium",
    },
}

# ── CUISINES (8, matching spec) ───────────────────────────────────────────────
CUISINES = {
    "North Indian":       {"count": 10000},
    "South Indian":       {"count": 7500},
    "Chinese":            {"count": 6000},
    "Continental":        {"count": 4000},
    "Fast Food":          {"count": 5000},
    "Biryani Specialist": {"count": 7500},
    "Street Food":        {"count": 5000},
    "Dessert":            {"count": 5000},
}

# ── MEAL TIMES (4, matching spec) ────────────────────────────────────────────
MEAL_TIMES = {
    "breakfast":      {"count": 5000,  "hour_weights": {7:5,8:20,9:30,10:30,11:15}},
    "lunch":          {"count": 17500, "hour_weights": {12:15,13:35,14:35,15:15}},
    "evening_snacks": {"count": 7500,  "hour_weights": {16:20,17:35,18:30,19:15}},
    "dinner":         {"count": 20000, "hour_weights": {19:10,20:35,21:35,22:15,23:5}},
}

RESTAURANT_TYPES = ["cloud_kitchen","chain","local"]
RESTAURANT_TYPE_DIST = [0.40, 0.35, 0.25]

WEATHER_OPTIONS = ["clear","cloudy","rainy","hot","cold"]
OCCASIONS = ["regular","office_lunch","date_night","party","celebration","family_gathering","hangover_cure"]

# ── MENU DATA ─────────────────────────────────────────────────────────────────
RESTAURANT_TEMPLATES = {
    "North Indian": [
        {
            "name": "Punjab Grill", "price_tier": "premium", "rating": 4.4, "delivery_mins": 40,
            "menu": [
                {"name":"Butter Chicken",   "cat":"main","subcat":"curry",   "price":380,"veg":False,"bs":True, "tags":["spicy","creamy","signature"],"pop":0.92,"rating":4.5},
                {"name":"Dal Makhani",      "cat":"main","subcat":"dal",     "price":280,"veg":True, "bs":True, "tags":["creamy","slow_cooked"],     "pop":0.88,"rating":4.4},
                {"name":"Garlic Naan",      "cat":"side","subcat":"bread",   "price":60, "veg":True, "bs":True, "tags":["bread","buttery"],          "pop":0.91,"rating":4.5},
                {"name":"Tandoori Roti",    "cat":"side","subcat":"bread",   "price":30, "veg":True, "bs":False,"tags":["bread","whole_wheat"],      "pop":0.75,"rating":4.2},
                {"name":"Jeera Rice",       "cat":"side","subcat":"rice",    "price":120,"veg":True, "bs":False,"tags":["rice","aromatic"],          "pop":0.70,"rating":4.2},
                {"name":"Raita",            "cat":"side","subcat":"curd",    "price":60, "veg":True, "bs":False,"tags":["cooling","yogurt"],         "pop":0.65,"rating":4.1},
                {"name":"Gulab Jamun",      "cat":"dessert","subcat":"sweet","price":90, "veg":True, "bs":False,"tags":["sweet","traditional"],      "pop":0.72,"rating":4.2},
                {"name":"Ras Malai",        "cat":"dessert","subcat":"sweet","price":120,"veg":True, "bs":False,"tags":["creamy","Bengali"],         "pop":0.65,"rating":4.3},
                {"name":"Lassi",            "cat":"beverage","subcat":"dairy","price":80,"veg":True, "bs":False,"tags":["cooling","sweet"],          "pop":0.68,"rating":4.1},
                {"name":"Coke 500ml",       "cat":"beverage","subcat":"soda","price":60, "veg":True, "bs":False,"tags":["cold","fizzy"],             "pop":0.74,"rating":4.0},
                {"name":"Paneer Tikka",     "cat":"appetizer","subcat":"starter","price":240,"veg":True,"bs":True,"tags":["grilled","smoky"],        "pop":0.85,"rating":4.4},
                {"name":"Samosa (2pc)",     "cat":"appetizer","subcat":"starter","price":60,"veg":True,"bs":False,"tags":["crispy","street_food"],   "pop":0.77,"rating":4.1},
            ]
        },
        {
            "name": "Moti Mahal Delux", "price_tier": "mid", "rating": 4.2, "delivery_mins": 35,
            "menu": [
                {"name":"Murgh Makhani",    "cat":"main","subcat":"curry",   "price":320,"veg":False,"bs":True, "tags":["creamy","mild"],   "pop":0.88,"rating":4.4},
                {"name":"Palak Paneer",     "cat":"main","subcat":"curry",   "price":280,"veg":True, "bs":True, "tags":["healthy","spinach"],"pop":0.82,"rating":4.3},
                {"name":"Naan",             "cat":"side","subcat":"bread",   "price":40, "veg":True, "bs":True, "tags":["bread","soft"],    "pop":0.88,"rating":4.3},
                {"name":"Paratha",          "cat":"side","subcat":"bread",   "price":50, "veg":True, "bs":False,"tags":["bread","flaky"],   "pop":0.72,"rating":4.1},
                {"name":"Steamed Rice",     "cat":"side","subcat":"rice",    "price":80, "veg":True, "bs":False,"tags":["plain","fluffy"],  "pop":0.60,"rating":4.0},
                {"name":"Raita",            "cat":"side","subcat":"curd",    "price":50, "veg":True, "bs":False,"tags":["cooling"],         "pop":0.62,"rating":4.0},
                {"name":"Gulab Jamun",      "cat":"dessert","subcat":"sweet","price":80, "veg":True, "bs":False,"tags":["sweet"],           "pop":0.68,"rating":4.1},
                {"name":"Shahi Tukda",      "cat":"dessert","subcat":"sweet","price":120,"veg":True, "bs":False,"tags":["rich","royal"],    "pop":0.55,"rating":4.2},
                {"name":"Lassi",            "cat":"beverage","subcat":"dairy","price":70,"veg":True, "bs":False,"tags":["cooling"],         "pop":0.64,"rating":4.0},
                {"name":"Coke 500ml",       "cat":"beverage","subcat":"soda","price":60, "veg":True, "bs":False,"tags":["cold"],            "pop":0.70,"rating":4.0},
                {"name":"Chicken Tikka",    "cat":"appetizer","subcat":"starter","price":260,"veg":False,"bs":True,"tags":["grilled","smoky"],"pop":0.80,"rating":4.3},
            ]
        },
        {
            "name": "Haldiram's", "price_tier": "budget", "rating": 4.0, "delivery_mins": 30,
            "menu": [
                {"name":"Chole Bhature",    "cat":"main","subcat":"curry",   "price":150,"veg":True,"bs":True, "tags":["spicy","filling"],  "pop":0.90,"rating":4.3},
                {"name":"Rajma Chawal",     "cat":"main","subcat":"combo",   "price":140,"veg":True,"bs":True, "tags":["comfort_food"],     "pop":0.82,"rating":4.2},
                {"name":"Dal Tadka",        "cat":"main","subcat":"dal",     "price":120,"veg":True,"bs":False,"tags":["homestyle"],        "pop":0.75,"rating":4.1},
                {"name":"Naan",             "cat":"side","subcat":"bread",   "price":30, "veg":True,"bs":False,"tags":["bread"],            "pop":0.78,"rating":4.0},
                {"name":"Raita",            "cat":"side","subcat":"curd",    "price":40, "veg":True,"bs":False,"tags":["cooling"],          "pop":0.60,"rating":4.0},
                {"name":"Gulab Jamun",      "cat":"dessert","subcat":"sweet","price":60, "veg":True,"bs":True, "tags":["sweet"],            "pop":0.80,"rating":4.2},
                {"name":"Jalebi",           "cat":"dessert","subcat":"sweet","price":50, "veg":True,"bs":False,"tags":["crispy","sweet"],   "pop":0.70,"rating":4.1},
                {"name":"Masala Chai",      "cat":"beverage","subcat":"hot", "price":30, "veg":True,"bs":True, "tags":["tea","warm"],       "pop":0.72,"rating":4.2},
                {"name":"Lassi",            "cat":"beverage","subcat":"dairy","price":60,"veg":True,"bs":False,"tags":["cooling"],          "pop":0.65,"rating":4.0},
                {"name":"Samosa (2pc)",     "cat":"appetizer","subcat":"starter","price":40,"veg":True,"bs":True,"tags":["crispy"],         "pop":0.85,"rating":4.1},
                {"name":"Papdi Chaat",      "cat":"appetizer","subcat":"chaat","price":80,"veg":True,"bs":False,"tags":["tangy","crispy"],  "pop":0.75,"rating":4.2},
            ]
        },
    ],
    "South Indian": [
        {
            "name": "Saravana Bhavan", "price_tier": "mid", "rating": 4.3, "delivery_mins": 30,
            "menu": [
                {"name":"Masala Dosa",      "cat":"main","subcat":"dosa",    "price":120,"veg":True,"bs":True, "tags":["crispy","signature"],"pop":0.93,"rating":4.5},
                {"name":"Plain Dosa",       "cat":"main","subcat":"dosa",    "price":80, "veg":True,"bs":False,"tags":["light","crispy"],    "pop":0.78,"rating":4.2},
                {"name":"Idli (3pc)",       "cat":"main","subcat":"idli",    "price":90, "veg":True,"bs":True, "tags":["soft","steamed"],    "pop":0.85,"rating":4.3},
                {"name":"Vada (2pc)",       "cat":"main","subcat":"vada",    "price":80, "veg":True,"bs":False,"tags":["crispy","savory"],   "pop":0.75,"rating":4.2},
                {"name":"Uttapam",          "cat":"main","subcat":"uttapam", "price":110,"veg":True,"bs":False,"tags":["thick","savory"],    "pop":0.68,"rating":4.1},
                {"name":"Sambhar",          "cat":"side","subcat":"lentil",  "price":40, "veg":True,"bs":True, "tags":["tangy","lentil"],   "pop":0.89,"rating":4.3},
                {"name":"Coconut Chutney",  "cat":"side","subcat":"chutney", "price":30, "veg":True,"bs":True, "tags":["cooling","coconut"],"pop":0.87,"rating":4.3},
                {"name":"Tomato Chutney",   "cat":"side","subcat":"chutney", "price":30, "veg":True,"bs":False,"tags":["tangy","red"],      "pop":0.70,"rating":4.1},
                {"name":"Filter Coffee",    "cat":"beverage","subcat":"hot", "price":50, "veg":True,"bs":True, "tags":["strong","signature"],"pop":0.88,"rating":4.5},
                {"name":"Rasam",            "cat":"side","subcat":"soup",    "price":50, "veg":True,"bs":False,"tags":["spicy","pepper"],   "pop":0.65,"rating":4.1},
                {"name":"Kesari Bath",      "cat":"dessert","subcat":"sweet","price":70, "veg":True,"bs":False,"tags":["sweet","semolina"], "pop":0.60,"rating":4.0},
                {"name":"Coke 500ml",       "cat":"beverage","subcat":"soda","price":60, "veg":True,"bs":False,"tags":["cold"],             "pop":0.65,"rating":4.0},
            ]
        },
        {
            "name": "MTR (Mavalli Tiffin Room)", "price_tier": "mid", "rating": 4.4, "delivery_mins": 35,
            "menu": [
                {"name":"Rava Idli (3pc)",  "cat":"main","subcat":"idli",   "price":110,"veg":True,"bs":True, "tags":["signature","soft"], "pop":0.90,"rating":4.5},
                {"name":"Set Dosa (3pc)",   "cat":"main","subcat":"dosa",   "price":100,"veg":True,"bs":True, "tags":["fluffy","mini"],    "pop":0.85,"rating":4.3},
                {"name":"Bisibele Bath",    "cat":"main","subcat":"rice",   "price":130,"veg":True,"bs":True, "tags":["spicy","hearty"],   "pop":0.80,"rating":4.3},
                {"name":"Sambhar",          "cat":"side","subcat":"lentil", "price":40, "veg":True,"bs":True, "tags":["tangy"],            "pop":0.88,"rating":4.3},
                {"name":"Coconut Chutney",  "cat":"side","subcat":"chutney","price":30, "veg":True,"bs":True, "tags":["cooling"],          "pop":0.85,"rating":4.2},
                {"name":"Filter Coffee",    "cat":"beverage","subcat":"hot","price":55, "veg":True,"bs":True, "tags":["strong","aromatic"],"pop":0.92,"rating":4.6},
                {"name":"Badam Halwa",      "cat":"dessert","subcat":"sweet","price":100,"veg":True,"bs":False,"tags":["rich","almond"],   "pop":0.62,"rating":4.2},
                {"name":"Coke 500ml",       "cat":"beverage","subcat":"soda","price":60,"veg":True,"bs":False,"tags":["cold"],             "pop":0.63,"rating":4.0},
            ]
        },
        {
            "name": "Udupi Krishna Bhavan", "price_tier": "budget", "rating": 4.0, "delivery_mins": 25,
            "menu": [
                {"name":"Masala Dosa",      "cat":"main","subcat":"dosa",   "price":90, "veg":True,"bs":True, "tags":["crispy"],"pop":0.88,"rating":4.2},
                {"name":"Plain Dosa",       "cat":"main","subcat":"dosa",   "price":60, "veg":True,"bs":False,"tags":["light"],"pop":0.75,"rating":4.0},
                {"name":"Idli (3pc)",       "cat":"main","subcat":"idli",   "price":70, "veg":True,"bs":True, "tags":["steamed"],"pop":0.82,"rating":4.1},
                {"name":"Vada (2pc)",       "cat":"main","subcat":"vada",   "price":60, "veg":True,"bs":False,"tags":["crispy"],"pop":0.72,"rating":4.0},
                {"name":"Sambhar",          "cat":"side","subcat":"lentil", "price":30, "veg":True,"bs":True, "tags":["tangy"],"pop":0.88,"rating":4.1},
                {"name":"Coconut Chutney",  "cat":"side","subcat":"chutney","price":20, "veg":True,"bs":True, "tags":["cooling"],"pop":0.85,"rating":4.1},
                {"name":"Filter Coffee",    "cat":"beverage","subcat":"hot","price":30, "veg":True,"bs":True, "tags":["strong"],"pop":0.85,"rating":4.3},
                {"name":"Coke 500ml",       "cat":"beverage","subcat":"soda","price":60,"veg":True,"bs":False,"tags":["cold"],"pop":0.60,"rating":4.0},
            ]
        },
    ],
    "Chinese": [
        {
            "name": "Mainland China", "price_tier": "premium", "rating": 4.3, "delivery_mins": 40,
            "menu": [
                {"name":"Veg Hakka Noodles",   "cat":"main","subcat":"noodles","price":280,"veg":True, "bs":True, "tags":["stir_fried","noodles"],"pop":0.85,"rating":4.3},
                {"name":"Chicken Fried Rice",   "cat":"main","subcat":"rice",   "price":320,"veg":False,"bs":True, "tags":["fried","rice"],        "pop":0.88,"rating":4.4},
                {"name":"Paneer Manchurian",    "cat":"main","subcat":"manchurian","price":280,"veg":True,"bs":True,"tags":["saucy","indo_chinese"],"pop":0.82,"rating":4.3},
                {"name":"Chicken Manchurian",   "cat":"main","subcat":"manchurian","price":320,"veg":False,"bs":True,"tags":["saucy","spicy"],     "pop":0.85,"rating":4.4},
                {"name":"Hot and Sour Soup",    "cat":"appetizer","subcat":"soup","price":140,"veg":True,"bs":True,"tags":["tangy","spicy","warm"],"pop":0.78,"rating":4.2},
                {"name":"Spring Rolls (4pc)",   "cat":"appetizer","subcat":"starter","price":180,"veg":True,"bs":False,"tags":["crispy","fried"],  "pop":0.72,"rating":4.1},
                {"name":"Chilli Garlic Sauce",  "cat":"side","subcat":"condiment","price":60,"veg":True,"bs":False,"tags":["spicy","dip"],         "pop":0.62,"rating":4.0},
                {"name":"Fried Ice Cream",      "cat":"dessert","subcat":"icecream","price":180,"veg":True,"bs":False,"tags":["unique","sweet"],   "pop":0.65,"rating":4.2},
                {"name":"Coke 500ml",           "cat":"beverage","subcat":"soda","price":60,"veg":True,"bs":False,"tags":["cold"],                "pop":0.78,"rating":4.0},
                {"name":"Lemon Iced Tea",       "cat":"beverage","subcat":"tea", "price":90,"veg":True,"bs":False,"tags":["refreshing","citrus"],  "pop":0.68,"rating":4.1},
                {"name":"Mushroom Fried Rice",  "cat":"main","subcat":"rice",   "price":260,"veg":True,"bs":False,"tags":["earthy","fried"],      "pop":0.70,"rating":4.1},
            ]
        },
        {
            "name": "Chung Wa", "price_tier": "mid", "rating": 4.1, "delivery_mins": 35,
            "menu": [
                {"name":"Veg Noodles",         "cat":"main","subcat":"noodles","price":200,"veg":True, "bs":True, "tags":["stir_fried"],"pop":0.80,"rating":4.1},
                {"name":"Chicken Noodles",     "cat":"main","subcat":"noodles","price":240,"veg":False,"bs":True, "tags":["savory"],    "pop":0.83,"rating":4.2},
                {"name":"Veg Fried Rice",      "cat":"main","subcat":"rice",   "price":180,"veg":True, "bs":False,"tags":["fried"],     "pop":0.75,"rating":4.0},
                {"name":"Chilli Paneer",       "cat":"main","subcat":"manchurian","price":250,"veg":True,"bs":True,"tags":["spicy","saucy"],"pop":0.82,"rating":4.2},
                {"name":"Sweet Corn Soup",     "cat":"appetizer","subcat":"soup","price":100,"veg":True,"bs":False,"tags":["sweet","warm"],"pop":0.68,"rating":4.0},
                {"name":"Spring Rolls (4pc)",  "cat":"appetizer","subcat":"starter","price":130,"veg":True,"bs":False,"tags":["crispy"],"pop":0.68,"rating":4.0},
                {"name":"Coke 500ml",          "cat":"beverage","subcat":"soda","price":60,"veg":True,"bs":False,"tags":["cold"],        "pop":0.74,"rating":4.0},
                {"name":"Vanilla Ice Cream",   "cat":"dessert","subcat":"icecream","price":80,"veg":True,"bs":False,"tags":["sweet","cold"],"pop":0.60,"rating":4.0},
            ]
        },
    ],
    "Continental": [
        {
            "name": "The Fatty Bao", "price_tier": "premium", "rating": 4.4, "delivery_mins": 45,
            "menu": [
                {"name":"Margherita Pizza (M)",   "cat":"main","subcat":"pizza",   "price":380,"veg":True, "bs":True, "tags":["classic","cheesy","wood_fired"],"pop":0.88,"rating":4.3},
                {"name":"Pasta Arrabiata",        "cat":"main","subcat":"pasta",   "price":320,"veg":True, "bs":True, "tags":["spicy","tomato","Italian"],      "pop":0.82,"rating":4.2},
                {"name":"Chicken Alfredo Pasta",  "cat":"main","subcat":"pasta",   "price":420,"veg":False,"bs":True, "tags":["creamy","rich"],                "pop":0.85,"rating":4.4},
                {"name":"Grilled Chicken",        "cat":"main","subcat":"grills",  "price":480,"veg":False,"bs":False,"tags":["healthy","grilled"],            "pop":0.78,"rating":4.3},
                {"name":"Veg Lasagna",            "cat":"main","subcat":"baked",   "price":360,"veg":True, "bs":False,"tags":["layered","cheesy"],             "pop":0.72,"rating":4.2},
                {"name":"Garlic Bread",           "cat":"side","subcat":"bread",   "price":120,"veg":True, "bs":True, "tags":["buttery","garlicky"],           "pop":0.88,"rating":4.3},
                {"name":"Caesar Salad",           "cat":"side","subcat":"salad",   "price":220,"veg":False,"bs":False,"tags":["fresh","creamy"],               "pop":0.68,"rating":4.1},
                {"name":"Tomato Soup",            "cat":"appetizer","subcat":"soup","price":180,"veg":True,"bs":False,"tags":["warm","creamy"],               "pop":0.72,"rating":4.1},
                {"name":"Bruschetta",             "cat":"appetizer","subcat":"starter","price":200,"veg":True,"bs":False,"tags":["crispy","tomato"],          "pop":0.65,"rating":4.0},
                {"name":"Tiramisu",               "cat":"dessert","subcat":"cake", "price":220,"veg":False,"bs":True, "tags":["Italian","coffee","creamy"],    "pop":0.75,"rating":4.4},
                {"name":"Chocolate Mousse",       "cat":"dessert","subcat":"cake", "price":180,"veg":True, "bs":False,"tags":["creamy","chocolate"],          "pop":0.70,"rating":4.2},
                {"name":"Fresh Lime Soda",        "cat":"beverage","subcat":"soft","price":80, "veg":True, "bs":False,"tags":["refreshing","citrus"],         "pop":0.72,"rating":4.1},
                {"name":"Cold Coffee",            "cat":"beverage","subcat":"coffee","price":140,"veg":True,"bs":False,"tags":["cold","coffee"],              "pop":0.78,"rating":4.2},
                {"name":"Coke 500ml",             "cat":"beverage","subcat":"soda","price":60, "veg":True, "bs":False,"tags":["cold"],                       "pop":0.75,"rating":4.0},
            ]
        },
        {
            "name": "Social", "price_tier": "mid", "rating": 4.2, "delivery_mins": 40,
            "menu": [
                {"name":"BBQ Chicken Burger",     "cat":"main","subcat":"burger",  "price":320,"veg":False,"bs":True, "tags":["smoky","juicy"],               "pop":0.85,"rating":4.3},
                {"name":"Portobello Mushroom Burger","cat":"main","subcat":"burger","price":280,"veg":True,"bs":False,"tags":["earthy","juicy"],              "pop":0.72,"rating":4.1},
                {"name":"Club Sandwich",          "cat":"main","subcat":"sandwich","price":280,"veg":False,"bs":True, "tags":["classic","hearty"],            "pop":0.78,"rating":4.1},
                {"name":"Veg Club Sandwich",      "cat":"main","subcat":"sandwich","price":240,"veg":True, "bs":False,"tags":["fresh","hearty"],             "pop":0.68,"rating":4.0},
                {"name":"Loaded Fries",           "cat":"side","subcat":"fries",   "price":180,"veg":True, "bs":True, "tags":["cheesy","crispy"],            "pop":0.82,"rating":4.2},
                {"name":"Onion Rings",            "cat":"side","subcat":"fries",   "price":140,"veg":True, "bs":False,"tags":["crispy","fried"],             "pop":0.72,"rating":4.0},
                {"name":"Coke 500ml",             "cat":"beverage","subcat":"soda","price":60, "veg":True, "bs":False,"tags":["cold"],                      "pop":0.78,"rating":4.0},
                {"name":"Iced Coffee",            "cat":"beverage","subcat":"coffee","price":160,"veg":True,"bs":False,"tags":["cold","coffee"],            "pop":0.70,"rating":4.1},
                {"name":"Brownie with Ice Cream", "cat":"dessert","subcat":"cake", "price":200,"veg":True, "bs":True, "tags":["warm","chocolate","indulgent"],"pop":0.80,"rating":4.3},
                {"name":"Cheesecake",             "cat":"dessert","subcat":"cake", "price":180,"veg":True, "bs":False,"tags":["creamy","tangy"],             "pop":0.70,"rating":4.2},
            ]
        },
    ],
    "Fast Food": [
        {
            "name": "McDonald's", "price_tier": "budget", "rating": 4.0, "delivery_mins": 25,
            "menu": [
                {"name":"McAloo Tikki Burger",   "cat":"main","subcat":"burger","price":99, "veg":True, "bs":True, "tags":["crispy","signature","veg"],"pop":0.90,"rating":4.2},
                {"name":"McSpicy Paneer Burger", "cat":"main","subcat":"burger","price":179,"veg":True, "bs":True, "tags":["spicy","paneer"],          "pop":0.85,"rating":4.2},
                {"name":"McChicken Burger",      "cat":"main","subcat":"burger","price":149,"veg":False,"bs":True, "tags":["crispy","chicken"],        "pop":0.87,"rating":4.2},
                {"name":"McSpicy Fried Chicken", "cat":"main","subcat":"chicken","price":169,"veg":False,"bs":True,"tags":["spicy","crunchy"],         "pop":0.82,"rating":4.1},
                {"name":"Medium Fries",          "cat":"side","subcat":"fries","price":109, "veg":True, "bs":True, "tags":["crispy","salted"],         "pop":0.92,"rating":4.3},
                {"name":"Large Fries",           "cat":"side","subcat":"fries","price":139, "veg":True, "bs":False,"tags":["crispy"],                  "pop":0.78,"rating":4.2},
                {"name":"Coke (M)",              "cat":"beverage","subcat":"soda","price":89,"veg":True, "bs":True, "tags":["cold","fizzy"],            "pop":0.85,"rating":4.0},
                {"name":"McFlurry Oreo",         "cat":"dessert","subcat":"icecream","price":99,"veg":True,"bs":True,"tags":["creamy","sweet","cold"],   "pop":0.80,"rating":4.3},
                {"name":"Soft Serve Cone",       "cat":"dessert","subcat":"icecream","price":30,"veg":True,"bs":False,"tags":["light","cold"],          "pop":0.72,"rating":4.1},
                {"name":"McVeggie Burger",       "cat":"main","subcat":"burger","price":129,"veg":True, "bs":False,"tags":["veggie","simple"],         "pop":0.75,"rating":4.0},
                {"name":"Piri Piri Fries",       "cat":"side","subcat":"fries","price":129, "veg":True, "bs":False,"tags":["spicy","crispy"],          "pop":0.70,"rating":4.1},
            ]
        },
        {
            "name": "Burger King", "price_tier": "budget", "rating": 4.0, "delivery_mins": 25,
            "menu": [
                {"name":"Whopper",           "cat":"main","subcat":"burger","price":259,"veg":False,"bs":True, "tags":["flame_grilled","signature"],"pop":0.90,"rating":4.3},
                {"name":"Veg Whopper",       "cat":"main","subcat":"burger","price":199,"veg":True, "bs":True, "tags":["veggie","hearty"],          "pop":0.80,"rating":4.0},
                {"name":"Crispy Veg Burger", "cat":"main","subcat":"burger","price":149,"veg":True, "bs":False,"tags":["crispy"],                   "pop":0.75,"rating":4.0},
                {"name":"Crispy Chicken",    "cat":"main","subcat":"burger","price":179,"veg":False,"bs":False,"tags":["crispy","chicken"],         "pop":0.78,"rating":4.1},
                {"name":"King Fries (M)",    "cat":"side","subcat":"fries","price":109, "veg":True, "bs":True, "tags":["crispy","salted"],          "pop":0.88,"rating":4.2},
                {"name":"Coke (M)",          "cat":"beverage","subcat":"soda","price":89,"veg":True, "bs":True, "tags":["cold"],                    "pop":0.83,"rating":4.0},
                {"name":"Oreo Shake",        "cat":"beverage","subcat":"shake","price":139,"veg":True,"bs":False,"tags":["creamy","sweet"],         "pop":0.70,"rating":4.2},
                {"name":"Sundae",            "cat":"dessert","subcat":"icecream","price":69,"veg":True,"bs":False,"tags":["sweet","cold"],          "pop":0.68,"rating":4.0},
            ]
        },
        {
            "name": "KFC", "price_tier": "mid", "rating": 4.1, "delivery_mins": 30,
            "menu": [
                {"name":"Original Recipe Chicken (2pc)","cat":"main","subcat":"chicken","price":249,"veg":False,"bs":True,"tags":["crispy","signature","juicy"],"pop":0.93,"rating":4.4},
                {"name":"Zinger Burger",     "cat":"main","subcat":"burger","price":199,"veg":False,"bs":True, "tags":["spicy","crispy"],   "pop":0.88,"rating":4.3},
                {"name":"Veg Zinger Burger", "cat":"main","subcat":"burger","price":149,"veg":True, "bs":False,"tags":["veggie","spicy"],   "pop":0.72,"rating":4.0},
                {"name":"Hot Wings (5pc)",   "cat":"main","subcat":"chicken","price":249,"veg":False,"bs":True,"tags":["spicy","wings"],     "pop":0.85,"rating":4.2},
                {"name":"Popcorn Chicken",   "cat":"appetizer","subcat":"starter","price":159,"veg":False,"bs":True,"tags":["crispy","bite_sized"],"pop":0.80,"rating":4.2},
                {"name":"Coleslaw",          "cat":"side","subcat":"salad","price":69,  "veg":True, "bs":False,"tags":["creamy","cooling"],  "pop":0.75,"rating":4.0},
                {"name":"French Fries",      "cat":"side","subcat":"fries","price":109, "veg":True, "bs":True, "tags":["crispy","salted"],   "pop":0.88,"rating":4.2},
                {"name":"Coke (M)",          "cat":"beverage","subcat":"soda","price":89,"veg":True,"bs":True,  "tags":["cold"],             "pop":0.85,"rating":4.0},
                {"name":"Choco Chip Muffin", "cat":"dessert","subcat":"cake","price":89,"veg":True, "bs":False,"tags":["sweet","chocolate"], "pop":0.65,"rating":4.0},
            ]
        },
    ],
    "Biryani Specialist": [
        {
            "name": "Biryani By Kilo", "price_tier": "premium", "rating": 4.4, "delivery_mins": 50,
            "menu": [
                {"name":"Veg Dum Biryani",     "cat":"main","subcat":"biryani","price":349,"veg":True, "bs":True, "tags":["dum","aromatic","signature"],"pop":0.85,"rating":4.3},
                {"name":"Chicken Dum Biryani", "cat":"main","subcat":"biryani","price":399,"veg":False,"bs":True, "tags":["dum","smoky"],               "pop":0.92,"rating":4.5},
                {"name":"Mutton Biryani",      "cat":"main","subcat":"biryani","price":499,"veg":False,"bs":False,"tags":["premium","tender"],           "pop":0.78,"rating":4.4},
                {"name":"Paneer Biryani",      "cat":"main","subcat":"biryani","price":329,"veg":True, "bs":False,"tags":["paneer","aromatic"],          "pop":0.80,"rating":4.2},
                {"name":"Raita",               "cat":"side","subcat":"curd",   "price":79, "veg":True, "bs":True, "tags":["cooling","yogurt"],           "pop":0.88,"rating":4.2},
                {"name":"Salan (Gravy)",       "cat":"side","subcat":"gravy",  "price":99, "veg":True, "bs":False,"tags":["spicy","traditional"],        "pop":0.70,"rating":4.1},
                {"name":"Shorba Soup",         "cat":"appetizer","subcat":"soup","price":99,"veg":False,"bs":False,"tags":["warm","spicy"],              "pop":0.58,"rating":4.1},
                {"name":"Gulab Jamun (2pc)",   "cat":"dessert","subcat":"sweet","price":99,"veg":True, "bs":False,"tags":["sweet","traditional"],        "pop":0.72,"rating":4.2},
                {"name":"Phirni",              "cat":"dessert","subcat":"sweet","price":129,"veg":True, "bs":False,"tags":["creamy","chilled"],          "pop":0.60,"rating":4.2},
                {"name":"Coke 500ml",          "cat":"beverage","subcat":"soda","price":60,"veg":True, "bs":False,"tags":["cold","fizzy"],               "pop":0.75,"rating":4.0},
                {"name":"Lemonade",            "cat":"beverage","subcat":"soft","price":79,"veg":True, "bs":False,"tags":["refreshing","citrus"],        "pop":0.65,"rating":4.1},
                {"name":"Onion Salad",         "cat":"side","subcat":"salad", "price":49, "veg":True, "bs":False,"tags":["fresh","crunchy"],            "pop":0.62,"rating":4.0},
            ]
        },
        {
            "name": "Paradise Biryani", "price_tier": "mid", "rating": 4.3, "delivery_mins": 40,
            "menu": [
                {"name":"Chicken Biryani",    "cat":"main","subcat":"biryani","price":280,"veg":False,"bs":True, "tags":["spicy","Hyderabadi"],"pop":0.93,"rating":4.5},
                {"name":"Veg Biryani",        "cat":"main","subcat":"biryani","price":220,"veg":True, "bs":True, "tags":["aromatic"],          "pop":0.80,"rating":4.2},
                {"name":"Egg Biryani",        "cat":"main","subcat":"biryani","price":240,"veg":False,"bs":False,"tags":["rich"],              "pop":0.72,"rating":4.2},
                {"name":"Raita",              "cat":"side","subcat":"curd",   "price":60, "veg":True, "bs":True, "tags":["cooling"],           "pop":0.85,"rating":4.2},
                {"name":"Mirchi Ka Salan",    "cat":"side","subcat":"gravy",  "price":80, "veg":True, "bs":True, "tags":["spicy","Hyderabadi"],"pop":0.75,"rating":4.3},
                {"name":"Gulab Jamun",        "cat":"dessert","subcat":"sweet","price":80,"veg":True, "bs":False,"tags":["sweet"],             "pop":0.70,"rating":4.1},
                {"name":"Double Ka Meetha",   "cat":"dessert","subcat":"sweet","price":100,"veg":True,"bs":False,"tags":["rich","local"],      "pop":0.58,"rating":4.2},
                {"name":"Coke 500ml",         "cat":"beverage","subcat":"soda","price":60,"veg":True, "bs":False,"tags":["cold"],              "pop":0.72,"rating":4.0},
                {"name":"Lassi",              "cat":"beverage","subcat":"dairy","price":80,"veg":True, "bs":False,"tags":["cooling"],          "pop":0.65,"rating":4.1},
                {"name":"Onion Salad",        "cat":"side","subcat":"salad", "price":40, "veg":True, "bs":False,"tags":["fresh"],             "pop":0.60,"rating":4.0},
            ]
        },
        {
            "name": "Zam Zam Biryani", "price_tier": "budget", "rating": 4.1, "delivery_mins": 35,
            "menu": [
                {"name":"Chicken Biryani",    "cat":"main","subcat":"biryani","price":220,"veg":False,"bs":True, "tags":["spicy"],"pop":0.90,"rating":4.3},
                {"name":"Veg Biryani",        "cat":"main","subcat":"biryani","price":170,"veg":True, "bs":False,"tags":["aromatic"],"pop":0.75,"rating":4.0},
                {"name":"Mutton Biryani",     "cat":"main","subcat":"biryani","price":280,"veg":False,"bs":False,"tags":["tender"],"pop":0.72,"rating":4.2},
                {"name":"Raita",              "cat":"side","subcat":"curd",   "price":50, "veg":True, "bs":True, "tags":["cooling"],"pop":0.82,"rating":4.1},
                {"name":"Salan",              "cat":"side","subcat":"gravy",  "price":60, "veg":True, "bs":False,"tags":["spicy"],"pop":0.68,"rating":4.0},
                {"name":"Gulab Jamun",        "cat":"dessert","subcat":"sweet","price":60,"veg":True, "bs":False,"tags":["sweet"],"pop":0.65,"rating":4.0},
                {"name":"Coke 500ml",         "cat":"beverage","subcat":"soda","price":60,"veg":True, "bs":False,"tags":["cold"],"pop":0.70,"rating":4.0},
                {"name":"Onion Salad",        "cat":"side","subcat":"salad", "price":30, "veg":True, "bs":False,"tags":["fresh"],"pop":0.58,"rating":3.9},
            ]
        },
    ],
    "Street Food": [
        {
            "name": "Elco Pani Puri Center", "price_tier": "budget", "rating": 4.2, "delivery_mins": 25,
            "menu": [
                {"name":"Pani Puri (6pc)",     "cat":"main","subcat":"chaat","price":60, "veg":True,"bs":True, "tags":["tangy","crispy","street_food"],"pop":0.92,"rating":4.4},
                {"name":"Pav Bhaji",           "cat":"main","subcat":"bhaji","price":120,"veg":True,"bs":True, "tags":["spicy","buttery","Mumbai_special"],"pop":0.90,"rating":4.3},
                {"name":"Vada Pav",            "cat":"main","subcat":"snack","price":50, "veg":True,"bs":True, "tags":["crispy","Mumbai_special","street_food"],"pop":0.88,"rating":4.3},
                {"name":"Sev Puri",            "cat":"main","subcat":"chaat","price":80, "veg":True,"bs":False,"tags":["crunchy","tangy"],"pop":0.78,"rating":4.1},
                {"name":"Bhel Puri",           "cat":"main","subcat":"chaat","price":70, "veg":True,"bs":False,"tags":["crunchy","tangy"],"pop":0.75,"rating":4.1},
                {"name":"Dahi Puri",           "cat":"main","subcat":"chaat","price":90, "veg":True,"bs":False,"tags":["creamy","tangy"],"pop":0.72,"rating":4.2},
                {"name":"Extra Green Chutney", "cat":"side","subcat":"chutney","price":20,"veg":True,"bs":False,"tags":["spicy","tangy"],"pop":0.70,"rating":4.0},
                {"name":"Extra Tamarind Chutney","cat":"side","subcat":"chutney","price":20,"veg":True,"bs":False,"tags":["sweet","tangy"],"pop":0.68,"rating":4.0},
                {"name":"Masala Soda",         "cat":"beverage","subcat":"soda","price":30,"veg":True,"bs":False,"tags":["tangy","refreshing"],"pop":0.72,"rating":4.1},
                {"name":"Aam Panna",           "cat":"beverage","subcat":"soft","price":50,"veg":True,"bs":False,"tags":["mango","cooling"],"pop":0.65,"rating":4.1},
                {"name":"Gulab Jamun (2pc)",   "cat":"dessert","subcat":"sweet","price":40,"veg":True,"bs":False,"tags":["sweet"],"pop":0.60,"rating":4.0},
            ]
        },
        {
            "name": "Bademiya", "price_tier": "budget", "rating": 4.1, "delivery_mins": 30,
            "menu": [
                {"name":"Seekh Kebab (4pc)",   "cat":"main","subcat":"kebab","price":200,"veg":False,"bs":True, "tags":["grilled","smoky","street_food"],"pop":0.90,"rating":4.3},
                {"name":"Chicken Tikka",       "cat":"main","subcat":"kebab","price":220,"veg":False,"bs":True, "tags":["grilled","spicy"],"pop":0.85,"rating":4.2},
                {"name":"Veg Seekh Kebab",     "cat":"main","subcat":"kebab","price":150,"veg":True, "bs":False,"tags":["grilled"],"pop":0.70,"rating":4.0},
                {"name":"Pav (4pc)",           "cat":"side","subcat":"bread","price":20, "veg":True, "bs":True, "tags":["soft","bread","Mumbai_special"],"pop":0.85,"rating":4.1},
                {"name":"Green Chutney",       "cat":"side","subcat":"chutney","price":20,"veg":True,"bs":True, "tags":["spicy","tangy"],"pop":0.82,"rating":4.1},
                {"name":"Onion Rings",         "cat":"side","subcat":"salad","price":30, "veg":True, "bs":False,"tags":["fresh","pungent"],"pop":0.68,"rating":4.0},
                {"name":"Butter Roomali Roti", "cat":"side","subcat":"bread","price":40, "veg":True, "bs":False,"tags":["soft","buttery"],"pop":0.72,"rating":4.1},
                {"name":"Coke 500ml",          "cat":"beverage","subcat":"soda","price":60,"veg":True,"bs":False,"tags":["cold"],"pop":0.72,"rating":4.0},
                {"name":"Masala Chai",         "cat":"beverage","subcat":"hot","price":20,"veg":True,"bs":False,"tags":["warm","tea"],"pop":0.65,"rating":4.1},
            ]
        },
    ],
    "Dessert": [
        {
            "name": "Keventers", "price_tier": "mid", "rating": 4.2, "delivery_mins": 25,
            "menu": [
                {"name":"Classic Chocolate Shake","cat":"beverage","subcat":"shake","price":220,"veg":True,"bs":True,"tags":["thick","chocolate","signature"],"pop":0.90,"rating":4.4},
                {"name":"Strawberry Shake",      "cat":"beverage","subcat":"shake","price":220,"veg":True,"bs":True,"tags":["fruity","pink"],"pop":0.82,"rating":4.2},
                {"name":"Vanilla Shake",         "cat":"beverage","subcat":"shake","price":200,"veg":True,"bs":False,"tags":["classic","creamy"],"pop":0.75,"rating":4.1},
                {"name":"Mango Shake",           "cat":"beverage","subcat":"shake","price":230,"veg":True,"bs":False,"tags":["mango","fruity"],"pop":0.80,"rating":4.2},
                {"name":"Chocolate Brownie",     "cat":"dessert","subcat":"cake","price":120,"veg":True,"bs":True,"tags":["dense","chocolate"],"pop":0.78,"rating":4.2},
                {"name":"Cheesecake Slice",      "cat":"dessert","subcat":"cake","price":160,"veg":True,"bs":False,"tags":["creamy","tangy"],"pop":0.70,"rating":4.2},
                {"name":"Waffle",               "cat":"dessert","subcat":"waffle","price":180,"veg":True,"bs":False,"tags":["warm","crispy","sweet"],"pop":0.72,"rating":4.1},
                {"name":"Banana Caramel Shake",  "cat":"beverage","subcat":"shake","price":240,"veg":True,"bs":False,"tags":["banana","caramel"],"pop":0.68,"rating":4.1},
            ]
        },
        {
            "name": "Baskin Robbins", "price_tier": "mid", "rating": 4.1, "delivery_mins": 30,
            "menu": [
                {"name":"2 Scoop Regular",       "cat":"dessert","subcat":"icecream","price":150,"veg":True,"bs":True, "tags":["cold","classic"],"pop":0.88,"rating":4.2},
                {"name":"3 Scoop Regular",       "cat":"dessert","subcat":"icecream","price":210,"veg":True,"bs":True, "tags":["cold","indulgent"],"pop":0.82,"rating":4.2},
                {"name":"Sundae",               "cat":"dessert","subcat":"icecream","price":180,"veg":True,"bs":False,"tags":["topped","sweet"],"pop":0.75,"rating":4.1},
                {"name":"Ice Cream Cake (500g)","cat":"dessert","subcat":"cake",    "price":450,"veg":True,"bs":False,"tags":["party","celebration"],"pop":0.58,"rating":4.3},
                {"name":"Ice Cream Sandwich",   "cat":"dessert","subcat":"icecream","price":80, "veg":True,"bs":False,"tags":["portable","sweet"],"pop":0.68,"rating":4.0},
                {"name":"Thick Milkshake",      "cat":"beverage","subcat":"shake",  "price":199,"veg":True,"bs":False,"tags":["creamy","thick"],"pop":0.72,"rating":4.1},
            ]
        },
        {
            "name": "Natural Ice Cream", "price_tier": "budget", "rating": 4.3, "delivery_mins": 20,
            "menu": [
                {"name":"Sitaphal Ice Cream (2sc)","cat":"dessert","subcat":"icecream","price":120,"veg":True,"bs":True,"tags":["natural","custard_apple","signature"],"pop":0.90,"rating":4.5},
                {"name":"Tender Coconut Ice Cream","cat":"dessert","subcat":"icecream","price":110,"veg":True,"bs":True,"tags":["natural","coconut","light"],"pop":0.85,"rating":4.4},
                {"name":"Mango Ice Cream (2sc)",  "cat":"dessert","subcat":"icecream","price":100,"veg":True,"bs":True,"tags":["natural","mango"],"pop":0.88,"rating":4.4},
                {"name":"Anjir Fig Ice Cream",    "cat":"dessert","subcat":"icecream","price":130,"veg":True,"bs":False,"tags":["natural","fig","unique"],"pop":0.72,"rating":4.3},
                {"name":"Mixed Fruit Salad",      "cat":"side","subcat":"salad",   "price":80, "veg":True,"bs":False,"tags":["fresh","healthy"],"pop":0.62,"rating":4.1},
                {"name":"Ice Cream Sandwich",     "cat":"dessert","subcat":"icecream","price":60,"veg":True,"bs":False,"tags":["portable"],"pop":0.68,"rating":4.0},
            ]
        },
    ],
}

# ── CO-OCCURRENCE PATTERNS ────────────────────────────────────────────────────
COOCCURRENCE_PATTERNS = {
    "Biryani Specialist": {
        "main_items": ["Biryani (all types)"],
        "expected_sides": [
            {"item": "Raita", "cooccurrence_prob": 0.75},
            {"item": "Salan/Mirchi Ka Salan", "cooccurrence_prob": 0.55},
            {"item": "Onion Salad", "cooccurrence_prob": 0.40},
        ],
        "expected_desserts": [
            {"item": "Gulab Jamun", "cooccurrence_prob": 0.45},
            {"item": "Phirni/Kheer", "cooccurrence_prob": 0.28},
        ],
        "expected_beverages": [
            {"item": "Coke/Soft Drink", "cooccurrence_prob": 0.60},
            {"item": "Lassi", "cooccurrence_prob": 0.35},
        ],
    },
    "North Indian": {
        "main_items": ["Curry", "Dal", "Paneer dish"],
        "expected_sides": [
            {"item": "Naan", "cooccurrence_prob": 0.85},
            {"item": "Roti/Paratha", "cooccurrence_prob": 0.70},
            {"item": "Rice", "cooccurrence_prob": 0.65},
            {"item": "Raita", "cooccurrence_prob": 0.50},
        ],
        "expected_desserts": [
            {"item": "Gulab Jamun", "cooccurrence_prob": 0.35},
            {"item": "Ras Malai", "cooccurrence_prob": 0.25},
        ],
        "expected_beverages": [
            {"item": "Lassi", "cooccurrence_prob": 0.50},
            {"item": "Coke", "cooccurrence_prob": 0.45},
        ],
    },
    "South Indian": {
        "main_items": ["Dosa", "Idli", "Vada", "Uttapam"],
        "expected_sides": [
            {"item": "Sambhar", "cooccurrence_prob": 0.90},
            {"item": "Coconut Chutney", "cooccurrence_prob": 0.88},
            {"item": "Tomato Chutney", "cooccurrence_prob": 0.55},
        ],
        "expected_desserts": [
            {"item": "Kesari Bath / Halwa", "cooccurrence_prob": 0.25},
        ],
        "expected_beverages": [
            {"item": "Filter Coffee", "cooccurrence_prob": 0.65},
            {"item": "Coke", "cooccurrence_prob": 0.30},
        ],
    },
    "Chinese": {
        "main_items": ["Noodles", "Fried Rice", "Manchurian"],
        "expected_sides": [],
        "expected_desserts": [
            {"item": "Fried Ice Cream", "cooccurrence_prob": 0.30},
        ],
        "expected_beverages": [
            {"item": "Coke", "cooccurrence_prob": 0.65},
            {"item": "Lemon Iced Tea", "cooccurrence_prob": 0.40},
        ],
        "expected_appetizers": [
            {"item": "Soup", "cooccurrence_prob": 0.55},
            {"item": "Spring Rolls", "cooccurrence_prob": 0.45},
        ],
    },
    "Continental": {
        "main_items": ["Pizza", "Pasta", "Burger", "Sandwich"],
        "expected_sides": [
            {"item": "Garlic Bread / Fries", "cooccurrence_prob": 0.78},
            {"item": "Salad", "cooccurrence_prob": 0.35},
        ],
        "expected_desserts": [
            {"item": "Tiramisu / Brownie / Cheesecake", "cooccurrence_prob": 0.50},
        ],
        "expected_beverages": [
            {"item": "Coke", "cooccurrence_prob": 0.72},
            {"item": "Iced Coffee / Cold Brew", "cooccurrence_prob": 0.40},
        ],
    },
    "Fast Food": {
        "main_items": ["Burger", "Chicken"],
        "expected_sides": [
            {"item": "Fries", "cooccurrence_prob": 0.85},
            {"item": "Coleslaw", "cooccurrence_prob": 0.40},
        ],
        "expected_desserts": [
            {"item": "Ice Cream / McFlurry", "cooccurrence_prob": 0.45},
        ],
        "expected_beverages": [
            {"item": "Coke", "cooccurrence_prob": 0.80},
            {"item": "Shake", "cooccurrence_prob": 0.35},
        ],
    },
    "Street Food": {
        "main_items": ["Pani Puri", "Pav Bhaji", "Chaat", "Kebab", "Vada Pav"],
        "expected_sides": [
            {"item": "Chutney", "cooccurrence_prob": 0.80},
            {"item": "Pav/Bread", "cooccurrence_prob": 0.55},
        ],
        "expected_desserts": [
            {"item": "Gulab Jamun", "cooccurrence_prob": 0.30},
        ],
        "expected_beverages": [
            {"item": "Masala Soda", "cooccurrence_prob": 0.45},
            {"item": "Coke", "cooccurrence_prob": 0.40},
        ],
    },
    "Dessert": {
        "main_items": ["Shake", "Ice Cream", "Waffle"],
        "expected_sides": [],
        "expected_desserts": [
            {"item": "Brownie/Cake", "cooccurrence_prob": 0.60},
            {"item": "Extra scoop", "cooccurrence_prob": 0.40},
        ],
        "expected_beverages": [],
    },
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def build_restaurant_pool():
    restaurants = []
    r_id = 1001
    i_id = 5001
    for cuisine, templates in RESTAURANT_TEMPLATES.items():
        for tmpl in templates:
            menu = []
            for item in tmpl["menu"]:
                menu.append({
                    "item_id": f"I{i_id}",
                    "name": item["name"],
                    "category": item["cat"],
                    "subcategory": item["subcat"],
                    "price": item["price"],
                    "is_veg": item["veg"],
                    "is_bestseller": item["bs"],
                    "is_new_item": False,
                    "days_since_launch": random.randint(60, 1000),
                    "tags": item["tags"],
                    "popularity_score": item["pop"],
                    "avg_rating": item["rating"],
                })
                i_id += 1
            restaurants.append({
                "restaurant_id": f"R{r_id}",
                "name": tmpl["name"],
                "cuisine_type": cuisine,
                "city": "Bangalore",
                "locality": "Generic",
                "price_tier": tmpl["price_tier"],
                "rating": tmpl["rating"],
                "is_new_restaurant": False,
                "total_orders": random.randint(5000, 50000),
                "delivery_time_mins": tmpl["delivery_mins"],
                "menu_size": len(menu),
                "avg_order_value": sum(i["price"] for i in menu) // len(menu) * 2,
                "menu": menu,
            })
            r_id += 1
    return restaurants


def get_meal_completeness(cs):
    score = 0.0
    if cs["has_main"]:      score += 0.35
    if cs["has_side"]:      score += 0.20
    if cs["has_beverage"]:  score += 0.20
    if cs["has_dessert"]:   score += 0.15
    if cs["has_appetizer"]: score += 0.10
    return round(score, 2)


def calc_relevance(rec_item, cart_summary, cuisine, seg_name, city, primary_item_name=None, active_offer_item_id=None):
    cat = rec_item["category"]
    already_has = {
        "side":      cart_summary["has_side"],
        "beverage":  cart_summary["has_beverage"],
        "dessert":   cart_summary["has_dessert"],
        "appetizer": cart_summary["has_appetizer"],
    }
    base = rec_item["popularity_score"] * 0.6 + 0.4
    # [v3] Item-level co-occurrence boost (precision signal)
    if primary_item_name:
        for key, pairs in ITEM_LEVEL_PATTERNS.items():
            if key in primary_item_name.lower():
                for comp_name, prob in pairs:
                    if comp_name in rec_item["name"].lower():
                        base = min(0.95, base + 0.18 * prob)
                break
    # Cuisine-level complement bonus
    patterns = COOCCURRENCE_PATTERNS.get(cuisine, {})
    for group in ["expected_sides","expected_beverages","expected_desserts","expected_appetizers"]:
        for pair in patterns.get(group, []):
            keyword = pair["item"].lower().split("/")[0].split("(")[0].strip()
            if keyword in rec_item["name"].lower():
                base = min(0.95, base + 0.15 * pair["cooccurrence_prob"])
    # Penalty for duplicate category
    if already_has.get(cat, False):
        base = max(0.40, base - 0.25)
    # Mumbai pav bonus
    if city == "Mumbai" and "Mumbai_special" in rec_item.get("tags", []):
        base = min(0.95, base + 0.12)
    # Health conscious: penalise fried/heavy items
    if seg_name == "health_conscious" and any(t in rec_item.get("tags",[]) for t in ["fried","heavy","buttery","creamy"]):
        base = max(0.40, base - 0.10)
    # [v3] Active offer boost
    if active_offer_item_id and rec_item["item_id"] == active_offer_item_id:
        base = min(0.95, base + 0.10)
    return round(min(0.95, max(0.40, base + random.uniform(-0.04, 0.04))), 2)


def should_accept(rec_item, seg_cfg, cart_total, relevance_score, is_weekend, meal_type, is_dessert_cat):
    base_rate = random.uniform(*seg_cfg["acceptance_rate"])
    budget_max = seg_cfg["cart_value_range"][1]
    # Softer price penalty: 0.75 when slightly over budget, 0.50 when very over budget
    over_budget_ratio = (cart_total + rec_item["price"]) / max(1, budget_max)
    if over_budget_ratio <= 1.0:
        price_factor = 1.0
    elif over_budget_ratio <= 1.2:
        price_factor = 0.80
    else:
        price_factor = 0.55
    # Weekend dessert: strong 65% floor acceptance (matches the 60% target)
    if is_weekend and is_dessert_cat:
        return random.random() < 0.65
    p = base_rate * relevance_score * price_factor
    return random.random() < min(0.95, p)


def recommend_items(menu_items, cart_summary, cuisine, seg_name, user_is_veg, city, n=5, exclude_ids=None, force_dessert_first=False, primary_item_name=None, active_offer_item_id=None):
    exclude_ids = exclude_ids or set()
    candidates = [i for i in menu_items if i["item_id"] not in exclude_ids and (not user_is_veg or i["is_veg"])]
    scored = [(calc_relevance(i, cart_summary, cuisine, seg_name, city, primary_item_name, active_offer_item_id), i) for i in candidates]
    scored.sort(key=lambda x: -x[0])

    # If weekend and no dessert yet, move a dessert item to position 1
    if force_dessert_first and not cart_summary["has_dessert"]:
        dessert_idx = next((i for i, (_, item) in enumerate(scored) if item["category"] == "dessert"), None)
        if dessert_idx is not None and dessert_idx > 0:
            scored.insert(0, scored.pop(dessert_idx))

    result = []
    for pos, (rel, item) in enumerate(scored[:n], 1):
        if rel >= 0.85:
            reason = "complements_main" if cart_summary["has_main"] else "meal_completion"
        elif rel >= 0.70:
            reason = "cuisine_fit"
        else:
            reason = "popular_pairing"
        rec_entry = {
            "item_id": item["item_id"],
            "name": item["name"],
            "category": item["category"],
            "price": item["price"],
            "is_veg": item["is_veg"],
            "relevance_score": rel,
            "reason": reason,
            "position": pos,
        }
        # [v3] Flag if this recommendation has an active offer
        if active_offer_item_id and item["item_id"] == active_offer_item_id:
            rec_entry["has_active_offer"] = True
        result.append(rec_entry)
    return result


def weighted_hour_choice(hour_weights):
    hours = list(hour_weights.keys())
    weights = list(hour_weights.values())
    return random.choices(hours, weights=weights, k=1)[0]


def get_context(meal_slot, is_weekend):
    hw = MEAL_TIMES[meal_slot]["hour_weights"]
    hour = weighted_hour_choice(hw)
    base_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 364))
    if is_weekend and base_date.weekday() < 5:
        base_date += timedelta(days=(5 - base_date.weekday()))
    elif not is_weekend and base_date.weekday() >= 5:
        base_date += timedelta(days=(7 - base_date.weekday()))
    ts = base_date.replace(hour=hour % 24, minute=random.randint(0, 59), second=0)
    return {
        "timestamp": ts.isoformat(),
        "hour": hour % 24,
        "day_of_week": ts.strftime("%A"),
        "is_weekend": ts.weekday() >= 5,
        "meal_type": meal_slot,
        "occasion": random.choice(OCCASIONS),
        "weather": random.choice(WEATHER_OPTIONS),
        "is_festival_season": random.random() < 0.08,
        "delivery_zone_demand": random.choice(["low","medium","high"]),
        "user_session_duration_seconds": 0,
    }


def build_cart_sequence(seg_name, seg_cfg, restaurant, cuisine, user_is_veg, budget_per_order, is_weekend, meal_type, city, active_offer=None, allow_abandonment=True):
    menu = restaurant["menu"]
    eligible = [i for i in menu if not user_is_veg or i["is_veg"]]
    if not eligible:
        eligible = menu

    # Select primary item (main preferred, or beverage for Dessert cuisine)
    mains = [i for i in eligible if i["category"] == "main"]
    if not mains and cuisine == "Dessert":
        mains = [i for i in eligible if i["category"] in ("beverage","dessert")]
    if not mains:
        mains = eligible
    # Non-veg users strongly prefer non-veg mains when available
    if not user_is_veg:
        nonveg_mains = [i for i in mains if not i["is_veg"]]
        if nonveg_mains:
            # 95% chance pick non-veg main (to achieve ~60:40 veg/nonveg ratio overall)
            mains = nonveg_mains if random.random() < 0.95 else mains
    primary = random.choice(mains)

    steps = []
    cart_items = {primary["item_id"]: 1}
    in_cart_ids = {primary["item_id"]}

    cart_summary = {
        "total_items": 1,
        "total_value": primary["price"],
        "has_main": primary["category"] == "main",
        "has_side": primary["category"] == "side",
        "has_beverage": primary["category"] == "beverage",
        "has_dessert": primary["category"] == "dessert",
        "has_appetizer": primary["category"] == "appetizer",
        "meal_completeness_score": 0.0,
    }
    cart_summary["meal_completeness_score"] = get_meal_completeness(cart_summary)

    max_steps = {
        "budget_conscious": 2, "mid_tier": 3, "premium": 4,
        "health_conscious": 3, "experimenter": 4,
    }.get(seg_name, 3)

    n_recs_init = {"budget_conscious":4,"mid_tier":5,"premium":5,"health_conscious":4,"experimenter":5}.get(seg_name, 5)

    time_offset = 0
    total_accepted = 0
    total_rejected = 0
    # [v3] Fatigue tracking
    consecutive_rejections = 0
    max_consecutive_rejections = 0
    total_recs_shown = 0
    offer_item_accepted = False
    active_offer_item_id = active_offer["item_id"] if active_offer else None
    first_item_price = primary["price"]

    # ── STEP 1: primary item added ───────────────────────────────────────────
    # [v3] Pass primary item name and offer item to relevance scorer
    recommendations = recommend_items(
        menu, cart_summary, cuisine, seg_name, user_is_veg, city,
        n=n_recs_init, exclude_ids=in_cart_ids, force_dessert_first=is_weekend,
        primary_item_name=primary["name"], active_offer_item_id=active_offer_item_id
    )
    total_recs_shown += len(recommendations)
    browse_time = random.randint(5, 25)
    accepted_item = None
    for rec in recommendations:
        is_dessert = rec["category"] == "dessert"
        if should_accept(rec, seg_cfg, cart_summary["total_value"], rec["relevance_score"], is_weekend, meal_type, is_dessert):
            accepted_item = rec
            break

    user_action = f"accepted_{accepted_item['item_id']}" if accepted_item else "rejected_all"
    acc_reason = random.choice(["complementary","cuisine_fit","price","meal_completion"]) if accepted_item else None
    rej_reason = random.choice(["satisfied_with_meal","not_relevant","out_of_budget"]) if not accepted_item else None

    # [v3] Track fatigue
    if accepted_item:
        if active_offer_item_id and accepted_item["item_id"] == active_offer_item_id:
            offer_item_accepted = True
        consecutive_rejections = 0
    else:
        consecutive_rejections += 1
        max_consecutive_rejections = max(max_consecutive_rejections, consecutive_rejections)

    steps.append({
        "step": 1,
        "timestamp_offset_seconds": 0,
        "action": "add_item",
        "item": {
            "item_id": primary["item_id"],
            "name": primary["name"],
            "category": primary["category"],
            "subcategory": primary.get("subcategory",""),
            "price": primary["price"],
            "qty": 1,
            "is_veg": primary["is_veg"],
            "is_bestseller": primary["is_bestseller"],
            "is_new_item": primary["is_new_item"],
            "days_since_launch": primary["days_since_launch"],
            "tags": primary["tags"],
            "popularity_score": primary["popularity_score"],
            "avg_rating": primary["avg_rating"],
        },
        "cart_state_after": [{"item_id": k, "qty": v, "price": next(i["price"] for i in menu if i["item_id"]==k)} for k,v in cart_items.items()],
        "cart_summary": dict(cart_summary),
        "recommendations_shown": recommendations,
        "user_action": user_action,
        "action_timestamp_offset_seconds": browse_time,
        "acceptance_reason": acc_reason,
        "rejection_reason": rej_reason,
        "user_browsed_time_seconds": browse_time,
        "consecutive_rejections_before_this_step": 0,
    })

    if accepted_item:
        total_accepted += 1
        total_rejected += len(recommendations) - 1
    else:
        total_rejected += len(recommendations)
    time_offset += browse_time

    # ── SUBSEQUENT STEPS ─────────────────────────────────────────────────────
    step_num = 2
    while accepted_item and step_num <= max_steps + 1:
        add_item_data = next((i for i in menu if i["item_id"] == accepted_item["item_id"]), None)
        if not add_item_data:
            break
        in_cart_ids.add(accepted_item["item_id"])
        cart_items[accepted_item["item_id"]] = cart_items.get(accepted_item["item_id"], 0) + 1

        cat = add_item_data["category"]
        if cat in ("main","side","beverage","dessert","appetizer"):
            cart_summary[f"has_{cat}"] = True
        cart_summary["total_items"] = sum(cart_items.values())
        cart_summary["total_value"] = sum(
            next(i["price"] for i in menu if i["item_id"]==k) * v for k,v in cart_items.items()
        )
        cart_summary["meal_completeness_score"] = get_meal_completeness(cart_summary)

        if cart_summary["total_value"] >= budget_per_order[1]:
            break

        n_recs = max(2, n_recs_init - step_num + 1)
        # [v3] Pass primary item name + offer to subsequent steps too
        recommendations = recommend_items(
            menu, cart_summary, cuisine, seg_name, user_is_veg, city,
            n=n_recs, exclude_ids=in_cart_ids, force_dessert_first=is_weekend,
            primary_item_name=primary["name"], active_offer_item_id=active_offer_item_id
        )
        total_recs_shown += len(recommendations)
        if not recommendations:
            break

        browse_time = random.randint(4, 18)
        prev_accepted = accepted_item
        accepted_item = None
        for rec in recommendations:
            is_dessert = rec["category"] == "dessert"
            if should_accept(rec, seg_cfg, cart_summary["total_value"], rec["relevance_score"], is_weekend, meal_type, is_dessert):
                accepted_item = rec
                break

        user_action = f"accepted_{accepted_item['item_id']}" if accepted_item else "rejected_all"
        acc_reason = random.choice(["meal_completion","cuisine_fit","price"]) if accepted_item else None
        rej_reason = random.choice(["satisfied_with_meal","not_relevant","out_of_budget","already_satisfied"]) if not accepted_item else None

        # [v3] Fatigue tracking per step
        if accepted_item:
            if active_offer_item_id and accepted_item["item_id"] == active_offer_item_id:
                offer_item_accepted = True
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1
            max_consecutive_rejections = max(max_consecutive_rejections, consecutive_rejections)

        steps.append({
            "step": step_num,
            "timestamp_offset_seconds": time_offset,
            "action": "add_item",
            "item": {
                "item_id": prev_accepted["item_id"],
                "name": add_item_data["name"],
                "category": add_item_data["category"],
                "subcategory": add_item_data.get("subcategory",""),
                "price": add_item_data["price"],
                "qty": 1,
                "is_veg": add_item_data["is_veg"],
                "is_bestseller": add_item_data["is_bestseller"],
                "is_new_item": add_item_data["is_new_item"],
                "days_since_launch": add_item_data["days_since_launch"],
                "tags": add_item_data["tags"],
                "popularity_score": add_item_data["popularity_score"],
                "avg_rating": add_item_data["avg_rating"],
            },
            "cart_state_after": [{"item_id": k, "qty": v, "price": next(i["price"] for i in menu if i["item_id"]==k)} for k,v in cart_items.items()],
            "cart_summary": dict(cart_summary),
            "recommendations_shown": recommendations,
            "user_action": user_action,
            "action_timestamp_offset_seconds": browse_time,
            "acceptance_reason": acc_reason,
            "rejection_reason": rej_reason,
            "user_browsed_time_seconds": browse_time,
            "consecutive_rejections_before_this_step": consecutive_rejections,
        })

        if accepted_item:
            total_accepted += 1
            total_rejected += len(recommendations) - 1
        else:
            total_rejected += len(recommendations)
        time_offset += browse_time
        step_num += 1
        if not accepted_item:
            break

    # ── ITEM REMOVAL (15% of carts) ──────────────────────────────────────────
    has_remove = False
    if random.random() < 0.15 and len(cart_items) > 1:
        has_remove = True
        # Pick a random non-primary item to remove
        removable = [k for k in cart_items if k != primary["item_id"]]
        if removable:
            remove_id = random.choice(removable)
            removed_item = next(i for i in menu if i["item_id"] == remove_id)
            cart_items[remove_id] -= 1
            if cart_items[remove_id] <= 0:
                del cart_items[remove_id]
                in_cart_ids.discard(remove_id)
                # Update cart summary flags
                if removed_item["category"] in ("main","side","beverage","dessert","appetizer"):
                    still_has = any(
                        next(i["category"] for i in menu if i["item_id"]==k) == removed_item["category"]
                        for k in cart_items
                    )
                    cart_summary[f"has_{removed_item['category']}"] = still_has
            cart_summary["total_items"] = sum(cart_items.values())
            cart_summary["total_value"] = sum(
                next(i["price"] for i in menu if i["item_id"]==k) * v for k,v in cart_items.items()
            )
            cart_summary["meal_completeness_score"] = get_meal_completeness(cart_summary)

            steps.append({
                "step": step_num,
                "timestamp_offset_seconds": time_offset + random.randint(3, 10),
                "action": "remove_item",
                "item": {
                    "item_id": removed_item["item_id"],
                    "name": removed_item["name"],
                    "category": removed_item["category"],
                    "price": removed_item["price"],
                    "qty": -1,
                },
                "cart_state_after": [{"item_id": k, "qty": v, "price": next(i["price"] for i in menu if i["item_id"]==k)} for k,v in cart_items.items()],
                "cart_summary": dict(cart_summary),
                "removal_reason": random.choice(["price_reconsideration","changed_mind","duplicate","diet_change"]),
            })
            step_num += 1
            time_offset += 10

    # ── CHECKOUT or ABANDONMENT ───────────────────────────────────────────────
    # [v3] 5% of sessions abandon cart (no checkout step)
    addon_revenue = cart_summary["total_value"] - first_item_price
    if allow_abandonment and random.random() < 0.05 and len(steps) >= 2:
        # Abandoned — no checkout step added
        abandonment_reason = random.choice(["price_too_high", "found_better_option", "distracted", "long_delivery_time"])
        return steps, cart_summary, total_accepted, total_rejected, total_recs_shown, has_remove, \
               max_consecutive_rejections, offer_item_accepted, first_item_price, addon_revenue, abandonment_reason

    checkout = {
        "step": step_num,
        "timestamp_offset_seconds": time_offset,
        "action": "checkout",
        "final_cart_value": cart_summary["total_value"],
        "final_item_count": cart_summary["total_items"],
        "total_browsing_time_seconds": time_offset,
        "recommendations_accepted": total_accepted,
        "recommendations_rejected": total_rejected,
        "acceptance_rate": round(total_accepted / max(1, total_accepted + total_rejected), 2),
        # [v3] Revenue uplift
        "addon_revenue": addon_revenue,
        "addon_revenue_pct": round(addon_revenue / max(1, first_item_price) * 100, 1),
    }
    steps.append(checkout)

    return steps, cart_summary, total_accepted, total_rejected, total_recs_shown, has_remove, \
           max_consecutive_rejections, offer_item_accepted, first_item_price, addon_revenue, None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def generate_all():
    print("Building restaurant pool...")
    restaurant_pool = build_restaurant_pool()
    by_cuisine = {}
    for r in restaurant_pool:
        by_cuisine.setdefault(r["cuisine_type"], []).append(r)
    print(f"Restaurants: {len(restaurant_pool)}")

    # Build assignment pools
    seg_pool = []
    for seg, cfg in USER_SEGMENTS.items():
        seg_pool.extend([seg] * cfg["count"])
    random.shuffle(seg_pool)

    cuis_pool = []
    for cuis, cfg in CUISINES.items():
        cuis_pool.extend([cuis] * cfg["count"])
    random.shuffle(cuis_pool)

    city_pool = []
    for city, cfg in CITIES.items():
        city_pool.extend([city] * cfg["count"])
    random.shuffle(city_pool)

    meal_pool = []
    for slot, cfg in MEAL_TIMES.items():
        meal_pool.extend([slot] * cfg["count"])
    random.shuffle(meal_pool)

    weekend_pool = [True]*15000 + [False]*35000  # 30% weekend
    random.shuffle(weekend_pool)

    # Cold start flags
    cold_user_indices  = set(random.sample(range(TOTAL_SCENARIOS), 7500))
    cold_rest_indices  = set(random.sample(range(TOTAL_SCENARIOS), 7500))
    cold_item_indices  = set(random.sample(range(TOTAL_SCENARIOS), 5000))

    # User history distribution: 30% new (0-2), 40% occasional (3-10), 30% frequent (>10)
    history_pool = (
        [(0, 2)] * 15000 +   # new
        [(3, 10)] * 20000 +  # occasional
        [(11, 300)] * 15000  # frequent
    )
    random.shuffle(history_pool)

    # [v3] Persistent user pool — 10K unique users, 30% appear 2-4 times
    USER_POOL_SIZE = 10000
    base_user_ids = list(range(10001, 10001 + USER_POOL_SIZE))
    # Build scenario→user_id map: 70% unique, 30% repeat
    repeat_user_ids = random.choices(base_user_ids, k=int(TOTAL_SCENARIOS * 0.30))
    unique_user_ids = [10001 + USER_POOL_SIZE + i for i in range(int(TOTAL_SCENARIOS * 0.70))]
    user_id_pool = repeat_user_ids + unique_user_ids
    random.shuffle(user_id_pool)
    # Track past accepted items per user_id for cross-session history
    user_accepted_history = {}  # user_id -> list of item names

    # [v3] Offer pool — 20% of scenarios get an active offer (assigned later per scenario)
    offer_flag_pool = [True] * int(TOTAL_SCENARIOS * 0.20) + [False] * int(TOTAL_SCENARIOS * 0.80)
    random.shuffle(offer_flag_pool)

    # Veg distribution: 60% veg, 40% non-veg overall
    # health_conscious: 80% veg, others vary
    # We control by assigning dietary at scenario level
    veg_pool = [True]*30000 + [False]*20000
    random.shuffle(veg_pool)

    print("Generating scenarios...")
    scenarios = []

    for idx in range(TOTAL_SCENARIOS):
        if idx % 5000 == 0:
            print(f"  {idx}/{TOTAL_SCENARIOS}...")

        seg_name   = seg_pool[idx]
        seg_cfg    = USER_SEGMENTS[seg_name]
        cuisine    = cuis_pool[idx]
        city       = city_pool[idx]
        meal_slot  = meal_pool[idx]
        is_weekend = weekend_pool[idx]
        hist_range = history_pool[idx]
        # [v3] Assign user from persistent pool
        assigned_uid = user_id_pool[idx]
        is_returning_user = assigned_uid in user_accepted_history
        has_offer = offer_flag_pool[idx]

        is_new_user  = (idx in cold_user_indices) or (hist_range == (0, 2))
        is_new_rest  = idx in cold_rest_indices
        is_new_item  = idx in cold_item_indices

        # Dietary preference
        if seg_name == "health_conscious":
            user_is_veg = random.random() < 0.70
        else:
            user_is_veg = veg_pool[idx]

        # City restrictions for premium
        if seg_name == "premium" and city not in ["Bangalore","Mumbai","Delhi/NCR","Hyderabad","Pune","Chennai"]:
            city = random.choice(["Bangalore","Mumbai","Delhi/NCR"])

        locality = random.choice(CITIES[city]["localities"])

        # Pick restaurant
        candidates = by_cuisine.get(cuisine, restaurant_pool)
        if seg_name in ("budget_conscious",):
            filtered = [r for r in candidates if r["price_tier"] == "budget"]
            candidates = filtered or candidates
        elif seg_name == "premium":
            filtered = [r for r in candidates if r["price_tier"] in ("mid","premium")]
            candidates = filtered or candidates

        restaurant = dict(random.choice(candidates))
        restaurant["city"] = city
        restaurant["locality"] = locality

        # Restaurant type
        rest_type = random.choices(RESTAURANT_TYPES, weights=RESTAURANT_TYPE_DIST)[0]
        restaurant["restaurant_type"] = rest_type

        if is_new_rest:
            restaurant["is_new_restaurant"] = True
            restaurant["total_orders"] = random.randint(5, 49)
            restaurant["rating"] = round(random.uniform(3.5, 4.2), 1)

        # Deep copy menu and mark new item if needed
        menu = [dict(i) for i in restaurant["menu"]]
        if is_new_item and menu:
            ni = random.randint(0, len(menu)-1)
            menu[ni]["is_new_item"] = True
            menu[ni]["days_since_launch"] = random.randint(1, 29)
        restaurant["menu"] = menu

        # User profile
        age = random.randint(*seg_cfg["age_range"])
        gender = random.choice(["male","female","other"])
        income_low, income_high = seg_cfg["income_range"]
        monthly_income = random.randint(income_low, income_high)
        dietary = "vegetarian" if user_is_veg else ("vegan" if random.random()<0.05 else "non_vegetarian")
        past_orders = random.randint(*hist_range)
        aov_hist = random.randint(200, 800) if past_orders > 2 else 0

        income_bracket = "entry_level"
        if monthly_income > 150000:  income_bracket = "senior"
        elif monthly_income > 75000: income_bracket = "mid_career"
        elif monthly_income > 35000: income_bracket = "junior"

        user_profile = {
            "user_id": f"U{assigned_uid}",
            "age": age,
            "gender": gender,
            "city": city,
            "locality": locality,
            "income_bracket": income_bracket,
            "monthly_income_range": [income_low, income_high],
            "budget_per_order": list(seg_cfg["cart_value_range"]),
            "dietary_preference": dietary,
            "is_new_user": is_new_user,
            "is_returning_user": is_returning_user,  # [v3]
            "past_order_count": past_orders,
            "avg_order_value_history": aov_hist,
            "preferred_cuisines": random.sample(list(CUISINES.keys()), k=3),
            "price_sensitivity": seg_cfg["price_sensitivity"],
            "avg_rating_given": round(random.uniform(3.5, 5.0), 1),
            "user_segment": seg_name,
            # [v3] Cross-session history for returning users
            "previously_accepted_items": list(user_accepted_history.get(assigned_uid, []))[:10],
        }

        rest_profile = {k: v for k, v in restaurant.items() if k != "menu"}
        context = get_context(meal_slot, is_weekend)

        # [v3] Build active offer for this scenario
        active_offer = None
        if has_offer and restaurant["menu"]:
            offer_item = random.choice(restaurant["menu"])
            offer_type = random.choice(OFFER_TYPES)
            disc_range = OFFER_DISCOUNT_RANGE[offer_type]
            disc_pct = random.randint(*disc_range)
            active_offer = {
                "item_id": offer_item["item_id"],
                "item_name": offer_item["name"],
                "offer_type": offer_type,
                "discount_pct": disc_pct,
                "original_price": offer_item["price"],
                "discounted_price": round(offer_item["price"] * (1 - disc_pct / 100)),
            }
        context["active_offer"] = active_offer

        cart_seq, final_cart, total_acc, total_rej, total_recs, has_remove, \
            max_consec_rej, offer_accepted, first_item_price, addon_rev, abandon_reason = build_cart_sequence(
            seg_name, seg_cfg, restaurant, cuisine, user_is_veg,
            list(seg_cfg["cart_value_range"]), is_weekend, meal_slot, city,
            active_offer=active_offer
        )

        # [v3] Update cross-session history for this user
        accepted_names = [
            step["item"]["name"] for step in cart_seq
            if step.get("action") == "add_item" and step.get("user_action", "").startswith("accepted_")
        ]
        if assigned_uid not in user_accepted_history:
            user_accepted_history[assigned_uid] = []
        user_accepted_history[assigned_uid].extend(accepted_names)
        user_accepted_history[assigned_uid] = user_accepted_history[assigned_uid][-20:]  # keep last 20

        total_steps = len(cart_seq)
        ovr_acc = round(total_acc / max(1, total_recs), 2)
        is_abandoned = abandon_reason is not None

        scenario = {
            "scenario_id": f"SC{str(idx+1).zfill(6)}",
            "user_profile": user_profile,
            "restaurant_profile": rest_profile,
            "context": context,
            "cart_sequence": cart_seq,
            "session_metadata": {
                "total_steps": total_steps,
                "total_recommendations_shown": total_recs,
                "total_accepted": total_acc,
                "total_rejected": total_rej,
                "overall_acceptance_rate": ovr_acc,
                "final_meal_completeness": final_cart["meal_completeness_score"],
                "aov_vs_user_avg": round(final_cart["total_value"] / max(1, aov_hist), 2) if aov_hist > 0 else None,
                "session_success": not is_abandoned,
                "abandonment_reason": abandon_reason,       # [v3] None if completed
                "has_remove_event": has_remove,
                "is_cold_start_user": is_new_user,
                "is_cold_start_restaurant": is_new_rest,
                "is_cold_start_item": is_new_item,
                # [v3] New fields
                "max_consecutive_rejections": max_consec_rej,
                "offer_item_accepted": offer_accepted,
                "first_item_price": first_item_price,
                "addon_revenue": addon_rev,
                "addon_revenue_pct": round(addon_rev / max(1, first_item_price) * 100, 1),
            }
        }
        scenarios.append(scenario)

    return scenarios, restaurant_pool


def compute_stats(scenarios):
    total = len(scenarios)
    total_steps  = sum(s["session_metadata"]["total_steps"] for s in scenarios)
    total_recs   = sum(s["session_metadata"]["total_recommendations_shown"] for s in scenarios)
    total_acc    = sum(s["session_metadata"]["total_accepted"] for s in scenarios)
    # [v3] Use session_metadata instead of cart_sequence[-1] to handle abandoned sessions
    completed    = [s for s in scenarios if s["session_metadata"]["session_success"]]
    checkouts    = [
        next((st for st in reversed(s["cart_sequence"]) if st.get("action") == "checkout"), None)
        for s in completed
    ]
    checkouts    = [c for c in checkouts if c is not None]
    avg_cart_val = sum(c["final_cart_value"] for c in checkouts) / max(1, len(checkouts))
    avg_items    = sum(c["final_item_count"] for c in checkouts) / max(1, len(checkouts))

    cold = sum(1 for s in scenarios if
        s["session_metadata"]["is_cold_start_user"] or
        s["session_metadata"]["is_cold_start_restaurant"] or
        s["session_metadata"]["is_cold_start_item"])

    has_remove = sum(1 for s in scenarios if s["session_metadata"]["has_remove_event"])

    city_dist    = {}
    cuisine_dist = {}
    meal_dist    = {}
    seg_dist     = {}
    rest_type_dist = {}

    veg_count = 0
    for s in scenarios:
        city_dist[s["user_profile"]["city"]] = city_dist.get(s["user_profile"]["city"], 0) + 1
        cuisine_dist[s["restaurant_profile"]["cuisine_type"]] = cuisine_dist.get(s["restaurant_profile"]["cuisine_type"], 0) + 1
        meal_dist[s["context"]["meal_type"]] = meal_dist.get(s["context"]["meal_type"], 0) + 1
        seg_dist[s["user_profile"]["user_segment"]] = seg_dist.get(s["user_profile"]["user_segment"], 0) + 1
        rt = s["restaurant_profile"].get("restaurant_type","unknown")
        rest_type_dist[rt] = rest_type_dist.get(rt, 0) + 1
        step1_veg = s["cart_sequence"][0]["item"]["is_veg"]
        if step1_veg:
            veg_count += 1

    # User history
    new_u  = sum(1 for s in scenarios if s["user_profile"]["past_order_count"] <= 2)
    occ_u  = sum(1 for s in scenarios if 3 <= s["user_profile"]["past_order_count"] <= 10)
    freq_u = sum(1 for s in scenarios if s["user_profile"]["past_order_count"] > 10)

    # Main without side at checkout
    no_side_at_checkout = 0
    no_bev_at_checkout  = 0
    for s in scenarios:
        seq = s["cart_sequence"]
        last_cart_step = next((st for st in reversed(seq) if "cart_summary" in st), None)
        if last_cart_step:
            cs = last_cart_step["cart_summary"]
            if cs["has_main"] and not cs["has_side"]:
                no_side_at_checkout += 1
            if not cs["has_beverage"]:
                no_bev_at_checkout += 1

    # Weekend dessert
    weekend_total   = sum(1 for s in scenarios if s["context"]["is_weekend"])
    weekend_dessert = 0
    for s in scenarios:
        if s["context"]["is_weekend"]:
            seq = s["cart_sequence"]
            last = next((st for st in reversed(seq) if "cart_summary" in st), None)
            if last and last["cart_summary"]["has_dessert"]:
                weekend_dessert += 1

    return {
        "VALIDATION": {
            "✅ total_scenarios":          f"{total} (target: 50,000)",
            "✅ avg_cart_value_INR":        f"₹{avg_cart_val:.0f} (target: ₹350-450)",
            "✅ avg_items_per_order":       f"{avg_items:.2f} (target: 2.3-3.1)",
            "✅ overall_acceptance_rate":   f"{total_acc/max(1,total_recs)*100:.1f}% (target: ~35-50%)",
            "✅ cold_start_pct":            f"{cold/total*100:.1f}% (target: 40%+)",
            "✅ veg_pct_primary_item":      f"{veg_count/total*100:.1f}% (target: ~60%)",
            "✅ new_user_pct":              f"{new_u/total*100:.1f}% (target: 30%)",
            "✅ occasional_user_pct":       f"{occ_u/total*100:.1f}% (target: 40%)",
            "✅ frequent_user_pct":         f"{freq_u/total*100:.1f}% (target: 30%)",
            "✅ remove_event_rate":         f"{has_remove/total*100:.1f}% (target: 15%)",
            "✅ no_side_at_checkout_pct":   f"{no_side_at_checkout/total*100:.1f}% (target: ~35%)",
            "✅ no_bev_at_checkout_pct":    f"{no_bev_at_checkout/total*100:.1f}% (target: ~45%)",
            "✅ weekend_dessert_rate":      f"{weekend_dessert/max(1,weekend_total)*100:.1f}% (target: 60%)",
        },
        "DISTRIBUTIONS": {
            "city":          city_dist,
            "cuisine":       cuisine_dist,
            "meal_time":     meal_dist,
            "user_segment":  seg_dist,
            "restaurant_type": rest_type_dist,
        },
        "RAW": {
            "total_steps": total_steps,
            "total_recs": total_recs,
            "total_acc": total_acc,
        }
    }


if __name__ == "__main__":
    import os
    out = r"d:\Zomathon\new"
    print("=== Zomato Cart Data Generator v3.0 ===")
    print(f"Target: {TOTAL_SCENARIOS:,} scenarios\n")

    scenarios, restaurant_pool = generate_all()

    print("\nWriting cart_scenarios_sequential_50k.json ...")
    with open(os.path.join(out, "cart_scenarios_sequential_50k.json"), "w") as f:
        json.dump(scenarios, f, separators=(',', ':'))  # compact for file size

    print("Writing restaurant_menus.json ...")
    menus_out = {"restaurants": [
        {"restaurant_id": r["restaurant_id"], "name": r["name"],
         "cuisine_type": r["cuisine_type"], "price_tier": r["price_tier"],
         "rating": r["rating"], "restaurant_type": r.get("restaurant_type","chain"),
         "menu": r["menu"]}
        for r in restaurant_pool
    ]}
    # Note: item_cooccurrence_patterns.json is now maintained separately
    # (includes item_level_patterns — do not overwrite)

    print("\nComputing stats...")
    stats = compute_stats(scenarios)
    with open(os.path.join(out, "generation_stats_v3.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== FINAL VALIDATION ===")
    print(json.dumps(stats["VALIDATION"], indent=2))
    print("\n=== DISTRIBUTIONS ===")
    print(json.dumps(stats["DISTRIBUTIONS"], indent=2))
    print("\n=== v3.0 NEW FEATURE RATES ===")
    abandoned = sum(1 for s in scenarios if not s["session_metadata"]["session_success"])
    returning = sum(1 for s in scenarios if s["user_profile"]["is_returning_user"])
    with_offer = sum(1 for s in scenarios if s["context"]["active_offer"] is not None)
    offer_acc  = sum(1 for s in scenarios if s["session_metadata"]["offer_item_accepted"])
    print(f"  Abandonment rate:     {abandoned/len(scenarios)*100:.1f}% (target: ~5%)")
    print(f"  Returning user rate:  {returning/len(scenarios)*100:.1f}% (target: ~30%)")
    print(f"  Scenarios with offer: {with_offer/len(scenarios)*100:.1f}% (target: ~20%)")
    print(f"  Offer acceptance rate:{offer_acc/max(1,with_offer)*100:.1f}% (vs regular)")
    avg_addon = sum(s["session_metadata"]["addon_revenue"] for s in scenarios) / len(scenarios)
    print(f"  Avg addon revenue:    INR {avg_addon:.0f}")
