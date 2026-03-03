"""
locustfile.py
=============
Benchmark suite for CSAO End-to-End System Architecture (Section 5.3)
Run locally: `locust -f locustfile.py --host=http://localhost:8000`
"""

import json
import random
from locust import HttpUser, task, between, events
from locust.env import Environment

CUISINES = ["North Indian", "South Indian", "Chinese", "Italian", "Fast Food"]
ITEMS = [
    {"item_id": "I1", "name": "Butter Chicken", "price": 350, "category": "main"},
    {"item_id": "I2", "name": "Naan", "price": 50, "category": "bread"},
    {"item_id": "I3", "name": "Biryani", "price": 250, "category": "main"},
    {"item_id": "I4", "name": "Coke", "price": 60, "category": "beverage"},
    {"item_id": "I5", "name": "Gulab Jamun", "price": 100, "category": "dessert"}
]

class CSAOUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def sustained_load_balanced(self):
        """Test 1: Sustained Load - Balanced Mode"""
        cart_size = random.randint(1, 3)
        cart = random.sample(ITEMS, cart_size)
        
        payload = {
            "user_id": f"U{random.randint(1, 10000)}",
            "cart_items": cart,
            "restaurant": {"restaurant_id": "R123", "cuisine_type": random.choice(CUISINES)},
            "context": {"hour": 19, "city": "Mumbai", "meal_type": "dinner"},
            "top_n": 8,
            "mode": "balanced"
        }
        
        self.client.post("/recommend", json=payload, name="/recommend (balanced)")

    @task(2)
    def spike_load_fast(self):
        """Test 2: Spike Load - Fast Mode (simulate latency-sensitive peaks)"""
        cart = random.sample(ITEMS, 2)
        payload = {
            "user_id": f"U{random.randint(1, 10000)}",
            "cart_items": cart,
            "restaurant": {"restaurant_id": "R124", "cuisine_type": "Fast Food"},
            "top_n": 10,
            "mode": "fast"
        }
        self.client.post("/recommend", json=payload, name="/recommend (fast)")

    @task(1)
    def failure_mode_quality(self):
        """Test 3: Failure Mode - Forcing fallback by requesting obscure context/quality mode"""
        payload = {
            "user_id": "U_COLD",
            "cart_items": [],
            "restaurant": {"restaurant_id": "R999", "cuisine_type": "Unknown"},
            "top_n": 5,
            "mode": "quality" # Pushes SLM to limit, increasing chance of timeout/fallback
        }
        self.client.post("/recommend", json=payload, name="/recommend (quality fallback)")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting Zomathon CSAO Load Test...")
