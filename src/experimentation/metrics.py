"""
Metrics Engine — CSAO A/B Testing
===================================
Defines every metric (primary, secondary, guardrail) used in the
experimentation framework, along with computation logic that works
on simulated or real order-event data.

Metric hierarchy:
  Primary    → AOV lift, CSAO attach rate, C2O ratio
  Secondary  → Recommendation CTR, add-to-cart rate, revenue/session
  Guardrail  → Cart abandonment, repeat-order rate, restaurant fairness
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


# ═════════════════════════════════════════════════════════════════════════════
#  METRIC DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricDefinition:
    """Schema for a single tracked metric."""
    name: str
    display_name: str
    tier: str                       # "primary" | "secondary" | "guardrail"
    unit: str                       # "ratio", "currency_inr", "percentage", "seconds"
    direction: str                  # "higher_is_better" | "lower_is_better"
    minimum_detectable_effect: float  # relative MDE (e.g. 0.02 = 2 %)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "tier": self.tier,
            "unit": self.unit,
            "direction": self.direction,
            "mde": self.minimum_detectable_effect,
            "description": self.description,
        }


# ── METRIC CATALOGUE ────────────────────────────────────────────────────────

METRIC_CATALOGUE: dict[str, MetricDefinition] = {
    # ── Primary ──────────────────────────────────────────────────────────────
    "aov": MetricDefinition(
        name="aov",
        display_name="Average Order Value",
        tier="primary",
        unit="currency_inr",
        direction="higher_is_better",
        minimum_detectable_effect=0.02,
        description="Mean order value (₹).  Target: ≥2 % lift over control.",
    ),
    "csao_attach_rate": MetricDefinition(
        name="csao_attach_rate",
        display_name="CSAO Rail Attach Rate",
        tier="primary",
        unit="ratio",
        direction="higher_is_better",
        minimum_detectable_effect=0.03,
        description="Fraction of orders containing ≥1 CSAO-recommended item.",
    ),
    "c2o_ratio": MetricDefinition(
        name="c2o_ratio",
        display_name="Cart-to-Order Ratio",
        tier="primary",
        unit="ratio",
        direction="higher_is_better",
        minimum_detectable_effect=0.01,
        description="Fraction of carts that convert to placed orders.",
    ),

    # ── Secondary ────────────────────────────────────────────────────────────
    "rec_ctr": MetricDefinition(
        name="rec_ctr",
        display_name="Recommendation CTR",
        tier="secondary",
        unit="ratio",
        direction="higher_is_better",
        minimum_detectable_effect=0.05,
        description="Click-through rate on CSAO rail items.",
    ),
    "add_to_cart_rate": MetricDefinition(
        name="add_to_cart_rate",
        display_name="Add-to-Cart Rate (CSAO)",
        tier="secondary",
        unit="ratio",
        direction="higher_is_better",
        minimum_detectable_effect=0.04,
        description="Fraction of CSAO impressions that result in an add-to-cart.",
    ),
    "revenue_per_session": MetricDefinition(
        name="revenue_per_session",
        display_name="Revenue per Session",
        tier="secondary",
        unit="currency_inr",
        direction="higher_is_better",
        minimum_detectable_effect=0.02,
        description="Total revenue (₹) divided by number of sessions.",
    ),
    "order_completion_time": MetricDefinition(
        name="order_completion_time",
        display_name="Order Completion Time",
        tier="secondary",
        unit="seconds",
        direction="lower_is_better",
        minimum_detectable_effect=0.05,
        description="Seconds from first item added to order placed.",
    ),
    "items_per_order": MetricDefinition(
        name="items_per_order",
        display_name="Items per Order",
        tier="secondary",
        unit="ratio",
        direction="higher_is_better",
        minimum_detectable_effect=0.03,
        description="Average number of distinct items per placed order.",
    ),

    # ── Guardrail ────────────────────────────────────────────────────────────
    "cart_abandonment_rate": MetricDefinition(
        name="cart_abandonment_rate",
        display_name="Cart Abandonment Rate",
        tier="guardrail",
        unit="ratio",
        direction="lower_is_better",
        minimum_detectable_effect=0.02,
        description="Must NOT increase. Fraction of carts that are abandoned.",
    ),
    "repeat_order_rate": MetricDefinition(
        name="repeat_order_rate",
        display_name="Repeat Order Rate (7-day)",
        tier="guardrail",
        unit="ratio",
        direction="higher_is_better",
        minimum_detectable_effect=0.02,
        description="Fraction of users who place ≥1 order in the next 7 days.",
    ),
    "restaurant_fairness": MetricDefinition(
        name="restaurant_fairness",
        display_name="Restaurant Recommendation Fairness",
        tier="guardrail",
        unit="ratio",
        direction="higher_is_better",
        minimum_detectable_effect=0.05,
        description=(
            "Gini coefficient of recommendation impressions across "
            "restaurants. Lower Gini = fairer. Must not worsen."
        ),
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class MetricsEngine:
    """
    Computes all experiment metrics from a list of order-event dicts.

    Each event is expected to have (at minimum):
        session_id, user_id, variant, order_placed (bool),
        cart_value, order_value, items_in_cart, csao_items_added,
        csao_impressions, csao_clicks, completion_time_s.

    Optional segment fields:
        user_segment, restaurant_type, city_tier, meal_type, time_slot.
    """

    def __init__(self) -> None:
        self.catalogue = METRIC_CATALOGUE

    # ── PUBLIC ──────────────────────────────────────────────────────────────

    def compute_all(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Compute every metric for each variant present in *events*."""
        by_variant: dict[str, list[dict]] = {}
        for e in events:
            by_variant.setdefault(e["variant"], []).append(e)

        results: dict[str, dict] = {}
        for variant, vevents in by_variant.items():
            results[variant] = self._compute_variant(vevents)
        return results

    def compute_segment_breakdown(
        self,
        events: list[dict[str, Any]],
        segment_key: str,
    ) -> dict[str, dict[str, dict]]:
        """
        Compute metrics broken down by a segment field
        (e.g. 'user_segment', 'city_tier', 'meal_type').
        Returns {segment_value: {variant: {metric: value}}}.
        """
        by_segment: dict[str, list[dict]] = {}
        for e in events:
            seg = e.get(segment_key, "unknown")
            by_segment.setdefault(seg, []).append(e)

        breakdown: dict[str, dict] = {}
        for seg, seg_events in by_segment.items():
            breakdown[seg] = self.compute_all(seg_events)
        return breakdown

    # ── VARIANT-LEVEL COMPUTATION ──────────────────────────────────────────

    def _compute_variant(self, events: list[dict]) -> dict[str, Any]:
        n = len(events)
        if n == 0:
            return {}

        placed = [e for e in events if e.get("order_placed", False)]
        n_placed = len(placed)

        # ── Primary ─────────────────────────────────────────────────────────
        order_values = [e.get("order_value", 0) for e in placed]
        aov = statistics.mean(order_values) if order_values else 0.0

        csao_orders = sum(
            1 for e in placed if e.get("csao_items_added", 0) > 0
        )
        csao_attach = csao_orders / max(n_placed, 1)

        c2o = n_placed / max(n, 1)

        # ── Secondary ──────────────────────────────────────────────────────
        total_impressions = sum(e.get("csao_impressions", 0) for e in events)
        total_clicks = sum(e.get("csao_clicks", 0) for e in events)
        rec_ctr = total_clicks / max(total_impressions, 1)

        total_added = sum(e.get("csao_items_added", 0) for e in events)
        add_to_cart_rate = total_added / max(total_impressions, 1)

        rev_per_session = sum(e.get("order_value", 0) for e in events) / max(n, 1)

        completion_times = [
            e.get("completion_time_s", 0) for e in placed if e.get("completion_time_s")
        ]
        avg_completion = (
            statistics.mean(completion_times) if completion_times else 0.0
        )

        items_counts = [e.get("items_in_cart", 0) for e in placed]
        items_per_order = statistics.mean(items_counts) if items_counts else 0.0

        # ── Guardrail ──────────────────────────────────────────────────────
        abandoned = sum(1 for e in events if not e.get("order_placed", False))
        abandon_rate = abandoned / max(n, 1)

        # Repeat-order proxy: unique users with > 1 session
        user_sessions: dict[str, int] = {}
        for e in placed:
            uid = e.get("user_id", "")
            user_sessions[uid] = user_sessions.get(uid, 0) + 1
        repeat_users = sum(1 for c in user_sessions.values() if c > 1)
        repeat_rate = repeat_users / max(len(user_sessions), 1)

        # Restaurant fairness (Gini of impressions across restaurants)
        rest_impressions: dict[str, int] = {}
        for e in events:
            rid = e.get("restaurant_id", "unknown")
            rest_impressions[rid] = rest_impressions.get(rid, 0) + e.get("csao_impressions", 1)
        gini = self._gini_coefficient(list(rest_impressions.values()))

        return {
            "n_sessions": n,
            "n_orders": n_placed,
            # Primary
            "aov": round(aov, 2),
            "csao_attach_rate": round(csao_attach, 4),
            "c2o_ratio": round(c2o, 4),
            # Secondary
            "rec_ctr": round(rec_ctr, 4),
            "add_to_cart_rate": round(add_to_cart_rate, 4),
            "revenue_per_session": round(rev_per_session, 2),
            "order_completion_time": round(avg_completion, 1),
            "items_per_order": round(items_per_order, 2),
            # Guardrail
            "cart_abandonment_rate": round(abandon_rate, 4),
            "repeat_order_rate": round(repeat_rate, 4),
            "restaurant_fairness_gini": round(gini, 4),
        }

    # ── HELPERS ─────────────────────────────────────────────────────────────

    @staticmethod
    def _gini_coefficient(values: list[int | float]) -> float:
        """Compute Gini coefficient (0 = perfect equality, 1 = max inequality)."""
        if not values or sum(values) == 0:
            return 0.0
        sorted_v = sorted(values)
        n = len(sorted_v)
        cumsum = 0.0
        weighted_sum = 0.0
        for i, v in enumerate(sorted_v):
            cumsum += v
            weighted_sum += (i + 1) * v
        total = cumsum
        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
        return max(0.0, min(1.0, gini))

    def get_metric_definitions(self, tier: str | None = None) -> list[dict]:
        """Return metric definitions, optionally filtered by tier."""
        defs = list(self.catalogue.values())
        if tier:
            defs = [d for d in defs if d.tier == tier]
        return [d.to_dict() for d in defs]
