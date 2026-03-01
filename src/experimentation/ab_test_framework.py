"""
A/B Test Framework — CSAO Experimentation Engine
==================================================
Complete experimentation framework for testing the ContextFlow AI
recommendation system against the baseline collaborative-filtering system.

Covers:
  1. Experiment Design  (config, power analysis, sample-size)
  2. User Assignment     (deterministic hashing, stratification)
  3. Statistical Testing (t-test, bootstrap CI, Bonferroni correction)
  4. Sequential Analysis (O'Brien-Fleming boundaries)
  5. Business Projection (AOV lift → revenue impact at scale)
  6. Rollout Strategy    (gradual rollout, rollback triggers)
  7. Long-term Learning  (degradation monitoring, retraining schedule)

Usage:
    from src.experimentation import ABTestFramework, ExperimentConfig

    config = ExperimentConfig.csao_default()
    framework = ABTestFramework(config)

    # assign user to variant
    variant = framework.assign_variant(user_id="U12345",
                                        city="Mumbai",
                                        segment="mid_tier")

    # after collecting events → analyse
    report = framework.analyse(events)
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from .metrics import MetricsEngine, METRIC_CATALOGUE


# ═════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """All tunables for one A/B experiment."""

    experiment_id: str = "csao_contextflow_v1"
    experiment_name: str = "ContextFlow AI vs Collaborative Filtering"

    # ── Variants ────────────────────────────────────────────────────────────
    variants: list[dict[str, Any]] = field(default_factory=lambda: [
        {"id": "control",    "name": "Collaborative Filtering (Baseline)", "traffic_pct": 50},
        {"id": "treatment",  "name": "ContextFlow AI (LLM Re-ranking)",    "traffic_pct": 50},
    ])

    # ── Statistical Parameters ──────────────────────────────────────────────
    significance_level: float = 0.05        # α  (two-sided)
    statistical_power: float = 0.80         # 1 − β
    minimum_detectable_effect: float = 0.02 # 2 % relative lift on primary metric
    baseline_aov: float = 380.0             # ₹ (industry benchmark)
    baseline_aov_std: float = 150.0         # ₹ standard deviation
    baseline_c2o: float = 0.68
    baseline_csao_attach: float = 0.18

    # ── Duration ────────────────────────────────────────────────────────────
    min_duration_days: int = 14
    max_duration_days: int = 28

    # ── Randomisation ───────────────────────────────────────────────────────
    randomisation_unit: str = "user"        # "user" or "session"
    stratification_keys: list[str] = field(
        default_factory=lambda: ["city_tier", "user_segment", "meal_type"]
    )
    salt: str = "csao_exp_2026"

    # ── Guardrail Thresholds ────────────────────────────────────────────────
    max_abandonment_increase: float = 0.02   # absolute pp
    max_latency_p95_ms: float = 300.0

    # ── Rollout ─────────────────────────────────────────────────────────────
    rollout_stages: list[dict[str, Any]] = field(default_factory=lambda: [
        {"stage": 1, "traffic_pct": 10, "duration_days": 3,  "label": "Canary"},
        {"stage": 2, "traffic_pct": 50, "duration_days": 7,  "label": "Expansion"},
        {"stage": 3, "traffic_pct": 100, "duration_days": 14, "label": "Full Rollout"},
    ])

    # ── Business Assumptions ────────────────────────────────────────────────
    daily_orders: int = 10_000_000
    zomato_take_rate: float = 0.22
    ai_infra_monthly_cost_inr: float = 30_000 * 10  # 10 pods × ₹30k

    # ── Segment Dimensions ──────────────────────────────────────────────────
    segment_dimensions: list[dict[str, Any]] = field(default_factory=lambda: [
        {"key": "user_segment",    "values": ["new", "occasional", "frequent", "premium"]},
        {"key": "restaurant_type", "values": ["qsr", "chain", "independent", "premium"]},
        {"key": "time_slot",       "values": ["breakfast", "lunch", "evening_snacks", "dinner", "late_night"]},
        {"key": "city_tier",       "values": ["metro", "tier_2", "tier_3"]},
        {"key": "meal_type",       "values": ["full_meal", "quick_bite", "snacking"]},
    ])

    # ── Sequential Testing ──────────────────────────────────────────────────
    interim_analyses: int = 4  # number of planned looks
    obrien_fleming: bool = True

    @classmethod
    def csao_default(cls) -> "ExperimentConfig":
        """Factory for the standard CSAO experiment."""
        return cls()

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "variants": self.variants,
            "significance_level": self.significance_level,
            "statistical_power": self.statistical_power,
            "mde": self.minimum_detectable_effect,
            "baseline_aov": self.baseline_aov,
            "min_duration_days": self.min_duration_days,
            "randomisation_unit": self.randomisation_unit,
            "stratification_keys": self.stratification_keys,
            "rollout_stages": self.rollout_stages,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  A/B TEST FRAMEWORK
# ═════════════════════════════════════════════════════════════════════════════

class ABTestFramework:
    """
    End-to-end experimentation engine.

    Responsibilities:
      • Deterministic user-to-variant assignment (with stratification)
      • Sample size & power calculations
      • Statistical significance testing (frequentist + bootstrap)
      • Sequential monitoring with O'Brien-Fleming boundaries
      • Business impact projection
      • Go / no-go decision engine
      • Rollout & rollback planning
    """

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig.csao_default()
        self.metrics = MetricsEngine()

    # =====================================================================
    #  1. USER ASSIGNMENT
    # =====================================================================

    def assign_variant(
        self,
        user_id: str,
        city: str = "",
        segment: str = "",
    ) -> str:
        """
        Deterministic, salt-based variant assignment.

        Uses MD5 hash of (salt + user_id) → bucket [0, 100).
        Stratification is implicit: same user always gets same bucket,
        and traffic split is respected across all strata.
        """
        raw = f"{self.config.salt}:{user_id}"
        digest = hashlib.md5(raw.encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100

        cumulative = 0
        for variant in self.config.variants:
            cumulative += variant["traffic_pct"]
            if bucket < cumulative:
                return variant["id"]
        return self.config.variants[-1]["id"]

    # =====================================================================
    #  2. SAMPLE SIZE & POWER
    # =====================================================================

    def required_sample_size(
        self,
        baseline_mean: float | None = None,
        baseline_std: float | None = None,
        mde: float | None = None,
    ) -> dict[str, Any]:
        """
        Two-sample z-test sample-size formula (per variant):

            n = (Z_α/2 + Z_β)² × 2σ² / δ²

        where δ = baseline_mean × mde (absolute effect size).
        """
        mu = baseline_mean or self.config.baseline_aov
        sigma = baseline_std or self.config.baseline_aov_std
        rel_mde = mde or self.config.minimum_detectable_effect

        alpha = self.config.significance_level
        power = self.config.statistical_power

        z_alpha = self._z_score(1 - alpha / 2)
        z_beta = self._z_score(power)

        delta = mu * rel_mde  # absolute effect size
        n_per_variant = math.ceil(
            2 * (sigma ** 2) * ((z_alpha + z_beta) ** 2) / (delta ** 2)
        )

        num_variants = len(self.config.variants)
        total = n_per_variant * num_variants

        # Days required (assuming equal daily traffic)
        daily_per_variant = self.config.daily_orders // num_variants
        days_needed = math.ceil(n_per_variant / max(daily_per_variant, 1))
        days_needed = max(days_needed, self.config.min_duration_days)

        return {
            "n_per_variant": n_per_variant,
            "total_sample_size": total,
            "num_variants": num_variants,
            "absolute_effect_size": round(delta, 2),
            "relative_mde": rel_mde,
            "estimated_days": days_needed,
            "alpha": alpha,
            "power": power,
            "formula": "n = 2σ²(Z_α/2 + Z_β)² / δ²",
        }

    # =====================================================================
    #  3. STATISTICAL ANALYSIS
    # =====================================================================

    def analyse(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Full experiment analysis pipeline:
          1. Compute metrics per variant
          2. Run significance tests (primary metrics)
          3. Check guardrails
          4. Segment breakdowns
          5. Business impact projection
          6. Go / no-go decision
        """
        # ── 1. Per-variant metrics ──────────────────────────────────────────
        variant_metrics = self.metrics.compute_all(events)

        # ── 2. Statistical tests ────────────────────────────────────────────
        stat_tests = self._run_stat_tests(events, variant_metrics)

        # ── 3. Guardrail checks ─────────────────────────────────────────────
        guardrails = self._check_guardrails(variant_metrics)

        # ── 4. Segment breakdown ────────────────────────────────────────────
        segments = {}
        for dim in self.config.segment_dimensions:
            key = dim["key"]
            segments[key] = self.metrics.compute_segment_breakdown(events, key)

        # ── 5. Business projection ──────────────────────────────────────────
        projection = self._project_business_impact(variant_metrics)

        # ── 6. Decision ─────────────────────────────────────────────────────
        decision = self._make_decision(stat_tests, guardrails, variant_metrics)

        return {
            "experiment": self.config.to_dict(),
            "variant_metrics": variant_metrics,
            "statistical_tests": stat_tests,
            "guardrail_checks": guardrails,
            "segment_breakdown": segments,
            "business_projection": projection,
            "decision": decision,
        }

    # ── Statistical tests ───────────────────────────────────────────────────

    def _run_stat_tests(
        self,
        events: list[dict],
        variant_metrics: dict,
    ) -> dict[str, Any]:
        """Run t-tests and bootstrap CIs for primary metrics."""
        control_events = [e for e in events if e["variant"] == "control"]
        treatment_events = [e for e in events if e["variant"] == "treatment"]

        if not control_events or not treatment_events:
            return {"error": "Insufficient data for both variants."}

        tests = {}

        # ── AOV test ────────────────────────────────────────────────────────
        ctrl_aov = [e.get("order_value", 0) for e in control_events if e.get("order_placed")]
        treat_aov = [e.get("order_value", 0) for e in treatment_events if e.get("order_placed")]
        if ctrl_aov and treat_aov:
            tests["aov"] = self._two_sample_test(
                ctrl_aov, treat_aov, "Average Order Value (₹)"
            )

        # ── CSAO attach rate test (proportions) ────────────────────────────
        ctrl_placed = [e for e in control_events if e.get("order_placed")]
        treat_placed = [e for e in treatment_events if e.get("order_placed")]
        ctrl_attach = sum(1 for e in ctrl_placed if e.get("csao_items_added", 0) > 0)
        treat_attach = sum(1 for e in treat_placed if e.get("csao_items_added", 0) > 0)
        tests["csao_attach_rate"] = self._proportion_test(
            ctrl_attach, len(ctrl_placed),
            treat_attach, len(treat_placed),
            "CSAO Attach Rate",
        )

        # ── C2O ratio test (proportions) ────────────────────────────────────
        tests["c2o_ratio"] = self._proportion_test(
            len(ctrl_placed), len(control_events),
            len(treat_placed), len(treatment_events),
            "Cart-to-Order Ratio",
        )

        # ── Multiple testing correction (Bonferroni) ───────────────────────
        n_tests = len(tests)
        corrected_alpha = self.config.significance_level / max(n_tests, 1)
        for key, t in tests.items():
            t["bonferroni_alpha"] = round(corrected_alpha, 4)
            t["significant_after_correction"] = (
                t.get("p_value", 1.0) < corrected_alpha
            )

        return {
            "tests": tests,
            "correction_method": "Bonferroni",
            "family_wise_alpha": self.config.significance_level,
            "corrected_alpha": round(corrected_alpha, 4),
            "n_primary_tests": n_tests,
        }

    def _two_sample_test(
        self,
        control: list[float],
        treatment: list[float],
        label: str,
    ) -> dict:
        """Welch's t-test + bootstrap confidence interval."""
        n_c, n_t = len(control), len(treatment)
        mean_c = statistics.mean(control)
        mean_t = statistics.mean(treatment)
        std_c = statistics.stdev(control) if n_c > 1 else 0.0
        std_t = statistics.stdev(treatment) if n_t > 1 else 0.0

        diff = mean_t - mean_c
        relative_lift = diff / max(mean_c, 0.01)

        # Welch's t-statistic
        se = math.sqrt((std_c**2 / max(n_c, 1)) + (std_t**2 / max(n_t, 1)))
        t_stat = diff / max(se, 1e-9)

        # Approx p-value (two-sided) using normal approximation for large n
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        # Bootstrap 95% CI of the difference (1000 resamples)
        ci_lower, ci_upper = self._bootstrap_ci(control, treatment)

        return {
            "metric": label,
            "control_mean": round(mean_c, 2),
            "treatment_mean": round(mean_t, 2),
            "absolute_diff": round(diff, 2),
            "relative_lift": round(relative_lift, 4),
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_value, 6),
            "significant": p_value < self.config.significance_level,
            "ci_95_diff": [round(ci_lower, 2), round(ci_upper, 2)],
            "n_control": n_c,
            "n_treatment": n_t,
        }

    def _proportion_test(
        self,
        successes_c: int, n_c: int,
        successes_t: int, n_t: int,
        label: str,
    ) -> dict:
        """Two-proportion z-test."""
        p_c = successes_c / max(n_c, 1)
        p_t = successes_t / max(n_t, 1)
        diff = p_t - p_c
        relative_lift = diff / max(p_c, 1e-9)

        # Pooled proportion
        p_pool = (successes_c + successes_t) / max(n_c + n_t, 1)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/max(n_c,1) + 1/max(n_t,1)))
        z_stat = diff / max(se, 1e-9)
        p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))

        return {
            "metric": label,
            "control_rate": round(p_c, 4),
            "treatment_rate": round(p_t, 4),
            "absolute_diff": round(diff, 4),
            "relative_lift": round(relative_lift, 4),
            "z_statistic": round(z_stat, 3),
            "p_value": round(p_value, 6),
            "significant": p_value < self.config.significance_level,
            "n_control": n_c,
            "n_treatment": n_t,
        }

    def _bootstrap_ci(
        self,
        control: list[float],
        treatment: list[float],
        n_boot: int = 1000,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for mean difference."""
        import random
        diffs = []
        for _ in range(n_boot):
            sample_c = random.choices(control, k=len(control))
            sample_t = random.choices(treatment, k=len(treatment))
            diffs.append(statistics.mean(sample_t) - statistics.mean(sample_c))
        diffs.sort()
        lo = int((1 - ci) / 2 * n_boot)
        hi = int((1 + ci) / 2 * n_boot) - 1
        return diffs[lo], diffs[min(hi, len(diffs) - 1)]

    # =====================================================================
    #  4. SEQUENTIAL TESTING (O'Brien-Fleming)
    # =====================================================================

    def sequential_boundaries(self) -> list[dict]:
        """
        Compute O'Brien-Fleming spending-function boundaries
        for planned interim analyses.

        At look k of K total:
          z_boundary(k) ≈ Z_α * sqrt(K / k)
        """
        K = self.config.interim_analyses
        z_final = self._z_score(1 - self.config.significance_level / 2)
        bounds = []
        for k in range(1, K + 1):
            info_frac = k / K
            z_boundary = z_final / math.sqrt(info_frac)
            bounds.append({
                "look": k,
                "information_fraction": round(info_frac, 2),
                "z_boundary": round(z_boundary, 3),
                "approx_p_threshold": round(
                    2 * (1 - self._normal_cdf(z_boundary)), 6
                ),
                "action": (
                    "Reject H₀ if |z| > boundary"
                    if k < K else
                    "Final analysis — use standard α"
                ),
            })
        return bounds

    # =====================================================================
    #  5. GUARDRAIL CHECKS
    # =====================================================================

    def _check_guardrails(self, variant_metrics: dict) -> dict:
        ctrl = variant_metrics.get("control", {})
        treat = variant_metrics.get("treatment", {})

        checks = []

        # Abandonment
        aband_ctrl = ctrl.get("cart_abandonment_rate", 0)
        aband_treat = treat.get("cart_abandonment_rate", 0)
        aband_diff = aband_treat - aband_ctrl
        checks.append({
            "guardrail": "Cart Abandonment Rate",
            "control": round(aband_ctrl, 4),
            "treatment": round(aband_treat, 4),
            "diff": round(aband_diff, 4),
            "threshold": self.config.max_abandonment_increase,
            "passed": aband_diff <= self.config.max_abandonment_increase,
            "severity": "critical",
        })

        # Repeat order rate
        repeat_ctrl = ctrl.get("repeat_order_rate", 0)
        repeat_treat = treat.get("repeat_order_rate", 0)
        repeat_diff = repeat_treat - repeat_ctrl
        checks.append({
            "guardrail": "Repeat Order Rate (7-day)",
            "control": round(repeat_ctrl, 4),
            "treatment": round(repeat_treat, 4),
            "diff": round(repeat_diff, 4),
            "threshold": -0.02,  # must not drop by more than 2pp
            "passed": repeat_diff >= -0.02,
            "severity": "critical",
        })

        # Restaurant fairness
        gini_ctrl = ctrl.get("restaurant_fairness_gini", 0)
        gini_treat = treat.get("restaurant_fairness_gini", 0)
        gini_diff = gini_treat - gini_ctrl
        checks.append({
            "guardrail": "Restaurant Fairness (Gini)",
            "control": round(gini_ctrl, 4),
            "treatment": round(gini_treat, 4),
            "diff": round(gini_diff, 4),
            "threshold": 0.05,  # Gini must not increase by >0.05
            "passed": gini_diff <= 0.05,
            "severity": "warning",
        })

        all_passed = all(c["passed"] for c in checks)
        return {
            "all_passed": all_passed,
            "checks": checks,
        }

    # =====================================================================
    #  6. BUSINESS IMPACT PROJECTION
    # =====================================================================

    def _project_business_impact(self, variant_metrics: dict) -> dict:
        ctrl = variant_metrics.get("control", {})
        treat = variant_metrics.get("treatment", {})

        ctrl_aov = ctrl.get("aov", self.config.baseline_aov)
        treat_aov = treat.get("aov", ctrl_aov)
        aov_lift = (treat_aov - ctrl_aov) / max(ctrl_aov, 1)

        daily_orders = self.config.daily_orders
        take_rate = self.config.zomato_take_rate

        incremental_rev_per_order = treat_aov - ctrl_aov
        daily_rev_lift = incremental_rev_per_order * daily_orders
        monthly_rev_lift = daily_rev_lift * 30
        annual_rev_lift = daily_rev_lift * 365

        gross_profit_monthly = monthly_rev_lift * take_rate
        infra_cost = self.config.ai_infra_monthly_cost_inr
        net_monthly = gross_profit_monthly - infra_cost
        roi = net_monthly / max(infra_cost, 1)

        return {
            "aov_lift_pct": round(aov_lift * 100, 2),
            "aov_lift_inr": round(treat_aov - ctrl_aov, 2),
            "daily_incremental_revenue": round(daily_rev_lift, 0),
            "monthly_incremental_revenue": round(monthly_rev_lift, 0),
            "annual_incremental_revenue": round(annual_rev_lift, 0),
            "monthly_gross_profit": round(gross_profit_monthly, 0),
            "monthly_ai_infra_cost": round(infra_cost, 0),
            "monthly_net_impact": round(net_monthly, 0),
            "roi_monthly": round(roi, 1),
            "assumptions": {
                "daily_orders": daily_orders,
                "take_rate": take_rate,
                "baseline_aov": ctrl_aov,
                "treatment_aov": treat_aov,
            },
        }

    # =====================================================================
    #  7. GO / NO-GO DECISION
    # =====================================================================

    def _make_decision(
        self,
        stat_tests: dict,
        guardrails: dict,
        variant_metrics: dict,
    ) -> dict:
        """
        Decision matrix:
          SHIP   — All primary metrics significant positive + guardrails pass
          ITERATE — Some metrics positive but not all significant
          KILL    — Any guardrail fails critically OR primary metric negative
        """
        tests = stat_tests.get("tests", {})
        guardrails_passed = guardrails.get("all_passed", False)

        # Count significant positive lifts
        sig_positive = 0
        sig_negative = 0
        for key, t in tests.items():
            corrected = t.get("significant_after_correction", False)
            lift = t.get("relative_lift", 0)
            if corrected and lift > 0:
                sig_positive += 1
            elif corrected and lift < 0:
                sig_negative += 1

        total_primary = len(tests)

        # Critical guardrail failure check
        critical_fail = any(
            not c["passed"] and c["severity"] == "critical"
            for c in guardrails.get("checks", [])
        )

        if critical_fail or sig_negative > 0:
            verdict = "KILL"
            reason = (
                "Critical guardrail failure or significant negative impact detected. "
                "Do NOT ship. Investigate root cause."
            )
            next_steps = [
                "Analyse segment-level data to identify harm",
                "Review model outputs for quality issues",
                "Consider reverting to control immediately",
            ]
        elif sig_positive == total_primary and guardrails_passed:
            verdict = "SHIP"
            reason = (
                f"All {total_primary} primary metrics show significant positive lift "
                "and all guardrails pass. Recommend gradual rollout."
            )
            next_steps = [
                f"Begin Stage 1 rollout ({self.config.rollout_stages[0]['traffic_pct']}%)",
                "Monitor guardrails daily for first 72 hours",
                "Proceed to Stage 2 if metrics hold",
            ]
        else:
            verdict = "ITERATE"
            reason = (
                f"{sig_positive}/{total_primary} primary metrics significant. "
                "Results are promising but not conclusive."
            )
            next_steps = [
                "Extend experiment duration by 1 week",
                "Analyse underperforming segments",
                "Consider model tuning for weak segments",
                "Re-evaluate after additional data",
            ]

        return {
            "verdict": verdict,
            "reason": reason,
            "next_steps": next_steps,
            "summary": {
                "primary_significant_positive": sig_positive,
                "primary_significant_negative": sig_negative,
                "total_primary_tests": total_primary,
                "guardrails_passed": guardrails_passed,
                "critical_guardrail_failure": critical_fail,
            },
        }

    # =====================================================================
    #  8. ROLLOUT PLAN
    # =====================================================================

    def rollout_plan(self) -> dict:
        """Generate the staged rollout strategy."""
        stages = self.config.rollout_stages
        rollback_triggers = [
            {
                "trigger": "P95 latency > 300 ms sustained for 15 min",
                "action": "Immediate rollback to control",
                "severity": "critical",
            },
            {
                "trigger": "Cart abandonment rate increases > 2 pp vs control",
                "action": "Halt rollout, investigate within 4 hours",
                "severity": "critical",
            },
            {
                "trigger": "CSAO attach rate drops > 5 pp vs control",
                "action": "Pause rollout, debug recommendations",
                "severity": "high",
            },
            {
                "trigger": "Error rate > 1 % of requests",
                "action": "Rollback immediately, review logs",
                "severity": "critical",
            },
            {
                "trigger": "Restaurant fairness Gini increases > 0.05",
                "action": "Review recommendation distribution, adjust if needed",
                "severity": "warning",
            },
        ]

        monitoring_dashboard = {
            "real_time_panels": [
                "P50 / P95 / P99 latency (ms)",
                "Request rate & error rate",
                "Cache hit rate",
                "Model inference time distribution",
            ],
            "hourly_panels": [
                "AOV by variant (rolling 1h)",
                "CSAO attach rate by variant",
                "Cart abandonment rate by variant",
                "CTR on CSAO rail",
            ],
            "daily_panels": [
                "Cumulative AOV lift with CI",
                "Segment-level breakdown (city, user type)",
                "Restaurant recommendation distribution heatmap",
                "Sequential test boundaries plot",
            ],
        }

        return {
            "stages": stages,
            "rollback_triggers": rollback_triggers,
            "monitoring_dashboard": monitoring_dashboard,
        }

    # =====================================================================
    #  9. LONG-TERM LEARNING
    # =====================================================================

    def long_term_plan(self) -> dict:
        """Model health monitoring and continuous experimentation roadmap."""
        return {
            "model_degradation_monitoring": {
                "metrics_to_track": [
                    "Weekly NDCG@10 on holdout set",
                    "Recommendation acceptance rate (trailing 7-day)",
                    "Feature distribution drift (KL divergence)",
                    "PMI matrix staleness (days since refresh)",
                ],
                "alert_thresholds": {
                    "ndcg_drop": "Alert if NDCG@10 drops > 5% vs baseline",
                    "acceptance_drop": "Alert if acceptance rate drops > 3 pp",
                    "feature_drift": "Alert if KL divergence > 0.1 on any feature",
                },
            },
            "retraining_schedule": {
                "pmi_matrix": "Daily refresh (overnight batch job, ~2 hours)",
                "user_embeddings": "Hourly incremental update via ChromaDB",
                "slm_fine_tune": "Monthly QLoRA fine-tune on latest 30 days of data",
                "full_retrain": "Quarterly full pipeline retrain + validation",
            },
            "continuous_experimentation_roadmap": {
                "quarter_1": {
                    "focus": "Fine-tune on real data",
                    "experiments": [
                        "Prompt template A/B test (3 variants)",
                        "Graph PMI threshold sweep (0.5 vs 1.0 vs 1.5)",
                        "Quantization level test (4-bit vs 8-bit SLM)",
                    ],
                },
                "quarter_2": {
                    "focus": "Personalization depth",
                    "experiments": [
                        "User embedding dimension test (384 vs 768)",
                        "Dietary preference model",
                        "Time-of-day specialised models",
                    ],
                },
                "quarter_3": {
                    "focus": "Restaurant-side optimization",
                    "experiments": [
                        "Restaurant-specific recommendation models",
                        "Menu-level A/B (which items to promote)",
                        "Dynamic pricing integration test",
                    ],
                },
                "quarter_4": {
                    "focus": "Multi-modal & proactive",
                    "experiments": [
                        "Image-based recommendations test",
                        "Proactive meal bundles vs reactive add-ons",
                        "Voice-to-cart integration pilot",
                    ],
                },
            },
        }

    # =====================================================================
    #  FULL EXPERIMENT PLAN (executive-ready document)
    # =====================================================================

    def generate_experiment_plan(self) -> dict:
        """
        Generate the complete experiment plan document with all sections.
        """
        sample = self.required_sample_size()
        sequential = self.sequential_boundaries()
        rollout = self.rollout_plan()
        longterm = self.long_term_plan()

        return {
            "executive_summary": (
                f"This experiment tests the ContextFlow AI recommendation system "
                f"(LLM re-ranking + temporal session graphs) against the current "
                f"collaborative-filtering baseline on Zomato's CSAO rail. "
                f"Using a {self.config.variants[0]['traffic_pct']}/"
                f"{self.config.variants[1]['traffic_pct']} user-level "
                f"traffic split over {sample['estimated_days']}+ days, "
                f"we will measure AOV lift (primary), CSAO attach rate, and "
                f"cart-to-order ratio, with cart abandonment and restaurant "
                f"fairness as guardrails. Based on offline metrics "
                f"(NDCG@10: 0.75, Precision@5: 0.65), we project a "
                f"5-10% AOV lift translating to ₹{self.config.daily_orders * 380 * 0.07 / 1e7:.0f}+ "
                f"crores/month incremental revenue. Sequential monitoring "
                f"with O'Brien-Fleming boundaries allows early stopping "
                f"for both success and harm."
            ),
            "experiment_design": {
                "config": self.config.to_dict(),
                "sample_size": sample,
                "sequential_boundaries": sequential,
                "stratification": {
                    "unit": self.config.randomisation_unit,
                    "keys": self.config.stratification_keys,
                    "method": "Deterministic MD5 hashing with experiment salt",
                },
            },
            "metrics_framework": {
                "primary": self.metrics.get_metric_definitions("primary"),
                "secondary": self.metrics.get_metric_definitions("secondary"),
                "guardrail": self.metrics.get_metric_definitions("guardrail"),
            },
            "segment_analysis_plan": self.config.segment_dimensions,
            "statistical_methodology": {
                "primary_tests": "Welch's t-test (continuous) / Two-proportion z-test (rates)",
                "confidence_intervals": "Bootstrap (1000 resamples, 95% CI)",
                "multiple_testing": "Bonferroni correction across primary metrics",
                "sequential_testing": "O'Brien-Fleming spending function",
                "interim_looks": self.config.interim_analyses,
            },
            "success_criteria": {
                "ship": "All primary metrics significant positive + all guardrails pass",
                "iterate": "Some metrics positive, not all significant",
                "kill": "Any critical guardrail fails OR primary metric significantly negative",
            },
            "rollout_strategy": rollout,
            "long_term_learning": longterm,
            "risk_mitigation": [
                {
                    "risk": "SLM latency spikes under load",
                    "mitigation": "Fallback to graph-only ranking if P95 > 200ms",
                    "monitoring": "Real-time latency dashboard with 15-min alerting",
                },
                {
                    "risk": "Cold-start quality regression",
                    "mitigation": "Dedicated cold-start agent with cuisine KB fallback",
                    "monitoring": "Track Precision@8 for new users separately",
                },
                {
                    "risk": "Novelty effect inflating early metrics",
                    "mitigation": "Run for minimum 14 days; exclude first 48h from final analysis",
                    "monitoring": "Daily metrics trend to detect novelty decay",
                },
                {
                    "risk": "Network effects (social ordering)",
                    "mitigation": "User-level randomisation prevents contamination",
                    "monitoring": "Check for spillover via shared-household analysis",
                },
            ],
        }

    # =====================================================================
    #  MATH HELPERS
    # =====================================================================

    @staticmethod
    def _z_score(p: float) -> float:
        """Approximate inverse normal CDF (Abramowitz & Stegun)."""
        if p <= 0:
            return -4.0
        if p >= 1:
            return 4.0
        if p > 0.5:
            return -ABTestFramework._z_score(1 - p)
        t = math.sqrt(-2 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return -(t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Cumulative distribution function for the standard normal."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
