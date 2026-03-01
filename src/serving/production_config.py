"""
Production Configuration — CSAO Serving Infrastructure
========================================================
Central config for latency budgets, scaling parameters, cost model,
deployment settings, and all tunable production knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ═════════════════════════════════════════════════════════════════════════════
#  LATENCY BUDGET
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LatencyBudget:
    """End-to-end latency breakdown (milliseconds)."""
    feature_retrieval_ms: int = 30
    candidate_generation_ms: int = 50
    llm_reranking_ms: int = 150
    post_processing_ms: int = 20
    total_budget_ms: int = 250      # hard ceiling
    p95_target_ms: int = 300        # allowable P95
    p99_target_ms: int = 500        # allowable P99

    def to_dict(self) -> dict:
        return {
            "feature_retrieval_ms": self.feature_retrieval_ms,
            "candidate_generation_ms": self.candidate_generation_ms,
            "llm_reranking_ms": self.llm_reranking_ms,
            "post_processing_ms": self.post_processing_ms,
            "total_budget_ms": self.total_budget_ms,
            "p95_target_ms": self.p95_target_ms,
            "p99_target_ms": self.p99_target_ms,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  SCALING CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScalingConfig:
    """Kubernetes horizontal autoscaling parameters."""
    # Traffic
    daily_predictions: int = 10_000_000
    peak_rps: int = 50_000
    avg_rps: int = 115              # 10M / 86400

    # Pod capacity
    rps_per_pod: int = 500          # each pod handles ~500 RPS
    min_replicas: int = 10
    max_replicas: int = 120
    target_cpu_pct: int = 65        # HPA target

    # Pod resources
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "4Gi"
    memory_limit: str = "8Gi"

    # Autoscale triggers
    scale_up_cpu_pct: int = 70
    scale_up_rps_threshold: int = 400   # per pod
    scale_down_delay_s: int = 300
    scale_up_delay_s: int = 30

    def pods_for_rps(self, target_rps: int) -> int:
        """Calculate required pod count for a given RPS target."""
        import math
        base = math.ceil(target_rps / self.rps_per_pod)
        # 1.3x headroom for burst
        return min(max(int(base * 1.3), self.min_replicas), self.max_replicas)

    def to_dict(self) -> dict:
        return {
            "daily_predictions": self.daily_predictions,
            "peak_rps": self.peak_rps,
            "rps_per_pod": self.rps_per_pod,
            "pods_at_peak": self.pods_for_rps(self.peak_rps),
            "pods_at_avg": self.pods_for_rps(self.avg_rps),
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_pct": self.target_cpu_pct,
            "pod_resources": {
                "cpu": f"{self.cpu_request} req / {self.cpu_limit} lim",
                "memory": f"{self.memory_request} req / {self.memory_limit} lim",
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
#  CACHING CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CacheConfig:
    """Multi-tier cache TTLs and size limits."""
    # L1: In-process (per pod) — LRU
    l1_max_items: int = 10_000
    l1_ttl_s: int = 60              # 1 min for hot items

    # L2: Redis cluster
    menu_cache_ttl_s: int = 3600    # 1 hour — menus change infrequently
    user_embedding_ttl_s: int = 300  # 5 min — refreshed in near-real-time
    llm_response_ttl_s: int = 1800  # 30 min — same cart pattern → same recs
    restaurant_meta_ttl_s: int = 7200  # 2 hours

    # L3: CDN / edge (if applicable)
    static_asset_ttl_s: int = 86400  # 24 hours

    # Redis cluster sizing
    redis_nodes: int = 6            # 3 primary + 3 replica
    redis_max_memory_gb: int = 32
    redis_eviction_policy: str = "allkeys-lru"

    def to_dict(self) -> dict:
        return {
            "l1_in_process": {"max_items": self.l1_max_items, "ttl_s": self.l1_ttl_s},
            "l2_redis": {
                "menu_ttl_s": self.menu_cache_ttl_s,
                "user_embedding_ttl_s": self.user_embedding_ttl_s,
                "llm_response_ttl_s": self.llm_response_ttl_s,
                "restaurant_meta_ttl_s": self.restaurant_meta_ttl_s,
            },
            "redis_cluster": {
                "nodes": self.redis_nodes,
                "max_memory_gb": self.redis_max_memory_gb,
                "eviction_policy": self.redis_eviction_policy,
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
#  COST MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CostModel:
    """Monthly infrastructure cost estimates in INR."""
    # Compute (GKE / EKS)
    pod_cost_monthly_inr: float = 12_000       # per pod/month
    avg_pod_count: int = 20

    # Redis cluster
    redis_monthly_inr: float = 80_000          # 6 nodes, 32 GB

    # PostgreSQL (feature store)
    pg_monthly_inr: float = 45_000             # managed, 2 read-replicas

    # Vector DB (ChromaDB / Weaviate cloud)
    vector_db_monthly_inr: float = 35_000

    # LLM API (Claude)
    llm_cost_per_1k_requests_inr: float = 25   # ~$0.30 per 1K calls
    daily_llm_calls: int = 2_000_000           # 20% of traffic hits LLM
    llm_cache_hit_rate: float = 0.40           # 40% cache hits → only 60% are API calls

    # Monitoring (Datadog / Grafana cloud)
    monitoring_monthly_inr: float = 30_000

    # CDN & networking
    network_monthly_inr: float = 20_000

    def compute_total(self) -> dict:
        compute = self.pod_cost_monthly_inr * self.avg_pod_count
        actual_llm_calls = self.daily_llm_calls * (1 - self.llm_cache_hit_rate)
        llm_monthly = (actual_llm_calls / 1000) * self.llm_cost_per_1k_requests_inr * 30
        total = (
            compute + self.redis_monthly_inr + self.pg_monthly_inr
            + self.vector_db_monthly_inr + llm_monthly
            + self.monitoring_monthly_inr + self.network_monthly_inr
        )
        return {
            "compute_inr": compute,
            "redis_inr": self.redis_monthly_inr,
            "postgresql_inr": self.pg_monthly_inr,
            "vector_db_inr": self.vector_db_monthly_inr,
            "llm_api_inr": round(llm_monthly),
            "monitoring_inr": self.monitoring_monthly_inr,
            "network_inr": self.network_monthly_inr,
            "total_monthly_inr": round(total),
            "total_monthly_usd": round(total / 84),  # approx exchange rate
            "llm_details": {
                "daily_raw_calls": self.daily_llm_calls,
                "cache_hit_rate": self.llm_cache_hit_rate,
                "daily_actual_api_calls": int(actual_llm_calls),
                "cost_per_1k_inr": self.llm_cost_per_1k_requests_inr,
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
#  DEPLOYMENT CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeploymentConfig:
    """CI/CD and deployment pipeline settings."""
    canary_stages: list[dict] = None  # type: ignore

    def __post_init__(self):
        if self.canary_stages is None:
            object.__setattr__(self, "canary_stages", [
                {"stage": 1, "traffic_pct": 5,   "duration_h": 2,  "label": "smoke_test"},
                {"stage": 2, "traffic_pct": 20,  "duration_h": 12, "label": "canary"},
                {"stage": 3, "traffic_pct": 50,  "duration_h": 24, "label": "expansion"},
                {"stage": 4, "traffic_pct": 100, "duration_h": 0,  "label": "full_rollout"},
            ])

    health_check_path: str = "/health"
    readiness_probe_delay_s: int = 10
    liveness_probe_interval_s: int = 15
    graceful_shutdown_s: int = 30
    rolling_update_max_surge: str = "25%"
    rolling_update_max_unavailable: str = "10%"

    def to_dict(self) -> dict:
        return {
            "canary_stages": self.canary_stages,
            "health_check": self.health_check_path,
            "rolling_update": {
                "max_surge": self.rolling_update_max_surge,
                "max_unavailable": self.rolling_update_max_unavailable,
            },
            "graceful_shutdown_s": self.graceful_shutdown_s,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER PRODUCTION CONFIG
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ProductionConfig:
    """Aggregated production configuration."""
    latency: LatencyBudget = field(default_factory=LatencyBudget)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    cost: CostModel = field(default_factory=CostModel)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    # SLA targets
    availability_target: float = 0.999   # 99.9 %
    error_budget_monthly_min: float = 43.2  # minutes of allowed downtime

    def to_dict(self) -> dict:
        return {
            "latency_budget": self.latency.to_dict(),
            "scaling": self.scaling.to_dict(),
            "caching": self.cache.to_dict(),
            "cost_estimate": self.cost.compute_total(),
            "deployment": self.deployment.to_dict(),
            "sla": {
                "availability": f"{self.availability_target * 100}%",
                "error_budget_monthly_min": self.error_budget_monthly_min,
            },
        }
