# MealMind — serving package
"""
CSAO Production Serving Infrastructure
=======================================
app               : FastAPI application — /recommend, /health endpoints
inference         : end-to-end inference pipeline (agents → orchestrator)
orchestrator      : request pipeline with parallelism + fallback chain
circuit_breaker   : 3-state CB + rate limiter + fallback manager
cache_manager     : L1 in-process LRU + L2 Redis-compatible cache
monitoring        : request tracing, latency histograms, alert engine
production_config : latency budgets, K8s scaling, cost model
"""
