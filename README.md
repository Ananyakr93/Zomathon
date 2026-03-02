<p align="center">
  <h1 align="center">🛒 MealMind — CSAO AI Recommender</h1>
  <p align="center">
    <strong>Context-Aware Cart Super Add-On System — Zomathon 2026</strong><br>
    GenAI-powered meal understanding · Multi-agent re-ranking · Production-grade serving
  </p>
</p>

---

## 📌 Problem Statement

When a Zomato customer is building their cart, recommend the **most relevant
add-on items** (beverages, desserts, sides, breads) that maximise order value
while feeling natural, culturally appropriate, and personalised.

## 💡 What Makes This Novel

| Innovation | Description |
|---|---|
| **3-Agent CSAO Architecture** | MealContextAgent → RerankerAgent → ColdStartAgent work as a pipeline — no single model handles everything |
| **Cultural Meal Intelligence** | System understands that Biryani needs Raita (not Salsa), Dosa needs Sambhar (not Soup) |
| **5-Level Graceful Degradation** | Full pipeline → Graph-only → CF-only → Cold Start → Popularity — the system never fails |
| **Statistical A/B Testing** | Bonferroni correction + O'Brien-Fleming sequential boundaries — production-grade experimentation |
| **Zero Cold-Start Failures** | Dedicated ColdStartAgent uses cuisine knowledge base — works even for brand-new users/restaurants |

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       FastAPI + Orchestrator                         │
│  POST /recommend    POST /recommend/sequential    GET /health        │
├───────────┬────────────────┬────────────────┬───────────────────────┤
│  Rate     │  L1 LRU Cache  │  Circuit       │  Request             │
│  Limiter  │  L2 Redis      │  Breakers      │  Tracer              │
├───────────┴────────────────┴────────────────┴───────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │ Feature      │   │ Candidate    │   │ LLM Re-ranking Pipeline  │ │
│  │ Retrieval    │──▶│ Generation   │──▶│                          │ │
│  │ (30ms SLA)   │   │ (50ms SLA)   │   │  MealContextAgent        │ │
│  └──────────────┘   └──────────────┘   │  ↓ analyze meal type     │ │
│         ▲                               │  ↓ cultural context      │ │
│         │                               │  RerankerAgent           │ │
│  ┌──────────────┐                      │  ↓ 4-axis weighted score │ │
│  │ ColdStart    │◀── fallback ─────────│  ↓ business rules        │ │
│  │ Agent        │                      │  (150ms SLA)             │ │
│  └──────────────┘                      └──────────────────────────┘ │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  Monitoring Dashboard │ Alert Engine │ P50/P95/P99 Histograms       │
└──────────────────────────────────────────────────────────────────────┘
```

**End-to-end latency budget: 250ms** | **Target: 10M predictions/day, 50K RPS peak**

## 🤖 The Three CSAO Agents

### 1. MealContextAgent (`src/agents/meal_context_agent.py`)
Analyses the cart to understand **what kind of meal** it is and **what's missing**.

- **Meal Type Identification:** Breakfast / Lunch / Dinner / Snack / Late-night (with confidence scores)
- **Completion Analysis:** Scores how "complete" the meal is (0-1) and identifies missing components
- **Cultural Intelligence:** Cuisine-specific patterns — North Indian (roti + raita), South Indian (sambhar + chutney), Chinese (soup + rice/noodles)

### 2. RerankerAgent (`src/agents/reranker_agent.py`)
Takes 50 candidates from collaborative filtering and produces the **optimal top 10**.

- **Meal Completion (40%):** Does this item fill a gap identified by MealContextAgent?
- **Contextual Relevance (25%):** Time-appropriate? Price-aligned? Cuisine-consistent?
- **Personalisation (20%):** Matches user dietary preferences and past patterns
- **Business Value (15%):** Item margin, platform promotions, diversity constraints

### 3. ColdStartAgent (`src/agents/cold_start_agent.py`)
Handles scenarios where ML models have **zero data** — using pure reasoning.

- **New Users:** No order history → cuisine knowledge base inference
- **New Restaurants:** Limited data → menu-based logical pairings
- **Unusual Carts:** No matching patterns → contextual reasoning (time + location + cuisine)

## 📊 A/B Testing Framework (`src/experimentation/`)

Production-grade experimentation engine:

| Feature | Implementation |
|---|---|
| **User Assignment** | Deterministic MD5 salt-based hashing (same user → same variant) |
| **Stat Tests** | Welch's t-test (continuous) + z-test (proportions) |
| **Confidence Intervals** | Bootstrap with 1000 resamples |
| **Multiple Testing** | Bonferroni correction across all primary metrics |
| **Sequential Testing** | O'Brien-Fleming spending function (4 interim looks) |
| **Decision Engine** | SHIP / ITERATE / KILL based on metrics + guardrails |
| **Metrics** | 11 metrics across 3 tiers (primary, secondary, guardrail) |

## 🛡️ Production Infrastructure (`src/serving/`)

| Component | File | What it does |
|---|---|---|
| **API Server** | `app.py` | FastAPI with health checks, request validation |
| **Orchestrator** | `orchestrator.py` | Wires the entire pipeline with parallelism |
| **Circuit Breaker** | `circuit_breaker.py` | 3-state CB (CLOSED→OPEN→HALF_OPEN) per dependency |
| **Cache** | `cache_manager.py` | L1 in-process LRU + L2 Redis with namespace TTLs |
| **Monitoring** | `monitoring.py` | Request tracing, latency histograms, alert engine |
| **Config** | `production_config.py` | Latency budgets, K8s scaling, cost model |
| **Pipeline** | `inference.py` | End-to-end inference connecting all agents |

### Latency Budget

| Stage | SLA | Optimization |
|---|---|---|
| Cache Lookup | 1ms | L1 in-process LRU, L2 Redis |
| Feature Retrieval | 30ms | Pre-computed, Redis feature store |
| Candidate Gen | 50ms | FAISS ANN + Graph PMI |
| LLM Re-ranking | 150ms | Agent pipeline with CB fallback |
| Post-processing | 20ms | Business rules engine |
| **Total** | **250ms** | **P95 target: 300ms** |

### Scaling

| Metric | Value |
|---|---|
| Daily Predictions | 10M |
| Peak RPS | 50,000 |
| RPS per Pod | 500 |
| Pods at Peak | 120 (HPA managed) |
| Pods at Average | 10 |
| Availability Target | 99.9% |

### Monthly Cost Estimate

| Component | Cost (₹) |
|---|---|
| Compute (K8s pods) | ₹2,40,000 |
| Redis Cluster (6 nodes) | ₹80,000 |
| PostgreSQL (+ 2 replicas) | ₹45,000 |
| Vector DB | ₹35,000 |
| LLM API (after 40% cache) | ₹9,00,000 |
| Monitoring | ₹30,000 |
| **Total** | **₹13,50,000 (~$16K)** |

## 📂 Project Structure

```
MealMind/
├── configs/
│   └── config.yaml                     # all hyperparams
├── src/
│   ├── agents/                         # 🤖 CSAO AI Agents
│   │   ├── meal_context_agent.py       #   Meal understanding + cultural context
│   │   ├── reranker_agent.py           #   Multi-axis candidate re-ranking
│   │   └── cold_start_agent.py         #   Zero-data fallback (13 cuisines)
│   ├── experimentation/                # 📊 A/B Testing
│   │   ├── ab_test_framework.py        #   Full experiment engine
│   │   └── metrics.py                  #   11 metrics, 3 tiers
│   ├── serving/                        # 🛡️ Production Infrastructure
│   │   ├── app.py                      #   FastAPI endpoints
│   │   ├── orchestrator.py             #   Pipeline orchestration
│   │   ├── inference.py                #   End-to-end inference
│   │   ├── circuit_breaker.py          #   Failure handling
│   │   ├── cache_manager.py            #   L1/L2 caching
│   │   ├── monitoring.py               #   Observability
│   │   └── production_config.py        #   Config & cost model
│   ├── data/
│   │   ├── synthetic_generator.py      #   LLM-augmented data gen
│   │   └── preprocessor.py             #   24-feature engineering pipeline
│   ├── features/
│   │   └── embeddings.py               #   MiniLM + FAISS
│   ├── models/
│   │   └── ranker.py                   #   LightGBM LambdaRank
│   └── evaluation/
│       ├── metrics.py                  #   NDCG, MRR, Hit@K
│       └── evaluate.py                 #   Full evaluation runner
├── models/
│   ├── saved/
│   │   └── cart_ranker.txt             #   Trained LightGBM model
│   └── results/
│       └── training_results.json       #   Training metrics (NDCG@10)
├── train_pipeline.py                   # 🏋️ End-to-end training runner
├── test_integration.py                 # ✅ 29-test integration suite
├── test_comprehensive.py               # 🧪 15+ real-world test cases
├── Dockerfile                          # 🐳 Multi-stage production build
├── requirements.txt
└── README.md
```

## 🎯 How We Address Every Evaluation Criterion

| # | Criterion | Our Approach |
|---|-----------|-------------|
| 1 | **Data Realism** | LLM-augmented synthetic data built on a real Indian food ontology. 50K sequential cart scenarios with co-purchase reasoning. |
| 2 | **AI Edge (LLM)** | 3-agent GenAI system with cultural meal intelligence, multi-axis re-ranking, and zero-data cold-start reasoning. |
| 3 | **Latency < 200ms** | Two-path architecture: fast path (FAISS + LightGBM <20ms) and full path (agent pipeline <250ms with circuit-breaker fallback). |
| 4 | **Cold-Start** | Dedicated ColdStartAgent with cuisine knowledge base (13 cuisines) — 5 reasoning strategies for new users, restaurants, and unusual carts. |
| 5 | **Sequential Cart Updates** | Every cart mutation triggers re-analysis and re-ranking. Exclusion sets prevent repeat recommendations. |
| 6 | **Overall Quality** | Production-grade: circuit breakers, caching, monitoring, A/B testing, Docker deployment, cost analysis. |

## 🚀 Quick Start

### 1. Setup

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_pipeline.py --n-scenarios 1000
```

This generates synthetic data, engineers 24 features, trains LightGBM LambdaRank, and saves the model to `models/saved/cart_ranker.txt`.

### 3. Run tests

```bash
python test_integration.py           # 29 end-to-end tests
python test_comprehensive.py         # 15+ real-world scenarios
python -m pytest tests/ -v           # unit tests
```

### 4. Start the API server

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Docker deployment

```bash
docker build -t mealmind-csao .
docker run -p 8000:8000 mealmind-csao
```

## 🔧 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **AI Agents** | Custom Python agents | Meal understanding → Re-ranking → Cold start pipeline |
| **Embeddings** | all-MiniLM-L6-v2 (22 MB) | Fast 384-d embeddings, <10ms/encode |
| **Retrieval** | FAISS (CPU) | Sub-millisecond ANN search |
| **Ranking** | LightGBM LambdaRank | NDCG-optimised, <1ms inference |
| **Serving** | FastAPI + Uvicorn | Async, production-grade, auto-docs |
| **Experimentation** | Custom A/B framework | Bonferroni + O'Brien-Fleming boundaries |
| **Infra** | Circuit breakers + caching | 5-level fallback, L1/L2 cache |
| **Deployment** | Docker multi-stage | Non-root, health checks, 4 workers |

## 💰 Cost

**Development: ₹0.** Everything runs on CPU with free/open-source models.
**Production estimate:** ~₹13.5L/month at full scale (10M daily predictions).

---

*Built for Zomathon 2026 🍕 — by Team MealMind*

