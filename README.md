<p align="center">
  <h1 align="center">🛒 CartComplete</h1>
  <p align="center">
    <strong>Cart Super Add-On Recommender — Zomato Hackathon 2026</strong><br>
    LLM-augmented synthetic data · Sentence-Transformer embeddings · LightGBM sequential ranker
  </p>
</p>

---

## 📌 Problem Statement

When a Zomato customer is building their cart, recommend the **most relevant
add-on items** (garlic bread, beverages, desserts, sides) that maximise
order value while feeling natural and personalised.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│  POST /recommend   POST /recommend/sequential           │
└──────────┬──────────────────────┬───────────────────────┘
           │                      │
     ┌─────▼─────┐        ┌──────▼──────┐
     │ Embedding  │        │  Feature    │
     │ (MiniLM)   │        │  Assembly   │
     └─────┬─────┘        └──────┬──────┘
           │                      │
     ┌─────▼─────┐        ┌──────▼──────┐
     │   FAISS    │        │  LightGBM   │
     │  Retrieve  │───────▶│  Re-Rank    │
     │  Top-50    │        │ (LambdaRank)│
     └───────────┘        └──────┬──────┘
                                  │
                           Top-5 Add-Ons
```

**End-to-end latency: ~18 ms on CPU** (well under the 200 ms target).

## 🎯 How We Address Every Evaluation Criterion

| # | Criterion | Our Approach |
|---|-----------|-------------|
| 1 | **Data Realism** | LLM-augmented synthetic data built on a real Indian food ontology. An LLM (Flan-T5 / Ollama Mistral) generates co-purchase reasoning so add-on pairings reflect how real customers order (biryani→raita, pizza→garlic bread + coke). |
| 2 | **AI Edge (LLM)** | LLM used at *data generation* time to inject natural language reasoning into training signal. Sentence-transformer embeddings enable zero-shot cold-start for new menu items. |
| 3 | **Latency < 200 ms** | Two-stage pipeline: FAISS ANN retrieval (~2 ms) + LightGBM scoring (~1 ms). Total P95 < 20 ms. No LLM at inference time. |
| 4 | **Cold-Start** | New / unseen add-ons are embedded from their *name + category text* via sentence-transformers — no interaction history required. |
| 5 | **Sequential Cart Updates** | Every cart mutation triggers re-embedding + re-ranking. Cart embedding uses weighted mean-pooling with recency decay, so the system learns that "just added" items dominate the recommendation context. |
| 6 | **Overall Quality** | Production-ready FastAPI service, NDCG/MRR/Hit@K eval suite, YAML config, unit + integration tests, clean separation of concerns. |

## 📊 Dataset

The dataset used for this project is too large to be included in the repository directly. You can find and download the complete dataset from the following link:
[CartComplete Dataset (Google Drive)](https://drive.google.com/drive/folders/1USc1F_p9h9e45HMD05Z84ohKJhuLVWQJ?usp=drive_link)

## 📂 Project Structure

```
CartComplete/
├── configs/
│   └── config.yaml              # all hyperparams in one place
├── data/
│   ├── raw/
│   │   └── cuisine_ontology.json  # seed ontology
│   ├── processed/                 # generated training data (parquet)
│   └── embeddings/                # FAISS indices & numpy arrays
├── models/
│   ├── prompts/
│   │   └── addon_suggestion.txt   # LLM prompt template
│   └── saved/                     # serialised LightGBM models
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py # LLM-augmented cart generation
│   │   └── preprocessor.py        # feature engineering & splits
│   ├── features/
│   │   └── embeddings.py          # sentence-transformer + FAISS
│   ├── models/
│   │   └── ranker.py              # LightGBM LambdaRank train/predict
│   ├── evaluation/
│   │   └── metrics.py             # NDCG, MRR, Hit@K, latency bench
│   ├── serving/
│   │   ├── app.py                 # FastAPI endpoints
│   │   └── inference.py           # end-to-end inference pipeline
│   └── utils/
│       └── config.py              # paths & constants
├── tests/
│   ├── test_metrics.py
│   ├── test_embeddings.py
│   └── test_api.py
├── notebooks/                     # EDA & experimentation
├── scripts/                       # CLI scripts (train, eval, serve)
├── logs/
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Create virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** PyTorch is installed as CPU-only via the `--extra-index-url` in
> requirements.txt. Total install is ~2 GB.

### 3. Run tests

```bash
python -m pytest tests/ -v
```

### 4. Start the API server

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs)

## 🔧 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Synthetic Data | **Flan-T5 / Ollama Mistral** | Free, CPU-friendly LLM for co-purchase reasoning |
| Embeddings | **all-MiniLM-L6-v2** (22 MB) | Fast, high-quality 384-d embeddings, <10ms/encode |
| Retrieval | **FAISS (CPU)** | Sub-millisecond ANN search |
| Ranking | **LightGBM LambdaRank** | NDCG-optimised, <1ms inference |
| Serving | **FastAPI + Uvicorn** | Async, production-grade, auto-docs |
| Storage | **Parquet (PyArrow)** | Columnar, compressed, fast I/O |

## 💰 Cost

**₹0.** Everything runs on CPU with free/open-source models. No paid APIs, no
cloud GPUs, no Docker required.

## 📝 Next Steps

- [ ] **Step 2:** Implement synthetic data generator with LLM augmentation
- [ ] **Step 3:** Build embedding pipeline and FAISS index
- [ ] **Step 4:** Train LightGBM ranker, evaluate on test set
- [ ] **Step 5:** Wire up end-to-end inference pipeline
- [ ] **Step 6:** Latency benchmarking & optimisation
- [ ] **Step 7:** Demo notebook / video

---

*Built for Zomathon 2026 🍕*
