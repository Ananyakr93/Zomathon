"""
config.py
=========
Project-wide configuration constants and helpers.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
PROMPT_DIR = PROJECT_ROOT / "models" / "prompts"
CONFIG_DIR = PROJECT_ROOT / "configs"
LOG_DIR = PROJECT_ROOT / "logs"

# ── Model defaults ────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
FAISS_TOP_K = 50           # candidates retrieved before LightGBM re-rank
RECOMMEND_TOP_N = 10       # final recommendations returned to user (N=8-10 per spec)
LATENCY_BUDGET_MS = 200    # hard ceiling

# ── Data generation ───────────────────────────────────────
DEFAULT_N_SESSIONS = 50_000
DEFAULT_SEED = 42

# ── LLM ───────────────────────────────────────────────────
LLM_BACKEND = "huggingface"   # "huggingface" | "ollama"
OLLAMA_MODEL = "mistral"       # if using Ollama
HF_MODEL_ID = "google/flan-t5-small"  # free, CPU-friendly


def ensure_dirs() -> None:
    """Create all project directories if they don't exist."""
    for d in (RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, MODEL_DIR, PROMPT_DIR,
              CONFIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
