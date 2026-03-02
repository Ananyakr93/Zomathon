"""
build_faiss_index.py
====================
Build and save a FAISS index from the menu items in the cuisine knowledge base.

This script encodes all add-on item names using the MiniLM model and saves:
  - data/embeddings/addon_index.faiss       (FAISS index)
  - data/embeddings/addon_metadata.json     (item metadata for lookup)

Usage:
    python build_faiss_index.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch  # Prevents transformers NameError: nn not defined

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features.embeddings import encode_items, build_faiss_index, save_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("build_faiss_index")

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

# ── Collect all add-on items from ColdStartAgent cuisine KB ──────────────

def collect_addon_items() -> list[dict]:
    """Gather all unique add-on items across all cuisines."""
    from src.agents.cold_start_agent import _CUISINE_COMPLEMENTS, _UNIVERSAL_SAFE

    items = []
    seen = set()

    for cuisine, categories in _CUISINE_COMPLEMENTS.items():
        for category, item_list in categories.items():
            for item_tuple in item_list:
                name = item_tuple[0]
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                items.append({
                    "name": name,
                    "category": category,
                    "price_bracket": item_tuple[1],
                    "is_veg": item_tuple[2],
                    "tags": item_tuple[3] if len(item_tuple) > 3 else [],
                    "source_cuisine": cuisine,
                    "text": f"{name} ({category}, {cuisine})",
                })

    # Add universal safe items
    for item_tuple in _UNIVERSAL_SAFE:
        name = item_tuple[0]
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        items.append({
            "name": name,
            "category": item_tuple[1],
            "price_bracket": item_tuple[2],
            "is_veg": item_tuple[3],
            "tags": item_tuple[4] if len(item_tuple) > 4 else [],
            "source_cuisine": "universal",
            "text": f"{name} ({item_tuple[1]}, universal)",
        })

    return items


def main():
    print("=" * 60)
    print("  FAISS INDEX BUILDER")
    print("=" * 60)

    # Step 1: Collect items
    items = collect_addon_items()
    logger.info("Collected %d unique add-on items across all cuisines", len(items))

    # Step 2: Encode item texts
    texts = [item["text"] for item in items]
    logger.info("Encoding %d items with MiniLM …", len(texts))
    t0 = time.perf_counter()
    embeddings = encode_items(texts)
    encode_time = time.perf_counter() - t0
    logger.info("Encoding took %.2f seconds, shape=%s", encode_time, embeddings.shape)

    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 4: Save index + metadata
    index_path = save_index(index, name="addon_index")

    # Save metadata for lookup during retrieval
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = EMBEDDINGS_DIR / "addon_metadata.json"
    metadata = []
    for i, item in enumerate(items):
        metadata.append({
            "idx": i,
            "name": item["name"],
            "category": item["category"],
            "price_bracket": item["price_bracket"],
            "is_veg": item["is_veg"],
            "tags": item["tags"],
            "source_cuisine": item["source_cuisine"],
        })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved -> %s", meta_path)

    # Save raw embeddings for potential reuse
    emb_path = EMBEDDINGS_DIR / "addon_embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info("Embeddings saved -> %s", emb_path)

    print(f"\n  Items indexed : {len(items)}")
    print(f"  Embedding dim : {embeddings.shape[1]}")
    print(f"  Encode time   : {encode_time:.2f}s")
    print(f"  Index path    : {index_path}")
    print(f"  Metadata path : {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
