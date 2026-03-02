# CartComplete — features package
"""
Modules
-------
embeddings        : sentence-transformer based item/cart embeddings + FAISS index
cart_features     : Tier 1 real-time cart composition, meal completeness, price positioning
session_graph     : Tier 1 temporal session graph (NetworkX + PMI co-occurrence)
user_features     : Tier 2 user behavioral embeddings (ChromaDB) + preference scoring
context_features  : Tier 3 temporal, geographic, restaurant, and candidate-item features
"""
