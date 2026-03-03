"""
data_splitter.py
================
Tier 4.1: Validation Methodology

Implements temporal train-test splitting to simulate real-world deployment
and prevent data leakage. Ensures that each user session remains atomic 
(never split across train/test segments).
"""

from __future__ import annotations
from typing import Any
import math
import logging

logger = logging.getLogger(__name__)

def temporal_split(
    events: list[dict[str, Any]], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1
) -> dict[str, list[dict[str, Any]]]:
    """
    Split event data temporally according to ratios.
    Assuming events are chronologically sorted or have a timestamp that can be sorted.
    For demonstration purposes, we sort by 'timestamp' if available, otherwise preserve order.
    """
    # Sort events if they have a timestamp to enforce temporal strictness
    if events and "timestamp" in events[0]:
        sorted_events = sorted(events, key=lambda x: x["timestamp"])
    else:
        # Fallback to provided order
        sorted_events = events

    # Group by session_id to maintain atomicity
    sessions: dict[str, list[dict]] = {}
    for e in sorted_events:
        sid = e.get("session_id", e.get("scenario_id", "unknown"))
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(e)
    
    # We now have a list of atomic sessions in rough temporal order of their first event
    ordered_sessions = list(sessions.values())
    
    n_sessions = len(ordered_sessions)
    train_end = math.floor(n_sessions * train_ratio)
    val_end = train_end + math.floor(n_sessions * val_ratio)

    train_sessions = ordered_sessions[:train_end]
    val_sessions = ordered_sessions[train_end:val_end]
    test_sessions = ordered_sessions[val_end:]

    return {
        "train": [e for session in train_sessions for e in session],
        "val": [e for session in val_sessions for e in session],
        "test": [e for session in test_sessions for e in session],
    }

def temporal_k_fold(
    events: list[dict[str, Any]], 
    k: int = 5
) -> list[dict[str, list[dict[str, Any]]]]:
    """
    5-fold temporal cross-validation on training set for hyperparameter tuning.
    Rolling window style:
    Fold 1: Train on chunk 1, Test on chunk 2
    Fold 2: Train on chunk 1-2, Test on chunk 3
    etc.
    """
    if events and "timestamp" in events[0]:
        sorted_events = sorted(events, key=lambda x: x["timestamp"])
    else:
        sorted_events = events

    sessions: dict[str, list[dict]] = {}
    for e in sorted_events:
        sid = e.get("session_id", e.get("scenario_id", "unknown"))
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append(e)
        
    ordered_sessions = list(sessions.values())
    n_sessions = len(ordered_sessions)
    
    # Total k+1 chunks required for k rolling windows
    chunk_size = math.floor(n_sessions / (k + 1))
    
    folds = []
    for i in range(1, k + 1):
        train_end = chunk_size * i
        test_end = chunk_size * (i + 1)
        
        train_sessions = ordered_sessions[:train_end]
        val_sessions = ordered_sessions[train_end:test_end]
        
        folds.append({
            "train": [e for session in train_sessions for e in session],
            "val": [e for session in val_sessions for e in session]
        })
        
    return folds
