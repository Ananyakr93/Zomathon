"""
Session State — Sequential Cart Tracking
==========================================
Tracks cart evolution across requests within a user session to support
the Sequential Contextual Ranking problem framing (Section 2.1).

Each session records:
  - Cart snapshots at every time step (t=0, t=1, t=2, ...)
  - Items added / removed between steps
  - Meal completeness trajectory over time
  - Recommendation history (what was shown at each step)

Example:
  t=0: Empty → Biryani added → completeness 32%
  t=1: Raita added → completeness 58%
  t=2: Gulab Jamun added → completeness 82% → lower intensity
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Session TTL: expire idle sessions after 30 minutes
SESSION_TTL_S = 1800


@dataclass
class CartSnapshot:
    """A frozen picture of the cart at one time step."""
    step: int
    timestamp: float
    items: list[dict[str, Any]]
    item_names: set[str]
    completeness: float  # 0-100
    items_added: list[str]  # vs previous step
    items_removed: list[str]  # vs previous step


@dataclass
class SessionState:
    """
    Tracks the evolution of a single user session.

    Updated on every ``serve_request`` call with the same ``session_id``.
    """
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    cart_history: list[CartSnapshot] = field(default_factory=list)
    recommendation_history: list[list[str]] = field(default_factory=list)

    @property
    def current_step(self) -> int:
        return len(self.cart_history)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_S

    def update(self, cart_items: list[dict], completeness: float = 0.0) -> CartSnapshot:
        """
        Record a new cart snapshot and compute the delta from the previous step.

        Parameters
        ----------
        cart_items : list[dict]
            Current cart items (each must have ``name``).
        completeness : float
            Current meal completeness score (0-100).

        Returns
        -------
        CartSnapshot for this time step.
        """
        self.last_active = time.time()
        current_names = {item.get("name", "").lower() for item in cart_items}

        if self.cart_history:
            prev_names = self.cart_history[-1].item_names
            added = [n for n in current_names if n not in prev_names]
            removed = [n for n in prev_names if n not in current_names]
        else:
            added = list(current_names)
            removed = []

        snapshot = CartSnapshot(
            step=self.current_step,
            timestamp=time.time(),
            items=cart_items,
            item_names=current_names,
            completeness=completeness,
            items_added=added,
            items_removed=removed,
        )
        self.cart_history.append(snapshot)
        return snapshot

    def record_recommendations(self, rec_names: list[str]) -> None:
        """Record which items were recommended at this step."""
        self.recommendation_history.append(rec_names)

    def get_context(self) -> dict[str, Any]:
        """
        Return sequential context metadata for downstream use.

        Includes step number, recent deltas, completeness trajectory,
        and whether the meal appears to be 'done'.
        """
        if not self.cart_history:
            return {
                "session_step": 0,
                "is_first_interaction": True,
                "completeness_trajectory": [],
                "meal_is_done": False,
                "items_added_last_step": [],
                "items_removed_last_step": [],
                "total_items_added": 0,
                "recommendation_fatigue": 0.0,
            }

        latest = self.cart_history[-1]
        trajectory = [s.completeness for s in self.cart_history]

        # Recommendation fatigue: if we've shown many recs and user
        # keeps not adding them, reduce intensity
        total_recs_shown = sum(len(r) for r in self.recommendation_history)
        total_items_added = sum(len(s.items_added) for s in self.cart_history)
        fatigue = 0.0
        if total_recs_shown > 0 and self.current_step >= 3:
            acceptance_rate = total_items_added / max(total_recs_shown, 1)
            fatigue = max(0.0, 1.0 - acceptance_rate * 10)  # 0-1 scale

        return {
            "session_step": latest.step,
            "is_first_interaction": latest.step == 0,
            "completeness_trajectory": trajectory,
            "meal_is_done": latest.completeness >= 90.0,
            "items_added_last_step": latest.items_added,
            "items_removed_last_step": latest.items_removed,
            "total_items_added": total_items_added,
            "recommendation_fatigue": round(fatigue, 2),
            "current_completeness": latest.completeness,
        }


class SessionStore:
    """
    In-memory session store with TTL-based expiry.

    Keyed by ``session_id``. Evicts expired sessions lazily.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        """Retrieve an existing session or create a new one."""
        self._evict_expired()
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if not session.is_expired:
                return session
        session = SessionState(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        """Retrieve a session if it exists and isn't expired."""
        session = self._sessions.get(session_id)
        if session and not session.is_expired:
            return session
        return None

    def _evict_expired(self) -> None:
        """Remove expired sessions (lazy cleanup)."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_active) > SESSION_TTL_S
        ]
        for sid in expired:
            del self._sessions[sid]

    @property
    def active_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)
