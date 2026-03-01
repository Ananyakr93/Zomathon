"""
Circuit Breaker & Failure Handling — CSAO Production
=====================================================
Implements three-state circuit breaker (CLOSED → OPEN → HALF_OPEN)
for external dependencies (LLM API, vector DB, feature store) with
automatic fallback chains and graceful degradation.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"          # normal operation
    OPEN = "open"              # all calls rejected, use fallback
    HALF_OPEN = "half_open"    # testing recovery


@dataclass
class CircuitBreakerConfig:
    """Tunable parameters for a circuit breaker."""
    failure_threshold: int = 5       # consecutive failures to trip
    recovery_timeout_s: float = 30.0  # how long to stay OPEN before HALF_OPEN
    half_open_max_calls: int = 3     # test calls before closing
    success_threshold: int = 2       # successes needed in HALF_OPEN to close
    timeout_s: float = 2.0           # individual call timeout


class CircuitBreaker:
    """
    Per-dependency circuit breaker.

    Usage:
        cb = CircuitBreaker("llm_api", config=CircuitBreakerConfig())

        result = cb.call(
            primary_fn=lambda: call_llm(prompt),
            fallback_fn=lambda: collaborative_filter_only(cart),
        )
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

        # Telemetry
        self._total_calls = 0
        self._total_failures = 0
        self._total_fallbacks = 0

    @property
    def state(self) -> CircuitState:
        """Check if we should transition from OPEN → HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.config.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
                logger.info(f"[CB:{self.name}] OPEN → HALF_OPEN after {elapsed:.1f}s")
        return self._state

    def call(
        self,
        primary_fn: Callable[[], Any],
        fallback_fn: Callable[[], Any] | None = None,
    ) -> Any:
        """
        Execute primary_fn if circuit is CLOSED/HALF_OPEN.
        If circuit is OPEN or primary fails, use fallback_fn.
        """
        self._total_calls += 1
        current_state = self.state

        if current_state == CircuitState.OPEN:
            logger.warning(f"[CB:{self.name}] Circuit OPEN — using fallback")
            self._total_fallbacks += 1
            if fallback_fn:
                return fallback_fn()
            raise CircuitOpenError(f"Circuit {self.name} is OPEN and no fallback provided")

        # CLOSED or HALF_OPEN — try the primary call
        try:
            result = primary_fn()
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            logger.warning(
                f"[CB:{self.name}] Primary failed ({exc.__class__.__name__}), "
                f"failures={self._failure_count}/{self.config.failure_threshold}"
            )
            self._total_fallbacks += 1
            if fallback_fn:
                return fallback_fn()
            raise

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"[CB:{self.name}] HALF_OPEN → CLOSED (recovered)")
        else:
            # CLOSED — reset streak
            self._failure_count = 0

    def _on_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN re-opens the circuit
            self._state = CircuitState.OPEN
            logger.warning(f"[CB:{self.name}] HALF_OPEN → OPEN (still failing)")
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                f"[CB:{self.name}] CLOSED → OPEN after {self._failure_count} failures"
            )

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_fallbacks": self._total_fallbacks,
            "fallback_rate": round(
                self._total_fallbacks / max(self._total_calls, 1), 4
            ),
        }

    def reset(self) -> None:
        """Force reset to CLOSED (manual intervention)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info(f"[CB:{self.name}] Manually reset to CLOSED")


class CircuitOpenError(Exception):
    """Raised when circuit is OPEN and no fallback is available."""
    pass


# ═════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ═════════════════════════════════════════════════════════════════════════════

class TokenBucketRateLimiter:
    """
    Token-bucket rate limiter for overload protection.

    rate       — tokens added per second
    capacity   — max burst size
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def allow(self) -> bool:
        """Return True if the request is allowed, False to shed load."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  FALLBACK STRATEGY MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class FallbackManager:
    """
    Manages graceful degradation with a prioritised fallback chain.

    Chain priority:
      1. LLM Re-ranking  (full quality)
      2. Graph-only Re-ranking  (no LLM, fast)
      3. Collaborative Filtering only  (baseline)
      4. Cold Start Agent  (no data at all)
      5. Popularity-based fallback  (emergency)
    """

    def __init__(self) -> None:
        self.breakers: dict[str, CircuitBreaker] = {
            "llm_api": CircuitBreaker(
                "llm_api",
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=30),
            ),
            "vector_db": CircuitBreaker(
                "vector_db",
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout_s=20),
            ),
            "feature_store": CircuitBreaker(
                "feature_store",
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout_s=15),
            ),
        }
        self.rate_limiter = TokenBucketRateLimiter(
            rate=50_000,    # 50K RPS max
            capacity=60_000  # burst headroom
        )

    def get_available_strategy(self) -> str:
        """
        Determine the highest-quality strategy currently available.
        Returns one of: "full", "graph_only", "cf_only", "cold_start", "popularity".
        """
        llm_ok = self.breakers["llm_api"].state != CircuitState.OPEN
        vector_ok = self.breakers["vector_db"].state != CircuitState.OPEN
        feature_ok = self.breakers["feature_store"].state != CircuitState.OPEN

        if llm_ok and vector_ok and feature_ok:
            return "full"
        elif vector_ok and feature_ok:
            return "graph_only"
        elif feature_ok:
            return "cf_only"
        elif vector_ok:
            return "cold_start"
        else:
            return "popularity"

    def check_rate_limit(self) -> bool:
        return self.rate_limiter.allow()

    def get_all_stats(self) -> dict:
        return {
            name: cb.get_stats() for name, cb in self.breakers.items()
        }
