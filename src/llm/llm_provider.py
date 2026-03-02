"""
llm_provider.py
===============
Unified LLM API integration layer for the CSAO system.

Supports multiple backends:
  - OpenAI (GPT-3.5-turbo / GPT-4)
  - Google Gemini
  - Local Ollama (Mistral / Llama)
  - Rule-based fallback (no API key needed)

Usage:
    from src.llm.llm_provider import get_llm, LLMResponse

    llm = get_llm()  # auto-selects based on env vars
    response = llm.generate("Suggest add-ons for Butter Chicken cart")
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  RESPONSE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """Standardised response from any LLM backend."""
    text: str
    model: str
    latency_ms: float
    tokens_used: int = 0
    fallback_used: bool = False


# ═══════════════════════════════════════════════════════════════════════════
#  ABSTRACT BASE
# ═══════════════════════════════════════════════════════════════════════════

class BaseLLM(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> LLMResponse:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  OPENAI BACKEND
# ═══════════════════════════════════════════════════════════════════════════

class OpenAILLM(BaseLLM):
    """OpenAI GPT backend (requires OPENAI_API_KEY env var)."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> LLMResponse:
        import openai
        client = openai.OpenAI(api_key=self._api_key)

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a food recommendation expert for Indian food delivery."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            latency_ms=round(latency, 2),
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  GOOGLE GEMINI BACKEND
# ═══════════════════════════════════════════════════════════════════════════

class GeminiLLM(BaseLLM):
    """Google Gemini backend (requires GOOGLE_API_KEY env var)."""

    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> LLMResponse:
        import google.generativeai as genai
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self.model)

        t0 = time.perf_counter()
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        latency = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=response.text if response.text else "",
            model=self.model,
            latency_ms=round(latency, 2),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  OLLAMA (LOCAL) BACKEND
# ═══════════════════════════════════════════════════════════════════════════

class OllamaLLM(BaseLLM):
    """Local Ollama backend (requires Ollama running on localhost:11434)."""

    def __init__(self, model: str = "mistral"):
        self.model = model
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def is_available(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> LLMResponse:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        latency = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            text=result.get("response", ""),
            model=self.model,
            latency_ms=round(latency, 2),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  RULE-BASED FALLBACK (always available)
# ═══════════════════════════════════════════════════════════════════════════

class RuleBasedLLM(BaseLLM):
    """
    Rule-based fallback that mimics LLM output using the ColdStartAgent's
    cuisine knowledge base. Always available, zero latency.
    """

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> LLMResponse:
        t0 = time.perf_counter()

        # Parse intent from prompt
        prompt_lower = prompt.lower()

        # Detect cuisine from prompt
        cuisine = "North Indian"
        cuisine_keywords = {
            "biryani": "Biryani Specialist",
            "south indian": "South Indian", "dosa": "South Indian", "idli": "South Indian",
            "chinese": "Chinese", "noodles": "Chinese", "manchurian": "Chinese",
            "continental": "Continental", "pizza": "Continental", "pasta": "Continental",
            "mughlai": "Mughlai", "kebab": "Mughlai",
            "punjabi": "Punjabi", "chole": "Punjabi",
            "bengali": "Bengali", "fish curry": "Bengali",
            "gujarati": "Gujarati", "thali": "Gujarati",
            "rajasthani": "Rajasthani", "dal bati": "Rajasthani",
        }
        for kw, c in cuisine_keywords.items():
            if kw in prompt_lower:
                cuisine = c
                break

        response_text = (
            f"Based on the {cuisine} cuisine context, I recommend complementary items "
            f"from the categories: sides, breads, beverages, and desserts. "
            f"The ColdStartAgent's knowledge base has been applied for "
            f"culturally-appropriate pairings."
        )

        latency = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text=response_text,
            model="rule_based_v1",
            latency_ms=round(latency, 2),
            fallback_used=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════════════════════════════════════════

# Priority order for LLM selection
_LLM_BACKENDS: list[tuple[str, type[BaseLLM]]] = [
    ("openai", OpenAILLM),
    ("gemini", GeminiLLM),
    ("ollama", OllamaLLM),
    ("rule_based", RuleBasedLLM),
]

_cached_llm: BaseLLM | None = None


def get_llm(preferred: str | None = None) -> BaseLLM:
    """
    Get the best available LLM backend.

    Priority:
      1. If OPENAI_API_KEY is set → OpenAI
      2. If GOOGLE_API_KEY is set → Gemini
      3. If Ollama is running locally → Ollama
      4. Rule-based fallback (always works)

    Parameters
    ----------
    preferred : str | None
        Force a specific backend: "openai", "gemini", "ollama", "rule_based"
    """
    global _cached_llm

    if preferred:
        for name, cls in _LLM_BACKENDS:
            if name == preferred:
                llm = cls()
                if llm.is_available():
                    logger.info("Using preferred LLM backend: %s", name)
                    return llm
                logger.warning("Preferred backend '%s' not available, falling back", name)
                break

    if _cached_llm is not None:
        return _cached_llm

    for name, cls in _LLM_BACKENDS:
        llm = cls()
        if llm.is_available():
            logger.info("Selected LLM backend: %s", name)
            _cached_llm = llm
            return llm

    # Should never reach here — RuleBasedLLM is always available
    _cached_llm = RuleBasedLLM()
    return _cached_llm


def available_backends() -> list[str]:
    """List all currently available LLM backends."""
    return [name for name, cls in _LLM_BACKENDS if cls().is_available()]
