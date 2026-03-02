"""
SLM Re-Ranking Agent — CSAO Agent #3 (Stage 3)
==============================================
Re-ranks Stage 1 top-20 candidates using a local Small Language Model (SLM)
via Ollama/HuggingFace to provide human-like reasoning.

Produces JSON output mapping candidate to Rank (1-20), Confidence
Score (0-1), and Reasoning string.

Usage:
    from src.agents.slm_reranker_agent import SLMRerankerAgent

    agent = SLMRerankerAgent()
    result = agent.rerank(candidates, cart, context, user_pref)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..llm.llm_provider import get_llm

logger = logging.getLogger(__name__)


class SLMRerankerAgent:
    """
    Stage 3: SLM Re-Ranking (The AI Edge)
     Constructs a prompt with current cart, context, and candidates.
     Requests the LLM to output a JSON ranking with reasoning.
     Parses and applies the new ranks to the candidates.
    """

    def __init__(self) -> None:
        self.llm = get_llm()
        self.system_prompt = (
            "You are a meal recommendation expert for Indian cuisine. "
            "Your goal is to re-rank the provided candidate items to best complement "
            "the user's current cart. Focus on creating a complete, satisfying meal "
            "while respecting the user's dietary preferences and contextual details. "
            "Output ONLY valid JSON. Do not include markdown codeblocks or extra text."
        )

    def rerank(
        self,
        candidates: list[dict],
        cart_items: list[dict],
        context: dict,
        user_preferences: dict | None,
    ) -> list[dict] | None:
        """
        Re-ranks top-20 candidates using the LLM.

        Returns
        -------
        list[dict] | None
            The re-ranked list of candidates adorned with 'slm_reasoning' and 'slm_rank'.
            Returns None if the LLM fails or the output is unparseable (triggering fallback).
        """
        if not candidates:
            return []

        prompt = self._build_prompt(candidates, cart_items, context, user_preferences)
        logger.debug("SLM Prompt length: %d chars", len(prompt))

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,  # Keep it deterministic
            )
            raw_text = response.text
            logger.debug("SLM Response length: %d chars", len(raw_text))

            parsed = self._parse_json_response(raw_text)
            if not parsed:
                logger.warning("SLM returned empty or unparseable JSON array.")
                return None

            return self._apply_llm_ranking(candidates, parsed)

        except Exception as exc:
            logger.error("SLM Re-ranking failed: %s", exc)
            return None

    def _build_prompt(
        self,
        candidates: list[dict],
        cart_items: list[dict],
        context: dict,
        user_preferences: dict | None,
    ) -> str:
        """Construct the prompt with cart, context, and candidate details."""
        # 1. Cart Context
        cart_desc = ", ".join(
            f"{i.get('name', '')} (₹{i.get('price', 0)}, {'Veg' if i.get('is_veg', True) else 'Non-Veg'})"
            for i in cart_items
        ) if cart_items else "Empty Cart"

        # 2. User Context
        city = context.get("city", "Unknown")
        meal_type = context.get("meal_type", "Unknown")
        diet = user_preferences.get("dietary_preference", "any") if user_preferences else "any"
        spice = user_preferences.get("preferred_spice_level", "any") if user_preferences else "any"
        
        user_desc = (
            f"Location: {city}, Meal Time: {meal_type}, "
            f"Diet: {diet}, Spice Pref: {spice}"
        )

        # 3. Candidates
        cand_list = []
        for i, c in enumerate(candidates):
            name = c.get('name', 'Unknown')
            price = c.get('price', 0)
            veg = 'Veg' if c.get('is_veg', True) else 'Non-Veg'
            cat = c.get('category', 'other')
            cand_list.append(f"{i+1}. {name} [₹{price}, {veg}, Category: {cat}]")
        
        cand_desc = "\n".join(cand_list)

        # 4. Instruction
        task = (
            "Task:\n"
            "Re-rank these candidate items from most suitable (Rank 1) to least suitable. "
            "Provide a Confidence score (0.0 to 1.0) and a 1-sentence Reasoning explaining "
            "why it completes or enhances the meal based on the context.\n\n"
            "Output Format:\n"
            "Return a JSON array of objects, where each object has the keys:\n"
            '- "name": (Exact string matching the candidate name)\n'
            '- "rank": (Integer from 1 to N)\n'
            '- "confidence": (Float from 0.0 to 1.0)\n'
            '- "reasoning": (String explanation)\n\n'
            "Strict rules:\n"
            "1. Output ONLY the JSON array (no markdown, no backticks, no 'Here is').\n"
            "2. Include EVERY candidate provided."
        )

        return (
            f"Current Cart:\n{cart_desc}\n\n"
            f"User Context:\n{user_desc}\n\n"
            f"Candidate Items:\n{cand_desc}\n\n"
            f"{task}"
        )

    def _parse_json_response(self, text: str) -> list[dict] | None:
        """Robustly parse the SLM JSON text."""
        cleaned = text.strip()
        
        # Remove markdown code block markers if the SLM ignored the prompt rule
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
            
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            return None
        except json.JSONDecodeError as exc:
            logger.debug("Failed to decode SLM JSON: %s. Raw text: %s", exc, text[:200])
            return None

    def _apply_llm_ranking(self, original_candidates: list[dict], parsed_ranking: list[dict]) -> list[dict]:
        """Align the JSON response with the original candidate dictionaries."""
        # Create lookup by lowercased name
        rank_lookup = {
            item.get("name", "").lower(): {
                "rank": item.get("rank", 999),
                "confidence": item.get("confidence", 0.5),
                "reasoning": item.get("reasoning", "No reason provided."),
            }
            for item in parsed_ranking
            if "name" in item
        }

        enriched_candidates = []
        for cand in original_candidates:
            name_key = cand.get("name", "").lower()
            llm_info = rank_lookup.get(name_key)
            
            cand_copy = cand.copy()
            if llm_info:
                cand_copy["slm_rank"] = llm_info["rank"]
                cand_copy["final_score"] = llm_info["confidence"]
                cand_copy["slm_reasoning"] = llm_info["reasoning"]
                cand_copy["slm_enriched"] = True
            else:
                # Fallback ranking for items the LLM missed
                cand_copy["slm_rank"] = 999
                cand_copy["final_score"] = cand.get("graph_score", 0.0)
                cand_copy["slm_reasoning"] = "Graph retrieval fallback (dropped by SLM)."
                cand_copy["slm_enriched"] = False
                
            enriched_candidates.append(cand_copy)

        # Sort by the new LLM rank
        enriched_candidates.sort(key=lambda x: x["slm_rank"])
        return enriched_candidates
