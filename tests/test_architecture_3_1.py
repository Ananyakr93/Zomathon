"""
Unit tests for Architecture 3.1: Stage 1 (Graph Generation) and Stage 3 (SLM Reranking).
"""
import pytest
from unittest.mock import MagicMock, patch

from src.serving.orchestrator import ServingOrchestrator
from src.agents.slm_reranker_agent import SLMRerankerAgent


class MockLLMResponse:
    def __init__(self, text):
        self.text = text


class TestStage1GraphGeneration:
    def test_generate_candidates_empty_cart_fallback(self):
        """Test that an empty cart falls back to the ColdStartAgent."""
        orch = ServingOrchestrator()
        
        # Ensure that it calls ColdStartAgent
        with patch("src.agents.cold_start_agent.ColdStartAgent.recommend") as mock_recommend:
            mock_recommend.return_value = {"recommendations": [{"name": "Mock New User Item"}]}
            
            candidates = orch._generate_candidates(
                cart_items=[], restaurant={"id": "r1"}, features={}
            )
            
            assert mock_recommend.called
            assert len(candidates) == 1
            assert candidates[0]["name"] == "Mock New User Item"

    def test_generate_candidates_graph_traversal(self):
        """Test that a non-empty cart correctly hits the graph."""
        orch = ServingOrchestrator()
        
        cart = [{"name": "Chicken Tikka"}]
        
        # Mock the graph yielding results to avoid falling back to ColdStartAgent
        with patch("src.serving.orchestrator.SessionGraph.get_candidate_scores", return_value=[{"name": "Garlic Naan", "graph_score": 0.85}]):
            candidates = orch._generate_candidates(cart, {"id": "r1"}, {})
            
            assert isinstance(candidates, list)
            assert len(candidates) <= 20
            if candidates:
                assert "category" in candidates[0]
                assert "graph_score" in candidates[0]
                assert candidates[0]["name"] == "Garlic Naan"


class TestStage3SLMReranking:
    def test_slm_reranker_agent_success(self):
        """Test SLM agent successfully parses a valid JSON response."""
        agent = SLMRerankerAgent()
        
        candidates = [{"name": "Naan"}, {"name": "Coke"}]
        cart = [{"name": "Butter Chicken"}]
        
        # Mock the LLM call to return valid JSON
        mock_json = '''
        [
            {"name": "Naan", "rank": 1, "confidence": 0.9, "reasoning": "Great with curry"},
            {"name": "Coke", "rank": 2, "confidence": 0.5, "reasoning": "Standard drink"}
        ]
        '''
        
        with patch.object(agent.llm, "generate", return_value=MockLLMResponse(mock_json)):
            result = agent.rerank(candidates, cart, {}, {})
            
            assert result is not None
            assert len(result) == 2
            assert result[0]["name"] == "Naan"  # Ranked 1
            assert result[0]["slm_rank"] == 1
            assert result[0]["slm_enriched"] is True
            assert result[1]["name"] == "Coke"
            assert result[1]["slm_rank"] == 2

    def test_slm_reranker_agent_markdown_wrapper(self):
        """Test SLM agent handles JSON wrapped in markdown blocks."""
        agent = SLMRerankerAgent()
        
        candidates = [{"name": "Mango Lassi"}]
        
        mock_json = '''
        Here is the output:
        ```json
        [
            {"name": "Mango Lassi", "rank": 1, "confidence": 0.8, "reasoning": "Sweet."}
        ]
        ```
        '''
        with patch.object(agent.llm, "generate", return_value=MockLLMResponse(mock_json)):
            result = agent.rerank(candidates, [], {}, {})
            assert result is not None
            assert result[0]["slm_rank"] == 1

    def test_slm_reranker_agent_failure_fallback(self):
        """Test SLM agent returns None on malformed output, triggering Orchestrator fallback."""
        agent = SLMRerankerAgent()
        candidates = [{"name": "Naan"}]
        
        mock_text = "I'm sorry, I cannot generate JSON right now."
        
        with patch.object(agent.llm, "generate", return_value=MockLLMResponse(mock_text)):
            result = agent.rerank(candidates, [], {}, {})
            # Should return None, telling Orchestrator to use fallback deterministic reranker
            assert result is None

    def test_orchestrator_llm_rerank_fallback(self):
        """Test that Orchestrator uses deterministic reranker when SLM returns None."""
        orch = ServingOrchestrator()
        
        candidates = [{"name": "Naan"}]
        
        # Make SLMRerankerAgent return None
        with patch("src.agents.slm_reranker_agent.SLMRerankerAgent.rerank", return_value=None):
            # The orchestrator should fall back to MealContextAgent + RerankerAgent
            # We don't need to mock those, let them run normally.
            result = orch._llm_rerank(candidates, [], {"id": "r1"}, {})
            
            assert result is not None
            assert len(result) == 1
            # If it fell back, it shouldn't have 'slm_rank' from the SLM agent
            assert "slm_rank" not in result[0]
