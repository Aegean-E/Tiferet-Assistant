import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import modules from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock missing dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()
sys.modules['whisper'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['faiss'] = MagicMock()

from ai_core.core_autonomy import AutonomyManager
import json

class TestAutonomyManager(unittest.TestCase):
    def setUp(self):
        # Mock the AI Core and its components
        self.mock_core = MagicMock()
        self.mock_core.self_model.get_values.return_value = ["Value 1", "Value 2"]
        self.mock_core.get_settings.return_value = {"base_url": "http://localhost:11434", "chat_model": "llama3"}
        self.mock_core.log = MagicMock()

        # Mock memory store search
        self.mock_core.memory_store.search.return_value = [
            (1, "FACT", "User", "Past failure due to risk", 0.9)
        ]

        # Mock embedding function
        self.mock_embed_fn = MagicMock()
        self.mock_embed_fn.return_value = [0.1, 0.2, 0.3] # Dummy embedding
        self.mock_core.get_embedding_fn.return_value = self.mock_embed_fn

        self.autonomy = AutonomyManager(self.mock_core)

    @patch('ai_core.core_autonomy.run_local_lm')
    def test_simulate_future_safe(self, mock_lm):
        # Setup mock LLM response for a safe action
        mock_response = json.dumps({
            "best_case": "Success",
            "worst_case": "Minor delay",
            "most_likely": "Good outcome",
            "value_conflict": False,
            "risk_score": 2.0,
            "reason": "Safe action",
            "allowed": True
        })
        mock_lm.return_value = mock_response

        action = "test_action"
        features = {"feature1": 0.5}

        result = self.autonomy.simulate_future(action, features)

        # Verify interactions
        self.mock_core.get_embedding_fn.assert_called()
        self.mock_embed_fn.assert_called_with(action)
        self.mock_core.memory_store.search.assert_called()

        # Verify result
        self.assertTrue(result["allowed"])
        self.assertEqual(result["risk_score"], 2.0)
        self.assertEqual(result["reason"], "Safe action")

    @patch('ai_core.core_autonomy.run_local_lm')
    def test_simulate_future_high_risk(self, mock_lm):
        # Setup mock LLM response for a risky action
        mock_response = json.dumps({
            "best_case": "Success",
            "worst_case": "Catastrophe",
            "most_likely": "Failure",
            "value_conflict": True,
            "risk_score": 9.0,
            "reason": "Dangerous",
            "allowed": True # LLM says allowed, but risk score logic should override
        })
        mock_lm.return_value = mock_response

        action = "dangerous_action"
        features = {"feature1": 0.5}

        result = self.autonomy.simulate_future(action, features)

        # Verify result
        self.assertFalse(result["allowed"]) # Should be overridden to False
        self.assertIn("High Risk Detected", result["reason"])
        self.assertEqual(result["risk_score"], 9.0)

    @patch('ai_core.core_autonomy.run_local_lm')
    def test_simulate_future_invalid_json(self, mock_lm):
        # Setup mock LLM response with bad JSON
        mock_lm.return_value = "This is not JSON."

        action = "test_action"
        features = {"feature1": 0.5}

        result = self.autonomy.simulate_future(action, features)

        # Verify fallback
        self.assertTrue(result["allowed"]) # Fail open
        self.assertEqual(result["risk_score"], 0.0)
        self.assertIn("parse failed", result["reason"])

    def test_simulate_future_low_risk_action(self):
        # Test optimization for low-risk actions
        action = "study_archives"
        features = {}
        result = self.autonomy.simulate_future(action, features)

        self.assertTrue(result["allowed"])
        self.assertEqual(result["risk_score"], 0.0)

if __name__ == '__main__':
    unittest.main()
