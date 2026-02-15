import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Create a dummy ai_core.lm module to prevent import errors during testing
mock_lm = MagicMock()
sys.modules['ai_core.lm'] = mock_lm

# Now we can import the modules under test
from ai_core.core_spotlight import GlobalWorkspace
from ai_core.core_phenomenology import Phenomenology

class TestConsciousness(unittest.TestCase):
    def setUp(self):
        self.mock_core = MagicMock()
        self.mock_core.get_settings.return_value = {
            "base_url": "http://test",
            "chat_model": "test-model",
            "embedding_model": "test-embed"
        }

        # Mock MemoryStore
        self.mock_memory_store = MagicMock()
        self.mock_core.memory_store = self.mock_memory_store

        # Mock SelfModel
        self.mock_self_model = MagicMock()
        self.mock_self_model.data = {
            "drives": {
                "circadian_phase": "day",
                "cognitive_energy": 0.8
            }
        }
        self.mock_core.self_model = self.mock_self_model

        # Mock EventBus
        self.mock_core.event_bus = MagicMock()

        # Initialize GlobalWorkspace
        # We mock os.makedirs and open to prevent file creation
        with patch('os.makedirs'), patch('builtins.open', new_callable=MagicMock):
            self.gw = GlobalWorkspace(self.mock_core)

    @patch('ai_core.core_spotlight.compute_embedding')
    def test_associative_resonance(self, mock_compute_embedding):
        # Setup
        mock_embedding = np.zeros(768)
        mock_compute_embedding.return_value = mock_embedding

        # Mock search results
        # Return list of tuples: (id, type, subject, text, sim)
        self.mock_memory_store.search.return_value = [
            (1, "FACT", "User", "Related Memory", 0.9)
        ]

        # Trigger resonance
        self.gw.associative_resonance("Test Thought")

        # Verify
        mock_compute_embedding.assert_called()
        self.mock_memory_store.search.assert_called()

        # Check if integrated
        found = False
        for item in self.gw.working_memory:
            if "Association: Related Memory" in item["content"]:
                found = True
                self.assertEqual(item["salience"], 0.3)
                break
        self.assertTrue(found)

    def test_integrate_sensory_stream(self):
        # Trigger sensory stream
        self.gw.integrate_sensory_stream()

        # Verify items in working memory
        found_circadian = False
        found_energy = False

        for item in self.gw.working_memory:
            if "Circadian Phase: day" in item["content"]:
                found_circadian = True
            if "Energy Level: 0.80" in item["content"]:
                found_energy = True

        self.assertTrue(found_circadian)
        self.assertTrue(found_energy)

    @patch('ai_core.core_spotlight.run_local_lm')
    def test_introspective_loop(self, mock_run_local_lm):
        # Add a dominant thought
        self.gw.integrate("Dominant Thought", "User", 0.9)

        # Mock LLM response (VALID)
        mock_run_local_lm.return_value = "[VALID]"

        # Force introspection (bypass random check by mocking random)
        with patch('ai_core.core_spotlight.random.random', return_value=0.1): # < 0.2
            self.gw.introspective_loop()

        # Verify salience boosted
        dominant = self.gw.get_dominant_thought()
        self.assertGreater(dominant["salience"], 0.9)

        # Mock LLM response (INVALID)
        mock_run_local_lm.return_value = "[INVALID]"

        # Force introspection again
        with patch('ai_core.core_spotlight.random.random', return_value=0.1):
            self.gw.introspective_loop()

        # Verify salience reduced
        dominant = self.gw.get_dominant_thought()
        self.assertLess(dominant["salience"], 1.0) # Should be around 0.7 (1.0 - 0.3)

    def test_phenomenology_vad_update(self):
        phenom = Phenomenology(self.mock_core)

        # Test "burdened" (New keyword)
        phenom.valence = 0.5
        phenom.dominance = 0.5
        phenom._update_vad("I feel burdened")

        self.assertLess(phenom.valence, 0.5)
        self.assertLess(phenom.dominance, 0.5)

        # Test "sharp" (New keyword)
        phenom.valence = 0.5
        phenom.dominance = 0.5
        phenom.arousal = 0.5
        phenom._update_vad("I feel sharp")

        self.assertGreater(phenom.valence, 0.5)
        self.assertGreater(phenom.dominance, 0.5)
        self.assertGreater(phenom.arousal, 0.5)

if __name__ == '__main__':
    unittest.main()
