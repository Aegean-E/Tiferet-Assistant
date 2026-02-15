import unittest
import sys
import os
import time
from unittest.mock import MagicMock

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import numpy
except ImportError:
    # If numpy is truly missing, we mock it, but give it a version
    sys.modules['numpy'] = MagicMock()
    sys.modules['numpy'].__version__ = "1.24.0"

# Mock only heavy/optional dependencies that might be missing in test env
# but keep core ones if available
sys.modules['faiss'] = MagicMock()
sys.modules['openai_whisper'] = MagicMock()
sys.modules['whisper'] = MagicMock()
sys.modules['pyvis'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['dowhy'] = MagicMock()
sys.modules['ttkbootstrap'] = MagicMock()
sys.modules['PyMuPDF'] = MagicMock()
sys.modules['fitz'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['PIL'] = MagicMock()

from ai_core.core_phenomenology import Phenomenology
from ai_core.core_self_model import SelfModel

class TestPhenomenology(unittest.TestCase):
    def setUp(self):
        self.mock_core = MagicMock()
        self.mock_core.event_bus = MagicMock()
        self.mock_core.log = MagicMock()
        self.mock_core.get_settings = MagicMock(return_value={})

        # Mock SelfModel
        self.self_model_path = "./tests/test_self_model_phenom.json"
        if os.path.exists(self.self_model_path):
            os.remove(self.self_model_path)
        self.self_model = SelfModel(file_path=self.self_model_path)
        self.mock_core.self_model = self.self_model

        # Mock Global Workspace
        self.mock_core.global_workspace = MagicMock()

        self.phenom = Phenomenology(self.mock_core)

    def tearDown(self):
        if os.path.exists(self.self_model_path):
            os.remove(self.self_model_path)
        if os.path.exists(self.self_model_path + ".tmp"):
             try:
                 os.remove(self.self_model_path + ".tmp")
             except: pass

    def test_initialization(self):
        self.assertEqual(self.phenom.valence, 0.5)
        self.assertEqual(self.phenom.arousal, 0.5)
        self.mock_core.event_bus.subscribe.assert_called_with("CONSCIOUS_CONTENT", self.phenom._process_experience)

    def test_update_vad_heuristic(self):
        # Test Positive
        self.phenom._update_vad("Great success achieved")
        self.assertGreater(self.phenom.valence, 0.5)

        # Test Negative
        self.phenom.valence = 0.5
        self.phenom._update_vad("Critical failure error")
        self.assertLess(self.phenom.valence, 0.5)

        # Test Arousal
        self.phenom.arousal = 0.5
        self.phenom._update_vad("Urgent panic alert")
        self.assertGreater(self.phenom.arousal, 0.5)

    def test_process_experience(self):
        # Create a mock event
        mock_event = MagicMock()
        mock_event.data = {"focus": "Success and progress", "full_context": "..."}

        # Initial mood
        self.self_model.data["drives"]["mood"] = 0.5

        # Process
        self.phenom._process_experience(mock_event)

        # Check VAD update
        self.assertGreater(self.phenom.valence, 0.5)

        # Check SelfModel update (Mood should drift up)
        self.assertGreater(self.self_model.data["drives"]["mood"], 0.5)

    def test_internal_monologue_integration(self):
        # Force high arousal to trigger monologue
        self.phenom.arousal = 0.9
        self.phenom.last_update = 0 # Force update

        # Mock run_local_lm
        import ai_core.core_phenomenology
        original_lm = ai_core.core_phenomenology.run_local_lm
        ai_core.core_phenomenology.run_local_lm = MagicMock(return_value="I feel good about this.")

        try:
            self.phenom._generate_internal_monologue("Something happened")

            # Check if integrated into GW
            self.mock_core.global_workspace.integrate.assert_called()
            args = self.mock_core.global_workspace.integrate.call_args
            self.assertEqual(args[1]['content'], "I feel good about this.")
            self.assertEqual(args[1]['source'], "Inner Voice")
        finally:
            ai_core.core_phenomenology.run_local_lm = original_lm

if __name__ == '__main__':
    unittest.main()
