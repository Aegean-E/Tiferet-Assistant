import unittest
import sys
import os
import shutil
import time
from unittest.mock import MagicMock

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies
sys.modules["numpy"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["faiss"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["wave"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.spatial.distance"] = MagicMock()

from ai_core.core_spotlight import GlobalWorkspace
from ai_core.core_self_model import SelfModel
from ai_core.ai_core import AICore

class TestConsciousness(unittest.TestCase):
    def setUp(self):
        # Mock AICore
        self.mock_core = MagicMock()
        self.mock_core.log = MagicMock()
        self.mock_core.event_bus = MagicMock()
        self.mock_core.get_settings = MagicMock(return_value={})

        # Setup SelfModel
        self.self_model_path = "./tests/test_self_model.json"
        if os.path.exists(self.self_model_path):
            os.remove(self.self_model_path)

        self.self_model = SelfModel(file_path=self.self_model_path)
        self.mock_core.self_model = self.self_model

        # Setup GlobalWorkspace
        self.gw = GlobalWorkspace(self.mock_core)
        self.gw.stream_file = "./tests/test_stream.md"
        self.mock_core.global_workspace = self.gw

        # Mock MemoryStore for gather_signals
        self.mock_core.memory_store = MagicMock()
        self.mock_core.memory_store.get_active_by_type.return_value = []

    def tearDown(self):
        if os.path.exists(self.self_model_path):
            os.remove(self.self_model_path)
        if os.path.exists(self.gw.stream_file):
            os.remove(self.gw.stream_file)
        if os.path.exists("./tests/test_self_model.json.tmp"):
             os.remove("./tests/test_self_model.json.tmp")

    def test_global_workspace_integration(self):
        # 1. Test Self-Projection
        projection = self.self_model.project_self()
        self.assertIn("IDENTITY", projection)
        self.assertIn("STATE", projection)
        self.assertIn("VALUES", projection)

        # 2. Test Integration into Workspace
        self.gw.integrate(projection, "SelfModel", 0.5)

        context = self.gw.get_context()
        self.assertIn("[SelfModel]", context)
        self.assertIn("IDENTITY", context)

        # 3. Test Broadcast
        if self.gw.working_memory:
            self.gw.broadcast(self.gw.working_memory[0])
            self.mock_core.event_bus.publish.assert_called()
            call_args = self.mock_core.event_bus.publish.call_args
            self.assertEqual(call_args[0][0], "CONSCIOUS_CONTENT")
            self.assertIn("full_context", call_args[1]['data'])

    def test_decay_and_pruning(self):
        self.gw.integrate("Fading Thought", "Test", 0.5)
        self.gw.last_update = time.time() - 10 # 10 seconds ago

        # Force decay
        self.gw.decay()

        # Should be pruned because salience <= 0.1
        self.assertEqual(len(self.gw.working_memory), 0)

    def test_capacity_limit(self):
        for i in range(10):
            self.gw.integrate(f"Thought {i}", "Test", 0.5 + (i * 0.01))

        self.gw.decay() # Sorts and prunes

        # Capacity is 7
        self.assertLessEqual(len(self.gw.working_memory), 7)
        # Should keep higher salience items (later ones)
        self.assertIn("Thought 9", self.gw.get_context())

if __name__ == '__main__':
    unittest.main()
