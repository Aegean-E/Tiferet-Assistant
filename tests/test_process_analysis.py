import unittest
from unittest.mock import MagicMock, patch
import time
from ai_core.core_spotlight import GlobalWorkspace
from treeoflife.tiferet_components.thought_generator import ThoughtGenerator

class TestProcessAnalysis(unittest.TestCase):
    def setUp(self):
        # Mock Core
        self.mock_core = MagicMock()
        self.mock_core.event_bus = MagicMock()
        self.mock_core.self_model = None
        self.mock_core.memory_store = MagicMock()

        # Init Workspace
        self.workspace = GlobalWorkspace(self.mock_core)

        # Mock Decider for ThoughtGenerator
        self.mock_decider = MagicMock()
        self.mock_decider.global_workspace = self.workspace
        self.mock_decider.log = MagicMock()
        self.mock_decider.get_settings.return_value = {"base_url": "", "chat_model": "test"}
        self.mock_decider.reasoning_store = MagicMock()
        self.mock_decider.event_bus = MagicMock()
        self.mock_decider.decision_maker = MagicMock()

        self.thought_gen = ThoughtGenerator(self.mock_decider)

    def test_process_history_tracking(self):
        # Add events
        self.workspace.integrate("Thought 1", "ThoughtGenerator", 0.5)
        self.workspace.integrate("Action 1", "ActionManager", 0.8)

        trace = self.workspace.get_process_trace()
        self.assertIn("[ThoughtGenerator] Thought 1", trace)
        self.assertIn("[ActionManager] Action 1", trace)

    @patch('treeoflife.tiferet_components.thought_generator.run_local_lm')
    def test_perform_process_analysis(self, mock_lm):
        # Setup trace
        self.workspace.integrate("Objective: Test", "User", 1.0)
        self.workspace.integrate("Thought: Analyzing", "ThoughtGenerator", 0.5)

        # Mock LLM response
        mock_lm.return_value = '{"status": "FLOW", "suggestion": "Continue testing", "objective": "Test"}'

        # Run analysis
        self.thought_gen.perform_process_analysis()

        # Verify Log
        self.mock_decider.log.assert_any_call("🔍 Analysis: FLOW - Continue testing")

        # Verify Store
        self.mock_decider.reasoning_store.add.assert_called()

        # Verify Event
        self.mock_decider.event_bus.publish.assert_called_with(
            "PROCESS_ANALYSIS",
            {"status": "FLOW", "suggestion": "Continue testing", "objective": "Test"},
            source="ThoughtGenerator",
            priority=9
        )

if __name__ == '__main__':
    unittest.main()
