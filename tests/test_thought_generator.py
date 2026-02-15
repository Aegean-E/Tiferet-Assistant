import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies if not installed
if 'numpy' not in sys.modules:
    sys.modules['numpy'] = MagicMock()
if 'requests' not in sys.modules:
    sys.modules['requests'] = MagicMock()

mock_lm_module = MagicMock()
sys.modules['ai_core.lm'] = mock_lm_module

from treeoflife.tiferet_components.thought_generator import ThoughtGenerator, ThinkingStrategy

class TestThoughtGenerator(unittest.TestCase):
    def setUp(self):
        self.decider = MagicMock()
        self.decider.get_settings.return_value = {
            "base_url": "http://mock",
            "chat_model": "mock-model",
            "embedding_model": "mock-embed",
        }
        self.decider.stop_check.return_value = False
        self.decider.daat.provide_reasoning_structure.return_value = "Step 1: Analyze."

        self.tg = ThoughtGenerator(self.decider)

    @patch('treeoflife.tiferet_components.thought_generator.run_local_lm')
    @patch('treeoflife.tiferet_components.thought_generator.compute_embedding')
    def test_perform_linear_chain(self, mock_embed, mock_lm):
        mock_lm.side_effect = [
            "1. Step One\n2. Step Two", # Reasoning
            "Summary of linear reasoning." # Synthesis
        ]

        self.tg.perform_thinking_chain("Test Topic", strategy="LINEAR")

        # Verify calls
        self.assertEqual(mock_lm.call_count, 2)

        # Check first call (Reasoning)
        call_args_1 = mock_lm.call_args_list[0]
        # Unpack call object
        args1, kwargs1 = call_args_1

        # messages might be positional or keyword
        if 'messages' in kwargs1:
            messages1 = kwargs1['messages']
        else:
            messages1 = args1[0]

        self.assertIn("Reason through this topic step-by-step", messages1[-1]['content'])

        # Check second call (Synthesis)
        call_args_2 = mock_lm.call_args_list[1]
        args2, kwargs2 = call_args_2

        if 'messages' in kwargs2:
            messages2 = kwargs2['messages']
        else:
            messages2 = args2[0]

        self.assertIn("Synthesize a clear, coherent, and actionable conclusion", messages2[-1]['content'])

        # Verify storage
        self.decider.command_executor.create_note.assert_called()
        self.decider.reasoning_store.add.assert_called()

    @patch('treeoflife.tiferet_components.thought_generator.run_local_lm')
    @patch('treeoflife.tiferet_components.thought_generator.compute_embedding')
    def test_perform_first_principles(self, mock_embed, mock_lm):
        mock_lm.side_effect = [
            "Axiom 1. Axiom 2.", # Deconstruct
            "Reconstructed Conclusion.", # Reconstruct
            "Synthesized Conclusion." # Synthesis
        ]

        self.tg.perform_thinking_chain("Test Topic", strategy="FIRST_PRINCIPLES")

        self.assertEqual(mock_lm.call_count, 3)
        self.decider.command_executor.create_note.assert_called()

    @patch('treeoflife.tiferet_components.thought_generator.run_local_lm')
    @patch('treeoflife.tiferet_components.thought_generator.compute_embedding')
    def test_perform_tree_of_thoughts(self, mock_embed, mock_lm):
        # Mocks for ToT:
        # 1. Expand (Depth 1) -> ["Step A"]
        # 2. Evaluate (Step A) -> Score 0.8
        # 3. Expand (Depth 2) -> ["[CONCLUSION] Done"]
        # 4. Synthesize Conclusion

        mock_lm.side_effect = [
            '["Step A"]', # Expand 1
            json.dumps({"score": 0.8}), # Evaluate Step A
            '["[CONCLUSION] Done"]', # Expand 2 (Conclusion found)
            "Synthesized ToT Conclusion." # Synthesis
        ]

        self.tg.perform_thinking_chain("Test Topic", strategy="TREE_OF_THOUGHTS", max_depth=2)

        # Verify flow
        self.decider.log.assert_any_call("🌳 Depth 1: Expanding 1 paths...")
        # Should stop after conclusion
        # Verify synthesis
        self.decider.command_executor.create_note.assert_called()
        call_args = self.decider.command_executor.create_note.call_args[0][0]
        self.assertIn("Synthesized ToT Conclusion", call_args)

    @patch('treeoflife.tiferet_components.thought_generator.run_local_lm')
    def test_choose_strategy(self, mock_lm):
        mock_lm.return_value = "FIRST_PRINCIPLES"
        strategy = self.tg._choose_strategy("Complex Topic")
        self.assertEqual(strategy, ThinkingStrategy.FIRST_PRINCIPLES)

if __name__ == '__main__':
    unittest.main()
