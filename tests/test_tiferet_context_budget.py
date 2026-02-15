import unittest
from unittest.mock import MagicMock
import sys
import os

# Ensure repo root is in path
sys.path.append(os.getcwd())

from treeoflife.tiferet import Decider

class TestEnforceContextBudget(unittest.TestCase):
    def setUp(self):
        # Mock dependencies required for Decider initialization
        self.mock_settings = {"default_temperature": 0.7, "default_max_tokens": 800}
        self.mock_get_settings = MagicMock(return_value=self.mock_settings)
        self.mock_update_settings = MagicMock()
        self.mock_memory_store = MagicMock()
        self.mock_document_store = MagicMock()
        self.mock_reasoning_store = MagicMock()
        self.mock_arbiter = MagicMock()
        self.mock_meta_memory_store = MagicMock()
        self.mock_actions = {}
        self.mock_log = MagicMock()

        self.decider = Decider(
            get_settings_fn=self.mock_get_settings,
            update_settings_fn=self.mock_update_settings,
            memory_store=self.mock_memory_store,
            document_store=self.mock_document_store,
            reasoning_store=self.mock_reasoning_store,
            arbiter=self.mock_arbiter,
            meta_memory_store=self.mock_meta_memory_store,
            actions=self.mock_actions,
            log_fn=self.mock_log
        )
        # Disable background executor
        self.decider.executor = None

    def test_fit_within_budget(self):
        """Test that blocks fitting within the budget are included fully."""
        blocks = ["A" * 50, "B" * 50]
        result = self.decider.chat_handler._enforce_context_budget(blocks, 150)
        self.assertEqual(len(result), 100)
        self.assertEqual(result, "A" * 50 + "B" * 50)

    def test_exact_fit(self):
        """Test exact budget fit."""
        blocks = ["A" * 50, "B" * 50]
        result = self.decider.chat_handler._enforce_context_budget(blocks, 100)
        self.assertEqual(len(result), 100)
        self.assertEqual(result, "A" * 50 + "B" * 50)

    def test_prioritize_recent_and_skip_when_remaining_small(self):
        """
        Test that recent blocks (later in list) are prioritized,
        and older blocks are skipped if remaining space is small (<= 100).
        """
        # Blocks: A (100), B (50), C (40)
        blocks = ["A" * 100, "B" * 50, "C" * 40]
        max_chars = 100

        # Processing order (reversed): C, B, A
        # 1. C (40) fits. Remaining: 60. Final: [C]
        # 2. B (50) fits. Remaining: 10. Final: [B, C] (prepended)
        # 3. A (100) doesn't fit. Remaining space = 10. 10 <= 100. Skip.

        result = self.decider.chat_handler._enforce_context_budget(blocks, max_chars)
        self.assertEqual(result, "B" * 50 + "C" * 40)

    def test_truncation_logic(self):
        """
        Test that if remaining space > 100, the block is truncated
        (keeping the end) and a marker is prepended.
        """
        blocks = ["A" * 200, "B" * 50]
        max_chars = 200

        # Processing order: B, A
        # 1. B (50) fits. Remaining: 150. Final: [B]
        # 2. A (200) doesn't fit. Remaining: 150.
        #    150 > 100, so truncate A to last 150 chars.
        #    Prepend marker.

        result = self.decider.chat_handler._enforce_context_budget(blocks, max_chars)

        marker = "... [Context Truncated] ...\n"
        truncated_part = ("A" * 200)[-150:]
        expected_result = marker + truncated_part + ("B" * 50)

        self.assertEqual(result, expected_result)
        # Note: result length exceeds max_chars by len(marker)
        self.assertEqual(len(result), max_chars + len(marker))

    def test_skip_empty_blocks(self):
        """Test that empty blocks are ignored."""
        blocks = ["", "A" * 50, ""]
        result = self.decider.chat_handler._enforce_context_budget(blocks, 100)
        self.assertEqual(result, "A" * 50)

    def test_order_preservation(self):
        """Test that original order of blocks is preserved in output."""
        blocks = ["1", "2", "3"]
        result = self.decider.chat_handler._enforce_context_budget(blocks, 10)
        self.assertEqual(result, "123")

if __name__ == '__main__':
    unittest.main()
