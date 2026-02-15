import unittest
from unittest.mock import MagicMock, patch, ANY
import sys
import os
import importlib
import time
from datetime import datetime

# Ensure path is set up if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSafetyExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create mocks
        cls.numpy_mock = MagicMock()
        cls.lm_mock = MagicMock()

        # Patch sys.modules to mock missing dependencies
        cls.patcher = patch.dict(sys.modules, {
            'numpy': cls.numpy_mock,
            'ai_core.lm': cls.lm_mock
        })
        cls.patcher.start()

        # Import/Reload the module under test
        import ai_core.core_actions
        importlib.reload(ai_core.core_actions)

        cls.ActionManager = ai_core.core_actions.ActionManager

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        self.mock_core = MagicMock()
        self.mock_core.thread_pool = MagicMock()

        # Mock ValueCore
        self.mock_core.value_core = MagicMock()
        self.mock_core.value_core.check_alignment.return_value = (True, 0.0, "Aligned")

        # Mock Settings
        self.mock_core.get_settings.return_value = {"permissions": {}}

        self.am = self.ActionManager(self.mock_core)

        # Mock Malkuth
        self.mock_core.malkuth = MagicMock()
        self.mock_core.malkuth.write_file.return_value = "File written"
        self.mock_core.malkuth.describe_image.return_value = "Image description"

    def test_high_risk_triggers_full_check(self):
        # WRITE_FILE is HIGH risk
        # This bypasses the actual file write logic because we are mocking malkuth,
        # but we want to test that process_tool_calls calls check_alignment with fast_check_only=False

        # We need to ensure _validate_path allows the write for the test to proceed to malkuth (or fail at safety check)
        # However, process_tool_calls calls check_alignment BEFORE executing the tool function.

        cmd = "[EXECUTE: WRITE_FILE, 'test.txt, content']"

        # process_tool_calls relies on am.get_tools()
        # We need to make sure WRITE_FILE is in get_tools()
        # It is by default.

        # We need to mock _validate_path if we want execution to succeed, but safety check happens first.

        self.am.process_tool_calls(cmd)

        self.mock_core.value_core.check_alignment.assert_called_with(
            proposal=ANY,
            context="Tool Execution (HIGH Risk)",
            fast_check_only=False
        )

    def test_medium_risk_triggers_fast_check(self):
        # SEARCH is MEDIUM risk
        cmd = "[EXECUTE: SEARCH, 'query']"

        # Mock safe_search
        self.am.safe_search = MagicMock(return_value="Search Result")

        self.am.process_tool_calls(cmd)

        self.mock_core.value_core.check_alignment.assert_called_with(
            proposal=ANY,
            context="Tool Execution (MEDIUM Risk)",
            fast_check_only=True
        )

    def test_safety_block_prevents_execution(self):
        # Simulate ValueCore rejecting the action
        self.mock_core.value_core.check_alignment.return_value = (False, 1.0, "Harmful content")

        cmd = "[EXECUTE: WRITE_FILE, 'virus.py, import os; os.system(...)']"

        result = self.am.process_tool_calls(cmd)

        self.assertIn("Safety Block", result)
        self.assertIn("Harmful content", result)

        # Ensure tool was NOT executed
        self.mock_core.malkuth.write_file.assert_not_called()

    def test_rate_limiting(self):
        # HIGH risk tool limit is 5
        tool_name = "WRITE_FILE"
        cmd = f"[EXECUTE: {tool_name}, 'test.txt, content']"

        # We assume _write_file_tool will fail or succeed, doesn't matter, we care about rate limit check
        # Mock _write_file_tool to do nothing to avoid side effects/errors
        self.am._write_file_tool = MagicMock(return_value="Written")

        # Call 5 times (allowed)
        for _ in range(5):
            self.am.process_tool_calls(cmd)

        # 6th call should be blocked
        result = self.am.process_tool_calls(cmd)

        self.assertIn("Rate limit exceeded", result)
        self.assertEqual(self.am._write_file_tool.call_count, 5)

if __name__ == '__main__':
    unittest.main()
