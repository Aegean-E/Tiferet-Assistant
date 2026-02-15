import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib

class TestCoreActions(unittest.TestCase):
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
        # We need to ensure 'ai_core' package exists
        import ai_core

        try:
            import ai_core.core_actions
            importlib.reload(ai_core.core_actions)
        except ImportError:
             # Should not happen if mocks are working
             pass

        cls.ActionManager = ai_core.core_actions.ActionManager

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def setUp(self):
        self.mock_core = MagicMock()
        self.am = self.ActionManager(self.mock_core)

    def test_safe_calculate_normal(self):
        self.assertEqual(self.am._safe_calculate("2 + 2"), "4")
        self.assertEqual(self.am._safe_calculate("2 * 3"), "6")
        self.assertEqual(self.am._safe_calculate("10 / 2"), "5.0")
        self.assertEqual(self.am._safe_calculate("2**10"), "1024") # Safe power

    def test_safe_calculate_dos(self):
        # 9**999999 should be rejected
        result = self.am._safe_calculate("9**999999")
        self.assertIn("Exponent too large", result)

        # 2**1001 should be rejected
        result = self.am._safe_calculate("2**1001")
        self.assertIn("Exponent too large", result)

    def test_safe_calculate_nested_dos(self):
        # 9**9**9 -> 9**(9**9) -> 9**387420489
        result = self.am._safe_calculate("9**9**9")
        self.assertIn("Exponent too large", result)

    def test_safe_calculate_negative_exponent(self):
        # 2**-2 -> 0.25
        self.assertEqual(self.am._safe_calculate("2**-2"), "0.25")

        # 2**-1001 -> rejected
        result = self.am._safe_calculate("2**-1001")
        self.assertIn("Exponent too large", result)

if __name__ == '__main__':
    unittest.main()
