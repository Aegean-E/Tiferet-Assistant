import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import importlib

# Ensure path is set up if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSafety(unittest.TestCase):
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
        self.am = self.ActionManager(self.mock_core)

        # Mock Malkuth
        self.mock_core.malkuth = MagicMock()
        self.mock_core.malkuth.describe_image.return_value = "Image description"
        self.mock_core.malkuth.write_file.return_value = "File written"

    def test_describe_image_unsafe_path(self):
        # Test with a path that should be blocked
        unsafe_path = os.path.abspath("/etc/passwd")

        result = self.am._describe_image_tool(unsafe_path)

        # Should fail with Access denied
        self.assertIn("Access denied", result)
        self.mock_core.malkuth.describe_image.assert_not_called()

    def test_write_file_unsafe_extension(self):
        unsafe_args = "hack.py, print('hack')"
        result = self.am._write_file_tool(unsafe_args)

        # Should return error and not call Malkuth
        self.assertIn("Access denied", result)
        self.mock_core.malkuth.write_file.assert_not_called()

    def test_write_file_valid_sanitization(self):
        # Test that traversal is neutralized (basename) and thus ALLOWED by ActionManager
        # because it resolves to a safe path in ./works
        unsafe_args = "../hack.txt, content"
        result = self.am._write_file_tool(unsafe_args)

        # Should succeed (delegating to Malkuth)
        # Verify Malkuth was called with sanitized filename
        self.mock_core.malkuth.write_file.assert_called_with("hack.txt", "content")

if __name__ == '__main__':
    unittest.main()
