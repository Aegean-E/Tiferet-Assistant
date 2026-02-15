import sys
from unittest.mock import MagicMock

# Create mocks for missing dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['dowhy'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['ttkbootstrap'] = MagicMock()
sys.modules['PyMuPDF'] = MagicMock()
sys.modules['fitz'] = MagicMock()

# Mock ai_core.lm but ensure run_local_lm is mockable
mock_lm = MagicMock()
mock_lm.run_local_lm = MagicMock(return_value="mocked_response")
sys.modules['ai_core.lm'] = mock_lm

import unittest
import os
import shutil
import json

# Ensure we can import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from treeoflife.malkuth import Malkuth
from ai_core.core_actions import ActionManager

class TestSelfModification(unittest.TestCase):
    def setUp(self):
        self.test_plugins_dir = os.path.abspath("./plugins")
        if not os.path.exists(self.test_plugins_dir):
            os.makedirs(self.test_plugins_dir)

        self.malkuth = Malkuth(log_fn=lambda x: None)
        # Mock dependencies
        self.malkuth.memory_store = MagicMock()

        # Mock AI Core for ActionManager
        self.mock_core = MagicMock()
        self.mock_core.malkuth = self.malkuth
        self.mock_core.get_settings.return_value = {"plugin_config": {}, "permissions": {}}
        self.mock_core.thread_pool = MagicMock()
        self.mock_core.crs = None
        self.mock_core.event_bus = None
        self.mock_core.internet_bridge = None # Safe search check
        self.mock_core.document_store = None

        self.action_manager = ActionManager(self.mock_core)

    def tearDown(self):
        # Clean up created plugins
        if os.path.exists(self.test_plugins_dir):
            for f in os.listdir(self.test_plugins_dir):
                if f.startswith("test_plugin"):
                    try:
                        os.remove(os.path.join(self.test_plugins_dir, f))
                    except: pass

    def test_write_plugin_safe(self):
        safe_code = """
def hello():
    return "Hello World"
"""
        result = self.malkuth.write_plugin("test_plugin_safe.py", safe_code)
        self.assertIn("Success", result)
        self.assertTrue(os.path.exists(os.path.join(self.test_plugins_dir, "test_plugin_safe.py")))

    def test_write_plugin_unsafe_import(self):
        unsafe_code = """
import os
def hack():
    os.system('rm -rf /')
"""
        result = self.malkuth.write_plugin("test_plugin_unsafe.py", unsafe_code)
        self.assertIn("Error", result)
        self.assertIn("Import of 'os' is forbidden", result)
        self.assertFalse(os.path.exists(os.path.join(self.test_plugins_dir, "test_plugin_unsafe.py")))

    def test_write_plugin_unsafe_call(self):
        unsafe_code = """
def hack():
    eval("print('hack')")
"""
        result = self.malkuth.write_plugin("test_plugin_eval.py", unsafe_code)
        self.assertIn("Error", result)
        self.assertIn("Call to 'eval' is forbidden", result)

    def test_action_manager_create_plugin(self):
        args = "test_plugin_tool.py, def run(): return 'ok'"
        result = self.action_manager._create_plugin_tool(args)
        self.assertIn("Success", result)
        self.assertTrue(os.path.exists(os.path.join(self.test_plugins_dir, "test_plugin_tool.py")))

    def test_action_manager_update_settings_valid(self):
        self.mock_core.update_settings = MagicMock()
        result = self.action_manager._update_settings_tool("temperature, 0.9")
        self.assertIn("Settings updated", result)
        self.mock_core.update_settings.assert_called_with({"temperature": 0.9})

    def test_action_manager_update_settings_invalid_key(self):
        result = self.action_manager._update_settings_tool("bot_token, secret")
        self.assertIn("Error", result)
        self.assertIn("not allowed", result)

    def test_action_manager_enable_plugin(self):
        # Create a dummy plugin first
        self.malkuth.write_plugin("test_plugin_enable.py", "pass")

        self.mock_core.update_settings = MagicMock()
        # Mock load_plugins to avoid import errors (it imports from current dir)
        self.action_manager.load_plugins = MagicMock()

        result = self.action_manager._enable_plugin_tool("test_plugin_enable")
        self.assertIn("enabled", result)
        self.mock_core.update_settings.assert_called()
        # Check that update_settings was called with plugin_config setting it to True
        call_args = self.mock_core.update_settings.call_args[0][0]
        self.assertTrue(call_args["plugin_config"]["test_plugin_enable"])

if __name__ == '__main__':
    unittest.main()
