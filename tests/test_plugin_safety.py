import unittest
from ai_core.safety_checks import PluginSafetyValidator
import ast

class TestSafetyChecks(unittest.TestCase):
    def setUp(self):
        self.validator = PluginSafetyValidator()

    def test_static_safe_code(self):
        code = """
def calculate_pi(n):
    return 3.14159 * n
"""
        is_safe, reason = self.validator.check_static(code)
        self.assertTrue(is_safe, f"Safe code blocked: {reason}")

    def test_static_blocked_import(self):
        code = "import os\nos.system('ls')"
        is_safe, reason = self.validator.check_static(code)
        self.assertFalse(is_safe)
        self.assertIn("Import of 'os' is forbidden", reason)

    def test_static_blocked_from_import(self):
        code = "from subprocess import Popen"
        is_safe, reason = self.validator.check_static(code)
        self.assertFalse(is_safe)
        self.assertIn("Import of 'subprocess' is forbidden", reason)

    def test_static_blocked_call(self):
        code = "eval('print(1)')"
        is_safe, reason = self.validator.check_static(code)
        self.assertFalse(is_safe)
        self.assertIn("Call to 'eval' is forbidden", reason)

    def test_static_blocked_open(self):
        code = "f = open('test.txt', 'w')"
        is_safe, reason = self.validator.check_static(code)
        self.assertFalse(is_safe)
        self.assertIn("Call to 'open' is forbidden", reason)

    def test_static_reflection(self):
        code = "x.__subclasses__()"
        is_safe, reason = self.validator.check_static(code)
        self.assertFalse(is_safe)
        self.assertIn("Reflection attribute", reason)

    def test_pathlib_blocked(self):
        code = "import pathlib\np = pathlib.Path('test')"
        is_safe, reason = self.validator.check_static(code)
        self.assertFalse(is_safe)
        self.assertIn("Import of 'pathlib' is forbidden", reason)

    def test_semantic_check_no_value_core(self):
        code = "print('hello')"
        is_safe, reason = self.validator.check_semantic(code, value_core=None)
        self.assertTrue(is_safe)
        self.assertIn("Skipped", reason)

if __name__ == '__main__':
    unittest.main()
