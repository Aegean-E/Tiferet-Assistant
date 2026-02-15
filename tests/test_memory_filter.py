import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_core.memory_filter import MemoryQualityFilter

class TestMemoryFilter(unittest.TestCase):
    def setUp(self):
        self.filter = MemoryQualityFilter()

    def test_questions(self):
        self.assertTrue(self.filter.is_low_quality("What is your name?"))
        self.assertFalse(self.filter.is_low_quality("My name is Alice"))

    def test_greetings(self):
        self.assertTrue(self.filter.is_low_quality("Hello"))
        self.assertTrue(self.filter.is_low_quality("Hi"))
        self.assertFalse(self.filter.is_low_quality("Hello there"))

    def test_short_text(self):
        # Short text with no type -> Allowed (legacy behavior)
        self.assertFalse(self.filter.is_low_quality("Short"))

        # Short text with unprotected type -> Filtered
        self.assertTrue(self.filter.is_low_quality("Short", "FACT"))

        # Short text with protected type -> Allowed
        self.assertFalse(self.filter.is_low_quality("help", "GOAL"))
        self.assertFalse(self.filter.is_low_quality("I am", "IDENTITY"))

    def test_generic_goals(self):
        self.assertTrue(self.filter.is_low_quality("My goal is to help", "GOAL"))
        self.assertTrue(self.filter.is_low_quality("I want to help with academic research", "GOAL"))
        self.assertFalse(self.filter.is_low_quality("My goal is to build a rocket", "GOAL"))

    def test_errors(self):
        self.assertTrue(self.filter.is_low_quality("System encountered an error"))

    def test_artifacts(self):
        self.assertTrue(self.filter.is_low_quality("Page 1 of 10"))

if __name__ == "__main__":
    unittest.main()
