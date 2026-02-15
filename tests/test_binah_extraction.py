import unittest
from treeoflife.binah import Binah

class TestBinahExtraction(unittest.TestCase):
    def test_extract_value_common_patterns(self):
        # " is "
        self.assertEqual(Binah._extract_value_from_text("Assistant name is Ada"), "Ada")
        self.assertEqual(Binah._extract_value_from_text("My car is red"), "red")

        # " lives in "
        self.assertEqual(Binah._extract_value_from_text("User lives in Van, Türkiye"), "Van, Türkiye")
        self.assertEqual(Binah._extract_value_from_text("He lives in Paris"), "Paris")

        # " works at "
        self.assertEqual(Binah._extract_value_from_text("She works at Google"), "Google")

        # " wants to "
        self.assertEqual(Binah._extract_value_from_text("The user wants to learn Python"), "learn Python")

        # " prefers "
        self.assertEqual(Binah._extract_value_from_text("He prefers tea over coffee"), "tea over coffee")

    def test_extract_value_case_sensitivity(self):
        # Current implementation is case-sensitive for splitting, but case-insensitive for detection
        # leading to fallback if case doesn't match the hardcoded patterns.
        self.assertEqual(Binah._extract_value_from_text("Assistant name IS Ada"), "Assistant name IS Ada")
        self.assertEqual(Binah._extract_value_from_text("User LIVES IN New York"), "User LIVES IN New York")

        # Mixed case that matches pattern exactly should work
        # But pattern is " is ", so " IS " won't work.
        # "Is" won't work either.
        self.assertEqual(Binah._extract_value_from_text("This test is passing"), "passing")

    def test_extract_value_fallback(self):
        # No pattern present
        self.assertEqual(Binah._extract_value_from_text("Just some random text"), "Just some random text")
        self.assertEqual(Binah._extract_value_from_text("No special keywords here"), "No special keywords here")

    def test_extract_value_edge_cases(self):
        # Empty string
        self.assertEqual(Binah._extract_value_from_text(""), "")

        # Pattern at the start
        # " is a test" -> strip() -> "is a test" -> " is " not found -> returns "is a test"
        # The method strips whitespace first, so patterns requiring surrounding spaces
        # won't match if they are at the start/end after stripping.
        self.assertEqual(Binah._extract_value_from_text(" is a test"), "is a test")

        # Pattern at the end
        # "The answer is " -> strip() -> "The answer is" -> " is " not found -> returns "The answer is"
        self.assertEqual(Binah._extract_value_from_text("The answer is "), "The answer is")

        # Multiple patterns: Takes the first one in the list.
        # patterns order: " is ", " lives in ", " works at ", " wants to ", " prefers "

        # "He is someone who lives in London"
        # " is " comes before " lives in " in the pattern list.
        # "He" / "someone who lives in London"
        self.assertEqual(Binah._extract_value_from_text("He is someone who lives in London"), "someone who lives in London")

        # "He wants to work where he is happy"
        # " is " is checked first.
        # "He wants to work where he" / "happy"
        self.assertEqual(Binah._extract_value_from_text("He wants to work where he is happy"), "happy")

    def test_extract_value_whitespace(self):
        # Leading/trailing whitespace should be stripped from result
        self.assertEqual(Binah._extract_value_from_text("My name is   Bond  "), "Bond")
