import unittest
import sys
import os

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_core.utils import parse_json_array_loose

class TestParseJsonArrayLoose(unittest.TestCase):
    """
    Tests for parse_json_array_loose function in ai_core/utils.py
    """

    def test_empty_or_none(self):
        """Should return empty list for None or empty/whitespace strings."""
        self.assertEqual(parse_json_array_loose(None), [])
        self.assertEqual(parse_json_array_loose(""), [])
        self.assertEqual(parse_json_array_loose("   "), [])

    def test_valid_json_array(self):
        """Should correctly parse standard JSON arrays."""
        self.assertEqual(parse_json_array_loose('[1, 2, 3]'), [1, 2, 3])
        self.assertEqual(parse_json_array_loose('["a", "b"]'), ["a", "b"])
        self.assertEqual(parse_json_array_loose('[true, false, null]'), [True, False, None])

    def test_markdown_blocks(self):
        """Should strip markdown code blocks."""
        # Standard markdown block
        raw = "```json\n[1, 2, 3]\n```"
        self.assertEqual(parse_json_array_loose(raw), [1, 2, 3])

        # Markdown block without language specifier
        raw_no_lang = "```\n['a', 'b']\n```"
        self.assertEqual(parse_json_array_loose(raw_no_lang), ['a', 'b'])

        # Markdown block with surrounding text handled by extraction logic
        raw_with_text = "Here is the list:\n```json\n[1, 2]\n```\nHope this helps."
        self.assertEqual(parse_json_array_loose(raw_with_text), [1, 2])

    def test_extra_text(self):
        """Should extract array from surrounding text."""
        raw = "Here is the list: [1, 2, 3]"
        self.assertEqual(parse_json_array_loose(raw), [1, 2, 3])

        raw_suffix = "[10, 20] is the result."
        self.assertEqual(parse_json_array_loose(raw_suffix), [10, 20])

    def test_python_literals(self):
        """Should handle single quotes and Python-specific literals via ast.literal_eval fallback."""
        # Single quotes (invalid JSON, valid Python)
        self.assertEqual(parse_json_array_loose("['apple', 'banana']"), ['apple', 'banana'])

        # Python booleans/None (True/False/None vs true/false/null)
        # Note: json.loads handles true/false/null -> True/False/None
        # ast.literal_eval handles True/False/None -> True/False/None
        self.assertEqual(parse_json_array_loose("[True, False, None]"), [True, False, None])

    def test_nested_arrays(self):
        """Should handle nested lists."""
        self.assertEqual(parse_json_array_loose("[[1, 2], [3, 4]]"), [[1, 2], [3, 4]])

    def test_non_list_json(self):
        """Should return empty list if valid JSON but not a list."""
        self.assertEqual(parse_json_array_loose('{"key": "value"}'), [])
        self.assertEqual(parse_json_array_loose('123'), [])
        self.assertEqual(parse_json_array_loose('"string"'), [])

    def test_malformed_input(self):
        """Should return empty list for unparseable strings."""
        self.assertEqual(parse_json_array_loose("[1, 2"), []) # Missing closing bracket
        self.assertEqual(parse_json_array_loose("just text"), [])
        self.assertEqual(parse_json_array_loose("1, 2, 3"), []) # No brackets

    def test_trailing_commas(self):
        """Should handle trailing commas via ast.literal_eval."""
        # Valid in Python, invalid in standard JSON
        self.assertEqual(parse_json_array_loose("[1, 2, 3,]"), [1, 2, 3])

    def test_mixed_quotes(self):
        """Should handle mixed double and single quotes."""
        self.assertEqual(parse_json_array_loose('["a", \'b\']'), ["a", "b"])

if __name__ == '__main__':
    unittest.main()
