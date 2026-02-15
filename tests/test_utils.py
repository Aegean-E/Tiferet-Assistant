from ai_core.utils import parse_json_object_loose

def test_parse_json_object_empty():
    assert parse_json_object_loose("") == {}
    assert parse_json_object_loose("   ") == {}

def test_parse_json_object_valid():
    assert parse_json_object_loose('{"key": "value"}') == {"key": "value"}
    assert parse_json_object_loose('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}

def test_parse_json_object_markdown():
    # With language tag
    raw = '```json\n{"key": "value"}\n```'
    assert parse_json_object_loose(raw) == {"key": "value"}

    # Without language tag
    raw = '```\n{"key": "value"}\n```'
    assert parse_json_object_loose(raw) == {"key": "value"}

def test_parse_json_object_extra_text():
    raw = 'Here is the result: {"key": "value"} and some more text.'
    assert parse_json_object_loose(raw) == {"key": "value"}

def test_parse_json_object_single_quotes():
    # json.loads will fail, ast.literal_eval should succeed
    raw = "{'key': 'value'}"
    assert parse_json_object_loose(raw) == {"key": "value"}

def test_parse_json_object_python_literals():
    # ast.literal_eval handles True, False, None
    raw = '{"is_active": True, "details": None}'
    assert parse_json_object_loose(raw) == {"is_active": True, "details": None}

def test_parse_json_object_nested():
    raw = '{"outer": {"inner": 42}}'
    assert parse_json_object_loose(raw) == {"outer": {"inner": 42}}

def test_parse_json_object_invalid():
    assert parse_json_object_loose('not a json') == {}
    assert parse_json_object_loose('{invalid') == {}
    assert parse_json_object_loose('["array", "not", "object"]') == {}

def test_parse_json_object_nested_markdown_complex():
    # Testing the regex for markdown blocks
    raw = """
Some text
```json
{
  "key": "value",
  "nested": {
    "a": 1
  }
}
```
More text
"""
    assert parse_json_object_loose(raw) == {"key": "value", "nested": {"a": 1}}
