import json
import re
import ast
from typing import List, Dict, Any, Union

def parse_json_array_loose(raw: str) -> list:
    """
    Robustly extract a JSON array from a string (e.g. LLM output).
    Handles markdown blocks, extra text, and single quotes.
    """
    if not raw:
        return []

    raw = raw.strip()

    # Strip markdown code blocks if present (e.g. ```json ... ```)
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
        raw = re.sub(r"\n```$", "", raw)
        raw = raw.strip()

    # Try direct parsing first (for valid JSON)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    # Try to extract array brackets
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    extracted = raw[start:end + 1]

    # Try parsing extracted array
    try:
        data = json.loads(extracted)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    # Try parsing as Python literal (handles single quotes, None, True/False)
    try:
        data = ast.literal_eval(extracted)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def parse_json_object_loose(raw: str) -> dict:
    """
    Robustly extract a JSON object from a string (e.g. LLM output).
    Handles markdown blocks, extra text, and single quotes.
    """
    if not raw:
        return {}

    raw = raw.strip()

    # Strip markdown code blocks
    if "```" in raw:
        match = re.search(r"```(?:\w+)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            raw = match.group(1)
        else:
            raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
            raw = re.sub(r"\n```$", "", raw)
            raw = raw.strip()

    # Try finding outer braces
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        extracted = raw[start:end + 1]
        try:
            return json.loads(extracted)
        except:
            try:
                data = ast.literal_eval(extracted)
                if isinstance(data, dict): return data
            except:
                pass

    return {}