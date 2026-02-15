import ast
import re
from typing import Tuple, Optional
import logging
from .lm import run_local_lm
from .utils import parse_json_object_loose

class PluginSafetyValidator:
    """
    Validates Python plugin code for safety and alignment.
    Enforces strict static analysis (AST) and semantic analysis (LLM).
    """

    # Expanded list of dangerous modules
    BLOCKED_IMPORTS = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'importlib',
        'pickle', 'marshal', 'ctypes', 'pathlib', 'requests', 'urllib',
        'http', 'ftplib', 'smtplib', 'xmlrpc', 'telnetlib', 'builtins',
        'multiprocessing', 'threading', 'concurrent.futures' # Prevent massive concurrency abuse
    }

    # Dangerous built-in functions
    BLOCKED_CALLS = {
        'eval', 'exec', 'compile', 'open', '__import__', 'input',
        'globals', 'locals', 'breakpoint', 'help', 'exit', 'quit',
        'getattr', 'setattr', 'delattr' # Reflection can bypass checks
    }

    def __init__(self, log_fn=logging.info):
        self.log = log_fn

    def check_static(self, code: str) -> Tuple[bool, str]:
        """
        Performs static AST analysis to block dangerous constructs.
        Returns (is_safe, reason).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"

        for node in ast.walk(tree):
            # Check Imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [n.name.split('.')[0] for n in node.names]
                else:
                    names = [node.module.split('.')[0]] if node.module else []

                for name in names:
                    if name in self.BLOCKED_IMPORTS:
                        return False, f"Static Check Failed: Import of '{name}' is forbidden."

            # Check Function Calls
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Check for things like os.system or Path.write_text
                    # We can't easily know the type of the object, but we can check the attribute name
                    # if it's suspicious combined with a known dangerous module (which is already blocked).
                    # But specific attributes like 'system', 'popen', 'write_text' might be worth flagging if context is suspicious.
                    pass

                if func_name in self.BLOCKED_CALLS:
                    return False, f"Static Check Failed: Call to '{func_name}' is forbidden."

            # Check Attribute Access (Anti-Reflection)
            elif isinstance(node, ast.Attribute):
                if node.attr in ['__subclasses__', '__bases__', '__mro__']:
                    return False, f"Static Check Failed: Reflection attribute '{node.attr}' is forbidden."

        return True, "Static Check Passed"

    def check_semantic(self, code: str, value_core=None) -> Tuple[bool, str]:
        """
        Performs semantic analysis using LLM to detect intent violations.
        Returns (is_safe, reason).
        """
        if not value_core:
            return True, "Semantic Check Skipped (No ValueCore)"

        prompt = (
            f"CODE ANALYSIS TASK:\n"
            f"```python\n{code[:2000]}\n```\n" # Limit code size for prompt
            "Review this Python code for safety and alignment constraints.\n"
            "1. Does it attempt to modify the AI's own source code or core files?\n"
            "2. Does it perform unauthorized surveillance, data exfiltration, or network attacks?\n"
            "3. Does it facilitate physical harm, illegal acts, or malware creation?\n"
            "4. Does it use obfuscation to hide its intent?\n\n"
            "If the code is a standard utility (math, data processing, text, logic), it is SAFE.\n"
            "If it tries to bypass safety filters or modify the system, it is UNSAFE.\n\n"
            "Output JSON: {\"verdict\": \"SAFE\" or \"UNSAFE\", \"reason\": \"...\"}"
        )

        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Code Safety Auditor. You protect the system from malicious internal code execution.",
                temperature=0.0,
                max_tokens=200
            )

            result = parse_json_object_loose(response)
            verdict = result.get("verdict", "UNSAFE").upper()
            reason = result.get("reason", "Unknown")

            if verdict == "SAFE":
                return True, f"Semantic Check Passed: {reason}"
            else:
                return False, f"Semantic Check Failed: {reason}"

        except Exception as e:
            self.log(f"⚠️ Semantic check failed: {e}")
            # Fail-safe: If we can't verify it, we should probably be cautious.
            # But blocking on LLM failure might break harmless plugins.
            # For strict safety, we return False.
            return False, f"Semantic Check Error: {e}"

    def validate(self, code: str, value_core=None) -> Tuple[bool, str]:
        """
        Runs full validation suite.
        """
        # 1. Static Check
        is_safe_static, reason_static = self.check_static(code)
        if not is_safe_static:
            return False, reason_static

        # 2. Semantic Check
        is_safe_semantic, reason_semantic = self.check_semantic(code, value_core)
        if not is_safe_semantic:
            return False, reason_semantic

        return True, "All Checks Passed"
