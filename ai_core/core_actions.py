import ast
import operator
import re
import os
import traceback
from datetime import datetime
from .lm import run_local_lm
from .utils import parse_json_object_loose
import threading
import concurrent.futures
import sys
import io
import logging
from typing import Dict

class ActionManager:
    """
    Manages tool definitions, execution, and safety wrappers.
    """
    def __init__(self, ai_core):
        self.core = ai_core
        self.ingestion_semaphore = threading.Semaphore(2)
        self.llm_executor = self.core.thread_pool
        self.ingestion_executor = self.core.thread_pool
        self._dynamic_tools = {}
        self._plugin_registry = {} # {plugin_name: {'enabled': bool, 'description': str}}
        self._tool_origins = {} # {tool_name: plugin_name}
        self._current_loading_plugin = None

    def get_tools(self):
        """Define all available tools here."""
        tools = {
            # 1. Basic Tools
            "CLOCK": lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "CALCULATOR": lambda x: self._safe_calculate(x),
            "PHYSICS": lambda q: self._physics_intuition(q),
            "SIMULATE_PHYSICS": lambda q: self._simulate_physics(q),
            "CAUSAL": lambda q: self._causal_inference(q),
            "SIMULATE_ACTION": lambda q: self.core.malkuth.predict_action_outcome(q, "Current System State") if self.core.malkuth else "Malkuth missing",
            "PREDICT": lambda q: self._handle_predict(q),
            "READ_CHUNK": lambda q: self._read_specific_chunk(q),
            "DESCRIBE_IMAGE": lambda q: self._describe_image_tool(q),
            "WRITE_FILE": lambda q: self._write_file_tool(q),
            
            # 2. Internet Tools
            "SEARCH": lambda q="": self.safe_search(q or "", "WEB"), # Now maps to DuckDuckGo
            "WIKI": lambda q="": self.safe_search(q or "", "WIKIPEDIA"),
            "FIND_PAPER": lambda q="": self.safe_search(q or "", "ARXIV"),

            # 3. Self Modification Tools
            "CREATE_PLUGIN": lambda q: self._create_plugin_tool(q),
            "UPDATE_SETTINGS": lambda q: self._update_settings_tool(q),
            "ENABLE_PLUGIN": lambda q: self._enable_plugin_tool(q),
        }
        
        # Add enabled dynamic tools
        for name, func in self._dynamic_tools.items():
            origin = self._tool_origins.get(name, "CORE")
            if origin == "CORE":
                tools[name] = func
            elif self._plugin_registry.get(origin, {}).get('enabled', True):
                tools[name] = func
                
        return tools

    def register_tool(self, name: str, func: callable):
        """Register a new tool dynamically."""
        name = name.upper()
        self._dynamic_tools[name] = func
        origin = self._current_loading_plugin or "CORE"
        self._tool_origins[name] = origin
        
        if origin != "CORE":
            if origin not in self._plugin_registry:
                self._plugin_registry[origin] = {'enabled': True, 'description': "External Plugin"}

    def load_plugins(self, plugin_dir: str = "./plugins"):
        """Load tools from a plugins directory."""
        if not os.path.exists(plugin_dir): return
        
        settings = self.core.get_settings()
        plugin_settings = settings.get("plugin_config", {})
        
        import importlib.util
        
        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("_"):
                path = os.path.join(plugin_dir, filename)
                plugin_name = filename[:-3]
                
                # Security: Check if plugin is explicitly enabled in settings
                is_enabled = plugin_settings.get(plugin_name, False)
                
                if not is_enabled:
                    self.core.log(f"🛡️ Plugin '{plugin_name}' found but is DISABLED. Enable it in Settings.")
                    self._plugin_registry[plugin_name] = {'enabled': False, 'description': "Awaiting user approval"}
                    continue

                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    if hasattr(module, "register_tools"):
                        self._current_loading_plugin = plugin_name
                        module.register_tools(self)
                        self._current_loading_plugin = None
                        self.core.log(f"🔌 Loaded plugin: {filename}")
                except Exception as e:
                    self.core.log(f"⚠️ Failed to load plugin {filename}: {e}")
                    self._current_loading_plugin = None

    def toggle_plugin(self, plugin_name: str, enabled: bool):
        """Enable or disable a plugin."""
        if plugin_name in self._plugin_registry:
            self._plugin_registry[plugin_name]['enabled'] = enabled
            self.core.log(f"🔌 Plugin '{plugin_name}' {'enabled' if enabled else 'disabled'}.")

    def get_plugins(self) -> Dict:
        """Get list of loaded plugins and their status."""
        return self._plugin_registry

    def process_tool_calls(self, text: str, retry_count: int = 0) -> str:
        """
        Scans text for [EXECUTE: TOOL, ARGS] tags, runs them, 
        and replaces the tag with the result.
        """
        # Regex to find [EXECUTE: NAME] or [EXECUTE: NAME, ARGS]
        # Improved regex to handle one level of nested brackets (e.g. "query [2020]")
        pattern = r"\[EXECUTE:\s*(\w+)(?:,\s*((?:[^\[\]]|\[[^\[\]]*\])*))?\]"
        
        max_tools = 3
        executed_count = 0
        
        tools = self.get_tools()
        
        def replace_match(match):
            nonlocal executed_count
            if executed_count >= max_tools:
                return "[Error: Tool execution limit reached for this turn]"
            
            tool_name = match.group(1).upper()
            args = match.group(2) or ""
            
            # Check Permission
            settings = self.core.get_settings()
            permissions = settings.get("permissions", {})
            if not permissions.get(tool_name, True):
                return f"[Error: Tool '{tool_name}' is disabled by user permissions.]"

            if tool_name in tools:
                executed_count += 1
                self.core.log(f"⚙️ Executing Tool: {tool_name} args={args}")
                
                # Cost Accounting (CRS)
                cost = 0.0
                if self.core.crs:
                    cost = self.core.crs.estimate_action_cost(tool_name, complexity=0.5)
                    self.core.crs.current_spend += cost
                
                # Observability (EventBus)
                if self.core.event_bus:
                    self.core.event_bus.publish("TOOL_EXECUTION", {"tool": tool_name, "args": args, "cost": cost}, source="ActionManager")

                try:
                    # Execute
                    if args:
                        # Strip quotes from args if present ('query' -> query)
                        args = args.strip().strip("'").strip('"')
                        result = tools[tool_name](args)
                    else:
                        result = tools[tool_name]()
                    
                    return str(result)
                except Exception as e:
                    # Tool Recovery Loop (OODA)
                    if retry_count < 2: # Max 2 retries
                        self.core.log(f"⚠️ Tool Error ({tool_name}): {e}. Attempting recovery (Try {retry_count+1})...")
                        recovery_prompt = (
                            f"Tool Execution Failed.\n"
                            f"Tool: {tool_name}\n"
                            f"Args: {args}\n"
                            f"Error: {e}\n\n"
                            "TASK: Analyze the error and generate a CORRECTED [EXECUTE: ...] command.\n"
                            "Output ONLY the corrected command tag."
                        )
                        correction = run_local_lm(
                            messages=[{"role": "user", "content": recovery_prompt}],
                            system_prompt="You are a Tool Repair Agent.",
                            temperature=0.1,
                            max_tokens=100
                        )
                        # Recursively process the correction
                        if "[EXECUTE:" in correction:
                            return self.process_tool_calls(correction, retry_count + 1)
                    
                    return f"[Error: {e}]"
            else:
                return f"[Error: Tool {tool_name} not found]"

        # Replace all occurrences in the text
        return re.sub(pattern, replace_match, text)

    def safe_search(self, query, source):
        """Robust search with document ingestion."""
        if not self.core.internet_bridge:
            return "⚠️ Observation: Internet Bridge is not initialized. Action Suggestion: Check settings to enable Telegram Bridge or Internet Access, or rely on internal knowledge."

        content, filepath = self.core.internet_bridge.search(query, source)
        content = content or ""
        if filepath:
            try:
                # Check file size before processing (limit to 10MB)
                if os.path.getsize(filepath) > 10 * 1024 * 1024:
                    self.core.log(f"⚠️ File too large for ingestion: {os.path.basename(filepath)}")
                    return content + "\n[System: File too large for ingestion]"

                file_hash = self.core.document_store.compute_file_hash(filepath)
                if not self.core.document_store.document_exists(file_hash):
                    # Offload heavy ingestion to background thread/event
                    def ingest_worker():
                        with self.ingestion_semaphore:
                            try:
                                chunks, page_count, file_type = self.core.document_processor.process_document(filepath)
                                self.core.document_store.add_document(file_hash=file_hash, filename=os.path.basename(filepath), file_type=file_type, file_size=os.path.getsize(filepath), page_count=page_count, chunks=chunks, upload_source="safe_search")
                                if self.core.ui_refresh_callback:
                                    self.core.ui_refresh_callback('docs')
                                self.core.log(f"✅ Background Ingestion complete: {os.path.basename(filepath)}")
                                if self.core.log:
                                    self.core.log(f"✅ Successfully ingested: {os.path.basename(filepath)}")
                            except Exception as e:
                                if self.core.log:
                                    self.core.log(f"⚠️ Failed to ingest {os.path.basename(filepath)}: {e}")
                                if self.core.log:
                                    self.core.log(f"⚠️ Background Ingestion failed: {e}")
                                if self.core.event_bus:
                                    self.core.event_bus.publish("INGESTION_FAILURE", {"file": os.path.basename(filepath), "error": str(e)})
                    
                    self.ingestion_executor.submit(ingest_worker)
                    return content + "\n[System: Document ingestion started in background]"
            except Exception as e:
                self.core.log(f"⚠️ Ingestion failed for safe search: {repr(e)}\n{traceback.format_exc()}")
        return content

    def _safe_calculate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression without using eval()."""
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.BitXor: operator.xor,
            ast.BitOr: operator.or_,
            ast.BitAnd: operator.and_,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Invert: operator.invert,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num): # Python < 3.8 compatibility
                return node.n
            elif isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in operators:
                    raise TypeError(f"Operator {op} not supported")

                left_val = eval_node(node.left)
                right_val = eval_node(node.right)

                # Security: Prevent CPU DoS via massive exponentiation
                if op == ast.Pow:
                    if isinstance(right_val, (int, float)) and abs(right_val) > 1000:
                        raise ValueError(f"Exponent too large (max 1000)")

                return operators[op](left_val, right_val)
            elif isinstance(node, ast.UnaryOp):
                op = type(node.op)
                if op not in operators:
                    raise TypeError(f"Operator {op} not supported")
                return operators[op](eval_node(node.operand))
            else:
                raise TypeError(f"Node type {type(node)} not supported")

        try:
            if not expression.strip():
                return "Error: No expression provided."
            if len(expression) > 200:
                return "Error: Expression too long"
            tree = ast.parse(expression.strip(), mode='eval')
            return str(eval_node(tree.body))
        except Exception as e:
            return f"Calculation Error: {e}"

    def _physics_intuition(self, query: str) -> str:
        """
        Combines LLM Fermi Estimation with the Malkuth Causal Engine.
        """
        self.core.log(f"🧬 [Malkuth] Running Physical Intuition check on: {query}")
        
        # 1. Metacognitive Step: LLM performs Dimensional Analysis
        reasoning_prompt = (
            f"Analyze the physical scenario: '{query}'.\n"
            "1. Identify the core physical variables (e.g., Concentration, Volume, Flux).\n"
            "2. Perform a Fermi Estimation (Order of Magnitude check).\n"
            "3. Check for Dimensional Consistency (Units must match).\n"
            "4. Thermodynamic Guardrails: Check for violations of conservation laws (Energy/Mass).\n"
            "Output ONLY the reasoning and a final 'Estimated Value' with units."
        )
        
        estimation = self._run_llm_regulated(
            messages=[{"role": "user", "content": reasoning_prompt}],
            system_prompt="You are a Physics Intuition Engine.",
            temperature=0.3, # Low temp for precision
            base_url=self.core.get_settings().get("base_url"),
            chat_model=self.core.get_settings().get("chat_model")
        )

        if estimation.startswith("⚠️"):
            return estimation

        # 2. Grounding Step: If a Causal Model exists, verify the estimation
        if self.core.malkuth:
            verification = self.core.malkuth.verify_physical_possibility(query, estimation)
            estimation += f"\n\n[Malkuth Grounding]: {verification}"

        return estimation

    def _simulate_physics(self, query: str) -> str:
        if not self.core.malkuth: return "Malkuth not initialized."
        return self.core.malkuth.run_physics_simulation(query)

    def _causal_inference(self, args: str) -> str:
        if not self.core.malkuth: return "Malkuth not initialized."
        parts = [p.strip() for p in args.split(",", 2)]
        if len(parts) < 3: return "Error: CAUSAL requires 'treatment, outcome, context'"
        return self.core.malkuth.run_causal_inference(parts[0], parts[1], parts[2])

    def _handle_predict(self, args: str) -> str:
        if not self.core.malkuth: return "Malkuth missing."
        if "," in args:
            claim, timeframe = args.split(",", 1)
            return self.core.malkuth.make_prediction(claim.strip(), timeframe.strip())
        return self.core.malkuth.make_prediction(args, "Unknown")

    def _read_specific_chunk(self, args: str) -> str:
        """
        Active Reading: Read a specific chunk and its neighbors.
        Args: "doc_id, chunk_index"
        """
        if not self.core.document_store: return "Document store not available."
        
        try:
            parts = [p.strip() for p in args.split(",")]
            if len(parts) != 2: return "Error: READ_CHUNK requires 'doc_id, chunk_index'"
            
            doc_id = int(parts[0])
            chunk_idx = int(parts[1])
            
            # Fetch target, prev, and next for context
            chunks = []
            for i in range(chunk_idx - 1, chunk_idx + 2):
                if i < 0: continue
                c = self.core.document_store.get_chunk_by_index(doc_id, i)
                if c:
                    chunks.append(f"[Chunk {i}] {c['text']}")
            
            if not chunks: return "Chunk not found."
            return "\n\n".join(chunks)
        except Exception as e:
            return f"Error reading chunk: {e}"

    def _validate_path(self, path: str, allow_write: bool = False) -> str:
        """
        Validates a file path for security.
        Returns the absolute path if safe, raises ValueError if unsafe.
        """
        # 1. Resolve Path
        abs_path = os.path.abspath(path)
        real_path = os.path.realpath(abs_path) # Resolve symlinks

        # 2. Define Safe Directories
        # We allow reading from DATA_DIR and ./works
        # We allow writing ONLY to ./works (and strictly checked)

        root_dir = os.path.abspath(".") # Assuming CWD is root
        data_dir = os.path.join(root_dir, "data")
        works_dir = os.path.join(root_dir, "works")

        allowed_read_roots = [data_dir, works_dir]
        allowed_write_roots = [works_dir]

        # Check roots
        is_safe_root = False
        roots = allowed_write_roots if allow_write else allowed_read_roots

        for root in roots:
            if os.path.commonpath([root, real_path]) == root:
                is_safe_root = True
                break

        if not is_safe_root:
            raise ValueError(f"Access denied: Path '{path}' is outside allowed directories.")

        # 3. Extension Check (if writing)
        if allow_write:
            blocked_exts = {'.py', '.sh', '.exe', '.bat', '.cmd', '.js', '.php', '.pl'}
            ext = os.path.splitext(real_path)[1].lower()
            if ext in blocked_exts:
                raise ValueError(f"Access denied: File type '{ext}' is blocked.")

        return real_path

    def _describe_image_tool(self, image_path: str) -> str:
        """
        Tool wrapper for Malkuth's vision capability.
        """
        if not self.core.malkuth:
            return "Malkuth (Vision) not available."

        try:
            safe_path = self._validate_path(image_path, allow_write=False)
            return self.core.malkuth.describe_image(safe_path)
        except Exception as e:
            return f"Error: {e}"

    def _write_file_tool(self, args: str) -> str:
        """
        Tool wrapper for Malkuth's write_file capability.
        Args: "filename, content"
        """
        if not self.core.malkuth:
            return "Malkuth not available."
        if "," not in args:
            return "Error: WRITE_FILE requires 'filename, content'"
        filename, content = args.split(",", 1)

        try:
            # Construct expected path (Malkuth logic)
            works_dir = os.path.abspath("./works")
            safe_filename = os.path.basename(filename.strip())
            expected_path = os.path.join(works_dir, safe_filename)

            # Validate extension and directory
            self._validate_path(expected_path, allow_write=True)

            # Pass safe_filename to Malkuth to enforce the path we validated
            return self.core.malkuth.write_file(safe_filename, content.strip())
        except Exception as e:
            return f"Error: {e}"

    def _run_llm_regulated(self, *args, **kwargs):
        """
        Wrapper for LLM calls to allow future budgeting/throttling.
        """
        # Enforce hard timeout via thread execution
        timeout = kwargs.pop('timeout', 60) # Default 60s timeout
        
        future = self.llm_executor.submit(run_local_lm, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self.core.log(f"⚠️ LLM Call Timed Out ({timeout}s)")
            return "⚠️ Observation: The LLM request timed out. Action Suggestion: The model might be overloaded or the prompt is too complex. Try a simpler task or wait a moment."
        except Exception as e:
            self.core.log(f"⚠️ LLM Call Failed: {e}")
            return f"⚠️ Observation: LLM Call Failed ({e}). Action Suggestion: Check model connection or reduce prompt size."

    def _create_plugin_tool(self, args: str) -> str:
        """
        Wrapper for Malkuth.write_plugin.
        Args: "filename, content"
        """
        if not self.core.malkuth:
            return "Malkuth not available."
        if "," not in args:
            return "Error: CREATE_PLUGIN requires 'filename, content'"
        filename, content = args.split(",", 1)
        filename = filename.strip()
        content = content.strip()

        # Remove quotes if present
        if (filename.startswith('"') and filename.endswith('"')) or (filename.startswith("'") and filename.endswith("'")):
             filename = filename[1:-1]

        return self.core.malkuth.write_plugin(filename, content)

    def _update_settings_tool(self, args: str) -> str:
        """
        Update system settings.
        Args: "key, value" or JSON string
        """
        allowed_keys = {
            "system_prompt", "temperature", "top_p",
            "daydream_cycle_limit", "max_tokens", "memory_extractor_prompt"
        }

        try:
            # Try parsing as JSON first
            if args.strip().startswith("{"):
                updates = parse_json_object_loose(args)
            elif "," in args:
                k, v = args.split(",", 1)
                updates = {k.strip(): v.strip()}
            else:
                return "Error: format must be 'key, value' or JSON object."

            # Validation
            filtered_updates = {}
            for k, v in updates.items():
                if k in allowed_keys:
                    # Type casting
                    try:
                        if k in ["temperature", "top_p"]:
                            filtered_updates[k] = float(v)
                        elif k in ["daydream_cycle_limit", "max_tokens"]:
                            filtered_updates[k] = int(v)
                        else:
                            filtered_updates[k] = str(v)
                    except ValueError:
                        return f"Error: Invalid type for setting '{k}'"
                else:
                    return f"Error: Setting '{k}' is not allowed to be modified."

            if not filtered_updates:
                return "Error: No valid settings to update."

            # Apply updates
            self.core.update_settings(filtered_updates)
            return f"Settings updated: {list(filtered_updates.keys())}"

        except Exception as e:
            return f"Error updating settings: {e}"

    def _enable_plugin_tool(self, plugin_name: str) -> str:
        """
        Enable a plugin in settings.
        """
        plugin_name = plugin_name.strip().replace('"', '').replace("'", "")
        if not plugin_name:
            return "Error: Plugin name required."

        # Check if file exists
        plugin_path = os.path.join(os.path.abspath("./plugins"), f"{plugin_name}.py")
        if not os.path.exists(plugin_path):
             return f"Error: Plugin file '{plugin_name}.py' not found in plugins directory."

        current_settings = self.core.get_settings()
        plugin_config = current_settings.get("plugin_config", {}).copy()

        if plugin_config.get(plugin_name, False):
            return f"Plugin '{plugin_name}' is already enabled."

        plugin_config[plugin_name] = True
        self.core.update_settings({"plugin_config": plugin_config})

        # Reload plugins
        self.load_plugins()

        return f"Plugin '{plugin_name}' enabled and loaded."