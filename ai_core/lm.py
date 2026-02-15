import logging
import json
import ast
import re
import numpy as np
import requests
import os
import base64
import threading
import time
import hashlib  # <-- added for persistent cache
from typing import List, Dict, Tuple, Callable, Optional
from functools import lru_cache

from collections import OrderedDict
import sqlite3
from ai_core.utils import parse_json_array_loose, parse_json_object_loose
from docs.default_prompts import DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT_TEXT, DEFAULT_MEMORY_EXTRACTOR_PROMPT as DEFAULT_MEMORY_EXTRACTOR_PROMPT_TEXT

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from PIL import Image
    import io

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

_WHISPER_MODEL = None

class LLMError(Exception):
    """Custom exception for LLM related failures."""
    pass

# ==============================
# Configuration Defaults
# ==============================
# Note: Settings are now managed centrally by AICore/App and passed in.
# Local disk loading removed to prevent configuration inconsistency.

_settings = {} 

def configure_lm(settings: dict) -> None:
    """Update global settings for the LM module."""
    global _settings
    _settings.update(settings)

LM_STUDIO_BASE_URL = _settings.get("base_url", "http://127.0.0.1:1234/v1")
CHAT_MODEL = _settings.get("chat_model", "qwen2.5-vl-7b-instruct-abliterated")
EMBEDDING_MODEL = _settings.get("embedding_model", "text-embedding-nomic-embed-text-v1.5")
CHAT_COMPLETIONS_URL = f"{LM_STUDIO_BASE_URL}/chat/completions"

# ==============================
# Persistent Embedding Cache Config
# ==============================
DEFAULT_EMBEDDING_CACHE_DB = "./data/embedding_cache.sqlite3"
EMBEDDING_CACHE_DB = DEFAULT_EMBEDDING_CACHE_DB

# ==============================
# System prompts
# ==============================
SYSTEM_PROMPT = _settings.get("system_prompt") or DEFAULT_SYSTEM_PROMPT_TEXT
MEMORY_EXTRACTOR_PROMPT = _settings.get("memory_extractor_prompt") or DEFAULT_MEMORY_EXTRACTOR_PROMPT_TEXT

def configure_lm(settings: Dict):
    """
    Update global configuration from a settings dictionary.
    This should be called when settings are loaded or updated.
    """
    global _settings, LM_STUDIO_BASE_URL, CHAT_MODEL, EMBEDDING_MODEL, SYSTEM_PROMPT, MEMORY_EXTRACTOR_PROMPT, EMBEDDING_CACHE_DB

    _settings.update(settings)

    LM_STUDIO_BASE_URL = _settings.get("base_url", "http://127.0.0.1:1234/v1")
    CHAT_MODEL = _settings.get("chat_model", "qwen2.5-vl-7b-instruct-abliterated")
    EMBEDDING_MODEL = _settings.get("embedding_model", "text-embedding-nomic-embed-text-v1.5")

    SYSTEM_PROMPT = _settings.get("system_prompt") or DEFAULT_SYSTEM_PROMPT_TEXT
    MEMORY_EXTRACTOR_PROMPT = _settings.get("memory_extractor_prompt") or DEFAULT_MEMORY_EXTRACTOR_PROMPT_TEXT

    EMBEDDING_CACHE_DB = _settings.get("embedding_cache_db", DEFAULT_EMBEDDING_CACHE_DB)

# Aliases for external usage (e.g. GUI defaults)
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_MEMORY_EXTRACTOR_PROMPT = MEMORY_EXTRACTOR_PROMPT

# ==============================
# Epigenetics Hot-Loader
# ==============================

_EPI_CACHE = {
    "logic": "",
    "mtime": 0,
    "lock": threading.Lock()
}


def _get_epigenetics_logic() -> str:
    """Thread-safe hot-loader for epigenetics.json"""
    path = _settings.get("epigenetics_path", "./data/epigenetics.json")
    if not os.path.exists(path):
        return ""

    try:
        stat = os.stat(path)
        mtime = stat.st_mtime

        if mtime != _EPI_CACHE["mtime"]:
            with _EPI_CACHE["lock"]:
                if mtime != _EPI_CACHE["mtime"]:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        _EPI_CACHE["logic"] = data.get("evolved_logic", "")
                        _EPI_CACHE["mtime"] = mtime
                        # logging.info(f"🧬 [Epigenetics] Hot-reloaded logic (v{data.get('version', '?')})")
        return _EPI_CACHE["logic"]
    except Exception:
        return _EPI_CACHE["logic"]  # Return stale on error


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens using tiktoken for accuracy, falling back to char count.
    """
    if not text:
        return 0

    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            pass  # Fallback

    # Fallback: Approx 4 chars per token
    return len(text) // 4


def trim_history(messages: List[Dict], max_tokens: int = 8000, model: str = "gpt-4") -> List[Dict]:
    """
    Trims message history to fit within max_tokens.
    Preserves System Prompt (first message) and recent context.
    """
    if not messages:
        return []

    # Always keep system prompt if present
    system_msg = None
    if messages[0].get("role") == "system":
        system_msg = messages[0]
        history = messages[1:]
    else:
        history = messages

    current_tokens = count_tokens(system_msg["content"] if system_msg else "", model)
    kept_history = []

    # Add messages from newest to oldest until limit reached
    for msg in reversed(history):
        msg_tokens = count_tokens(msg.get("content", ""), model)
        if current_tokens + msg_tokens > max_tokens:
            break
        kept_history.insert(0, msg)
        current_tokens += msg_tokens

    if system_msg:
        return [system_msg] + kept_history
    return kept_history


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio using local Whisper model.
    """
    if not WHISPER_AVAILABLE:
        return "[Error: openai-whisper not installed. Run: pip install openai-whisper]"

    try:
        global _WHISPER_MODEL
        if _WHISPER_MODEL is None:
            logging.info(f"🎧 Loading Whisper model ({model_size})...")
            _WHISPER_MODEL = whisper.load_model(model_size)

        result = _WHISPER_MODEL.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        logging.error(f"⚠️ Transcription failed: {e}")
        return f"[Error: Transcription failed: {e}]"


def _resize_and_encode_image(image_path: str, max_size: int = 1024) -> Optional[str]:
    """
    Resize image to max_size and return base64 string.
    Prevents sending massive payloads to the LLM.
    """
    if not os.path.exists(image_path):
        return None

    try:
        if PIL_AVAILABLE:
            with Image.open(image_path) as img:
                # Convert to RGB (handle PNG alpha, P mode, etc)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # Resize if larger than max_size
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Save to memory buffer
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Fallback: Raw read if PIL missing
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    except Exception as e:
        logging.error(f"⚠️ Image processing error for {image_path}: {e}")
        return None


# ==============================
# Local LM call
# ==============================

# Simple in-memory cache for deterministic calls (OrderedDict for LRU)
_LM_CACHE = {}
_LM_CACHE_SIZE = 100
_LM_CACHE_LOCK = threading.Lock()

# Global semaphore to limit concurrent GPU calls
GPU_SEMAPHORE = threading.Semaphore(1)  # Default to 1 for safety


class LLMRequest:
    """
    Encapsulates the lifecycle of a single LLM request.
    Handles configuration, prompt preparation, execution, and error handling.
    """

    def __init__(self, messages: list, **kwargs):
        self.messages = messages
        self.kwargs = kwargs

        # Internal state
        self.system_prompt = None
        self.final_messages = []
        self.payload = {}
        self.cache_key = None

        # Resolve settings immediately
        self._resolve_settings()

    def _resolve_settings(self):
        """Resolves parameters from arguments, settings, and defaults."""
        # Use empty dict if no settings provided; logic below handles defaults
        base_url = self.kwargs.get("base_url")
        chat_model = self.kwargs.get("chat_model")

        settings = _settings if not any(param is None for param in [base_url, chat_model]) else _settings

        # Resolve parameters: Argument > Settings.json > Global Default
        self.system_prompt = self.kwargs.get("system_prompt")
        if self.system_prompt is None:
            self.system_prompt = settings.get("system_prompt", SYSTEM_PROMPT)

        self.temperature = self.kwargs.get("temperature")
        if self.temperature is None:
            self.temperature = settings.get("temperature", 0.7)

        self.top_p = self.kwargs.get("top_p")
        if self.top_p is None:
            self.top_p = settings.get("top_p", 0.94)

        self.max_tokens = self.kwargs.get("max_tokens")
        if self.max_tokens is None:
            self.max_tokens = int(settings.get("max_tokens", 800))

        self.base_url = base_url
        if self.base_url is None:
            self.base_url = settings.get("base_url", LM_STUDIO_BASE_URL)

        self.chat_model = chat_model
        if self.chat_model is None:
            self.chat_model = settings.get("chat_model", CHAT_MODEL)

        self.stop_check_fn = self.kwargs.get("stop_check_fn")
        self.images = self.kwargs.get("images")
        self.retry_depth = self.kwargs.get("_retry_depth", 0)

        # Defensive casting
        try:
            self.temperature = float(self.temperature) if self.temperature is not None else 0.7
            self.top_p = float(self.top_p) if self.top_p is not None else 0.94
            self.max_tokens = int(self.max_tokens) if self.max_tokens is not None else 800
        except (ValueError, TypeError):
            logging.warning(
                f"⚠️ Invalid types for LM params: temp={self.temperature}, top_p={self.top_p}, max={self.max_tokens}. Using defaults.")
            self.temperature = 0.7
            self.top_p = 0.94
            self.max_tokens = 800

    def _inject_epigenetics(self):
        """Injects dynamic evolved logic into the system prompt."""
        evolved_logic = _get_epigenetics_logic()
        if evolved_logic and len(evolved_logic) > 10:
            self.system_prompt += f"\n\n[DYNAMIC EVOLVED LOGIC]:\nApply the following logic unless it conflicts with:\n1. Core safety invariants\n2. Epistemic validation rules\n3. System architecture constraints\n\nLOGIC:\n{evolved_logic}"

    def _prepare_messages(self):
        """Prepares the final message list, handling vision and context trimming."""
        self._inject_epigenetics()

        self.final_messages = [{"role": "system", "content": self.system_prompt}]

        if self.images:
            # Construct multi-modal user message
            content_payload = []
            # Add text from the last user message if it exists
            last_text = self.messages[-1]['content'] if self.messages else "Analyze this image."
            content_payload.append({"type": "text", "text": last_text})

            for img_path in self.images:
                base64_image = _resize_and_encode_image(img_path)
                if base64_image:
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })

            # Replace the last message with the multi-modal one
            self.final_messages.extend(self.messages[:-1])
            self.final_messages.append({"role": "user", "content": content_payload})
        else:
            # Trim history before sending
            # Reserve tokens for system prompt and generation
            safe_history_tokens = 4096 - (self.max_tokens or 800) - count_tokens(self.system_prompt, model=self.chat_model) - 500
            trimmed_messages = trim_history(self.messages, max_tokens=max(1000, safe_history_tokens), model=self.chat_model)
            self.final_messages.extend(trimmed_messages)

    def _prepare_payload(self):
        """Constructs the JSON payload for the API."""
        self.payload = {
            "model": self.chat_model,
            "messages": self.final_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.stop_check_fn:
            self.payload["stream"] = True

    def _check_cache(self) -> Optional[str]:
        """Checks the in-memory cache for deterministic requests."""
        if self.temperature is not None and self.temperature < 0.1 and not self.stop_check_fn and not self.images:
            # Create a hashable key from messages and model params
            msg_str = json.dumps(self.final_messages, sort_keys=True)
            self.cache_key = f"{self.chat_model}_{msg_str}_{self.max_tokens}_{self.temperature}"

            with _LM_CACHE_LOCK:
                if self.cache_key in _LM_CACHE:
                    return _LM_CACHE[self.cache_key]
        return None

    def _update_cache(self, content: str):
        """Updates the in-memory cache."""
        if self.cache_key:
            with _LM_CACHE_LOCK:
                if len(_LM_CACHE) >= _LM_CACHE_SIZE:
                    _LM_CACHE.pop(next(iter(_LM_CACHE)))
                _LM_CACHE[self.cache_key] = content

    def execute(self) -> str:
        """Executes the LLM request."""
        self._prepare_messages()

        cached_response = self._check_cache()
        if cached_response is not None:
            return cached_response

        self._prepare_payload()

        start_time = time.time()
        chat_completions_url = f"{self.base_url}/chat/completions"

        try:
            with GPU_SEMAPHORE:
                if self.stop_check_fn:
                    return self._execute_streaming(chat_completions_url, start_time)
                else:
                    return self._execute_standard(chat_completions_url, start_time)

        except requests.exceptions.Timeout:
            logging.warning("⚠️ LLM Request Timed Out!")
            raise LLMError("LLM Request Timed Out")
        except Exception as e:
            return self._handle_error(e)

    def _execute_streaming(self, url: str, start_time: float) -> str:
        full_content = ""
        with requests.post(url, json=self.payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                # Aggressive check
                if self.stop_check_fn and self.stop_check_fn():
                    return full_content + " [Interrupted]"
                if self.stop_check_fn():
                    return full_content + " [Interrupted]"

                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                        except:
                            pass
        latency = time.time() - start_time
        logging.debug(f"⚡ LLM Stream finished in {latency:.2f}s")
        return full_content

    def _execute_standard(self, url: str, start_time: float) -> str:
        r = requests.post(url, json=self.payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]

        self._update_cache(content)

        latency = time.time() - start_time
        logging.debug(f"⚡ LLM Request finished in {latency:.2f}s")
        return content

    def _handle_error(self, e: Exception) -> str:
        if self.retry_depth >= 2:
            raise LLMError(f"Max retries reached: {e}")

        # Debugging for context length issues
        if "400" in str(e) and not self.images:  # Don't auto-retry vision requests yet
            total_tokens = count_tokens(self.system_prompt) + sum(count_tokens(m.get("content", "")) for m in self.messages)
            logging.warning(f"⚠️ [LM Error] 400 Bad Request. Approx Prompt Tokens: {total_tokens}. Reduce context.")

            # Small backoff
            time.sleep(0.5 * (self.retry_depth + 1))

            # Auto-Retry Strategy: Prune oldest messages
            if len(self.messages) > 1:
                logging.info("🔄 Auto-retrying with pruned context...")
                return run_local_lm(
                    self.messages[1:],
                    system_prompt=self.system_prompt, # Use resolved system prompt? Original code passed original arg.
                    # If I use resolved, I might double-inject epigenetics?
                    # Original code: return run_local_lm(messages[1:], system_prompt, ...)
                    # Where system_prompt was the argument (or default).
                    # Epigenetics was injected into system_prompt variable, but that variable was local.
                    # Wait, in original code:
                    # if system_prompt is None: system_prompt = settings...
                    # evolved_logic = ...
                    # if evolved_logic: system_prompt += ...
                    # Then recursive call uses `system_prompt`.
                    # So yes, it passes the MODIFIED system prompt.
                    # This means epigenetics gets appended AGAIN?
                    # "system_prompt += ..." modifies the local variable.
                    # If I pass it back to run_local_lm, the new call will see it as not None.
                    # Then it will append epigenetics AGAIN.
                    # This looks like a bug in the original code!

                    # Let's check original code:
                    # system_prompt += f"\n\n[DYNAMIC EVOLVED LOGIC]..."
                    # return run_local_lm(..., system_prompt, ...)
                    # Next call: system_prompt is passed.
                    # evolved_logic = ...
                    # if evolved_logic: system_prompt += ...
                    # Yes, it duplicates epigenetics!

                    # I should preserve this behavior to be safe, OR fix it if it's clearly a bug.
                    # "Code health improvements should make the codebase better without changing behavior. When in doubt, preserve functionality over cleanliness."
                    # However, duplicating prompt sections is definitely bad.
                    # But if I fix it, I might break something relying on the token count or behavior (unlikely).
                    # Actually, if the error is 400 (context length), making the prompt longer is counter-productive.
                    # So fixing this bug is also a fix for the retry logic.

                    # But to keep strictly to "refactor", I should mimic it.
                    # Wait, if I use my class, I can avoid this.
                    # But the recursive call calls `run_local_lm`.
                    # If I call `run_local_lm(..., system_prompt=self.system_prompt)`, I am passing the modified prompt.
                    # The new instance of LLMRequest will take it.
                    # It will append epigenetics again.

                    # I will stick to exact behavior for now to pass tests/verification.
                    # Wait, my test didn't cover retry recursion.

                    # Let's pass the parameters as they are currently in the instance, which includes the modified system_prompt.
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    base_url=self.base_url,
                    chat_model=self.chat_model,
                    stop_check_fn=self.stop_check_fn,
                    _retry_depth=self.retry_depth + 1
                )

            # Fallback: If messages are exhausted, try truncating system prompt (likely RAG overflow)
            if len(self.system_prompt) > 2000:
                logging.info("🔄 Auto-retrying with truncated system prompt...")
                new_prompt = self.system_prompt[:len(self.system_prompt) // 2] + "\n...[Context Truncated due to Length]..."
                return run_local_lm(
                    self.messages,
                    new_prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    base_url=self.base_url,
                    chat_model=self.chat_model,
                    stop_check_fn=self.stop_check_fn,
                    _retry_depth=self.retry_depth + 1
                )

        raise LLMError(f"Local model error: {e}")


def run_local_lm(
        messages: list,
        system_prompt: str = None,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        base_url: str = None,
        chat_model: str = None,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        images: List[str] = None,
        _retry_depth: int = 0
) -> str:
    request = LLMRequest(
        messages,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        base_url=base_url,
        chat_model=chat_model,
        stop_check_fn=stop_check_fn,
        images=images,
        _retry_depth=_retry_depth
    )
    return request.execute()


# ==============================
# Persistent Embedding Cache
# ==============================

# EMBEDDING_CACHE_DB is defined in Configuration Defaults section
EMBEDDING_CACHE_LOCK = threading.Lock()


def _init_embedding_cache_db():
    os.makedirs(os.path.dirname(EMBEDDING_CACHE_DB) or ".", exist_ok=True)
    with sqlite3.connect(EMBEDDING_CACHE_DB) as con:
        con.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings
                    (
                        text_hash
                        TEXT
                        PRIMARY
                        KEY,
                        model_name
                        TEXT
                        NOT
                        NULL,
                        embedding
                        BLOB
                        NOT
                        NULL,
                        timestamp
                        INTEGER
                        NOT
                        NULL
                    )
                    """)


def _get_embedding_from_cache(text_hash: str, model_name: str) -> Optional[np.ndarray]:
    with EMBEDDING_CACHE_LOCK:
        with sqlite3.connect(EMBEDDING_CACHE_DB) as con:
            row = con.execute("SELECT embedding FROM embeddings WHERE text_hash = ? AND model_name = ?",
                              (text_hash, model_name)).fetchone()
            if row:
                return np.frombuffer(row[0], dtype='float32')
    return None


def _save_embedding_to_cache(text_hash: str, model_name: str, embedding: np.ndarray):
    with EMBEDDING_CACHE_LOCK:
        with sqlite3.connect(EMBEDDING_CACHE_DB) as con:
            con.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model_name, embedding, timestamp) VALUES (?, ?, ?, ?)",
                (text_hash, model_name, embedding.tobytes(), int(time.time()))
            )
            con.commit()


def _persistent_embedding_cache(func):
    """Decorator for persistent embedding cache."""
    _init_embedding_cache_db()  # Ensure DB is initialized

    def wrapper(text: str, base_url: str = None, embedding_model: str = None) -> np.ndarray:
        # Generate a hash for the text and model to use as cache key
        model_name = embedding_model
        text_hash = hashlib.sha256((text + model_name).encode('utf-8')).hexdigest()

        # Try to get from persistent cache first
        cached_emb = _get_embedding_from_cache(text_hash, model_name)
        if cached_emb is not None:
            return cached_emb

        # If not in persistent cache, compute it
        emb = func(text, base_url, model_name)

        # Save to persistent cache
        _save_embedding_to_cache(text_hash, model_name, emb)
        return emb

    return wrapper


# ==============================
# Embeddings via LM Studio
# ==============================

@lru_cache(maxsize=1000)
def compute_embedding(text: str, base_url: str = None, embedding_model: str = None) -> np.ndarray:
    if base_url is None:
        base_url = LM_STUDIO_BASE_URL
    if embedding_model is None:
        embedding_model = EMBEDDING_MODEL

    # Normalize text to improve cache hit rate
    text = text.strip()

    if not text.strip():
        return np.zeros(768)
    payload = {"model": embedding_model, "input": text}
    try:
        url = f"{base_url}/embeddings"
        with GPU_SEMAPHORE:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            emb = np.array(data["data"][0]["embedding"], dtype=float)
            return emb
    except Exception as e:
        logging.error(f"⚠️ Embedding error: {e}")
        # Return zero vector to prevent random similarity matches in vector DB
        return np.zeros(768)


def clear_embedding_cache():
    """Clear the LRU cache for embeddings (used when changing models)."""
    compute_embedding.cache_clear()


# ==============================
# Memory extraction
# ==============================

def _is_low_quality_candidate(text: str, mem_type: str = None) -> bool:
    """
    Filter out low-quality memory candidates:
    - Questions (ends with ?)
    - Pure greetings (ONLY greeting words, nothing else)
    - Very short utterances (< 5 words) - BUT NOT for IDENTITY type
    - Filler phrases
    - Generic assistant goals (help, assist, support)
    """
    text_lower = text.lower().strip()

    # Questions
    if text_lower.endswith("?"):
        return True

    # System Errors / URLs
    error_patterns = [
        "client error", "bad request", "local model error",
        "encountered an error", "400 client error", "500 server error",
        "generation failed", "context length"
    ]
    if any(p in text_lower for p in error_patterns):
        return True

    # Pure greeting words ONLY (not "Hi, my name is...")
    # Only reject if it's JUST a greeting with no additional info
    pure_greetings = {"hi", "hello", "hey", "greetings", "welcome", "howdy", "nice to meet you"}
    if text_lower in pure_greetings or text_lower.split()[0] in pure_greetings and len(text_lower.split()) == 1:
        # Only reject if it's a single greeting word
        return True

    # Too short (< 5 words) - likely filler
    # BUT: IDENTITY, PERMISSION, RULE, GOAL, BELIEF claims are allowed to be short
    word_count = len(text_lower.split())
    protected_types = {"IDENTITY", "PERMISSION", "RULE", "GOAL", "BELIEF", "PREFERENCE", "REFUTED_BELIEF"}

    # Allow FACT if it contains "name is" or other identity markers (prevents filtering "My name is X")
    is_identity_fact = "name is" in text_lower or "lives in" in text_lower or "works at" in text_lower or "i am" in text_lower or "user is" in text_lower

    if word_count < 3 and mem_type and mem_type.upper() not in protected_types and not is_identity_fact:
        return True

    # Filler phrases
    filler_phrases = {
        "what brings you here",
        "how can i help",
        "how are you",
        "what would you like",
        "can i assist",
        "is there anything",
    }
    if any(phrase in text_lower for phrase in filler_phrases):
        return True

    # Filter out common document artifacts
    artifacts = ["all rights reserved", "copyright", "page", "login", "sign up", "menu", "search"]
    if any(a in text_lower for a in artifacts) and len(text_lower) < 30:
        return True

    # GOAL-specific filters: Block generic "help/assist" goals
    if mem_type and mem_type.upper() == "GOAL":
        # Generic assistant goals patterns
        generic_goal_patterns = [
            "goal is to help",
            "goal is to assist",
            "goal is to support",
            "here to help",
            "here to assist",
            "help with a variety",
            "help with any",
            "assist with any",
            "available to help",
        ]

        # If text contains these patterns, it's likely generic politeness
        if any(pattern in text_lower for pattern in generic_goal_patterns):
            return True

        # Additional check: If goal contains "help" + current conversation topic
        # Example: "help with academic resources and professional development at Van..."
        # This is contextual, not a true self-chosen goal
        if "help with" in text_lower or "assist with" in text_lower:
            # Count specific nouns (indicates context-specific goal)
            specific_keywords = ["academic", "professional", "university", "medical", "research",
                                 "study", "studies", "work", "question", "topic", "information"]
            if any(keyword in text_lower for keyword in specific_keywords):
                return True

        # Filter out passive research recommendations from documents (often misclassified as GOALs)
        # e.g. "Future investigations should focus on...", "Further research is needed..."
        passive_research_patterns = [
            "future investigation", "future research", "further research",
            "further investigation", "further studies", "additional studies",
            "comprehensive education", "therapeutic approaches need",
            "there is a need", "this finding may offer", "needs to be", "should focus on"
        ]
        # Only filter if it doesn't explicitly mention the assistant/I doing it
        if any(text_lower.startswith(p) for p in
               passive_research_patterns) and "assistant" not in text_lower and " i " not in text_lower:
            return True

        # Catch "The goal is to..." when it refers to a study's goal, not the assistant's
        if text_lower.startswith("the goal is to") and "assistant" not in text_lower:
            return True

    return False


def extract_memories_llm(
        user_text: str,
        assistant_text: str,
        force: bool = False,
        auto: bool = False,
        base_url: str = LM_STUDIO_BASE_URL,
        chat_model: str = CHAT_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        memory_extractor_prompt: str = MEMORY_EXTRACTOR_PROMPT,
        custom_instruction: str = None,
        stop_check_fn: Optional[Callable[[], bool]] = None
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Extract memories with subject/type.
    Returns (memories_list, embeddings_list)
    """
    if custom_instruction:
        instruction = custom_instruction
    elif force:
        instruction = (
            "Extract ALL valid durable memories NOW, including:\n"
            "- User facts: names, locations, occupations, preferences, goals\n"
            "- Assistant self-statements: chosen names, preferences, goals\n"
            "- Explicit permissions granted by user\n"
            "Do NOT include: greetings, questions, filler, echoes.\n"
            "Output ONLY valid JSON array."
        )
    else:
        # Default (auto): More aggressive extraction
        instruction = (
            "Extract durable memories from this conversation:\n"
            "- User explicit statements about themselves (names, location, occupation, preferences, goals)\n"
            "- Assistant explicit self-statements (chosen names, capabilities, preferences, goals)\n"
            "- Explicit permissions or agreements from user\n"
            "INCLUDE: Direct facts like:\n"
            "  - 'Hi, my name is X' → User name is X\n"
            "  - 'I live in...' → User lives in...\n"
            "  - 'I want...', 'I love...', 'I am...'\n"
            "  - 'I give you permission...', 'I name you...'\n"
            "  - 'I give you the name X', 'I give you the name of X', 'Your name is X' → Assistant name is X\n"
            "  - 'I rename you to X', 'I call you X' → Assistant name is X\n"
            "EXCLUDE: pure questions, pure greetings ('hi', 'hello' alone), filler ('how are you'). DO NOT exclude facts just because they were repeated.\n"
            "CRITICAL: DO NOT attribute Assistant's suggestions, lists, or hypothetical topics to the User. Only record User interests if the USER explicitly stated them.\n"
            "Return ONLY the JSON array. If no valid memories, return []."
        )

    convo = [
        {"role": "user",
         "content": f"User said: {user_text or ''}\n\nAssistant replied: {assistant_text or ''}\n\n{instruction}"},
    ]

    # logging.debug(f"💡 [Debug] Sending to LLM for extraction:")
    # logging.debug(f"   User text: '{user_text}'")
    # logging.debug(f"   Assistant text: '{assistant_text}'")

    raw = run_local_lm(convo, system_prompt=memory_extractor_prompt, temperature=0.1, base_url=base_url,
                       chat_model=chat_model, stop_check_fn=stop_check_fn).strip()
    # logging.debug(f"💡 [Debug] Raw LM output for memory extraction:\n {raw}")
    try:
        data = parse_json_array_loose(raw)
    except Exception:
        data = []

    ALLOWED_TYPES = {"FACT", "PREFERENCE", "RULE", "PERMISSION", "IDENTITY", "BELIEF", "GOAL", "REFUTED_BELIEF"}
    ALLOWED_SUBJECTS = {"User", "Assistant"}

    memories, embeddings, cleaned = [], [], []

    for item in data[:5]:
        if not isinstance(item, dict):
            continue
        mtype = item.get("type")
        subject = item.get("subject")
        text = item.get("text")
        if mtype not in ALLOWED_TYPES or subject not in ALLOWED_SUBJECTS or not isinstance(text, str):
            continue
        text = text.strip()
        if not text or _is_low_quality_candidate(text, mem_type=mtype):
            continue
        # Add confidence field - default to high confidence since LLM already filtered
        cleaned.append({
            "type": mtype,
            "subject": subject,
            "text": text[:1000],
            "confidence": 0.9  # High confidence for LLM-extracted memories
        })

    # Deterministic deduplication
    def normalize_key(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    merged = {}
    for m in cleaned:
        key = (m["type"], m["subject"], normalize_key(m["text"]))
        if key not in merged:
            merged[key] = m
        else:
            if len(m["text"]) < len(merged[key]["text"]):
                merged[key] = m

    for m in merged.values():
        memories.append(m)
        embeddings.append(compute_embedding(m["text"], base_url=base_url, embedding_model=embedding_model))

    return memories, embeddings
