import threading
import time
from typing import List, Dict, Any, Optional

class Thalamus:
    """
    Sensory Gating System (The Thalamus).
    Buffers and prioritizes inputs (Chat, File, Web) before they reach conscious attention.
    Prevents cognitive overload by chunking or throttling high-volume streams.
    """
    def __init__(self, core):
        self.core = core
        self.input_buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 50
        self.processing_thread = None
        self.running = False
        self.focus_context = None
        
        # Subscribe to conscious content to bias perception
        if self.core.event_bus:
            self.core.event_bus.subscribe("CONSCIOUS_CONTENT", self._on_conscious_content)

    def _on_conscious_content(self, event):
        """Update focus based on Global Workspace."""
        data = event.data
        self.focus_context = data.get("content", "").lower()
        self.core.log(f"🧠 [Thalamus] Attention biased toward: {self.focus_context}")

    def start(self):
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True, name="ThalamusWorker")
        self.processing_thread.start()

    def stop(self):
        self.running = False

    def ingest(self, source: str, content: Any, metadata: Dict = None):
        """
        Receive raw sensory input.
        """
        with self.buffer_lock:
            # Overflow protection: Drop oldest low-priority inputs if full
            if len(self.input_buffer) >= self.max_buffer_size:
                self.input_buffer.pop(0)
            
            packet = {
                "source": source,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time()
            }
            self.input_buffer.append(packet)
            self.core.log(f"🧠 [Thalamus] Buffered input from {source}")

    def _process_loop(self):
        while self.running:
            time.sleep(0.5) # Tick rate
            
            packet = None
            with self.buffer_lock:
                if self.input_buffer:
                    packet = self.input_buffer.pop(0)
            
            if not packet:
                continue

            self._route_signal(packet)

    def _route_signal(self, packet):
        """
        Route signal to appropriate system.
        Prioritizes inputs related to focus_context (Attention Modulation).
        """
        source = packet["source"]
        content = packet["content"]
        
        # --- ATTENTION MODULATION ---
        # If input is related to current focus, boost its priority/visibility
        is_relevant = False
        if self.focus_context and isinstance(content, str):
            # Simple keyword matching for relevance
            focus_words = set(self.focus_context.split())
            content_words = set(content.lower().split())
            if focus_words.intersection(content_words):
                is_relevant = True
                self.core.log(f"🧠 [Thalamus] Input from {source} matches focus. Prioritizing.")
        
        # 1. Large Text Handling (e.g. pasted PDF content)
        if isinstance(content, str) and len(content) > 2000:
            self.core.log(f"🧠 [Thalamus] Large input detected ({len(content)} chars). Routing to Document Processor.")
            # Creation of temporary file/chunking would happen here
            pass
        
        # 2. Standard Routing with Attention Flags
        if self.core.event_bus:
            packet["is_focused"] = is_relevant
            self.core.event_bus.publish("SENSORY_INPUT", packet)