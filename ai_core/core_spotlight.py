import time
import os
import difflib
from typing import Dict, Any, List, Optional
from .lm import run_local_lm

class GlobalWorkspace:
    """
    The Global Workspace (Spotlight of Attention).
    Maintains a 'Working Memory' of active conscious items.
    Integrates competing signals (Drives, Goals, Sensory) and broadcasts
    the current mental state to all modules.
    """
    def __init__(self, core):
        self.core = core
        self.working_memory: List[Dict[str, Any]] = [] # List of active items
        self.capacity = 7 # The magical number 7 +/- 2
        self.last_update = time.time()
        self.update_interval = 2.0 # Faster update cycle for fluidity
        self.last_broadcast = time.time()
        self.stream_file = "./data/stream_of_consciousness.md"

        # Ensure data dir exists
        try:
            os.makedirs(os.path.dirname(self.stream_file), exist_ok=True)
            with open(self.stream_file, "w", encoding="utf-8") as f:
                f.write("# Stream of Consciousness\n\n")
        except Exception as e:
            print(f"Warning: Could not initialize stream file: {e}")

    def integrate(self, content: str, source: str, salience: float, metadata: Dict = None):
        """
        Add a new item to working memory.
        If item exists (by content), update its salience.
        """
        if not content: return

        # Normalize content
        content_key = content.strip().lower()

        # Check if exists (Semantic/Fuzzy Match)
        existing = None
        for item in self.working_memory:
            # Check exact match first
            if item["content"].strip().lower() == content_key:
                existing = item
                break

            # Check fuzzy match
            ratio = difflib.SequenceMatcher(None, item["content"], content).ratio()
            if ratio > 0.75:
                existing = item
                break

        if existing:
            # Boost salience (Reinforcement)
            existing["salience"] = min(1.0, existing["salience"] + (salience * 0.5))
            existing["timestamp"] = time.time() # Refresh recency
            existing["source"] = source # Update source
            if metadata:
                existing["metadata"].update(metadata)
        else:
            # Add new item
            self.working_memory.append({
                "content": content,
                "source": source,
                "salience": min(1.0, salience),
                "timestamp": time.time(),
                "metadata": metadata or {}
            })

    def decay(self):
        """
        Apply decay to all items in working memory.
        Items fade if not reinforced.
        """
        now = time.time()
        decay_rate = 0.05 # Decay per second

        time_delta = now - self.last_update

        active_items = []
        for item in self.working_memory:
            # Apply decay
            item["salience"] -= (decay_rate * time_delta)
            
            # Prune items with low salience or old age
            age = now - item["timestamp"]
            # Keep if salience > 0.1 AND age < 300 (5 mins)
            # OR if salience is very high (>0.8), keep longer
            if (item["salience"] > 0.1 and age < 300) or (item["salience"] > 0.8 and age < 600):
                active_items.append(item)
                
        self.working_memory = active_items

        # Enforce capacity (keep top K by salience)
        self.working_memory.sort(key=lambda x: x["salience"], reverse=True)
        if len(self.working_memory) > self.capacity:
            self.working_memory = self.working_memory[:self.capacity]

    def get_context(self) -> str:
        """
        Return a formatted string of the current Conscious State.
        Used by Decider and Self-Model.
        """
        if not self.working_memory:
            return "Mind is empty."

        lines = []
        for item in self.working_memory:
            # Format: [SOURCE] Content (Salience)
            s_val = int(item["salience"] * 100)
            lines.append(f"[{item['source']}] {item['content']} ({s_val}%)")

        return "\n".join(lines)

    def update(self):
        """
        Main cycle: Gather inputs, Decay, Broadcast.
        """
        if time.time() - self.last_update < self.update_interval:
            return

        # 1. Decay existing items
        self.decay()
        self.last_update = time.time()

        # 2. Poll Competing Signals (Pull-based)
        self._gather_signals()

        # 3. Broadcast if significant change
        if self.working_memory:
            top_item = self.working_memory[0]
            # Always broadcast periodically to keep system alive
            if time.time() - self.last_broadcast > 5.0:
                self.broadcast(top_item)

    def _gather_signals(self):
        """Gather inputs from Drives, Goals, Sensory."""

        # A. Drives
        if self.core.self_model:
            drives = self.core.self_model.get_drives()
            # Ingest high pressure drives
            for d, v in drives.items():
                if isinstance(v, (int, float)) and v > 0.7:
                     self.integrate(f"High Drive: {d}", "SelfModel", v * 0.8)

        # B. Sensory (User Input)
        if self.core.self_model:
             last = self.core.self_model.get_last_interaction()
             if time.time() - last < 30:
                 self.integrate("User is present and interacting", "Sensory", 0.9)

        # C. Active Goal (Focus)
        if self.core.memory_store:
             # get_active_by_type might return tuple (id, type, subject, text, ...)
             goals = self.core.memory_store.get_active_by_type("GOAL")
             if goals:
                 # Top goal
                 g = goals[0]
                 # Handle variable schema from MemoryStore (sometimes 3, sometimes 8 items)
                 # Typically: (id, type, subject, text, source, verified, flags, confidence)
                 # or basic: (id, type, text) depending on query
                 text = "Unknown Goal"
                 if len(g) > 3:
                     text = g[3]
                 elif len(g) > 2:
                     text = g[2]

                 self.integrate(f"Current Goal: {text}", "GoalManager", 0.6)

    def broadcast(self, top_item: Dict[str, Any]):
        """
        Broadcast the full conscious context.
        """
        context_str = self.get_context()

        # 1. Log to Stream of Consciousness File
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"**[{timestamp}]**\n{context_str}\n\n---\n\n"
            with open(self.stream_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            # Silent fail if file IO fails
            pass

        # 2. Event Bus
        if self.core.event_bus:
            self.core.log(f"🔦 Workspace: Focus on '{top_item['content']}'")
            # We publish the top item as the 'focus', but include full context in data
            self.core.event_bus.publish(
                "CONSCIOUS_CONTENT",
                data={
                    "focus": top_item["content"],
                    "salience": top_item["salience"],
                    "full_context": context_str,
                    "working_memory": self.working_memory
                },
                source="GlobalWorkspace",
                priority=10
            )

    def get_dominant_thought(self) -> Optional[Dict[str, Any]]:
        """Return the item with the highest salience."""
        if not self.working_memory:
            return None
        # Sort by salience desc
        sorted_mem = sorted(self.working_memory, key=lambda x: x["salience"], reverse=True)
        return sorted_mem[0]

    def evolve_thought(self, chat_model: str = None, base_url: str = None) -> str:
        """
        Evolve the dominant thought using LLM (Chain of Thought).
        """
        dominant = self.get_dominant_thought()
        if not dominant:
            return "Mind is empty."

        content = dominant["content"]

        prompt = (
            f"CURRENT THOUGHT: {content}\n\n"
            "You are the Inner Voice of the AI.\n"
            "Expand on this thought. Connect it to broader implications, or refine it into a specific question.\n"
            "Do not repeat the thought. Move it forward.\n"
            "Output ONLY the new thought (1-2 sentences)."
        )

        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Stream of Consciousness.",
            temperature=0.7,
            max_tokens=100,
            chat_model=chat_model,
            base_url=base_url
        )

        if response and not response.startswith("⚠️"):
            # Integrate back with high salience to maintain focus
            self.integrate(response, "GlobalWorkspace", 1.0)
            return response

        return content
