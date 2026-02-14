from typing import Optional, Callable, Any, Dict, List
import numpy as np
import logging

from .memory import MemoryStore
from .meta_memory import MetaMemoryStore


TYPE_PRECEDENCE = {
    "PERMISSION": 0,  # Highest priority: only user can grant
    "RULE": 1,        # Rules from/for assistant
    "IDENTITY": 2,    # Who is user/assistant
    "REFUTED_BELIEF": 3, # Explicitly rejected ideas (Negative Memory)
    "COMPLETED_GOAL": 3, # Finished objectives (Prevents re-generation)
    "PREFERENCE": 4,  # Likes/dislikes
    "GOAL": 5,        # Aims/desires
    "FACT": 6,        # Objective truths
    "BELIEF": 7,      # Opinions/convictions (lowest priority)
    "NOTE": 1,        # Assistant Notes (High priority, internal)
    "CURIOSITY_GAP": 1, # High priority: Questions for the user
}

CONFIDENCE_MIN = {
    "PERMISSION": 0.85,  # Very high: explicit user permission (only user can grant)
    "RULE": 0.9,         # Very high: guidelines for assistant behavior
    "IDENTITY": 0.8,     # High: identity claims (who are you)
    "REFUTED_BELIEF": 0.9, # Very high: explicit rejection
    "COMPLETED_GOAL": 0.9, # Very high: explicit completion
    "PREFERENCE": 0.6,   # Medium-high: preferences/likes/dislikes
    "GOAL": 0.7,         # High: goals/desires
    "FACT": 0.7,         # High: factual assertions
    "BELIEF": 0.5,       # Medium: beliefs/opinions/convictions
    "NOTE": 0.9,         # Very high: manual notes
    "CURIOSITY_GAP": 0.9, # Very high: system generated
}


class MemoryArbiter:
    """
    Autonomous bridge between ReasoningStore and MemoryStore.

    - Does NOT reason
    - Does NOT decide truth
    - Only enforces admission rules
    """

    def __init__(self, memory_store: MemoryStore, meta_memory_store: Optional[MetaMemoryStore] = None, embed_fn: Optional[Callable] = None, log_fn: Callable[[str], None] = logging.info, event_bus: Any = None, config: Dict = None):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.embed_fn = embed_fn
        self.log = log_fn
        self.event_bus = event_bus
        self.config = config or {}

        self.type_precedence = {
            "PERMISSION": 0,
            "RULE": 1,
            "IDENTITY": 2,
            "REFUTED_BELIEF": 3,
            "COMPLETED_GOAL": 3,
            "PREFERENCE": 4,
            "GOAL": 5,
            "FACT": 6,
            "BELIEF": 7,
            "NOTE": 1,
            "CURIOSITY_GAP": 1,
        }

        # Load confidence thresholds from config or use defaults
        defaults = {
            "PERMISSION": 0.85,
            "RULE": 0.9,
            "IDENTITY": 0.8,
            "REFUTED_BELIEF": 0.9,
            "COMPLETED_GOAL": 0.9,
            "PREFERENCE": 0.6,
            "GOAL": 0.7,
            "FACT": 0.7,
            "BELIEF": 0.5,
            "NOTE": 0.9,
            "CURIOSITY_GAP": 0.9,
        }
        self.confidence_min = self.config.get("arbiter_confidence_thresholds", defaults)

    # --------------------------
    # Public API
    # --------------------------
    def consider(
        self,
        text: str,
        mem_type: str,
        confidence: float,
        subject: str = "User",
        source: str = "reasoning",
        affect: float = 0.5,
        embedding: Optional[np.ndarray] = None,
        identity: Optional[str] = None
    ) -> Optional[int]:
        """
        Decide whether to promote reasoning into memory.

        Returns memory_id if stored, None otherwise.
        """

        mem_type = mem_type.upper()
        self.log(f"🔍 [Arbiter] Processing {mem_type} for {subject} (Conf: {confidence}):\n    \"{text}\"")

        if mem_type not in self.type_precedence:
            self.log(f"❌ [Arbiter] Type '{mem_type}' not in precedence table")
            return None

        # 0️⃣ Filter out meta-actions/refutations saved as text
        if text.strip().startswith("[Refuting Belief") or text.strip().startswith("Refuting Belief"):
            self.log(f"⛔ [Arbiter] BLOCKING meta-action text: \"{text[:50]}...\"")
            return None

        # 1️⃣ Confidence gate
        min_conf = self.confidence_min.get(mem_type, 0.7)
        if confidence < min_conf:
            self.log(f"❌ [Arbiter] Confidence gate failed: {confidence} < {min_conf} (required for {mem_type})")
            return None

        self.log(f"✅ [Arbiter] Passed confidence gate: {confidence} >= {min_conf}")

        # 2️⃣ Identity + version chaining
        if identity is None:
            identity = self.memory_store.compute_identity(text, mem_type=mem_type)
        previous_versions = self.memory_store.get_by_identity(identity)
        parent_id = previous_versions[-1]["id"] if previous_versions else None

        # 2.5️⃣ Exact duplicate guard (same text as latest version)
        if previous_versions:
            last_text = previous_versions[-1]["text"].strip()
            if last_text == text.strip() and source != "user_override":
                 self.log(f"❌ [Arbiter] Exact duplicate detected (ID: {previous_versions[-1]['id']})")
                 return None

        self.log(f"✅ [Arbiter] No duplicates. Previous versions: {len(previous_versions)}, parent_id: {parent_id}")

        # 2.7️⃣ Negative Knowledge Enforcement (Invariant 7)
        # Generate embedding early for checks
        if embedding is None and self.embed_fn:
            embedding = self.embed_fn(text)

        if mem_type not in ("REFUTED_BELIEF", "NOTE"):
            # Check via Embedding
            if embedding is not None:
                similar_mems = self.memory_store.search_refuted(embedding, limit=3)
                for mid, mtype, msubj, mtext, sim in similar_mems:
                    # search_refuted only returns REFUTED_BELIEF
                    if sim > 0.85:
                         self.log(f"⛔ [Arbiter] BLOCKING memory: Contradicts REFUTED_BELIEF (Sim: {sim:.2f})\n    New: \"{text}\"\n    Refuted: \"{mtext}\"")
                         return None

        # 3️⃣ Conflict detection (exact, conservative)
        
        # 2.6️⃣ Cross-Subject Identity Conflict Guard
        # Prevent Assistant from claiming User's name/identity and vice versa
        # Check this for IDENTITY type OR if the text looks like an identity claim
        extracted_val = self._extract_value(text)
        if extracted_val and (mem_type == "IDENTITY" or "name is" in text.lower()):
            # Check against all active identities (and FACTs that act like identities)
            active_identities = self.memory_store.get_active_by_type("IDENTITY")
            active_facts = self.memory_store.get_active_by_type("FACT")
            
            for item in (active_identities + active_facts):
                _, subj, txt, _ = item[:4] # Safe unpacking
                # If subject is different (e.g. User vs Assistant)
                if subj.lower() != subject.lower():
                    existing_val = self._extract_value(txt)
                    if existing_val and existing_val.lower() == extracted_val.lower():
                        self.log(f"❌ [Arbiter] Identity conflict: '{extracted_val}' is already assigned to {subj}")
                        return None

        # 2.8️⃣ Permission Overwrite Guard
        if previous_versions and previous_versions[-1]['type'] == 'PERMISSION' and mem_type != 'PERMISSION':
             if source != "user_override":
                 self.log(f"⛔ [Arbiter] BLOCKING overwrite of PERMISSION with {mem_type}. Source must be 'user_override'.")
                 return None

        conflicts = self.memory_store.find_conflicts_exact(text)
        self.log(f"🔍 [Arbiter] Found {len(conflicts)} conflicts")

        # 4️⃣ Precedence dampening
        adjusted_confidence = confidence
        for c in conflicts:
            if self.type_precedence.get(c["type"], 99) < self.type_precedence.get(mem_type, 99):
                self.log(f"⛔ [Arbiter] BLOCKING memory due to conflict with higher precedence {c['type']} (ID: {c['id']}):\n    Conflict: \"{c['text']}\"")
                return None

        # 5️⃣ Append-only write (versioned)
        self.log(f"✅ [Arbiter] Saving memory with adjusted_confidence={adjusted_confidence}")
        
        # Use consistent timestamp for both memory and meta-memory

        import time
        created_at = int(time.time())
        from datetime import datetime
        timestamp = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")

        memory_id = self.memory_store.add_entry(
            text=text,
            mem_type=mem_type,
            subject=subject,
            confidence=adjusted_confidence,
            source=source,
            identity=identity,
            parent_id=parent_id,
            conflicts=[c["id"] for c in conflicts],
            created_at=created_at,
            embedding=embedding,
            affect=affect
        )

        # If we just learned a HIGH CONFIDENCE Fact or Insight (Epiphany)
        # and it wasn't just explicitly told to us by the User (Source != User)
        if memory_id is not None and confidence > 0.92 and source != "User":
            
            # 1. Check importance (heuristic)
            is_exciting = any(w in text.lower() for w in ["breakthrough", "critical", "proven", "refuted", "connection found"])
            
            if is_exciting and self.event_bus:
                self.log(f"⚡ Epiphany detected! Signaling to user.")
                self.event_bus.publish(
                    "AI_SPEAK", 
                    f"💡 Insight: I just realized that {text} (Confidence: {confidence})"
                )

        # 6️⃣ Create meta-memory about this new memory creation
        if self.meta_memory_store and memory_id:
            # Extract the current value
            value = self._extract_value(text)
            
            # Check for previous version to show change
            old_value = None
            if previous_versions:
                old_value = self._extract_value(previous_versions[-1]["text"])

            # Create human-readable meta-memory
            if mem_type == 'IDENTITY':
                if 'name is' in text.lower():
                    if old_value:
                        meta_text = f"{subject} name changed from {old_value} to {value} on {timestamp}"
                    else:
                        meta_text = f"{subject} name set to {value} on {timestamp}"
                elif 'lives in' in text.lower():
                    if old_value:
                        meta_text = f"{subject} location changed from {old_value} to {value} on {timestamp}"
                    else:
                        meta_text = f"{subject} location set to {value} on {timestamp}"
                else:
                    if old_value:
                        meta_text = f"{subject} {mem_type.lower()} updated: '{old_value}' -> '{value}' on {timestamp}"
                    else:
                        meta_text = f"{subject} {mem_type.lower()} recorded: '{value}' on {timestamp}"
            elif old_value:
                meta_text = f"{subject} {mem_type.lower()} updated: '{old_value}' -> '{value}' on {timestamp}"
            else:
                # Default for new recordings
                type_labels = {
                    'PREFERENCE': 'preference',
                    'GOAL': 'goal',
                    'FACT': 'fact',
                    'RULE': 'rule',
                    'PERMISSION': 'permission',
                    'BELIEF': 'belief'
                }
                label = type_labels.get(mem_type, mem_type.lower())
                meta_text = f"{subject} {label} recorded: '{value}' on {timestamp}"

            self.meta_memory_store.add_meta_memory(
                event_type="VERSION_UPDATE" if old_value else "MEMORY_CREATED",
                memory_type=mem_type,
                subject=subject,
                text=meta_text,
                old_id=previous_versions[-1]["id"] if previous_versions else None,
                new_id=memory_id,
                old_value=old_value,
                new_value=value,
                metadata={"confidence": adjusted_confidence, "source": source}
            )
            self.log(f"      🧠 Meta-memory: {meta_text}")

        return memory_id

    def consider_batch(self, candidates: List[Dict]) -> List[int]:
        """
        Process a batch of memory candidates through the arbiter.
        Returns a list of IDs for successfully promoted memories.
        """
        promoted_ids = []
        if not candidates:
            return promoted_ids

        # Pre-process candidates to get embeddings and identities
        processed_candidates = []
        for c in candidates:
            mem_type = c.get("type", "FACT").upper()
            text = c.get("text", "")
            subject = c.get("subject", "User")
            confidence = c.get("confidence", 0.85)
            source = c.get("source", "reasoning")
            affect = c.get("affect", 0.5)

            if mem_type not in self.type_precedence:
                self.log(f"❌ [Arbiter] Type '{mem_type}' not in precedence table for batch candidate.")
                continue

            # Generate embedding and identity once
            embedding = self.embed_fn(text) if self.embed_fn else None
            identity = self.memory_store.compute_identity(text, mem_type)

            processed_candidates.append({
                "mem_type": mem_type, "text": text, "subject": subject,
                "confidence": confidence, "source": source, "affect": affect,
                "embedding": embedding, "identity": identity
            })

        # Now iterate and call single consider for each (can be optimized further with batch DB ops)
        for pc in processed_candidates:
            mid = self.consider(
                text=pc["text"], mem_type=pc["mem_type"], confidence=pc["confidence"],
                subject=pc["subject"], source=pc["source"], affect=pc["affect"],
                embedding=pc["embedding"], identity=pc["identity"]
            )
            if mid is not None:
                promoted_ids.append(mid)

        return promoted_ids

    @staticmethod
    def _extract_value(text: str) -> str:
        """Extract the value from memory text."""
        text = text.strip()
        patterns = [" is ", " lives in ", " works at ", " wants to ", " prefers ", " loves ", " allowed ", " granted ", " believes "]
        for pattern in patterns:
            if pattern in text.lower():
                parts = text.split(pattern, 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return text
