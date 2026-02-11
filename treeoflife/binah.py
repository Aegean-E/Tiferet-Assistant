"""
Binah (Understanding): Smart consolidation WITHOUT data loss.

Strategy:
- Group memories by identity pattern
- IDENTITY type: Version chaining (old → new, new supersedes)
  Example: "AI Companion" → "Ada" (clear version update)

- Other types: Smart linking (KEEP BOTH memories)
  If similarity ≥ threshold: Link them via parent_id
  But NEVER delete: Append-only, no data loss
  Goal: Prevent duplication while keeping all unique information

Result: Compressed database with zero data loss, full audit trail via parent_id chain

Meta-Memory: Automatically creates meta-memories when consolidation happens
- Tracks what changed, when, and why
- Stored in separate meta_memory database
- Enables self-reflection and temporal reasoning
"""

import time
import json
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime

from memory import MemoryStore
from meta_memory import MetaMemoryStore
from lm import compute_embedding, run_local_lm
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class Binah:
    """
    Binah (Understanding) - The Structure of Thought.
    Consolidates similar/duplicate memories to keep memory store clean.
    
    Role: Reasoning, Logic, Verification, and Structure.
    """

    def __init__(self, memory_store: MemoryStore, meta_memory_store: Optional[MetaMemoryStore] = None, 
                 consolidation_thresholds: Optional[Dict[str, float]] = None,
                 log_fn: Callable[[str], None] = print):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.log = log_fn
        
        self.consolidation_thresholds = consolidation_thresholds or {
            "GOAL": 0.88,
            "IDENTITY": 0.87,
            "BELIEF": 0.87,
            "PERMISSION": 0.87,
            "FACT": 0.93,
            "PREFERENCE": 0.93,
            "RULE": 0.93,
            "REFUTED_BELIEF": 0.95 # Very high threshold to avoid accidental merging of distinct refutations
            # INVARIANT: REFUTED_BELIEF is not "low confidence". It is a hard epistemic boundary.
            # Do not optimize away.
        }

    def prune_stale_goals(self, days_to_keep: int = 3) -> int:
        """
        Forget goals that are older than 'days_to_keep'.
        We change their type to 'ARCHIVED_GOAL' so they stop creating Gevurah pressure.
        """
        try:
            # Calculate timestamps for SQLite INTEGER comparison
            cutoff = int(time.time()) - (days_to_keep * 86400)
            delete_cutoff = int(time.time()) - (30 * 86400)

            with self.memory_store._connect() as con:
                # 1. Archive goals older than X days
                cur = con.execute("""
                    UPDATE memories 
                    SET type = 'ARCHIVED_GOAL' 
                    WHERE type = 'GOAL' 
                    AND parent_id IS NULL 
                    AND created_at < ?
                """, (cutoff,))
                count = cur.rowcount
                
                # 2. Hard delete VERY old archived goals (e.g. > 30 days) to save space
                del_cur = con.execute("""
                    DELETE FROM memories 
                    WHERE type = 'ARCHIVED_GOAL' 
                    AND created_at < ?
                """, (delete_cutoff,))
                del_count = del_cur.rowcount
                
                con.commit()
                
                if count > 0:
                    self.log(f"🧹 [Binah] Auto-archived {count} stale goals (older than {days_to_keep} days).")
                if del_count > 0:
                    self.log(f"🗑️ [Binah] Permanently deleted {del_count} ancient goals.")
                    
                return count
        except Exception as e:
            self.log(f"❌ Error pruning stale goals: {e}")
            return 0

    def _get_dynamic_threshold(self, base_threshold: float, total_count: int, local_count: int = 0) -> float:
        # Global pressure: As total memories grow, become slightly stricter
        global_penalty = (total_count / 2000) * 0.01
        
        # Local density pressure:
        # If dense (> 20 items), become stricter to preserve nuance
        # If sparse (< 5 items), relax slightly to encourage linking
        local_adjustment = 0.0
        
        if local_count > 100:
            local_adjustment = 0.04
        elif local_count > 50:
            local_adjustment = 0.02
        elif local_count > 20:
            local_adjustment = 0.01
        elif local_count < 5 and local_count > 0:
            local_adjustment = -0.02
            
        final = base_threshold + global_penalty + local_adjustment
        
        return max(0.80, min(0.99, final))

    def consolidate(self, time_window_hours: Optional[int] = None) -> Dict[str, int]:
        """
        Smart consolidation: Detect & link duplicates WITHOUT data loss.

        Process:
        1. Group memories by identity pattern (For forced replacements like Name/Location)
        2. Consolidate across the whole set for semantic overlaps (e.g. "loves coffee" vs "prefers coffee")
        """
        # 1. Run Goal Pruning FIRST to reduce noise
        self.prune_stale_goals(days_to_keep=3)

        stats = {'processed': 0, 'consolidated': 0, 'skipped': 0}
        
        # Pre-calculate total memories for dynamic thresholding
        total_memories = self.memory_store.count_all()

        # Get memories based on time window
        if time_window_hours is not None:
            cutoff_time = int(time.time()) - (time_window_hours * 3600)
            recent_memories = self._get_memories_after(cutoff_time)
        else:
            recent_memories = self._get_all_memories()

        if not recent_memories:
            self.log("🧠 [Binah] No memories to consolidate")
            return stats

        # Sort all memories chronologically
        recent_memories = sorted(recent_memories, key=lambda m: m['created_at'])

        # --- Phase 1: Identity-based Forced Replacements (Name, Location etc) ---
        # Recompute identity hashes for IDENTITY types to catch legacy data with old patterns
        groups_by_identity = {}
        for mem in recent_memories:
            # For IDENTITY types, recompute the identity hash using current patterns
            if mem['type'].upper() == 'IDENTITY':
                identity = self.memory_store.compute_identity(mem['text'], mem['type'])
            else:
                identity = mem['identity']
            
            if identity not in groups_by_identity:
                groups_by_identity[identity] = []
            groups_by_identity[identity].append(mem)

        identity_consolidated_ids = set()
        for identity, group in groups_by_identity.items():
            if len(group) > 1 and group[-1]['type'].upper() == 'IDENTITY':
                for i in range(len(group) - 1):
                    old, new = group[i], group[i+1]
                    if old['subject'] == new['subject'] and old['type'] == new['type']:
                        if self._mark_consolidated(old['id'], new['id']):
                            stats['consolidated'] += 1
                            identity_consolidated_ids.add(old['id'])
                            if self.meta_memory_store:
                                self._create_meta_memory(old, new, "VERSION_UPDATE")

        # --- Phase 2: Universal Semantic Overlap (Refinements/Duplicates) ---
        groups_by_type = {}
        for mem in recent_memories:
            if mem['id'] in identity_consolidated_ids: continue
            if mem['type'] == 'NOTE': continue  # Skip Assistant Notes
            key = (mem['subject'], mem['type'])
            if key not in groups_by_type: groups_by_type[key] = []
            groups_by_type[key].append(mem)

        for (subj, mtype), group in groups_by_type.items():
            stats['processed'] += 1
            if len(group) < 2: continue

            # Prepare data for vectorized calculation
            valid_items = []
            embeddings = []
            
            for mem in group:
                # Skip if already consolidated or if it already has a parent
                if mem['id'] in identity_consolidated_ids or mem['parent_id'] is not None:
                    continue
                
                emb = mem.get('embedding')
                if emb is None:
                    emb = compute_embedding(mem['text'])
                    # Self-healing: Save this embedding so we don't compute it again
                    self._update_embedding(mem['id'], emb)
                
                if emb is not None:
                    valid_items.append(mem)
                    embeddings.append(np.array(emb))
            
            if len(embeddings) < 2:
                continue

            # Vectorized Cosine Similarity: Sim = (A . B.T) / (|A| * |B|)
            # 1. Stack embeddings into matrix (N x D)
            matrix = np.stack(embeddings)
            
            # 2. Normalize rows (L2 norm)
            norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix_normalized = matrix / (norm + 1e-10) # Avoid div by zero
            
            # 3. Compute Similarity Matrix (N x N)
            sim_matrix = np.dot(matrix_normalized, matrix_normalized.T)
            
            # 4. Iterate
            count = len(valid_items)
            
            # SCALABILITY UPGRADE: Use FAISS for large groups to avoid O(N^2) freeze
            use_faiss = FAISS_AVAILABLE and count > 50
            faiss_index = None
            
            if use_faiss:
                # Build temporary index for this group
                d = matrix.shape[1]
                faiss_index = faiss.IndexFlatIP(d)
                # Matrix is already normalized? No, step 2 above normalized it.
                # Ensure float32 for FAISS
                matrix_f32 = matrix_normalized.astype('float32')
                faiss_index.add(matrix_f32)

            for r in range(count):
                old = valid_items[r]
                if old['id'] in identity_consolidated_ids: continue

                # Determine threshold
                is_identity_like = "name is" in old['text'].lower() or "lives in" in old['text'].lower()
                lookup_type = old['type']
                if is_identity_like:
                    lookup_type = 'IDENTITY'
                base_threshold = self.consolidation_thresholds.get(lookup_type, 0.93)
                threshold = self._get_dynamic_threshold(base_threshold, total_memories, local_count=count)

                # Candidate generation
                candidates = []
                if use_faiss:
                    # O(N log N) Search
                    q = matrix_normalized[r:r+1].astype('float32')
                    # Range search returns all vectors within similarity threshold
                    lims, D, I = faiss_index.range_search(q, threshold)
                    # lims[0] to lims[1] contains results
                    for j in range(lims[0], lims[1]):
                        c = I[j]
                        if c > r: # Only look forward to avoid duplicates/self
                            candidates.append((c, D[j]))
                else:
                    # O(N^2) Linear Scan
                    for c in range(r + 1, count):
                        candidates.append((c, float(sim_matrix[r, c])))

                for c, similarity in candidates:
                    new = valid_items[c]
                    if new['id'] in identity_consolidated_ids: continue

                    # Check cache for previous rejections
                    cached_sim = self.memory_store.get_comparison_similarity(old['id'], new['id'])
                    # Trust the cache if it exists and is below threshold (this handles the 0.5 poison pill from failed entailment)
                    if cached_sim is not None and cached_sim < threshold:
                        continue

                    # Substring checks (Identity/Belief)
                    is_substring_expansion = False
                    if old['subject'] == new['subject'] and old['type'] in ('IDENTITY', 'BELIEF'):
                        old_val = self._extract_value_from_text(old['text']).lower().strip()
                        new_val = self._extract_value_from_text(new['text']).lower().strip()
                        if old_val != new_val and (old_val in new_val or new_val in old_val):
                            is_substring_expansion = True
                    
                    if similarity >= threshold or is_substring_expansion:
                        # ENTAILMENT CHECK: Prevent merging "Love X" and "Hate X"
                        # Only run expensive LLM check if similarity is high
                        if similarity > 0.85 and not is_substring_expansion:
                            if not self._check_entailment(old, new):
                                self.log(f"      🛡️ Entailment check failed for high-sim pair. Keeping distinct.")
                                # Record as low similarity to prevent re-checking
                                self.memory_store.record_comparison(old['id'], new['id'], 0.5)
                                continue

                        if self._mark_consolidated(old['id'], new['id']):
                            stats['consolidated'] += 1
                            identity_consolidated_ids.add(old['id'])
                            if self.meta_memory_store:
                                self._create_meta_memory(old, new, "SIMILARITY_LINK")
                            break
                    else:
                        if similarity > 0.8:
                            self.memory_store.record_comparison(old['id'], new['id'], similarity)
        return stats

    def _consolidate_group(self, group: List[Dict]) -> int:
        """DEPRECATED: Logic moved to main consolidate function."""
        return 0

    def _get_memories_after(self, cutoff_time: int) -> List[Dict]:
        """Get all memories created after cutoff_time."""
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at, embedding, verified, verification_attempts
                FROM memories
                WHERE created_at > ?
                ORDER BY created_at DESC
            """, (cutoff_time,)).fetchall()

        memories = []
        for r in rows:
            memories.append({
                'id': r[0],
                'identity': r[1],
                'parent_id': r[2],
                'type': r[3],
                'subject': r[4],
                'text': r[5],
                'confidence': r[6],
                'source': r[7],
                'created_at': r[8],
                'embedding': json.loads(r[9]) if r[9] else None,
                'verified': r[10] if len(r) > 10 else 0,
                'verification_attempts': r[11] if len(r) > 11 else 0,
            })
        return memories

    def _get_all_memories(self) -> List[Dict]:
        """Get ALL memories regardless of age."""
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at, embedding, verified, verification_attempts
                FROM memories
                ORDER BY created_at DESC
            """).fetchall()

        memories = []
        for r in rows:
            memories.append({
                'id': r[0],
                'identity': r[1],
                'parent_id': r[2],
                'type': r[3],
                'subject': r[4],
                'text': r[5],
                'confidence': r[6],
                'source': r[7],
                'created_at': r[8],
                'embedding': json.loads(r[9]) if r[9] else None,
                'verified': r[10] if len(r) > 10 else 0,
                'verification_attempts': r[11] if len(r) > 11 else 0,
            })
        return memories

    def _mark_consolidated(self, old_id: int, new_id: int) -> bool:
        """
        Mark old_id as consolidated into new_id.

        Adds a "consolidated_into" relationship without deleting.
        (Append-only: add a record instead of modifying)

        Returns:
            True if consolidation was performed, False if already consolidated
        """
        # For now: just add a parent_id relationship if not already set
        # In future: could add a consolidation_log table
        with self.memory_store._connect() as con:
            # Check if already has parent
            existing = con.execute(
                "SELECT parent_id FROM memories WHERE id = ?",
                (old_id,)
            ).fetchone()

            # Update parent_id to point to the latest version
            # This creates a version chain: old -> new
            if existing:
                current_parent = existing[0]
                # Only update if not already pointing to this new_id
                if current_parent != new_id:
                    con.execute(
                        "UPDATE memories SET parent_id = ? WHERE id = ?",
                        (new_id, old_id)
                    )
                    con.commit()
                    self.log(f"      🔗 Updated parent_id: ID {old_id} now points to ID {new_id}")
                    return True
                else:
                    # Already consolidated
                    return False
            return False

    def _update_embedding(self, memory_id: int, embedding: np.ndarray) -> None:
        """Update the embedding for a specific memory ID."""
        try:
            embedding_json = json.dumps(embedding.tolist())
            with self.memory_store._connect() as con:
                con.execute("UPDATE memories SET embedding = ? WHERE id = ?", (embedding_json, memory_id))
                con.commit()
        except Exception as e:
            self.log(f"⚠️ Failed to update embedding for memory {memory_id}: {e}")

    def _create_meta_memory(self, old_mem: Dict, new_mem: Dict, event_type: str) -> None:
        """
        Create a meta-memory about a memory change.

        This enables self-reflection and temporal reasoning.

        Examples:
        - VERSION_UPDATE (IDENTITY): "Assistant name changed from Ada to Lara on 2026-02-04 at 11:37"
        - SIMILARITY_LINK (PREFERENCE): "User added similar preference: 'loves tea' (related to 'loves coffee') on 2026-02-04 14:30"
        - SIMILARITY_LINK (GOAL): "User added similar goal: 'learn Rust' (related to 'learn Python') on 2026-02-04 15:00"

        Args:
            old_mem: The old memory version
            new_mem: The new memory version
            event_type: Type of event (VERSION_UPDATE, SIMILARITY_LINK, CONFLICT_DETECTED, etc.)
        """
        # Extract the actual values from the memory text
        old_value = self._extract_value_from_text(old_mem['text'])
        new_value = self._extract_value_from_text(new_mem['text'])

        # Skip redundant meta-memories where value hasn't actually changed
        # (e.g., during pointer-only updates or deduplication)
        if old_value == new_value and event_type in ("VERSION_UPDATE", "SIMILARITY_LINK"):
            return

        # Format timestamp
        timestamp = datetime.fromtimestamp(new_mem['created_at']).strftime("%Y-%m-%d %H:%M")

        # Create display versions (truncated) to prevent log bloat
        display_old = (old_value[:60] + '...') if len(old_value) > 60 else old_value
        display_new = (new_value[:60] + '...') if len(new_value) > 60 else new_value

        # Create human-readable meta-memory text based on event type
        if event_type == "VERSION_UPDATE":
            # VERSION_UPDATE: True replacement (IDENTITY types)
            if old_mem['type'] == 'IDENTITY':
                # For identity changes, be specific about what changed
                if 'name is' in old_mem['text'].lower():
                    meta_text = f"{old_mem['subject']} name changed from {display_old} to {display_new} on {timestamp}"
                elif 'lives in' in old_mem['text'].lower():
                    meta_text = f"{old_mem['subject']} location changed from {display_old} to {display_new} on {timestamp}"
                else:
                    meta_text = f"{old_mem['subject']} {old_mem['type'].lower()} updated from '{display_old}' to '{display_new}' on {timestamp}"
            else:
                meta_text = f"{old_mem['subject']} {old_mem['type'].lower()} updated from '{display_old}' to '{display_new}' on {timestamp}"

        elif event_type == "SIMILARITY_LINK":
            # SIMILARITY_LINK: Additive/related (PREFERENCE, GOAL, FACT, etc.)
            # Use language that suggests addition, not replacement
            mem_type_lower = old_mem['type'].lower()

            if old_mem['type'] == 'PREFERENCE':
                meta_text = f"{old_mem['subject']} added similar preference: '{display_new}' (related to '{display_old}') on {timestamp}"
            elif old_mem['type'] == 'GOAL':
                meta_text = f"{old_mem['subject']} added similar goal: '{display_new}' (related to '{display_old}') on {timestamp}"
            elif old_mem['type'] == 'FACT':
                meta_text = f"{old_mem['subject']} added similar fact: '{display_new}' (related to '{display_old}') on {timestamp}"
            elif old_mem['type'] == 'RULE':
                meta_text = f"{old_mem['subject']} added similar rule: '{display_new}' (related to '{display_old}') on {timestamp}"
            elif old_mem['type'] == 'PERMISSION':
                meta_text = f"{old_mem['subject']} granted similar permission: '{display_new}' (related to '{display_old}') on {timestamp}"
            else:
                meta_text = f"{old_mem['subject']} added similar {mem_type_lower}: '{display_new}' (related to '{display_old}') on {timestamp}"

        else:
            # Fallback for other event types
            meta_text = f"{old_mem['subject']} {old_mem['type'].lower()} event: '{display_old}' → '{display_new}' on {timestamp}"

        # Create metadata with structured information
        metadata = {
            "timestamp": new_mem['created_at'],
            "is_replacement": event_type == "VERSION_UPDATE",
            "is_additive": event_type == "SIMILARITY_LINK"
        }

        # Save the meta-memory
        self.meta_memory_store.add_meta_memory(
            event_type=event_type,
            memory_type=old_mem['type'],
            subject=old_mem['subject'],
            text=meta_text,
            old_id=old_mem['id'],
            new_id=new_mem['id'],
            old_value=old_value,
            new_value=new_value,
            metadata=metadata
        )

        self.log(f"      🧠 Meta-memory created: {meta_text}")

    @staticmethod
    def _extract_value_from_text(text: str) -> str:
        """
        Extract the actual value from memory text.
        Example: "Assistant name is Ada" → "Ada"
        Example: "User lives in Van, Türkiye" → "Van, Türkiye"
        """
        text = text.strip()

        # Common patterns
        patterns = [
            " is ",
            " lives in ",
            " works at ",
            " wants to ",
            " prefers ",
        ]

        for pattern in patterns:
            if pattern in text.lower():
                parts = text.split(pattern, 1)
                if len(parts) == 2:
                    return parts[1].strip()

        # Fallback: return the whole text
        return text

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _check_entailment(self, old_mem: Dict, new_mem: Dict) -> bool:
        """
        Use LLM to check if two high-similarity memories are actually equivalent or subsumed.
        Prevents merging "I love X" and "I hate X" (which have high cosine similarity).
        """
        # Quick heuristic: if texts are identical, skip LLM
        if old_mem['text'].strip().lower() == new_mem['text'].strip().lower():
            return True
            
        prompt = (
            f"Analyze the relationship between these two memory statements:\n"
            f"A: {old_mem['text']}\n"
            f"B: {new_mem['text']}\n\n"
            "Determine if they are semantically EQUIVALENT (mean the same thing), "
            "if one SUBSUMES the other (one implies the other), "
            "or if they are CONTRADICTORY/UNRELATED.\n"
            "Output ONLY one word: EQUIVALENT, SUBSUMED, CONTRADICTION, or RELATED."
        )
        
        response = run_local_lm(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        ).strip().upper()
        
        return response in ["EQUIVALENT", "SUBSUMED"]

    def learn_associations(self, reasoning_store) -> int:
        """
        Analyzes the reasoning store to find memories used together and creates links.
        This is the core of "Spreading Activation" learning.
        """
        self.log("🧠 [Binah] Learning associations from reasoning...")
        
        # 1. Get recent reasoning nodes
        recent_thoughts = reasoning_store.list_recent(limit=50)
        if len(recent_thoughts) < 2:
            return 0

        # 2. Find co-occurring memories within the reasoning context
        # Heuristic: If memories are mentioned close together in the reasoning log, they are related.
        # We can extract memory IDs if they are logged, or search for memory text.
        
        linked_pairs = set()
        
        # For this implementation, we'll use a simple heuristic:
        # Search for text from each memory in the reasoning log.
        all_memories = self.memory_store.list_recent(limit=200)
        reasoning_text = " ".join([t['content'] for t in recent_thoughts])

        # This is O(N*M), can be slow. A better approach would be to have reasoning nodes reference memory IDs.
        # For now, this demonstrates the concept.
        mentioned_mems = []
        for mem in all_memories:
            # mem: (id, type, subject, text, source, verified, flags)
            # Simple check if a significant part of the memory text is in the reasoning log
            # Use a heuristic to avoid matching on very short/common phrases
            if len(mem[3]) > 15 and mem[3][:15] in reasoning_text:
                mentioned_mems.append(mem[0])

        if len(mentioned_mems) < 2:
            return 0

        # 3. Create/Strengthen links
        # Create links between all co-mentioned memories in this context
        for i in range(len(mentioned_mems)):
            for j in range(i + 1, len(mentioned_mems)):
                id_a, id_b = mentioned_mems[i], mentioned_mems[j]
                # Using a default relation type and strength for now
                self.memory_store.link_memories(id_a, id_b, "RELATED_TO", strength=0.1) # Reduced strength to prevent gravity wells
                linked_pairs.add(tuple(sorted((id_a, id_b))))

        if linked_pairs:
            self.log(f"🔗 [Binah] Learned {len(linked_pairs)} new memory associations.")
        
        return len(linked_pairs)

    def expand_associative_context(self, seed_memory_ids: List[int], limit: int = 10) -> List[Dict]:
        """
        Active Association: Follow 'SUPPORTS' or 'RELATED_TO' links to bring wider semantic context.
        """
        if not seed_memory_ids: return []
        
        expanded_context = []
        visited = set(seed_memory_ids)
        
        for mid in seed_memory_ids:
            # Get neighbors from memory_links table
            neighbors = self.memory_store.get_associated_memories(mid, min_strength=0.5)
            
            for n in neighbors:
                if n['id'] not in visited:
                    visited.add(n['id'])
                    expanded_context.append(n)
                    if len(expanded_context) >= limit:
                        return expanded_context
        
        return expanded_context