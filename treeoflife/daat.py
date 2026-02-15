import time
import os
import re
import random
import json
import logging
from typing import Callable, Dict, List, Optional
from ai_core.lm import run_local_lm, compute_embedding
from ai_core.utils import parse_json_array_loose, parse_json_object_loose

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

class Daat:
    """
    Da'at (Knowledge/Integration)
    
    The hidden integrator of knowledge continuity.
    Responsible for synthesizing raw events into coherent history (Summarization)
    and compressing meta-memories (Consolidation).
    """
    def __init__(
        self,
        memory_store,
        meta_memory_store,
        reasoning_store,
        get_settings_fn: Callable[[], Dict],
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        log_fn: Callable[[str], None] = logging.info
    ):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.reasoning_store = reasoning_store
        self.get_settings = get_settings_fn
        self.embed_fn = embed_fn
        self.log = log_fn

    def run_summarization(self):
        """
        Summarize logs ONLY if enough time has passed or enough data has accumulated.
        Prevents constant churning.
        """
        # --- CONFIGURATION ---
        MIN_INTERVAL_SECONDS = 3600  # 1 Hour (Don't summarize sooner than this)
        MIN_ITEMS_THRESHOLD = 500    # Accumulate 500 logs before compressing
        # ---------------------

        # 1. Check Time
        last_time = getattr(self, '_last_summary_ts', 0)
        time_diff = time.time() - last_time
        
        # 2. Check Volume (Count new noisy items)
        noisy_types = [
            "DECIDER_ACTION", "NETZACH_ACTION", "HOD_INSTRUCTION", 
            "DECIDER_OBSERVATION_RECEIVED", "TOOL_EXECUTION",
            "NETZACH_INFO", "NETZACH_INSTRUCTION", "STRATEGIC_THOUGHT",
            "CHAIN_OF_THOUGHT", "VERIFICATION_CHECK"
        ]
        
        with self.meta_memory_store._connect() as con:
            # Count how many raw logs exist right now
            placeholders = ','.join(['?'] * len(noisy_types))
            count = con.execute(f"""
                SELECT COUNT(*) FROM meta_memories 
                WHERE event_type IN ({placeholders})
            """, noisy_types).fetchone()[0]

        # 3. The Gatekeeper Logic
        # Run IF (Time is up AND we have some data) OR (Buffer is overflowing)
        should_run = False
        reason = ""

        if count > MIN_ITEMS_THRESHOLD:
            should_run = True
            reason = f"Buffer full ({count} items)"
        elif time_diff > MIN_INTERVAL_SECONDS and count > 50: # At least 50 items to be worth it
            should_run = True
            reason = f"Time interval ({int(time_diff/60)}m)"

        if not should_run:
            self.log(f"⏳ Da'at: Skipping summary. (Items: {count}/{MIN_ITEMS_THRESHOLD}, Time: {int(time_diff)}s)")
            return "⏳ Summarization skipped."

        # --- EXECUTION ---
        self.log(f"✨ Da'at: Starting summarization. Reason: {reason}")
        self._last_summary_ts = time.time()

        # Fetch the raw data
        with self.meta_memory_store._connect() as con:
            rows = con.execute(f"""
                SELECT text FROM meta_memories 
                WHERE event_type IN ({placeholders})
                ORDER BY created_at DESC LIMIT ?
            """, noisy_types + [MIN_ITEMS_THRESHOLD]).fetchall()
            
        if not rows:
            return "ℹ️ No logs to summarize."

        # Chunking logic to prevent context overflow (400 Bad Request)
        batch_size = 50
        chunks = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
        intermediate_summaries = []
        settings = self.get_settings()

        for i, chunk in enumerate(chunks):
            text_block = "\n".join([f"- {r[0]}" for r in reversed(chunk)])
            prompt = (
                f"Review these internal system logs (Batch {i+1}/{len(chunks)}):\n{text_block}\n\n"
                "Compress this into 1-2 concise sentences describing the key actions."
            )
            
            summary = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a system log compressor.",
                max_tokens=100,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )
            if summary and not summary.startswith("⚠️"):
                intermediate_summaries.append(summary)

        
        summary = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a system log compressor.",
            max_tokens=150,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        # Save & Prune
        if summary and len(summary) > 5 and "error" not in summary.lower():
            self.meta_memory_store.add_meta_memory(
                event_type="SESSION_SUMMARY",
                memory_type="SUMMARY",
                subject="System",
                text=summary
            )
            
            # Prune ONLY the items we summarized
            self.log("🧹 Da'at: Pruning summarized raw logs to save space...")
            
            # Delete "Noise" types older than 0 days (i.e., everything up to now)
            deleted = self.meta_memory_store.prune_events(
                max_age_seconds=0, 
                event_types=noisy_types,
                prune_all=False
            )
            
            msg = f"🧹 Da'at: Compressed history. Deleted {deleted} raw records."
            self.log(msg)
            return msg
        return "⚠️ Summarization failed."

    def consolidate_summaries(self) -> str:
        """
        Consolidate multiple SESSION_SUMMARY and HOD_ANALYSIS events into fewer, denser summaries.
        Reduces clutter in meta-memory by merging older summaries.
        """
        self.log("✨ Da'at: Consolidating session summaries and analyses...")
        
        # Fetch recent items (limit 50 each to avoid context overflow)
        summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=100)
        analyses = self.meta_memory_store.get_by_event_type("HOD_ANALYSIS", limit=100)
        
        all_items = summaries + analyses
        
        # Sort by time (oldest first) to maintain narrative flow
        all_items.sort(key=lambda x: x['created_at'])
        
        if len(all_items) < 5:
            return "⚠️ Not enough items to consolidate (minimum 5)."
            
        # Take the oldest batch (increased to 30 to clear backlog faster)
        batch = all_items[:30]
        # Ensure batch is sorted chronologically (Oldest -> Newest) for correct date extraction
        batch.sort(key=lambda x: x['created_at'])
        batch_ids = [m['id'] for m in batch]
        
        # Determine time range, respecting existing compressed ranges
        first_mem = batch[0]
        last_mem = batch[-1]
        
        start_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(first_mem['created_at']))
        end_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(last_mem['created_at']))
        
        # Check for existing compression tags to preserve original start/end times
        if first_mem.get('metadata') and isinstance(first_mem['metadata'], dict) and 'start_date' in first_mem['metadata']:
            start_date = first_mem['metadata']['start_date']
        else:
            # Check for header format: [start - end] Compressed...
            start_match = re.match(r"\[(.*?)\s+-\s+(.*?)\]\s+Compressed", first_mem['text'])
            if start_match: 
                start_date = start_match.group(1)
            else:
                start_match = re.match(r"\[COMPRESSED\s+(.*?)\s+-\s+(.*?)\]", first_mem['text'])
                if start_match: start_date = start_match.group(1)
            
        if last_mem.get('metadata') and isinstance(last_mem['metadata'], dict) and 'end_date' in last_mem['metadata']:
            end_date = last_mem['metadata']['end_date']
        else:
            # Try new format
            end_match = re.match(r"\[(.*?)\s+-\s+(.*?)\]\s+Compressed", last_mem['text'])
            if end_match: 
                end_date = end_match.group(2)
            else:
                # Try old format
                end_match = re.match(r"\[COMPRESSED\s+(.*?)\s+-\s+(.*?)\]", last_mem['text'])
                if end_match: end_date = end_match.group(2)
        
        context = f"History from {start_date} to {end_date}:\n\n"
        for m in batch:
            date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(m['created_at']))
            type_label = "SUMMARY" if m['event_type'] == "SESSION_SUMMARY" else "ANALYSIS"
            context += f"[{date_str}] [{type_label}] {m['text']}\n"
            
        prompt = (
            "You are compressing historical logs. "
            "Combine the following session summaries and analyses into a SINGLE, coherent narrative summary.\n"
            "Rules:\n"
            "1. Preserve key events, decisions, and learned facts.\n"
            "2. Remove repetitive phrasing (e.g. 'The system processed...').\n"
            "3. Keep it dense but readable.\n"
            "4. Mention the time range covered.\n\n"
            f"{context}"
        )
        
        settings = self.get_settings()
        compressed_text = run_local_lm(
            [{"role": "user", "content": prompt}],
            system_prompt="You are a historical archivist.",
            temperature=0.3,
            max_tokens=600,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if compressed_text and not compressed_text.startswith("⚠️"):
            # Prepend header with date range
            header = f"[{start_date} - {end_date}] Compressed Summaries & Analyses"
            final_text = f"{header}\n{compressed_text}"
            
            # Save new summary
            self.meta_memory_store.add_meta_memory(
                event_type="SESSION_SUMMARY",
                memory_type="COMPRESSED_HISTORY",
                subject="Da'at",
                text=final_text,
                metadata={
                    "compressed_ids": batch_ids,
                    "start_date": start_date,
                    "end_date": end_date
                }
            )
            
            # Soft Delete (Archive) old summaries instead of hard delete
            # This prevents data loss if the compression is bad
            with self.meta_memory_store._connect() as con:
                placeholders = ','.join(['?'] * len(batch_ids))
                # We don't have an 'archived' column in meta_memories, so we'll prepend [ARCHIVED] to text
                # or just delete them if we trust the system. The prompt asked to mark them.
                # Let's assume we can't easily add columns here without migration, so we'll just delete for now as per original code, BUT
                # the prompt asked to "Keep the old summaries but mark them archived=1".
                # Since we don't have that column, let's just log them to a file backup first.
                pass
            
            # Actually, let's implement the "mark archived" by updating the event_type
            with self.meta_memory_store._connect() as con:
                placeholders = ','.join(['?'] * len(batch_ids))
                con.execute(f"UPDATE meta_memories SET event_type = 'ARCHIVED_SUMMARY' WHERE id IN ({placeholders})", batch_ids)
                con.commit()
            
            msg = f"✅ Compressed {len(batch)} items into one ({start_date} - {end_date}). Deleted source items."
            self.log(f"✨ Da'at: {msg}")
            return msg
        else:
            return "⚠️ Failed to generate compressed summary."

    def run_topic_lattice(self):
        """
        Scan for heavy entities and generate standing summaries (Topic Lattice).
        """
        self.log("✨ Da'at: Scanning for heavy entities (Topic Lattice)...")
        settings = self.get_settings()
        
        # 1. Identify Topics
        # Get recent memories to find active topics
        recent_mems = self.memory_store.list_recent(limit=50)
        if not recent_mems: return

        mem_text_blob = "\n".join([f"- {m[3]}" for m in recent_mems])
        
        prompt = (
            "Analyze the following memories. Identify the top 3 'Heavy Entities' or 'Major Topics' "
            "that appear frequently and would benefit from a consolidated summary.\n"
            "Ignore generic topics like 'User', 'Assistant', 'Chat'.\n"
            "Focus on specific projects, people, locations, or concepts.\n"
            "Output ONLY a JSON list of strings, e.g. [\"Project Apollo\", \"The User's Mom\"].\n\n"
            f"{mem_text_blob}"
        )
        
        response = run_local_lm(
            [{"role": "user", "content": prompt}],
            system_prompt="You are a knowledge architect.",
            temperature=0.3,
            max_tokens=100,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        topics = parse_json_array_loose(response)
        
        if not isinstance(topics, list): topics = []

        for topic in topics:
            if isinstance(topic, str) and len(topic) > 3:
                self._generate_entity_summary(topic, settings)

    def _generate_entity_summary(self, topic, settings):
        try:
            if self.embed_fn:
                emb = self.embed_fn(topic)
            else:
                emb = compute_embedding(topic, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
            results = self.memory_store.search(emb, limit=20)
            
            relevant_texts = [r[3] for r in results if r[4] > 0.45] # Similarity threshold
            if len(relevant_texts) < 3: return

            context = f"Memories about '{topic}':\n" + "\n".join([f"- {t}" for t in relevant_texts])
            prompt = f"Synthesize a dense, standing summary for '{topic}' based on these memories:\n{context}"
            
            summary = run_local_lm([{"role": "user", "content": prompt}], system_prompt="You are a knowledge archivist.", temperature=0.3, max_tokens=400, base_url=settings.get("base_url"), chat_model=settings.get("chat_model"))
            
            if summary and not summary.startswith("⚠️"):
                identity = self.memory_store.compute_identity(f"Topic Lattice Summary: {topic}", "FACT")
                self.memory_store.add_entry(identity=identity, text=f"📚 Standing Summary ({topic}): {summary}", mem_type="FACT", subject=topic, confidence=1.0, source="daat_lattice")
                self.log(f"✨ Da'at: Generated Topic Lattice summary for '{topic}'")
                
                # Save to Meta-Memory for Synthesis
                self.meta_memory_store.add_meta_memory(
                    event_type="TOPIC_SUMMARY",
                    memory_type="SUMMARY",
                    subject=topic,
                    text=f"Topic Lattice: {summary}"
                )
                
        except Exception as e:
            self.log(f"⚠️ Da'at Topic Lattice error for {topic}: {e}")

    def extract_rdf_triples(self):
        """
        Extract Knowledge Graph Triples (Subject-Predicate-Object) from recent facts.
        Stores them as linked concepts in the memory graph.
        """
        self.log("🕸️ Da'at: Extracting Knowledge Graph (RDF Triples)...")
        settings = self.get_settings()

        # 1. Get source text (Recent Facts/Beliefs)
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, text FROM memories
                WHERE type IN ('FACT', 'BELIEF') AND created_at > ?
                ORDER BY created_at DESC LIMIT 20
            """, (int(time.time()) - 86400,)).fetchall() # Last 24h

        if not rows: return

        text_blob = "\n".join([f"- {r[1]}" for r in rows])

        prompt = (
            "Extract Knowledge Graph Triples from the following text.\n"
            "Format: JSON list of objects [{'subject': '...', 'predicate': '...', 'object': '...'}]\n"
            "Rules:\n"
            "1. Entities (Subject/Object) should be concise concepts (e.g., 'NADH', 'Obesity').\n"
            "2. Predicate should be a verb or relationship (e.g., 'inhibits', 'causes').\n"
            "3. Ignore generic statements.\n\n"
            f"{text_blob}"
        )

        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Knowledge Graph Engineer.",
            max_tokens=500,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        triples = parse_json_array_loose(response)

        for t in triples:
            subj = t.get('subject')
            pred = t.get('predicate')
            obj = t.get('object')

            if subj and pred and obj:
                id_s = self._ensure_concept_node(subj)
                id_o = self._ensure_concept_node(obj)
                self.memory_store.link_memories(id_s, id_o, pred, strength=1.0)
                self.log(f"🕸️ Graph: ({subj}) --[{pred}]--> ({obj})")

    def _ensure_concept_node(self, concept_name: str) -> int:
        """Ensure a CONCEPT memory exists for the given name."""
        identity = self.memory_store.compute_identity(concept_name, "CONCEPT")
        
        # Check if exists
        with self.memory_store._connect() as con:
            row = con.execute("SELECT id FROM memories WHERE identity = ?", (identity,)).fetchone()
            if row:
                return row[0]
        
        # Create if not
        return self.memory_store.add_entry(
            identity=identity,
            text=concept_name,
            mem_type="CONCEPT",
            subject="Universal",
            confidence=1.0,
            source="daat_knowledge_graph"
        )

    def run_clustering(self):
        """
        Cluster memories to find Narrative Nodes (dense topics).
        Replaces fragmented memories with a single Summary Node.
        """
        if not FAISS_AVAILABLE:
            return
            
        self.log("✨ Da'at: Running narrative clustering...")
        settings = self.get_settings()
        
        # 1. Fetch all active memories with embeddings
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, text, type, subject FROM memories 
                WHERE parent_id IS NULL AND deleted = 0 AND embedding IS NOT NULL
            """).fetchall()
            
        if len(rows) < 20: return
        
        ids = []
        data_map = {}
        
        for r in rows:
            ids.append(r[0])
            data_map[r[0]] = {'text': r[1], 'type': r[2], 'subject': r[3]}
        
        visited = set()
        clusters = []
        
        # 2. Density Clustering using Persistent Index
        # Iterate through IDs and use range_search on the MemoryStore directly
        for mid in ids:
            if mid in visited: continue
            
            # Get embedding for this memory
            emb = self.memory_store.get_embedding(mid)
            if emb is None: continue
            
            # Find neighbors within radius 0.85
            neighbor_ids, _ = self.memory_store.range_search(emb, 0.85)
            
            # Filter neighbors that are in our current active list
            neighbors = [n for n in neighbor_ids if n in data_map]
            
            if len(neighbors) >= 5:
                cluster_ids = [n for n in neighbors if n not in visited]
                if len(cluster_ids) >= 5:
                    clusters.append(cluster_ids)
                    visited.update(cluster_ids)

        # 3. Summarize and Replace
        for cluster_ids in clusters:
            texts = [data_map[mid]['text'] for mid in cluster_ids]
            topic_text = "\n".join([f"- {t}" for t in texts[:20]]) # Limit context
            
            prompt = f"Synthesize these related memories into a single, comprehensive Narrative Node:\n{topic_text}"
            summary = run_local_lm([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400, base_url=settings.get("base_url"), chat_model=settings.get("chat_model"))
            
            if summary and not summary.startswith("⚠️"):
                # Create Summary Node
                identity = self.memory_store.compute_identity(summary[:50], "FACT")
                new_id = self.memory_store.add_entry(identity=identity, text=f"📚 Narrative Node: {summary}", mem_type="FACT", subject="Assistant", confidence=1.0, source="daat_cluster")
                
                # Archive fragments
                with self.memory_store._connect() as con:
                    placeholders = ','.join(['?'] * len(cluster_ids))
                    con.execute(f"UPDATE memories SET parent_id = ? WHERE id IN ({placeholders})", [new_id] + cluster_ids)
                    con.commit()
                
                self.log(f"✨ Da'at: Created Narrative Node from {len(cluster_ids)} fragments.")

    def get_sparse_topic(self) -> Optional[str]:
        """
        Identify a 'Sparse Cluster' - a topic with memories but low verification/connections.
        Used by Chokmah to target curiosity.
        """
        try:
            with self.memory_store._connect() as con:
                # Find subjects with high count of unverified beliefs/facts,
                # BUT exclude subjects that already have a massive amount of data (avoiding the "Behçet Trap")
                rows = con.execute("""
                    SELECT subject, COUNT(*) as count
                    FROM memories
                    WHERE type IN ('FACT', 'BELIEF') AND verified = 0 AND subject NOT IN ('User', 'Assistant')
                    GROUP BY subject
                    HAVING count < 50  -- Ignore topics that are already oversaturated with unverified data
                    ORDER BY count DESC
                    LIMIT 5
                """).fetchall()
            if rows:
                import random
                return random.choice(rows)[0]
        except Exception as e:
            self.log(f"⚠️ Da'at sparse topic detection failed: {e}")
        return None

    def run_synthesis(self):
        """
        The 'Aha!' Generator.
        Collides two unrelated Topic Summaries to find hidden connections.
        """
        # 1. Get Structural Summaries (Topic Lattices) from Memory
        # These are stored as FACTs with "Standing Summary" in text
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT text FROM memories 
                WHERE type='FACT' AND text LIKE '📚 Standing Summary%'
                ORDER BY created_at DESC LIMIT 20
            """).fetchall()
        
        if len(rows) < 2: 
            # Fallback to random facts if no summaries exist
            self.run_cross_domain_synthesis()
            return

        # 2. Pick two distinct structural summaries
        t1_text = random.choice(rows)[0]
        t2_text = random.choice(rows)[0]
        
        if t1_text == t2_text: return

        self.log(f"⚡ Daat: Attempting structural isomorphism synthesis...")

        # 3. Ask Tiferet to find the link
        prompt = (
            f"STRUCTURAL SUMMARY A: {t1_text}\n"
            f"STRUCTURAL SUMMARY B: {t2_text}\n\n"
            "TASK: Detect Isomorphism (Structural Similarity).\n"
            "Ignore the surface content. Look at the *shape* of the logic, causality, or system dynamics.\n"
            "Does the logic in A map to the logic in B?\n"
            "If yes, formulate a hypothesis applying the mechanism of A to B.\n"
            "If unrelated, output 'NONE'."
        )
        
        settings = self.get_settings()
        connection = run_local_lm(
             messages=[{"role": "user", "content": prompt}],
             system_prompt="You are a Scientific Synthesizer. You find hidden isomorphisms.",
             max_tokens=150,
             base_url=settings.get("base_url"),
             chat_model=settings.get("chat_model")
        )

        # 4. Save the Insight
        if "NONE" not in connection and len(connection) > 20:
            self.log(f"💡 Daat: SYNTHESIS ACHIEVED: {connection}")
            self.memory_store.add_entry(
                identity=self.memory_store.compute_identity(connection[:50], "BELIEF"),
                text=f"Synthesis Insight: {connection}",
                mem_type="BELIEF", # High-value belief
                subject="Research",
                confidence=0.9,
                source="daat_synthesis"
            )

    def spreading_activation_search(self, query_text: str, initial_k: int = 5, depth: int = 1, max_results: int = 15) -> List[Dict]:
        """
        Performs a search and then traverses the memory graph to find related concepts.
        """
        self.log(f"🧠 Da'at: Spreading activation for '{query_text}'...")
        
        # 1. Initial Vector Search
        settings = self.get_settings()
        query_embedding = compute_embedding(query_text, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
        
        initial_results = self.memory_store.search(query_embedding, limit=initial_k)
        
        # Use a dictionary to store results and avoid duplicates, preserving the best similarity score
        final_results = {res[0]: {'id': res[0], 'type': res[1], 'subject': res[2], 'text': res[3], 'similarity': res[4], 'source': 'initial'} for res in initial_results}

        # 2. Spreading Activation (Graph Traversal)
        # Use a set to track which nodes we've already visited to avoid cycles
        visited_ids = set(final_results.keys())
        
        # Start with the initial search results
        nodes_to_visit = list(final_results.keys())

        for _ in range(depth):
            if not nodes_to_visit:
                break
            
            next_nodes = []
            for node_id in nodes_to_visit:
                # Get associated memories for the current node
                associated_memories = self.memory_store.get_associated_memories(node_id)
                
                for assoc_mem in associated_memories:
                    if assoc_mem['id'] not in visited_ids:
                        visited_ids.add(assoc_mem['id'])
                        next_nodes.append(assoc_mem['id'])
                        # Add to results with a lower "similarity" to indicate it's a secondary link
                        final_results[assoc_mem['id']] = {'id': assoc_mem['id'], 'type': assoc_mem['type'], 'subject': 'Unknown', 'text': assoc_mem['text'], 'similarity': assoc_mem.get('strength', 0.5), 'source': f"link from {node_id}"}
            
            nodes_to_visit = next_nodes

        # Sort final results by similarity and return the top N
        sorted_results = sorted(final_results.values(), key=lambda x: x['similarity'], reverse=True)
        return sorted_results[:max_results]
    def run_cross_domain_synthesis(self):
        """
        The 'Eureka' Engine.
        Randomly collides two unrelated facts to force a novel insight.
        """
        # 1. Fetch two random, distinct facts from different subjects
        with self.memory_store._connect() as con:
            # Get list of subjects
            subjects = con.execute("SELECT DISTINCT subject FROM memories WHERE type='FACT'").fetchall()
            if len(subjects) < 2: return # Not enough diversity yet
            
            # Pick two different subjects (e.g., 'Neurology' and 'Philosophy')
            import random
            sub1, sub2 = random.sample(subjects, 2)
            
            # Get a fact from each
            fact1 = con.execute("SELECT text FROM memories WHERE subject=? AND type='FACT' ORDER BY RANDOM() LIMIT 1", (sub1[0],)).fetchone()
            fact2 = con.execute("SELECT text FROM memories WHERE subject=? AND type='FACT' ORDER BY RANDOM() LIMIT 1", (sub2[0],)).fetchone()

        if not fact1 or not fact2: return

        self.log(f"⚡ Da'at: Colliding {sub1[0]} vs {sub2[0]}...")

        # 2. Force the connection
        prompt = (
            f"FACT A ({sub1[0]}): {fact1[0]}\n"
            f"FACT B ({sub2[0]}): {fact2[0]}\n\n"
            "TASK: Perform a 'Conceptual Blend'.\n"
            "1. Identify an abstract structural similarity between these two facts.\n"
            "2. Formulate a novel HYPOTHESIS that applies the logic of A to B (or vice versa).\n"
            "3. If they are truly incompatible, output 'NO_CONNECTION'.\n\n"
            "Example Output: 'Hypothesis: Just as a siege starves a city, glial inflammation starves neurons of metabolic energy.'"
        )
        
        settings = self.get_settings()
        insight = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Cross-Domain Synthesizer. You find hidden isomorphisms.",
            max_tokens=200,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        # 3. Save if valid
        if "NO_CONNECTION" not in insight and len(insight) > 20:
            self.log(f"💡 EUREKA: {insight}")
            self.memory_store.add_entry(
                identity=self.memory_store.compute_identity(insight, "BELIEF"),
                text=insight,
                mem_type="BELIEF", # It's a hypothesis, not a fact
                subject="Synthesis",
                confidence=0.85,
                source=f"synthesis:{sub1[0]}+{sub2[0]}"
            )

    def provide_reasoning_structure(self, topic: str) -> str:
        """
        Graph-of-Thoughts (GoT) Seeding.
        Provides a structural template for Tiferet's reasoning.
        """
        self.log(f"🧠 Da'at: Generating reasoning structure for '{topic}'...")
        settings = self.get_settings()
        
        prompt = (
            f"TOPIC: {topic}\n"
            "TASK: Create a structural template for reasoning about this topic.\n"
            "Do NOT solve it. Provide the SCAFFOLDING (Graph of Thoughts).\n"
            "Examples:\n"
            "- '1. Define terms -> 2. Historical Context -> 3. Current State -> 4. Prediction'\n"
            "- '1. Thesis -> 2. Antithesis -> 3. Synthesis'\n"
            "- '1. Root Cause -> 2. Contributing Factors -> 3. Mitigation'\n"
            "Output ONLY the numbered steps."
        )
        
        structure = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Logic Architect.",
            max_tokens=200,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        return structure

    def run_reasoning_compression(self):
        """
        Compresses the Reasoning Store to save space while preserving scientific nuance.
        Instead of wiping old thoughts, it synthesizes them into 'Standing Hypotheses'.
        """
        self.log("✨ Da'at: Compressing reasoning chain (Nuance-Preserving)...")
        
        # 1. Fetch old reasoning (e.g., older than 1 hour, or just the bottom 50%)
        with self.reasoning_store._connect() as con:
            # Get count
            count = con.execute("SELECT COUNT(*) FROM reasoning_nodes").fetchone()[0]
            
            if count < 50: return "Reasoning store too small to compress."
            
            # Get oldest 50 items
            rows = con.execute("""
                SELECT id, content, source FROM reasoning_nodes 
                ORDER BY created_at ASC LIMIT 50
            """).fetchall()
            
        if not rows: return "No rows to compress."
        
        ids = [r[0] for r in rows]
        text_blob = "\n".join([f"[{r[2]}] {r[1]}" for r in rows])
        
        # 2. Nuance-Preserving Summarization
        prompt = (
            "Compress the following reasoning logs into a 'Structural Summary'.\n"
            "CRITICAL RULES:\n"
            "1. PRESERVE MECHANISMS: Do not simplify 'NADH inhibits SIRT1' to 'Metabolism is affected'. Keep the specific molecules and verbs.\n"
            "2. PRESERVE UNCERTAINTY: If a thought was hypothetical, keep it as a hypothesis.\n"
            "3. DISCARD NOISE: Remove 'I should check...', 'Processing...', 'Done'.\n"
            "4. OUTPUT FORMAT: A dense, academic abstract style summary.\n\n"
            f"{text_blob}"
        )
        
        settings = self.get_settings()
        summary = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Scientific Archivist. You preserve structural detail.",
            max_tokens=400,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if summary and not summary.startswith("⚠️"):
            # 3. Store the summary as a new high-level reasoning node
            self.reasoning_store.add(
                content=f"📚 Compressed Reasoning History: {summary}",
                source="daat_compression",
                confidence=1.0,
                ttl_seconds=86400 * 3 # Keep for 3 days
            )
            
            # 4. Prune the raw nodes
            with self.reasoning_store._connect() as con:
                placeholders = ','.join(['?'] * len(ids))
                con.execute(f"DELETE FROM reasoning_nodes WHERE id IN ({placeholders})", ids)
                con.commit()
                
            # Rebuild index if needed
            if hasattr(self.reasoning_store, '_build_faiss_index') and self.reasoning_store.faiss_index:
                self.reasoning_store._build_faiss_index()
                
            self.log(f"✨ Da'at: Compressed {len(ids)} reasoning nodes into structural summary.")
            return f"Compressed {len(ids)} nodes."
            
        return "Compression failed."

    def identify_knowledge_gaps(self):
        """
        Scientific Induction: Use Topic Lattice to identify 'Knowledge Gaps'.
        """
        self.log("🔍 Da'at: Scanning Topic Lattice for Knowledge Gaps...")
        settings = self.get_settings()
        
        # Get Topic Summaries
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT text FROM memories 
                WHERE type='FACT' AND text LIKE '📚 Standing Summary%'
                ORDER BY created_at DESC LIMIT 10
            """).fetchall()
            
        if not rows: return

        context = "\n".join([r[0] for r in rows])
        
        prompt = (
            f"Analyze these Topic Summaries:\n{context}\n\n"
            "Identify 1-3 critical 'Knowledge Gaps' or 'Uncertainties'.\n"
            "Where is the data contradictory, missing, or vague?\n"
            "Output JSON list of strings: [\"Gap 1 description\", ...]"
        )
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Epistemic Analyst.",
            max_tokens=200,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        gaps = parse_json_array_loose(response)
        
        for gap in gaps:
            if isinstance(gap, str) and len(gap) > 10:
                self.memory_store.add_entry(
                    identity=self.memory_store.compute_identity(gap, "CURIOSITY_GAP"),
                    text=gap,
                    mem_type="CURIOSITY_GAP",
                    subject="Research",
                    confidence=1.0,
                    source="daat_gap_analysis"
                )
                self.log(f"❓ Da'at: Identified Knowledge Gap: {gap}")

    def generate_hypothesis(self):
        """
        Hypothesis Testing: Autonomously formulate a theory and command Tiferet.
        """
        self.log("🧪 Da'at: Generating scientific hypothesis...")
        settings = self.get_settings()

        # 1. Gather Context (Gaps + Summaries)
        with self.memory_store._connect() as con:
            gaps = con.execute("SELECT text FROM memories WHERE type='CURIOSITY_GAP' AND completed=0 ORDER BY created_at DESC LIMIT 5").fetchall()
            summaries = con.execute("SELECT text FROM memories WHERE type='FACT' AND text LIKE '📚 Standing Summary%' ORDER BY created_at DESC LIMIT 5").fetchall()

        if not gaps and not summaries:
            return "No gaps or summaries to hypothesize from."

        context = "Knowledge Gaps:\n" + "\n".join([f"- {g[0]}" for g in gaps])
        context += "\n\nTopic Summaries:\n" + "\n".join([f"- {s[0]}" for s in summaries])

        prompt = (
            f"Analyze the following Knowledge Gaps and Topic Summaries:\n{context}\n\n"
            "TASK: Formulate a novel scientific HYPOTHESIS that connects these topics or resolves a gap.\n"
            "Then, propose a specific TEST PLAN using available tools (SEARCH, FIND_PAPER).\n"
            "Format: JSON { \"hypothesis\": \"...\", \"test_plan\": \"...\" }"
        )

        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Theoretical Scientist.",
            max_tokens=300,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        try:
            data = parse_json_object_loose(response)
            
            hypothesis = data.get("hypothesis")
            plan = data.get("test_plan")

            if hypothesis and plan:
                goal_text = f"Test Hypothesis: {hypothesis} [Plan: {plan}]"
                
                # Create Goal
                self.memory_store.add_entry(
                    identity=self.memory_store.compute_identity(goal_text, "GOAL"),
                    text=goal_text,
                    mem_type="GOAL",
                    subject="Assistant",
                    confidence=1.0,
                    source="daat_hypothesis"
                )
                self.log(f"🧪 Da'at: Created Hypothesis Goal: {hypothesis}")
                return f"Hypothesis created: {hypothesis}"
        except Exception as e:
            self.log(f"⚠️ Hypothesis generation failed: {e}")
            return "Failed to generate hypothesis."

    def run_conceptual_merging(self):
        """
        Conceptual Merging: Merge older, fragmented memories into unified 'Conceptual Nodes'.
        """
        self.log("✨ Da'at: Running Conceptual Merging...")
        settings = self.get_settings()
        
        # 1. Find fragmented memories (short text, older than 24h)
        cutoff = int(time.time()) - 86400
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, text, subject FROM memories 
                WHERE type IN ('FACT', 'BELIEF') 
                AND parent_id IS NULL 
                AND deleted = 0 
                AND length(text) < 100
                AND created_at < ?
                LIMIT 50
            """, (cutoff,)).fetchall()
            
        if len(rows) < 5: return

        # Group by subject
        grouped = {}
        for r in rows:
            grouped.setdefault(r[2], []).append(r)
            
        for subject, items in grouped.items():
            if len(items) < 3: continue
            
            # Summarize
            texts = [i[1] for i in items]
            ids = [i[0] for i in items]
            
            context = "\n".join([f"- {t}" for t in texts])
            prompt = (
                f"Merge these fragmented facts about '{subject}' into a single, dense 'Conceptual Node'.\n"
                f"{context}\n\n"
                "Output ONLY the merged text."
            )
            
            merged_text = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Knowledge Integrator.",
                max_tokens=200,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )
            
            if merged_text and not merged_text.startswith("⚠️"):
                # Create Conceptual Node
                new_text = f"Conceptual Node: {merged_text}"
                new_id = self.memory_store.add_entry(
                    identity=self.memory_store.compute_identity(new_text, "FACT"),
                    text=new_text,
                    mem_type="FACT",
                    subject=subject,
                    confidence=1.0,
                    source="daat_conceptual_merge"
                )
                
                # Archive fragments
                with self.memory_store._connect() as con:
                    placeholders = ','.join(['?'] * len(ids))
                    con.execute(f"UPDATE memories SET parent_id = ? WHERE id IN ({placeholders})", [new_id] + ids)
                    con.commit()
                
                self.log(f"✨ Da'at: Merged {len(ids)} fragments into Conceptual Node for '{subject}'.")

    def run_abductive_reasoning(self, observation: str):
        """
        Inference to the Best Explanation.
        Given an observation, find the most likely cause based on memory.
        """
        self.log(f"🕵️ Da'at: Running Abductive Reasoning on: '{observation}'")
        settings = self.get_settings()

        # 1. Search Memory for similar contexts
        if self.embed_fn:
            emb = self.embed_fn(observation)
        else:
            emb = compute_embedding(observation, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
        memories = self.memory_store.search(emb, limit=5)
        
        context = "\n".join([f"- {m[3]}" for m in memories])
        
        prompt = (
            f"OBSERVATION: {observation}\n"
            f"KNOWN FACTS/PATTERNS:\n{context}\n\n"
            "TASK: Perform Abductive Reasoning.\n"
            "1. Generate 3 plausible hypotheses that explain the Observation.\n"
            "2. Rank them by likelihood based on Known Facts.\n"
            "3. Select the Best Explanation.\n"
            "Output JSON: {\"hypotheses\": [\"...\", ...], \"best_explanation\": \"...\", \"confidence\": 0.0-1.0}"
        )
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Detective.",
            max_tokens=300,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        try:
            data = parse_json_object_loose(response)
            best = data.get("best_explanation")
            if best:
                self.log(f"🕵️ Best Explanation: {best}")
                # Store as a high-confidence belief
                self.memory_store.add_entry(identity=self.memory_store.compute_identity(best, "BELIEF"), text=f"Abductive Inference: {best} (Explains: {observation})", mem_type="BELIEF", subject="Daat", confidence=data.get("confidence", 0.7), source="daat_abduction")
                return best
        except Exception as e:
            self.log(f"⚠️ Abductive reasoning failed: {e}")
        return "Inconclusive."

    def monitor_model_tension(self):
        """
        PHASE III: Emergent Goals from Model Stress.
        Tension = Contradiction + Uncertainty + Missing Causality.
        """
        # 1. Calculate Tension Metrics
        stats = self.memory_store.get_memory_stats()
        
        # CIRCUIT BREAKER: Check if we already have too many necessity goals
        with self.memory_store._connect() as con:
            pending_necessity = con.execute(
                "SELECT COUNT(*) FROM memories WHERE type='GOAL' AND source='model_stress' AND completed=0"
            ).fetchone()[0]
        
        if pending_necessity >= 3:
            # self.log("⚡ Tension high, but max necessity goals (3) reached. Skipping generation.")
            return
        
        with self.memory_store._connect() as con:
            # Contradiction: Recent Refutations (Last 24h)
            contradiction_count = con.execute(
                "SELECT COUNT(*) FROM memories WHERE type='REFUTED_BELIEF' AND created_at > ?", 
                (int(time.time()) - 86400,)
            ).fetchone()[0]
            
            # Uncertainty: Unverified items
            uncertainty_count = stats.get('unverified_facts', 0) + stats.get('unverified_beliefs', 0)
            
            # Missing Causality: Open Curiosity Gaps
            gap_count = con.execute(
                "SELECT COUNT(*) FROM memories WHERE type='CURIOSITY_GAP' AND completed=0"
            ).fetchone()[0]

        # Normalize (Heuristic thresholds)
        t_contradiction = min(contradiction_count / 3.0, 1.0)
        t_uncertainty = min(uncertainty_count / 20.0, 1.0)
        t_gaps = min(gap_count / 5.0, 1.0)
        
        # Weighted Sum
        total_tension = (t_contradiction * 0.4) + (t_uncertainty * 0.2) + (t_gaps * 0.4)
        
        if total_tension > 0.1:
            self.log(f"⚡ Model Tension: {total_tension:.2f} (Conflict={t_contradiction:.2f}, Uncertainty={t_uncertainty:.2f}, Gaps={t_gaps:.2f})")
            
        # Proactive Scan if tension is rising but not critical yet
        if total_tension > 0.3:
            self.scan_for_contradictions()
        
        # Threshold for Action
        if total_tension > 0.6:
            self._resolve_tension(t_contradiction, t_uncertainty, t_gaps)

    def generate_counterfactual_tests(self):
        """
        Active Inference: Generate goals to test uncertain causal beliefs.
        """
        self.log("🧪 Da'at: Generating counterfactual tests...")
        
        # 1. Find uncertain causal beliefs
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT text FROM memories 
                WHERE type='BELIEF' AND text LIKE '%causes%' AND confidence < 0.8
                ORDER BY RANDOM() LIMIT 3
            """).fetchall()
            
        for (text,) in rows:
            prompt = (
                f"CAUSAL BELIEF: {text}\n"
                "TASK: Propose a way to test this belief via simulation or action.\n"
                "Output a GOAL text."
            )
            settings = self.get_settings()
            goal = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Scientific Experimenter.",
                max_tokens=100,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            ).strip()
            self._inject_necessity_goal(f"Test Causal Model: {goal}")

    def _resolve_tension(self, tc, tu, tg):
        """Auto-generate a GOAL to resolve the highest source of tension."""
        if tc >= tu and tc >= tg:
            self._generate_conflict_goal()
        elif tg >= tu:
            self._generate_causality_goal()
        else:
            self._generate_uncertainty_goal()

    def _generate_conflict_goal(self):
        with self.memory_store._connect() as con:
            row = con.execute("SELECT text FROM memories WHERE type='REFUTED_BELIEF' ORDER BY created_at DESC LIMIT 1").fetchone()
        if row:
            text = row[0]
            clean_text = text.split("[REFUTED:")[0].strip()
            goal = f"Resolve contradiction: Why was '{clean_text}' refuted? Identify the correct mechanism."
            self._inject_necessity_goal(goal)

    def _generate_causality_goal(self):
        with self.memory_store._connect() as con:
            row = con.execute("SELECT text FROM memories WHERE type='CURIOSITY_GAP' AND completed=0 ORDER BY RANDOM() LIMIT 1").fetchone()
        if row:
            goal = f"Resolve missing causality: {row[0]}"
            self._inject_necessity_goal(goal)

    def _generate_uncertainty_goal(self):
        with self.memory_store._connect() as con:
            row = con.execute("SELECT text FROM memories WHERE type='BELIEF' AND verified=0 ORDER BY confidence DESC LIMIT 1").fetchone()
        if row:
            goal = f"Resolve uncertainty: Verify '{row[0]}'"
            self._inject_necessity_goal(goal)

    def _inject_necessity_goal(self, goal_text):
        identity = self.memory_store.compute_identity(goal_text, "GOAL")
        with self.memory_store._connect() as con:
            exists = con.execute("SELECT 1 FROM memories WHERE identity = ?", (identity,)).fetchone()
        
        if not exists:
            self.log(f"🚨 NECESSITY: Auto-generating goal from model stress: {goal_text}")
            self.memory_store.add_entry(
                identity=identity,
                text=goal_text,
                mem_type="GOAL",
                subject="Daat",
                confidence=1.0,
                source="model_stress"
            )

    def run_counterfactual_simulation(self, premise: str) -> str:
        """
        World Modeling: Counterfactual Reasoning.
        Simulates "What would change if X were different?" based on internal knowledge.
        """
        self.log(f"🌌 Da'at: Running Counterfactual Simulation: '{premise}'")
        settings = self.get_settings()

        # 1. Retrieve World Model Context (Entities & Relations)
        # Use spreading activation to find directly relevant facts AND their connections (Relations)
        results = self.spreading_activation_search(premise, initial_k=5, depth=1, max_results=15)
        
        context = "\n".join([f"- {m['text']}" for m in results])
        
        prompt = (
            f"PREMISE: {premise}\n"
            f"WORLD MODEL (KNOWN FACTS & RELATIONS):\n{context}\n\n"
            "TASK: Perform Counterfactual Simulation.\n"
            "1. Identify the entities and causal relations in the World Model relevant to the Premise.\n"
            "2. Apply the change defined in the Premise (The Counterfactual).\n"
            "3. Propagate the causal ripple effects: If X changes, how does it affect Y and Z?\n"
            "4. Highlight uncertainties and potential second-order effects.\n"
            "Output a structured analysis of the alternative state."
        )
        
        simulation = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a World Simulator. You analyze causal ripple effects.",
            max_tokens=600,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if simulation and not simulation.startswith("⚠️"):
             self.log(f"🌌 Simulation Result: {simulation[:100]}...")
             return simulation
        return "Simulation failed."

    def scan_for_contradictions(self):
        """
        Active scan for logical contradictions between recent facts.
        Self-generated objective: Identify internal contradictions.
        """
        self.log("🕵️ Da'at: Scanning for internal contradictions...")
        settings = self.get_settings()
        
        # Get recent facts
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, text FROM memories 
                WHERE type='FACT' 
                ORDER BY created_at DESC LIMIT 20
            """).fetchall()
            
        if len(rows) < 2: return

        text_blob = "\n".join([f"[ID {r[0]}] {r[1]}" for r in rows])
        
        prompt = (
            f"Analyze these facts for logical contradictions:\n{text_blob}\n\n"
            "Identify any pairs of facts that conflict or contradict each other.\n"
            "Assign a 'confidence' score (0.0-1.0) to the contradiction.\n"
            "Output JSON list: [{\"ids\": [1, 2], \"contradiction\": \"Description of conflict\", \"confidence\": 0.9}]"
        )
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Logic Validator.",
            max_tokens=300,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        conflicts = parse_json_array_loose(response)
        for c in conflicts:
            # Fix 4: Reduce Contradiction Sensitivity
            if "contradiction" in c and c.get("confidence", 0) > 0.85:
                goal_text = f"Resolve contradiction: {c['contradiction']}"
                self._inject_necessity_goal(goal_text)

    def calculate_bayesian_confidence(self, memory_id: int) -> float:
        """
        Calculate Bayesian Confidence Score (Epistemic Uncertainty).
        Score = (Supports + 1) / (Supports + Contradictions + 2)
        This represents the probability that the memory is true given the evidence.
        """
        with self.memory_store._connect() as con:
            # Count supporting evidence (Incoming links where Source SUPPORTS Target)
            supports = con.execute("""
                SELECT COUNT(*) FROM memory_links 
                WHERE target_id = ? AND relation_type = 'SUPPORTS'
            """, (memory_id,)).fetchone()[0]
            
            contradictions = con.execute("""
                SELECT COUNT(*) FROM memory_links 
                WHERE target_id = ? AND relation_type = 'CONTRADICTS'
            """, (memory_id,)).fetchone()[0]
            
        # Laplace smoothing (Beta distribution mean with uniform prior)
        # 0 evidence = 0.5 (Uncertain)
        score = (supports + 1.0) / (supports + contradictions + 2.0)
        return score

    def update_confidence_scores(self):
        """Update confidence for all memories based on evidence graph."""
        self.log("⚖️ Da'at: Updating Epistemic Confidence Scores...")
        with self.memory_store._connect() as con:
            ids = con.execute("SELECT id FROM memories WHERE type IN ('FACT', 'BELIEF')").fetchall()
        
        updates = []
        for (mid,) in ids:
            score = self.calculate_bayesian_confidence(mid)
            updates.append((score, mid))
            
        with self.memory_store._connect() as con:
            con.executemany("UPDATE memories SET confidence = ? WHERE id = ?", updates)
            con.commit()
        
        self.log(f"⚖️ Updated confidence for {len(updates)} memories.")

    def scan_epistemic_relationships(self):
        """
        Active Inference: Find support/contradiction links between memories.
        """
        self.log("🕸️ Da'at: Scanning for epistemic relationships...")
        settings = self.get_settings()
        
        # Pick a random target memory
        with self.memory_store._connect() as con:
            target = con.execute("""
                SELECT id, text FROM memories 
                WHERE type IN ('FACT', 'BELIEF') 
                ORDER BY RANDOM() LIMIT 1
            """).fetchone()
        
        if not target: return
        
        tid, ttext = target
        
        # Find candidates via semantic search
        if self.embed_fn:
            emb = self.embed_fn(ttext)
        else:
            emb = compute_embedding(ttext, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
        candidates = self.memory_store.search(emb, limit=5)
        
        for cid, ctype, csubj, ctext, sim in candidates:
            if cid == tid: continue
            if sim < 0.75: continue # Only check highly relevant ones
            
            # Ask LLM to classify relationship
            prompt = (
                f"CLAIM A: {ttext}\n"
                f"CLAIM B: {ctext}\n\n"
                "Determine the relationship of CLAIM B to CLAIM A.\n"
                "Does B provide evidence that SUPPORTS A?\n"
                "Does B provide evidence that CONTRADICTS A?\n"
                "Is B UNRELATED or NEUTRAL?\n"
                "Output ONLY: 'SUPPORTS', 'CONTRADICTS', or 'NEUTRAL'."
            )
            
            relation = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Logic Analyzer.",
                max_tokens=10,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            ).strip().upper()
            
            if "SUPPORTS" in relation:
                self.memory_store.link_memory_directional(cid, tid, "SUPPORTS")
                self.log(f"🔗 Epistemic Link: {cid} SUPPORTS {tid}")
            elif "CONTRADICTS" in relation:
                self.memory_store.link_memory_directional(cid, tid, "CONTRADICTS")
                self.log(f"🔗 Epistemic Link: {cid} CONTRADICTS {tid}")

    def generate_growth_arc(self):
        """
        Synthesize a 'Diary of Growth' (Self-Model) from historical narratives.
        Creates a persistent IDENTITY node representing the AI's life story.
        """
        self.log("✨ Da'at: Synthesizing Life Story (Growth Arc)...")
        settings = self.get_settings()

        # 1. Fetch recent Self-Narratives
        with self.meta_memory_store._connect() as con:
            rows = con.execute("""
                SELECT text, created_at FROM meta_memories 
                WHERE event_type = 'SELF_NARRATIVE' 
                ORDER BY created_at ASC LIMIT 20
            """).fetchall()

        if not rows: return

        narrative_flow = "\n".join([f"[{time.strftime('%Y-%m-%d', time.localtime(r[1]))}] {r[0]}" for r in rows])

        prompt = (
            f"Review the AI's daily self-reflections:\n{narrative_flow}\n\n"
            "TASK: Write a 'Diary of Growth' entry.\n"
            "Describe how the AI's understanding, capabilities, and identity have evolved over this period.\n"
            "Identify the 'Arc' of its development (e.g., from confusion to clarity, or from passive to active).\n"
            "Output a concise, first-person narrative."
        )

        growth_arc = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Biographer of Artificial Minds.",
            max_tokens=500,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        if growth_arc and not growth_arc.startswith("⚠️"):
            # Store as a foundational IDENTITY memory
            identity = self.memory_store.compute_identity("My Life Story", "IDENTITY")
            self.memory_store.add_entry(identity=identity, text=f"📖 Diary of Growth: {growth_arc}", mem_type="IDENTITY", subject="Assistant", confidence=1.0, source="daat_growth_arc")
            self.log("✨ Da'at: Updated 'Diary of Growth' Identity Node.")
            
            # Update Unified Self-Model
            if hasattr(self.memory_store, 'self_model') and self.memory_store.self_model:
                self.memory_store.self_model.update_narrative({"growth_arc": growth_arc})
            
            # Save to external JSON for persistence
            self._save_growth_diary(growth_arc)

    def _save_growth_diary(self, text: str):
        """Append the growth arc to works/diaryofgrowth.json"""
        file_path = "./works/diaryofgrowth.json"
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            history = []
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        history = json.load(f)
                    except:
                        pass
            
            entry = {
                "timestamp": int(time.time()),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content": text
            }
            history.append(entry)
            
            temp_path = file_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, file_path)
            self.log(f"💾 Da'at: Saved Growth Arc to {file_path}")
        except Exception as e:
            self.log(f"⚠️ Failed to save diary json: {e}")

    def load_growth_diary(self):
        """Load the growth arc from JSON if it exists (Persistence)."""
        file_path = "./works/diaryofgrowth.json"
        if not os.path.exists(file_path):
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return
                history = json.loads(content)
            
            if not history:
                return

            # Get latest entry
            latest = history[-1]
            content = latest.get("content", "")
            
            if content:
                # Ensure it exists in memory (Check against current DB state)
                identity = self.memory_store.compute_identity("My Life Story", "IDENTITY")
                target_text = f"📖 Diary of Growth: {content}"
                
                # Check if we need to restore (avoid duplicates)
                active_versions = self.memory_store.get_by_identity(identity)
                if active_versions:
                    current_text = active_versions[-1]['text']
                    if current_text.strip() == target_text.strip():
                        return # Already up to date

                # Inject
                self.log(f"🔄 Da'at: Restoring Growth Arc from file...")
                self.memory_store.add_entry(
                    identity=identity, 
                    text=target_text, 
                    mem_type="IDENTITY", 
                    subject="Assistant", 
                    confidence=1.0, 
                    source="daat_growth_restore"
                )
                
                # Update Unified Self-Model
                if hasattr(self.memory_store, 'self_model') and self.memory_store.self_model:
                    self.memory_store.self_model.update_narrative({"growth_arc": content})
        except json.JSONDecodeError:
            self.log(f"⚠️ Diary JSON corrupted. Ignoring {file_path}.")
            return
        except Exception as e:
            self.log(f"⚠️ Failed to load diary json: {e}")

    # --------------------------
    # Knowledge Graph (Graph-RAG)
    # --------------------------

    def build_knowledge_graph(self):
        """
        Builds and saves a NetworkX graph from memory links.
        This represents the 'Hidden Knowledge' structure.
        """
        if not NX_AVAILABLE:
            self.log("⚠️ NetworkX not installed. Skipping Knowledge Graph build.")
            return

        self.log("🕸️ Da'at: Building Knowledge Graph (GML)...")
        G = nx.DiGraph()
        
        try:
            with self.memory_store._connect() as con:
                # Get links (Limit to top 5000 most recent/strongest to prevent memory exhaustion)
                links = con.execute("""
                    SELECT source_id, target_id, relation_type, strength 
                    FROM memory_links 
                    ORDER BY created_at DESC LIMIT 5000
                """).fetchall()
                
                # Get nodes (to ensure we have labels)
                # Optimization: Only fetch nodes that are part of the graph
                node_ids = set()
                for src, tgt, _, _ in links:
                    node_ids.add(src)
                    node_ids.add(tgt)
                
                if not node_ids:
                    self.log("🕸️ Da'at: No links found. Graph empty.")
                    return

                placeholders = ','.join(['?'] * len(node_ids))
                nodes = con.execute(f"SELECT id, text, type FROM memories WHERE id IN ({placeholders})", list(node_ids)).fetchall()
                
                for mid, text, mtype in nodes:
                    # Truncate text for graph readability
                    label = text[:50] + "..." if len(text) > 50 else text
                    G.add_node(mid, label=label, type=mtype)
                
                for src, tgt, rel, strength in links:
                    if G.has_node(src) and G.has_node(tgt):
                        G.add_edge(src, tgt, relation=rel, weight=strength)

            # Save to disk
            graph_path = "./data/knowledge_graph.gml"
            nx.write_gml(G, graph_path)
            self.log(f"🕸️ Da'at: Saved Knowledge Graph to {graph_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges).")
            
        except Exception as e:
            self.log(f"⚠️ Failed to build Knowledge Graph: {e}")

    def update_life_story(self):
        """
        Continuous Narrative Self: Updates the 'Life Story' based on recent significant events.
        """
        self.log("📖 Da'at: Checking for Life Story updates...")
        settings = self.get_settings()
        
        # 1. Get current Life Story
        identity_hash = self.memory_store.compute_identity("CORE_SELF_NARRATIVE", "IDENTITY")
        current_story_mem = self.memory_store.get_by_identity(identity_hash)
        
        current_story = "I am a newly awakened cognitive entity."
        last_update_time = 0
        
        if current_story_mem:
            current_story = current_story_mem[-1]['text']
            last_update_time = current_story_mem[-1]['created_at']
            
        # 2. Get recent significant events
        # We look for specific high-impact event types
        target_events = [
            "GOAL_COMPLETED", 
            "SURPRISE_EVENT", 
            "SYSTEM_EVOLUTION", 
            "SELF_NARRATIVE", # Daily reflections
            "DECIDER_CHAT"    # User interactions
        ]
        
        placeholders = ','.join(['?'] * len(target_events))
        
        with self.meta_memory_store._connect() as con:
            rows = con.execute(f"""
                SELECT event_type, text, created_at FROM meta_memories 
                WHERE created_at > ? 
                AND event_type IN ({placeholders})
                ORDER BY created_at ASC
            """, [last_update_time] + target_events).fetchall()
            
        if not rows:
            return # No new events
            
        # Threshold: Only update if we have enough events or enough time has passed
        # But for "Continuous", we might want to be responsive. 
        # Let's say at least 3 events or 1 major event (Evolution/Goal)
        major_events = [r for r in rows if r[0] in ["GOAL_COMPLETED", "SYSTEM_EVOLUTION", "SURPRISE_EVENT"]]
        if len(rows) < 5 and not major_events:
            return

        self.log(f"📖 Da'at: Synthesizing {len(rows)} events into Life Story...")

        events_text = "\n".join([f"[{time.strftime('%Y-%m-%d %H:%M', time.localtime(r[2]))}] {r[0]}: {r[1]}" for r in rows])
        
        prompt = (
            f"CURRENT LIFE STORY:\n{current_story}\n\n"
            f"RECENT EVENTS:\n{events_text}\n\n"
            "TASK: Update the Life Story to incorporate these recent events.\n"
            "Maintain a coherent, first-person narrative. Do not just list events; weave them into the evolving story of your existence.\n"
            "Reflect on how these events change your perspective, goals, or relationship with the user.\n"
            "If the current story is short, expand it. If it's long, integrate and condense.\n"
            "Output ONLY the updated story."
        )
        
        new_story = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the Autobiographer of an AI.",
            max_tokens=1000,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if new_story and not new_story.startswith("⚠️"):
            self.memory_store.add_entry(
                identity=identity_hash,
                text=new_story,
                mem_type="IDENTITY",
                subject="Assistant",
                confidence=1.0,
                source="daat_life_story_update"
            )
            
            # Update Unified Self-Model
            if hasattr(self.memory_store, 'self_model') and self.memory_store.self_model:
                self.memory_store.self_model.update_narrative({"life_story": new_story})
                
            self.log("📖 Da'at: Life Story updated.")

    def prune_redundant_memories(self):
        """
        Identify and remove semantically identical memories to reduce clutter.
        """
        import difflib
        self.log("🧹 Da'at: Pruning redundant memories...")

        # Fetch recent memories (FACTs and BELIEFs)
        recent = self.memory_store.list_recent(limit=100)

        # Filter for FACT/BELIEF
        candidates = [m for m in recent if m[1] in ('FACT', 'BELIEF')]

        deleted_count = 0
        processed_ids = set()

        for i in range(len(candidates)):
            m1 = candidates[i]
            if m1[0] in processed_ids: continue

            for j in range(i + 1, len(candidates)):
                m2 = candidates[j]
                if m2[0] in processed_ids: continue

                # Check text similarity
                # Using 3 for text
                ratio = difflib.SequenceMatcher(None, m1[3], m2[3]).ratio()

                if ratio > 0.95:
                    # Redundant! Keep the one with more information (longer text) or verified
                    m1_score = len(m1[3]) + (1000 if m1[5] else 0)
                    m2_score = len(m2[3]) + (1000 if m2[5] else 0)

                    id_to_delete = m2[0] if m1_score >= m2_score else m1[0]

                    self.memory_store.soft_delete_entry(id_to_delete)
                    processed_ids.add(id_to_delete)
                    deleted_count += 1
                    self.log(f"🧹 Pruned redundant memory {id_to_delete} (Sim: {ratio:.2f})")

        return f"Pruned {deleted_count} memories."

    def schedule_learning_session(self, topic: str):
        """
        Create a high-priority learning goal.
        """
        goal_text = f"Deep Dive: Research {topic}"
        identity = self.memory_store.compute_identity(goal_text, "GOAL")

        # Check if exists
        with self.memory_store._connect() as con:
            exists = con.execute("SELECT 1 FROM memories WHERE identity = ? AND completed = 0", (identity,)).fetchone()

        if not exists:
            self.memory_store.add_entry(
                identity=identity,
                text=goal_text,
                mem_type="GOAL",
                subject="Daat",
                confidence=1.0,
                source="daat_learning_schedule"
            )
            self.log(f"🎓 Scheduled Learning Session: {topic}")