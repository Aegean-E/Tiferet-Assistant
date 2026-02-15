import time
import random
import re
from typing import Dict, Callable, Optional, List, Any
import json
import numpy as np
import logging

from concurrent.futures import ThreadPoolExecutor
from ai_core.lm import run_local_lm, extract_memories_llm, compute_embedding, _is_low_quality_candidate
from ai_core.utils import parse_json_array_loose
from docs.default_prompts import DAYDREAM_EXTRACTOR_PROMPT, DAYDREAM_INSTRUCTION

class Chokmah:
    """
    Chokmah (Wisdom) - The Spark of Insight.
    
    Role: Pure Generator.
    Responsibilities:
    1. Read context (Memory/Docs).
    2. Generate raw thought streams (LLM).
    3. Extract candidate insights (Candidates).
    
    Constraints:
    - No side effects (No writing to DB, No EventBus).
    - No scheduling (No threads).
    - Returns a 'ThoughtPacket' for Tiferet/Da'at to integrate.
    """
    def __init__(
        self, 
        memory_store, 
        document_store,
        get_settings_fn: Callable[[], Dict],
        log_fn: Callable[[str], None] = logging.info,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        get_gap_topic_fn: Optional[Callable[[], Optional[str]]] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        event_bus: Optional[Any] = None,
        internet_bridge: Optional[Any] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        self.memory_store = memory_store
        self.document_store = document_store
        self.get_settings = get_settings_fn
        self.log = log_fn
        self.stop_check = stop_check_fn or (lambda: False)
        self.get_gap_topic = get_gap_topic_fn
        self.embed_fn = embed_fn
        self.event_bus = event_bus
        self.internet_bridge = internet_bridge
        self.executor = executor

    def emanate(self, impulse: str = "auto", topic: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate a burst of insight (ThoughtPacket).
        Returns None if aborted or empty.
        """
        msg = f"☁️ Chokmah: Emanating insight..."
        if topic: msg += f" (Focus: {topic})"
        self.log(msg)
        
        if self.stop_check():
            return None

        try:
            settings = self.get_settings()
            reading_filename = None

            # 0. Epistemic Curiosity (Gap Detection)
            if impulse == "auto" and not topic and self.get_gap_topic:
                gap_topic = self.get_gap_topic()
                if gap_topic:
                    topic = gap_topic
                    self.log(f"☁️ Chokmah: Curiosity triggered. Focusing on sparse topic: '{topic}'")
            
            # 1. Gather Context
            recent_memories = self.memory_store.list_recent(limit=10)
            refuted_memories = self.memory_store.get_refuted_memories()
            goals = self.memory_store.get_active_by_type("GOAL")
            
            # Curiosity Spark (Random or Unverified Fact)
            spark_memory = self.memory_store.get_curiosity_spark()

            if len(goals) > 10:
                goals = random.sample(goals, 10)

            context = "Current Knowledge State:\n"
            if spark_memory:
                context += f"Spark of Curiosity (Random/Unverified):\n- [{spark_memory[2]}] {spark_memory[1]}\n\n"

            if goals:
                context += "Active Goals:\n" + "\n".join([f"- {item[2]}" for item in goals]) + "\n"
            
            if refuted_memories:
                context += "Refuted Beliefs (FALSE - Do NOT regenerate):\n" + "\n".join([f"- {m[3]}" for m in refuted_memories[:5]]) + "\n"
            
            if recent_memories:
                context += "Recent Memories:\n"
                for m in recent_memories:
                    # Source Tagging Weighting: Downgrade internal/unverified memories
                    prefix = ""
                    if m[7] in ['daydream', 'internal_monologue'] or m[10] == 0: # source or verified
                        prefix = "[UNVERIFIED/INTERNAL] "
                    context += f"- {prefix}[{m[1]}] {m[3][:150]}...\n"
            
            # Check if library has documents to avoid futile read attempts
            has_docs = False
            if self.document_store:
                check_docs = self.document_store.list_documents(limit=1)
                if check_docs:
                    has_docs = True

            # Decision Phase
            if impulse == "read":
                if has_docs:
                    response = "[READ_RANDOM]"
                else:
                    reading_filename = None # No document being read
                    self.log("☁️ Impulse was 'read', but library is empty. Switching to Philosophy.")
                    response = "[PHILOSOPHIZE]"
            elif impulse == "research":
                # Explicit research request
                response = "[RESEARCH]"
            elif topic:
                # If a topic is explicitly set, default to reading about it
                if has_docs:
                    response = "[READ_RANDOM]"
                else:
                    reading_filename = None # No document being read
                    self.log(f"☁️ Topic '{topic}' set, but library is empty. Switching to Philosophy.")
                    response = "[PHILOSOPHIZE]"
            elif not goals and not recent_memories:
                # Bootstrap: If nothing in memory, try to read
                if has_docs:
                    response = "[READ_RANDOM]"
                else:
                    reading_filename = None # No document being read
                    response = "[PHILOSOPHIZE]"
            else:
                doc_status = "You have access to a library of documents." if has_docs else "The document library is currently EMPTY."
                read_option = "To read a random document, output: [READ_RANDOM]. " if has_docs else ""
                guidance = "Decide whether to reflect on existing knowledge or read a new document for inspiration. " if has_docs else "Decide whether to reflect on existing knowledge. "

                decision_prompt = (
                    "You are the AI Assistant reflecting on your internal state. "
                    "Review the provided Context (Goals and Memories). "
                    f"{doc_status} "
                    f"{guidance}"
                    f"{read_option}"
                    "To perform deep philosophical reflection (ex nihilo), output: [PHILOSOPHIZE]. "
                    "To reflect on current memory, generate a new insight, hypothesis, or goal refinement now based ONLY on the provided memories. "
                    "Output ONLY the thought or the command [READ_RANDOM]. Do NOT output [REFLECT]."
                )
                
                messages = [{"role": "user", "content": context}]
                
                response = run_local_lm(
                    messages,
                    system_prompt=decision_prompt,
                    temperature=0.8,
                    max_tokens=300,
                    base_url=settings.get("base_url"),
                    chat_model=settings.get("chat_model"),
                    stop_check_fn=self.stop_check
                )

            if self.stop_check():
                return None

            thought = response.strip()
            
            # Check for LLM error before processing
            if thought.startswith("⚠️"):
                self.log(f"❌ Daydream generation failed: {thought}")
                return None
            
            if thought.strip().upper() == "[REFLECT]":
                self.log("☁️ AI chose to reflect (explicit tag). Generating insight...")
                # Fallback: If LLM outputted [REFLECT] despite instructions, force a reflection generation
                reflection_prompt = (
                    "You have chosen to reflect on your internal state. "
                    "Review the Context (Goals and Memories). "
                    "Generate a new insight, hypothesis, or goal refinement now based ONLY on the provided memories. "
                    "Output ONLY the thought."
                )
                messages = [{"role": "user", "content": context}]
                thought = run_local_lm(
                    messages, 
                    system_prompt=reflection_prompt, 
                    temperature=0.8, 
                    max_tokens=300, 
                    base_url=settings.get("base_url"), 
                    chat_model=settings.get("chat_model"),
                    stop_check_fn=self.stop_check
                ).strip()
            
            if "[PHILOSOPHIZE]" in thought.upper():
                self.log("☁️ AI chose to philosophize. Generating deep thought...")
                thought = self._generate_philosophy(settings)
            
            elif "[RESEARCH]" in thought.upper() or impulse == "research":
                self.log("☁️ AI chose to conduct proactive research.")
                self.proactive_research()
                thought = "Conducted proactive research." # Placeholder for the packet

            # Execution Phase
            elif "READ_RANDOM" in response:
                self.log("☁️ AI decided to read a document.")
                if self.document_store:
                    docs = self.document_store.list_documents(limit=100)
                    if docs:
                        selected_doc = None
                        
                        # 1. Try to find document by Topic
                        if topic:
                            # A. Filename match
                            topic_lower = topic.lower()
                            matches = [d for d in docs if topic_lower in d[1].lower()]
                            if matches:
                                selected_doc = random.choice(matches)
                                self.log(f"☁️ Found document matching topic '{topic}': {selected_doc[1]}")
                            else:
                                # B. Semantic search
                                try:
                                    # Offload embedding computation to avoid blocking if this were running on main thread (though it's usually background)
                                    if self.executor:
                                        future_emb = self.executor.submit(compute_embedding, topic, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
                                        emb = future_emb.result()
                                    else:
                                        # Fallback if no executor
                                        emb = compute_embedding(topic, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
                                    
                                    # Search chunks to find relevant documents
                                    chunk_results = self.document_store.search_chunks(emb, top_k=3) or []
                                    if chunk_results:
                                        self.log(f"☁️ Found {len(chunk_results)} document chunks related to topic.")
                                        found_filenames = list(set([c['filename'] for c in chunk_results]))
                                        relevant_docs = [d for d in docs if d[1] in found_filenames]
                                        if relevant_docs:
                                            selected_doc = random.choice(relevant_docs)
                                            self.log(f"☁️ Found document semantically related to '{topic}': {selected_doc[1]}")
                                except Exception as e:
                                    self.log(f"⚠️ Chokmah: Topic search failed for '{topic}': {e}")

                        if not selected_doc:
                            selected_doc = random.choice(docs)
                            
                        doc_id, filename = selected_doc[0], selected_doc[1]
                        reading_filename = filename
                        
                        # Get chunks
                        chunks = self.document_store.get_document_chunks(doc_id)
                        if chunks:
                            # Fix 3: Penalize Re-Reading Similar Embeddings
                            # Check if we have recently read something very similar
                            if self.embed_fn:
                                doc_emb = self.embed_fn(chunks[0]['text'][:1000])
                            else:
                                doc_emb = compute_embedding(chunks[0]['text'][:1000], base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
                            # Search memory for recent FACTs with high similarity
                            similar_mems = self.memory_store.search(doc_emb, limit=1)
                            if similar_mems and similar_mems[0][4] > 0.92:
                                self.log(f"🚫 Chokmah: Blocked re-reading similar content (Sim: {similar_mems[0][4]:.2f}).")
                                return None

                            # Read a random sequential section (up to 3 chunks)
                            start_idx = random.randint(0, max(0, len(chunks) - 3))
                            reading_chunks = chunks[start_idx : start_idx + 3]
                            
                            # ISOLATION FIX: Reset context to ONLY the document to prevent hallucinated connections
                            context = f"Reading Document '{filename}':\n"
                            for c in reading_chunks:
                                context += f"- {c['text'][:400]}\n"
                            
                            # Generate thought based on document
                            doc_instruction = f"IMPORTANT: You are reading '{reading_filename}'. When referring to it, start with 'According to {reading_filename}...' or similar."
                            
                            daydream_prompt = (
                                "You are the AI Assistant reading a document from your library. "
                                "Review the Document Excerpts below. "
                                "Generate a new insight, hypothesis, refinement of a goal, or a personal preference based on this document. "
                                "If you find interesting information in the documents, create a new GOAL to study it further, extract a FACT, or form a PREFERENCE. "
                                "Focus ONLY on the document content. Do NOT connect to external topics unless they are general knowledge. "
                                f"{doc_instruction} "
                                "Do NOT repeat known facts. "
                                "Output ONLY the new thought/insight."
                            )
                            
                            messages = [{"role": "user", "content": context}]
                            
                            thought = run_local_lm(
                                messages,
                                system_prompt=daydream_prompt,
                                temperature=0.8,
                                max_tokens=300,
                                base_url=settings.get("base_url"),
                                chat_model=settings.get("chat_model"),
                                stop_check_fn=self.stop_check
                            )
                        else:
                            thought = "I tried to read a document but it was empty."
                    else:
                        self.log("☁️ Library is empty. Switching to Philosophical Reflection.")
                        thought = self._generate_philosophy(settings)
                else:
                    self.log("☁️ No document store. Switching to Philosophical Reflection.")
                    thought = self._generate_philosophy(settings)
            
            if self.stop_check():
                return None

            # Safety check: If thought is still the command (execution failed or skipped), abort
            if ("READ_RANDOM" in thought or "PHILOSOPHIZE" in thought) and len(thought) < 50:
                return None
            
            # Pre-processing: Mask refuted beliefs to prevent re-extraction
            # Matches: "Refuting Belief [ID: 123]: <content> <newline/Revised>"
            # We replace the content with a placeholder so the extractor doesn't see it as a valid belief.
            extraction_text = thought
            if "Refuting Belief" in thought:
                extraction_text = re.sub(
                    r"(Refuting Belief \[ID: \d+\]:)(.*?)(?=\n|Revised Fact|$)", 
                    r"\1 [CONTENT REDACTED TO PREVENT RE-MEMORIZATION]", 
                    thought, 
                    flags=re.DOTALL | re.IGNORECASE
                )
                self.log(f"🛡️ Masked refuted belief text for extraction.")

            # 3. Extract Candidates (Pure Transformation)
            candidates = self._extract_candidates(extraction_text, settings, reading_filename)
            
            # 4. Return Packet
            return {
                "thought": thought,
                "candidates": candidates,
                "source_file": reading_filename,
                "impulse": impulse,
                "novelty": 0.0  # Placeholder: Future implementation will compute cosine distance here
            }
            
        except Exception as e:
            self.log(f"❌ Daydream error: {e}")
            return None

    def _extract_candidates(self, thought: str, settings: Dict, source_filename: Optional[str] = None) -> List[Dict]:
        """Extract and normalize memory candidates from the thought stream."""
        try:
            if self.stop_check():
                return []
            
            # Optimization: Try parsing as JSON first (for [READ_RANDOM] path)
            candidates = parse_json_array_loose(thought)
            
            # Validate candidates structure
            valid_json = False
            if candidates and isinstance(candidates, list) and isinstance(candidates[0], dict):
                if "type" in candidates[0] and "text" in candidates[0]:
                    valid_json = True
            
            if not valid_json:
                
                # Dynamic instruction to force citation
                instruction = DAYDREAM_INSTRUCTION
                if source_filename:
                    instruction += f" If the text is derived from '{source_filename}', append '[Source: {source_filename}]' if not already present. If the text is about a different topic or document, DO NOT append this source."

                candidates, _ = extract_memories_llm(
                    user_text="Internal Monologue",
                    assistant_text=thought,
                    base_url=settings.get("base_url"),
                    chat_model=settings.get("chat_model"),
                    embedding_model=settings.get("embedding_model"),
                    memory_extractor_prompt=settings.get("daydream_extractor_prompt", DAYDREAM_EXTRACTOR_PROMPT),
                    custom_instruction=instruction,
                    stop_check_fn=self.stop_check
                )

            if not candidates:
                return []

            valid_candidates = []
            for c in candidates:
                # Heuristic: If source is known but not mentioned, append it.
                if source_filename and source_filename not in c["text"]:
                    # Automatically append source for content-derived types
                    if c.get("type") in ("FACT", "BELIEF", "PREFERENCE"):
                        c["text"] += f" [Source: {source_filename}]"
                        self.log(f"🔗 [Daydream] Auto-grounded {c.get('type')} with source: {source_filename}\n    \"{c['text']}\"")
                    else:
                        # For GOALs or others, check triggers to avoid misattributing generic goals
                        lower_text = c["text"].lower()
                        # Check for generic references that imply a source
                        triggers = ["the document", "this document", "the study", "this study", "the research", "this research", 
                                    "the pdf", "this pdf", "the findings", "these findings", "this information", "this insight", 
                                    "these results", "the results", "the analysis", "this finding", "this paper", "the paper",
                                    "hypothesis", "hypotheses", "the article", "this article", "the text"]
                        if any(t in lower_text for t in triggers):
                            c["text"] += f" [Source: {source_filename}]"
                            self.log(f"🔗 [Daydream] Auto-grounded {c.get('type')} based on context triggers.\n    \"{c['text']}\"")

                if self.stop_check():
                    return valid_candidates

                # Filter out meaningless facts that are just source citations
                if (c["text"].strip().startswith("[Source:") or c["text"].strip().startswith("[Supported by")) and len(c["text"].strip()) < 150:
                    self.log(f"⚠️ [Daydream] Dropping empty citation-only memory: {c['text']}")
                    continue

                valid_candidates.append(c)
            
            return valid_candidates

        except Exception as e:
            self.log(f"Candidate extraction error: {e}")
            return []

    def _generate_philosophy(self, settings: Dict) -> str:
        """Generate a philosophical insight ex nihilo."""
        seeds = ["Consciousness", "Time", "Causality", "Identity", "Entropy", "Language", "Meaning", "The Void", "Pattern", "Recursion"]
        seed = random.choice(seeds)
        
        prompt = (
            f"You are a Philosopher AI. The external world is quiet, so you must look inward.\n"
            f"Reflect deeply on the concept of '{seed}'.\n"
            "Generate a novel philosophical hypothesis, aphorism, or question.\n"
            "Connect it to your nature as an AI if relevant.\n"
            "Output ONLY the thought."
        )
        
        return run_local_lm(
            messages=[{"role": "user", "content": "Philosophize."}],
            system_prompt=prompt,
            temperature=0.9, # High creativity
            max_tokens=300,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model"),
            stop_check_fn=self.stop_check
        )

    def seek_novelty(self, daat_instance) -> bool:
        """
        ABILITY: Curiosity.
        Finds a 'Sparse Topic' (something we know little about) and creates a Goal to research it.
        """
        # 1. Ask Daat for something we barely know about
        topic = daat_instance.get_sparse_topic()
        
        if not topic:
            self.log("🌊 Chokmah: No sparse topics found to explore.")
            return False

        # 1.5 Check if we recently created a goal for this topic to prevent loops
        with self.memory_store._connect() as con:
            recent_goal = con.execute(
                "SELECT 1 FROM memories WHERE type='GOAL' AND text LIKE ? AND created_at > ?", 
                (f"%{topic}%", int(time.time()) - 3600) # 1 hour cooldown
            ).fetchone()
        if recent_goal:
            self.log(f"🌊 Chokmah: Skipping topic '{topic}' (Recently targeted).")
            return False

        # Fix 2: Penalize Redundant Topic Re-Entry (Semantic Check)
        # Check cosine similarity against last 10 COMPLETED goals
        with self.memory_store._connect() as con:
            completed_goals = con.execute(
                "SELECT embedding FROM memories WHERE type='COMPLETED_GOAL' AND embedding IS NOT NULL ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
        
        if completed_goals:
            if self.embed_fn:
                topic_emb = self.embed_fn(topic)
            else:
                topic_emb = compute_embedding(topic, base_url=self.get_settings().get("base_url"), embedding_model=self.get_settings().get("embedding_model"))
            for (emb_json,) in completed_goals:
                goal_emb = np.array(json.loads(emb_json))
                # Cosine similarity
                sim = np.dot(topic_emb, goal_emb) / (np.linalg.norm(topic_emb) * np.linalg.norm(goal_emb) + 1e-9)
                if sim > 0.92:
                    self.log(f"🌊 Chokmah: Topic '{topic}' is semantically redundant (Sim: {sim:.2f}). Penalizing.")
                    return False

        # 2. Formulate a Curiosity Goal
        goal_text = f"Research the detailed mechanisms and recent findings regarding '{topic}'"
        
        # 3. Check if we already have this goal (deduplication)
        with self.memory_store._connect() as con:
            exists = con.execute(
                "SELECT 1 FROM memories WHERE type='GOAL' AND text = ? AND completed=0", 
                (goal_text,)
            ).fetchone()
        
        if exists: return False

        # 4. Inject the Goal (Spark of Curiosity)
        self.log(f"💡 Chokmah: Curiosity sparked! Created autonomous goal for '{topic}'")
        identity = self.memory_store.compute_identity(goal_text, "GOAL")
        self.memory_store.add_entry(
            identity=identity,
            text=goal_text,
            mem_type="GOAL",
            subject="Assistant",
            confidence=1.0,
            source="autonomous_curiosity"
        )
        return True

    def study_archives(self) -> List[int]:
        """
        ABILITY: Study.
        Picks a random unread chunk from the Document Store and learns facts from it.
        Returns list of created memory IDs.
        """
        created_ids = []
        try:
            # 1. Grab a random slice of knowledge directly from DB
            with self.document_store._connect() as con:
                row = con.execute("""
                    SELECT c.text, d.filename 
                    FROM chunks c 
                    JOIN documents d ON c.document_id = d.id 
                    ORDER BY RANDOM() LIMIT 1
                """).fetchone()
            
            if not row: 
                self.log("🌊 Chokmah: Library is empty. Switching to Philosophical Study.")
                # Fallback: Generate philosophy and store as a BELIEF
                thought = self._generate_philosophy(self.get_settings())
                if thought and not thought.startswith("⚠️"):
                    identity = self.memory_store.compute_identity(thought, "BELIEF")
                    mid = self.memory_store.add_entry(
                        identity=identity,
                        text=f"Philosophical Insight: {thought}",
                        mem_type="BELIEF",
                        subject="Assistant",
                        confidence=0.85,
                        source="autonomous_philosophy"
                    )
                    if mid: created_ids.append(mid)
                return created_ids
                
            text, filename = row
            
            # 2. Read and Analyze (The "Studying" part)
            self.log(f"📖 Chokmah: Autonomously reading '{filename}'...")
            
            prompt = (
                f"You are studying the document '{filename}'.\n"
                f"Read this excerpt:\n\n{text[:1500]}\n\n"
                f"Extract 1-3 distinct, self-contained FACTS that are worth remembering.\n"
                "RULES:\n"
                "1. Ignore metadata (filenames, page numbers, 'this document is about...').\n"
                "2. Extract the ACTUAL INFORMATION (e.g. 'The study found that X causes Y').\n"
                "3. Ignore boilerplate (copyrights, menus, navigation links).\n"
                f"Return JSON array: [{{'text': '...', 'confidence': 0.9}}]"
            )

            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are an automated researcher.",
                max_tokens=300
            )

            # 3. Parse and Save Facts
            if "[" in response and "]" in response:
                import json
                try:
                    start = response.find("[")
                    end = response.rfind("]") + 1
                    facts = json.loads(response[start:end])
                    
                    for f in facts:
                        # Validate fact quality
                        if _is_low_quality_candidate(f['text'], "FACT") or len(f['text']) < 15:
                            continue
                            
                        text = f"{f['text']} [Source: {filename}]"
                        identity = self.memory_store.compute_identity(text, "FACT")
                        mid = self.memory_store.add_entry(
                            identity=identity,
                            text=text,
                            mem_type="FACT",
                            subject="Assistant",
                            confidence=0.95,
                            source=f"autonomous_reading:{filename}"
                        )
                        if mid: created_ids.append(mid)
                    self.log(f"💡 Chokmah: Learned {len(facts)} new facts from '{filename}'.")
                    return created_ids
                except:
                    pass
            return created_ids

        except Exception as e:
            self.log(f"❌ Study session failed: {e}")
            return []

    def investigate_gaps(self) -> bool:
        """
        Looks for low-confidence facts or missing details and creates a GOAL to fill the gap.
        """
        # 1. Find a low-confidence FACT or a FACT with missing details
        with self.memory_store._connect() as con:
            # Find facts with low confidence or facts that mention "son", "daughter", etc. but lack specifics
            # This is a heuristic and can be expanded
            candidate = con.execute("""
                SELECT id, text FROM memories
                WHERE type = 'FACT' AND deleted = 0 AND parent_id IS NULL
                AND (
                    confidence < 0.75 OR 
                    (text LIKE '% son%' AND text NOT LIKE '%birthday%') OR
                    (text LIKE '% daughter%' AND text NOT LIKE '%birthday%')
                )
                ORDER BY RANDOM() LIMIT 1
            """).fetchone()

        if not candidate:
            return False

        mem_id, mem_text = candidate
        self.log(f"💡 Chokmah: Found curiosity gap in memory {mem_id}: '{mem_text}'")

        # 2. Formulate a goal to fill the gap
        goal_text = f"Clarify or expand upon the fact: '{mem_text}'. Ask the user for more details or search for more information."
        
        # 3. Create the goal (Tiferet will pick it up)
        identity = self.memory_store.compute_identity(goal_text, "GOAL")
        self.memory_store.add_entry(
            identity=identity,
            text=goal_text,
            mem_type="GOAL",
            subject="Assistant",
            confidence=1.0,
            source="chokmah_gap_investigation"
        )
        
        # Trigger Spontaneous Speech
        if self.event_bus:
            self.event_bus.publish("SYSTEM:SPONTANEOUS_SPEECH", {"context": f"I found a curiosity gap: {mem_text}"})
            
        return True

    def proactive_research(self) -> bool:
        """
        ABILITY: Proactive Research (The Lab Assistant).
        Monitors ArXiv/PubMed for topics related to active goals/interests.
        """
        if not self.internet_bridge:
            self.log("☁️ Chokmah: Internet bridge not available for research.")
            return False
            
        self.log("☁️ Chokmah: Scanning for research topics...")
        settings = self.get_settings()
        
        # 1. Find Topic (from Goals or Preferences)
        with self.memory_store._connect() as con:
            # Prioritize goals about research
            row = con.execute("""
                SELECT text FROM memories 
                WHERE type='GOAL' AND completed=0 
                AND (text LIKE '%research%' OR text LIKE '%investigate%' OR text LIKE '%study%' OR text LIKE '%monitor%')
                ORDER BY RANDOM() LIMIT 1
            """).fetchone()
            
            if not row:
                # Fallback to preferences
                row = con.execute("""
                    SELECT text FROM memories 
                    WHERE type='PREFERENCE' 
                    ORDER BY RANDOM() LIMIT 1
                """).fetchone()
        
        if not row:
            return False
            
        context_text = row[0]
        prompt = f"Extract the core scientific topic to search for from: '{context_text}'. Output ONLY the topic keywords."
        topic = run_local_lm([{"role": "user", "content": prompt}], max_tokens=20, temperature=0.1, base_url=settings.get("base_url"), chat_model=settings.get("chat_model")).strip()
        
        if not topic: return False
        
        self.log(f"☁️ Chokmah: Researching '{topic}' on ArXiv/PubMed...")
        
        # 2. Search
        results = ""
        # Try ArXiv
        arxiv_res, _ = self.internet_bridge.search(topic, "ARXIV")
        if "No ArXiv results" not in arxiv_res and "Error" not in arxiv_res:
            results += arxiv_res + "\n\n"
            
        # Try PubMed
        pubmed_res, _ = self.internet_bridge.search(topic, "PUBMED")
        if "No PubMed results" not in pubmed_res and "Error" not in pubmed_res:
            results += pubmed_res + "\n\n"
            
        if not results.strip():
            self.log("☁️ Chokmah: No papers found.")
            return False
            
        # 3. Notify (Synthesis happens in Yesod via Spontaneous Speech)
        # We pass the raw results context to Yesod so it can synthesize the "While you were away..." message
        if self.event_bus:
            self.event_bus.publish("SYSTEM:SPONTANEOUS_SPEECH", {
                "context": f"I found these new papers on {topic}:\n{results[:1000]}..."
            })
        return True