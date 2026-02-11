import time
import random
import re
from typing import Dict, Callable, Optional, List, Any

from lm import run_local_lm, extract_memory_candidates, DEFAULT_MEMORY_EXTRACTOR_PROMPT, _parse_json_array_loose, compute_embedding

# Specialized prompt for extracting insights from internal monologue
DAYDREAM_EXTRACTOR_PROMPT = (
    "Extract insights, goals, facts, and preferences from the Assistant's internal monologue. "
    "Return ONLY a valid JSON array.\n\n"
    "Memory Types:\n"
    "- GOAL: Specific, actionable objectives for the Assistant (e.g., 'Assistant plans to cross-reference X with Y'). Do NOT extract general statements like 'Future research should...' as GOALs; classify them as BELIEFS or FACTS instead.\n"
    "- FACT: Objective truths derived from documents or reasoning\n"
    "- BELIEF: Opinions, convictions, hypotheses, or research insights\n"
    "- REFUTED_BELIEF: Ideas explicitly proven false or rejected. (e.g. 'Assistant rejected the idea that...')\n"
    "- PREFERENCE: Personal likes/dislikes ONLY (e.g., 'Assistant enjoys sci-fi'). DO NOT use for research suggestions, hypotheses, or document relevance.\n\n"
    "Rules:\n"
    "1. Extract from the Assistant's text.\n"
    "2. Each object MUST have: \"type\", \"subject\" (must be 'Assistant'), \"text\".\n"
    "3. Use DOUBLE QUOTES for all keys and string values.\n"
    "4. Max 5 memories.\n"
    "5. MAKE MEMORIES SELF-CONTAINED: Replace pronouns like 'This', 'These', 'It' with specific nouns. Ensure the text makes sense without the surrounding context.\n"
    "6. Return ONLY the JSON array. If no new memories, return [].\n"
)

DAYDREAM_INSTRUCTION = (
    "Analyze the Internal Monologue above. "
    "Extract key insights as FACT, BELIEF, GOAL, or PREFERENCE memories for the Assistant. "
    "Format as JSON objects with keys: 'type', 'subject' (must be 'Assistant'), 'text'. "
    "Ensure the text includes the source document filename if mentioned. "
    "CRITICAL: Replace pronouns (e.g., 'This', 'These', 'It') with specific nouns to make the memory self-contained. "
    "Return ONLY a valid JSON array. Do not invent sources."
)

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
        log_fn: Callable[[str], None] = print,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        get_gap_topic_fn: Optional[Callable[[], Optional[str]]] = None
    ):
        self.memory_store = memory_store
        self.document_store = document_store
        self.get_settings = get_settings_fn
        self.log = log_fn
        self.stop_check = stop_check_fn or (lambda: False)
        self.get_gap_topic = get_gap_topic_fn

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
                context += "Active Goals:\n" + "\n".join([f"- {t}" for _, s, t, _ in goals]) + "\n"
            
            if refuted_memories:
                context += "Refuted Beliefs (FALSE - Do NOT regenerate):\n" + "\n".join([f"- {m[3]}" for m in refuted_memories[:5]]) + "\n"
            
            if recent_memories:
                context += "Recent Memories:\n" + "\n".join([f"- [{m[1]}] {m[3][:150]}..." for m in recent_memories]) + "\n"
            
            # Decision Phase
            if impulse == "read":
                response = "[READ_RANDOM]"
            elif topic:
                # If a topic is explicitly set, default to reading about it
                response = "[READ_RANDOM]"
            elif not goals and not recent_memories:
                # Bootstrap: If nothing in memory, try to read
                response = "[READ_RANDOM]"
            else:
                decision_prompt = (
                    "You are the AI Assistant reflecting on your internal state. "
                    "Review the provided Context (Goals and Memories). "
                    "You have access to a library of documents. "
                    "Decide whether to reflect on existing knowledge or read a new document for inspiration. "
                    "To read a random document, output: [READ_RANDOM] "
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
            
            # Execution Phase
            if "READ_RANDOM" in response:
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
                                    self.log(f"⚠️ Topic search failed: {e}")

                        if not selected_doc:
                            selected_doc = random.choice(docs)
                            
                        doc_id, filename = selected_doc[0], selected_doc[1]
                        reading_filename = filename
                        
                        # Get chunks
                        chunks = self.document_store.get_document_chunks(doc_id)
                        if chunks:
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
                        thought = "I wanted to read, but the library is empty."
                else:
                    thought = "I wanted to read, but I have no document store."
            
            if self.stop_check():
                return None

            # Safety check: If thought is still the command (execution failed or skipped), abort
            if "READ_RANDOM" in thought and len(thought) < 50:
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
            candidates = _parse_json_array_loose(thought)
            
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

                candidates = extract_memory_candidates(
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

    def seek_novelty(self, daat_instance, memory_store) -> bool:
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
        with memory_store._connect() as con:
            recent_goal = con.execute(
                "SELECT 1 FROM memories WHERE type='GOAL' AND text LIKE ? AND created_at > ?", 
                (f"%{topic}%", int(time.time()) - 3600) # 1 hour cooldown
            ).fetchone()
        if recent_goal:
            self.log(f"🌊 Chokmah: Skipping topic '{topic}' (Recently targeted).")
            return False

        # 2. Formulate a Curiosity Goal
        goal_text = f"Research the detailed mechanisms and recent findings regarding '{topic}'"
        
        # 3. Check if we already have this goal (deduplication)
        with memory_store._connect() as con:
            exists = con.execute(
                "SELECT 1 FROM memories WHERE type='GOAL' AND text = ? AND completed=0", 
                (goal_text,)
            ).fetchone()
        
        if exists: return False

        # 4. Inject the Goal (Spark of Curiosity)
        self.log(f"💡 Chokmah: Curiosity sparked! Created autonomous goal for '{topic}'")
        identity = memory_store.compute_identity(goal_text, "GOAL")
        memory_store.add_entry(
            identity=identity,
            text=goal_text,
            mem_type="GOAL",
            subject="Assistant",
            confidence=1.0,
            source="autonomous_curiosity"
        )
        return True

    def study_archives(self, document_store, memory_store) -> bool:
        """
        ABILITY: Study.
        Picks a random unread chunk from the Document Store and learns facts from it.
        """
        try:
            # 1. Grab a random slice of knowledge directly from DB
            with document_store._connect() as con:
                row = con.execute("""
                    SELECT c.text, d.filename 
                    FROM chunks c 
                    JOIN documents d ON c.document_id = d.id 
                    ORDER BY RANDOM() LIMIT 1
                """).fetchone()
            
            if not row: 
                self.log("🌊 Chokmah: Library is empty.")
                return False
                
            text, filename = row
            
            # 2. Read and Analyze (The "Studying" part)
            self.log(f"📖 Chokmah: Autonomously reading '{filename}'...")
            
            prompt = (
                f"You are studying the document '{filename}'.\n"
                f"Read this excerpt:\n\n{text[:1500]}\n\n"
                f"Extract 1-3 distinct, self-contained FACTS that are worth remembering.\n"
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
                        text = f"{f['text']} [Source: {filename}]"
                        identity = memory_store.compute_identity(text, "FACT")
                        memory_store.add_entry(
                            identity=identity,
                            text=text,
                            mem_type="FACT",
                            subject="Assistant",
                            confidence=0.95,
                            source=f"autonomous_reading:{filename}"
                        )
                    self.log(f"💡 Chokmah: Learned {len(facts)} new facts from '{filename}'.")
                    return True
                except:
                    pass
            return False

        except Exception as e:
            self.log(f"❌ Study session failed: {e}")
            return False

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
        return True