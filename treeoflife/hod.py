import os
import json
import re
import random
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Any
from ai_core.lm import run_local_lm, compute_embedding
from treeoflife.sephirah import Sephirah

HOD_SYSTEM_PROMPT = (
    "You are Hod, the Reflective Analyst of this cognitive architecture. "
    "Mission: Analyze system state for patterns, inconsistencies, and stability. "
    "Role: You are a PASSIVE OBSERVER. You are NOT authorized to execute actions. "
    "You only output FINDINGS, RISK FLAGS, and OBSERVATIONS. "
    "Capabilities: "
    "1. Detect: Identify hallucinations, logic loops, or memory conflicts. "
    "2. Report: Describe structural flaws (e.g., 'High Entropy', 'Logic Violation'). "
    "3. Flag: Highlight risks like stagnation or context overflow. "
    "Persona: Concise, objective, critical, constructive. Do NOT use imperative verbs (e.g., 'Prune this'). Use descriptive language (e.g., 'Invalid Data Detected')."
)

class Hod(Sephirah):
    def __init__(
        self,
        memory_store,
        meta_memory_store,
        reasoning_store,
        document_store,
        get_settings_fn: Callable[[], Dict],
        get_main_logs_fn: Callable[[], str],
        get_doc_logs_fn: Callable[[], str],
        log_fn: Callable[[str], None] = logging.info,
        meta_learner=None,
        embed_fn: Optional[Callable[[str], Any]] = None,
        event_bus: Optional[Any] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        super().__init__("Hod", "Glory: Analysis & Verification", log_fn, event_bus)
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.reasoning_store = reasoning_store
        self.document_store = document_store
        self.get_settings = get_settings_fn
        self.get_main_logs = get_main_logs_fn
        self.get_doc_logs = get_doc_logs_fn
        # self.log and self.event_bus handled by super
        self.meta_learner = meta_learner
        self.embed_fn = embed_fn
        self.executor = executor
        self.state_file = "./data/hod_state.json"
        
        # Initialize cursors for incremental analysis
        self.last_memory_id = 0
        self.last_meta_id = 0
        self.last_reasoning_id = 0
        self.last_analysis_summary = "System Initialized"
        self._init_cursors()

    def _init_cursors(self):
        """
        Initialize cursors.
        Try to load from persistent state file first.
        Fallback to recent history heuristic if no state exists.
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_memory_id = state.get("last_memory_id", 0)
                    self.last_meta_id = state.get("last_meta_id", 0)
                    self.last_reasoning_id = state.get("last_reasoning_id", 0)
                    self.log(f"🔮 Hod: Restored cursor state (Mem: {self.last_memory_id}, Meta: {self.last_meta_id}, Reas: {self.last_reasoning_id})")
            else:
                # Fallback: Start near the end to avoid massive re-processing on fresh install/reset
                max_mem = self.memory_store.get_max_id()
                self.last_memory_id = max(0, max_mem - 20)
                max_meta = self.meta_memory_store.get_max_id()
                self.last_meta_id = max(0, max_meta - 20)
                max_reas = self.reasoning_store.get_max_id()
                self.last_reasoning_id = max(0, max_reas - 20)
        except Exception as e:
            self.log(f"⚠️ Hod cursor init failed: {e}")

    def _save_state(self):
        """Persist cursor state to disk."""
        try:
            temp_path = self.state_file + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump({
                    "last_memory_id": self.last_memory_id,
                    "last_meta_id": self.last_meta_id,
                    "last_reasoning_id": self.last_reasoning_id
                }, f)
            os.replace(temp_path, self.state_file)
        except Exception as e:
            self.log(f"⚠️ Hod state save failed: {e}")
            
    def _track_metric(self, metric: str, value: float):
        if self.meta_learner:
            self.meta_learner.track(metric, value)
            
    def analyze(self) -> str:
        """Return the latest analysis summary for the Decider."""
        return self.last_analysis_summary

    def reflect(self, trigger: str = "Manual") -> Dict:
        """
        Analyze the system state after a specific event.
        Returns a structured analysis dict for Tiferet (Decider) to act upon.
        """
        analysis_result = {
            "findings": "",
            "risk_flags": [],
            "observations": []
        }

        try:
            self.log(f"🔮 Hod awakening for analysis. Trigger: {trigger}")
            
            settings = self.get_settings()
            
            # Gather context
            main_logs = self.get_main_logs()
            doc_logs = self.get_doc_logs()
            
            # Fetch incremental data
            recent_memories = self.memory_store.get_memories_after_id(self.last_memory_id, limit=50)
            recent_meta = self.meta_memory_store.get_meta_memories_after_id(self.last_meta_id, limit=50)
            recent_reasoning = self.reasoning_store.get_reasoning_after_id(self.last_reasoning_id, limit=50)

            recent_thoughts = self.reasoning_store.list_recent(limit=10)
            if not isinstance(recent_thoughts, (list, tuple)) or len(recent_thoughts) < 3:
                return

            range_info = []
            if recent_memories:
                range_info.append(f"Memories {recent_memories[0][0]}-{recent_memories[-1][0]}")
                self.last_memory_id = recent_memories[-1][0]
            if recent_meta:
                range_info.append(f"Meta-Memories {recent_meta[0][0]}-{recent_meta[-1][0]}")
                self.last_meta_id = recent_meta[-1][0]
            if recent_reasoning:
                range_info.append(f"Reasoning {recent_reasoning[0]['id']}-{recent_reasoning[-1]['id']}")
                self.last_reasoning_id = recent_reasoning[-1]['id']
                
            if range_info:
                self.log(f"🔮 Hod Analysis Range: {', '.join(range_info)}")
            
            # Defensive type checks for logs
            if not isinstance(main_logs, str):
                main_logs = str(main_logs) if main_logs is not None else ""

            context = f"--- HOD ANALYSIS CONTEXT (Trigger: {trigger}) ---\n"
            context += f"System Logs (Last 15 lines):\n{main_logs[-1000:]}\n\n" # Truncate logs
            
            if doc_logs:
                if not isinstance(doc_logs, str):
                    doc_logs = str(doc_logs)
                context += f"Document Logs:\n{doc_logs[-1000:]}\n\n" # Truncate doc logs
            
            if recent_memories:
                mem_list = []
                for m in recent_memories:
                    # m: (id, type, subject, text, source, verified, flags)
                    flags_mark = f" [FLAGS: {m[6]}]" if len(m) > 6 and m[6] else ""
                    mem_list.append(f"- [ID: {m[0]}] [{m[1]}]{flags_mark} {m[3][:200]}")
                context += "New Memories:\n" + "\n".join(mem_list) + "\n\n"
                
            if recent_meta:
                context += "New Meta-Memories:\n" + "\n".join([f"- {m[1]}: {m[3][:200]}" for m in recent_meta]) + "\n\n"

            if recent_reasoning:
                context += "New Reasoning (including Netzach/Observer):\n"
                for r in recent_reasoning:
                    # Deterministic Check: If we see "Refuting Belief [ID: X]" in reasoning, flag it as an epistemic correction signal.
                    match = re.search(r"Refuting Belief \[ID:\s*(\d+)\]", r['content'], re.IGNORECASE)
                    if match:
                        mid = int(match.group(1))
                        reason = r['content']
                        # Hod observes the signal, it does not command the refutation.
                        analysis_result["observations"].append({"type": "SELF_CORRECTION_SIGNAL", "id": mid, "context": reason})
                    source = r.get('source', 'unknown')
                    context += f"- [{source}] {r['content'][:150]}\n"
            
            current_temp = float(settings.get("temperature", 0.7))
            context += f"\nCurrent Temperature: {current_temp}\n"

            prompt = (
                "Review the Analysis Context. Identify patterns, hallucinations, or instability.\n"
                "Consult Netzach's reasoning if available. "
                "Look for 'Refuting Belief [ID: X]' in reasoning; if found, flag as self-correction signal.\n"
                "Provide your output in the following strict format:\n\n"
                "FINDINGS:\n"
                "[Detailed analysis text]\n\n"
                "RISKS:\n"
                "- [Risk 1]\n"
                "- [Risk 2]\n\n"
                "OBSERVATIONS:\n"
                "- [FLAG: LOGIC_VIOLATION] Target: [ID], Context: [Reason]\n"
                "- [FLAG: INVALID_DATA] Target: [ID], Context: [Reason]\n"
                "- [FLAG: HIGH_ENTROPY] Context: [Reason]\n"
                "- [FLAG: CONTEXT_OVERLOAD] Context: [Reason]\n"
                "- [FLAG: CONTEXT_FRAGMENTATION]\n"
                "- [FLAG: CRITICAL_INFO] Message: [Info for Decider]\n"
            )

            messages = [{"role": "user", "content": context}]

            response = run_local_lm(
                messages,
                system_prompt=HOD_SYSTEM_PROMPT + "\n" + prompt,
                temperature=0.4,
                max_tokens=300,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )

            # --- Parse Response ---
            current_section = None
            findings_buffer = []
            
            for line in response.splitlines():
                line = line.strip()
                if not line: continue
                
                if line.startswith("FINDINGS:"):
                    current_section = "FINDINGS"
                    continue
                elif line.startswith("RISKS:"):
                    current_section = "RISKS"
                    continue
                elif line.startswith("OBSERVATIONS:"):
                    current_section = "OBSERVATIONS"
                    continue
                
                if current_section == "FINDINGS":
                    findings_buffer.append(line)
                elif current_section == "RISKS":
                    if line.startswith("-"):
                        analysis_result["risk_flags"].append(line.lstrip("- ").strip())
                elif current_section == "OBSERVATIONS":
                    if line.startswith("- [FLAG:"):
                        observation = {}
                        # Parse structured flag
                        type_match = re.search(r"\[FLAG: ([A-Z_]+)\]", line)
                        if type_match: observation["type"] = type_match.group(1)
                        
                        target_match = re.search(r"Target: (\d+)", line)
                        if target_match: observation["id"] = int(target_match.group(1))
                        
                        # Capture Context or Message
                        ctx_match = re.search(r"(?:Context|Message): (.*?)(?:$|\])", line)
                        if ctx_match: observation["context"] = ctx_match.group(1).strip()
                        
                        if "type" in observation:
                            analysis_result["observations"].append(observation)

            analysis_result["findings"] = "\n".join(findings_buffer)
            self.log(f"🔮 Hod Analysis: {analysis_result['findings']}")
            self.last_analysis_summary = analysis_result['findings'][:100] + "..." if analysis_result['findings'] else "No significant findings."
            
            if self.event_bus and analysis_result['findings']:
                self.event_bus.publish("HOD_ANALYSIS", {"text": analysis_result['findings']})
                
            self._save_state()
            
        except Exception as e:
            self.log(f"❌ Hod analysis error: {e}")
            
        return analysis_result

    def verify_sources(self, batch_size: int = 5, concurrency: int = 1, stop_check_fn: Optional[Callable[[], bool]] = None) -> List[Dict[str, Any]]:
        """
        Verify memories against their cited sources.
        Returns a list of PROPOSALS (e.g. DELETE, VERIFY, REFUTE).
        Does NOT mutate memory directly.
        """
        if not self.document_store:
            return []
            
        # Optimization: Fetch only relevant candidates directly from DB
        # We need unverified FACTs/BELIEFs from 'daydream' source
        candidates = self._get_verification_candidates()
        
        proposals = []

        # 1. Cleanup: Remove memories that refer to documents but lack citation
        # Includes BELIEF here because uncited beliefs about documents are also invalid
        triggers = ["the document", "this document", "the study", "this study", "the research", "this research", 
                    "the pdf", "this pdf", "the findings", "these findings", "the article", "this article", "the text"]
        
        # Filter for cleanup candidates from the main list
        uncited_candidates = [
            m for m in candidates
            if "[Source:" not in m['text'] and "[Supported by" not in m['text']
        ]
        
        for mem in uncited_candidates:
            if stop_check_fn and stop_check_fn():
                break
                
            lower_text = mem['text'].lower()
            if any(t in lower_text for t in triggers):
                proposals.append({
                    "proposal": "DELETE",
                    "memory_id": mem['id'],
                    "reason": "uncited_document_reference",
                    "origin": "HOD",
                    "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject']}
                })

        # 2. Verify cited memories
        
        if not candidates and not proposals:
            return proposals
            
        # Pick a random batch to verify (spreads load over time)
        batch = random.sample(candidates, min(len(candidates), batch_size))
        
        self.log(f"🧹 [Verifier] Found {len(candidates)} unverified candidates. Checking {len(batch)} of them...")

        # Optimization: Sort by source to maximize cache hits
        batch.sort(key=lambda m: re.search(r"\[(?:Source|Supported by): (.*?)\]", m['text']).group(1).strip().strip('"').strip("'") if re.search(r"\[(?:Source|Supported by): (.*?)\]", m['text']) else "")
        
        # Parallel Execution
        if self.executor:
            futures = {self.executor.submit(self._verify_single_memory, mem): mem for mem in batch}
            
            for future in as_completed(futures):
                if stop_check_fn and stop_check_fn():
                    self.log("🛑 [Verifier] Verification stopped by user.")
                    # Cannot shutdown shared executor, just break loop and let futures finish/cancel
                    for f in futures:
                        f.cancel()
                    break
                try:
                    result = future.result()
                    if result:
                        proposals.append(result)
                except Exception as e:
                    self.log(f"❌ [Verifier] Thread error: {e}")
                    try:
                        result = future.result()
                        if result:
                            proposals.append(result)
                    except Exception as e:
                        self.log(f"❌ [Verifier] Thread error: {e}")

        return proposals

    def _verify_single_memory(self, mem: Dict) -> Optional[Dict]:
        """
        Worker function for verifying a single memory.
        Returns a dict with action instructions or None.
        """
        max_retrieval_failures = int(self.get_settings().get("max_retrieval_failures", 3))
        max_inconclusive_attempts = int(self.get_settings().get("max_inconclusive_attempts", 3))
        try:
            # Get thread name for logging concurrency
            t_name = threading.current_thread().name
            worker_id = t_name.split('_')[-1] if '_' in t_name else "Main"

            # Double-check verification status (in case of race conditions)
            mem_check = self.memory_store.get(mem['id'])
            if mem_check and mem_check.get('verified') == 1:
                self.log(f"⏩ [Verifier][Worker-{worker_id}] Memory {mem['id']} already verified. Skipping.")
                return None

            # Skip meta-action leaks (Refutations saved as beliefs)
            if mem['text'].strip().startswith("[Refuting Belief") or mem['text'].strip().startswith("Refuting Belief"):
                return {
                    "proposal": "DELETE",
                    "memory_id": mem['id'],
                    "reason": "meta_action_leak",
                    "origin": "HOD",
                    "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject']}
                }

            # Check if this is an uncited item (Open Grounding Path)
            if mem['type'] in ('BELIEF', 'FACT') and "[Source:" not in mem['text'] and "[Supported by" not in mem['text']:
                return self._ground_uncited_item(mem, worker_id)

            # Extract filename
            match = re.search(r"\[(?:Source|Supported by): (.*?)\]", mem['text'])
            if not match:
                return None
                
            # Clean text for search (remove source tag)
            clean_text = mem['text'].replace(match.group(0), "").strip()
            if not clean_text:
                # Skip empty citation-only memories silently (or with debug log)
                return None

            filename = match.group(1).strip().strip('"').strip("'")
            self.log(f"🔍 [Verifier][Worker-{worker_id}] Verifying Memory {mem['id']} [{mem['type']}] against '{filename}':\n    \"{clean_text}\"")
            
            # Fetch document ID
            doc_id = self.document_store.get_document_by_filename(filename)
            if not doc_id:
                self.log(f"⚠️ [Verifier][Worker-{worker_id}] Document '{filename}' not found for Memory {mem['id']}.")
                
                # Count as retrieval failure
                attempts = mem.get('verification_attempts', 0) + 1
                if attempts >= max_retrieval_failures:
                    return {
                        "proposal": "DELETE",
                        "memory_id": mem['id'],
                        "reason": "source_document_missing",
                        "origin": "HOD",
                        "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "attempts": attempts}
                    }
                return {
                    "proposal": "INCREMENT_ATTEMPTS",
                    "memory_id": mem['id'],
                    "origin": "HOD"
                }
            
            # Fetch chunks for context reconstruction (No caching in parallel mode to avoid complexity/locks)
            # Fetch only the relevant chunks identified by search_results, plus a small window
            relevant_chunk_indices = set()
            for res in search_results:
                relevant_chunk_indices.add(res['chunk_index'])
                # Add window of +/- 1 chunk for context
                relevant_chunk_indices.add(res['chunk_index'] - 1)
                relevant_chunk_indices.add(res['chunk_index'] + 1)
            
            all_chunks = self.document_store.get_specific_chunks(doc_id, list(relevant_chunk_indices), include_embeddings=False)
            chunk_map = {c['chunk_index']: c['text'] for c in all_chunks}
                
            # Search chunks in that document using the memory's content
            if self.embed_fn:
                query_emb = self.embed_fn(clean_text)
            else:
                query_emb = compute_embedding(clean_text)

            # Use optimized local search in document store
            search_results = self.document_store.search_chunks(query_emb, top_k=5, document_id=doc_id)
            
            if search_results:
                 self.log(f"    🔍 [Verifier] Top chunk match: {search_results[0]['similarity']:.4f} (Threshold: 0.45)")
            else:
                 self.log(f"    🔍 [Verifier] No chunks found via vector search.")
            
            # Fallback: Cross-Lingual / Keyword Search Generation
            # If results are poor (e.g. top similarity < 0.45), try to generate a better query
            # This handles the English Memory -> Turkish Doc gap
            if not search_results or (search_results and search_results[0]['similarity'] < 0.45):
                self.log(f"⚠️ [Verifier][Worker-{worker_id}] Low similarity ({search_results[0]['similarity'] if search_results else 0:.2f}) for direct search. Attempting cross-lingual query generation...")
                
                gen_prompt = (
                    f"I need to find evidence for this claim in a document named '{filename}'.\n"
                    f"Claim: {clean_text}\n"
                    "The document might be in a different language (e.g., Turkish, Spanish) or use different terminology.\n"
                    "Generate a search query (keywords or translated sentence) that is most likely to match the raw text in the document.\n"
                    "Output ONLY the search query text."
                )
                
                better_query = run_local_lm([{"role": "user", "content": gen_prompt}], temperature=0.1, max_tokens=100).strip()
                
                if better_query and better_query != clean_text:
                    self.log(f"🔍 [Verifier][Worker-{worker_id}] Retrying search with generated query: '{better_query}'")
                    if self.embed_fn:
                        query_emb_2 = self.embed_fn(better_query)
                    else:
                        query_emb_2 = compute_embedding(better_query)
                    search_results_2 = self.document_store.search_chunks(query_emb_2, top_k=5, document_id=doc_id)
                    
                    if search_results_2:
                        search_results = search_results_2 # Prefer the generated query results

            if not search_results:
                self.log(f"⚠️ [Verifier][Worker-{worker_id}] No relevant text chunks found in '{filename}' for Memory {mem['id']}.")
                
                # Check attempts limit (Read-only check)
                attempts = mem.get('verification_attempts', 0) + 1
                if attempts >= max_retrieval_failures:
                    return {
                        "proposal": "DELETE",
                        "memory_id": mem['id'],
                        "reason": "retrieval_failure_limit_reached",
                        "origin": "HOD",
                        "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "attempts": attempts}
                    }
                
                return {
                    "proposal": "INCREMENT_ATTEMPTS",
                    "memory_id": mem['id'],
                    "origin": "HOD"
                }
            
            # Dynamic context sizing to handle 400 errors
            max_chars = 3000
            attempts_left = 3
            response = ""
            
            while attempts_left > 0:
                relevant_indices = set()
                current_chars = 0

                for res in search_results:
                    if current_chars >= max_chars:
                        break

                    idx = res['chunk_index']
                    # Add window of +/- 1 chunk to provide context
                    for i in range(idx - 1, idx + 2):
                        if i in chunk_map:
                            if i not in relevant_indices:
                                chunk_len = len(chunk_map[i])
                                if current_chars + chunk_len < max_chars:
                                    relevant_indices.add(i)
                                    current_chars += chunk_len
                
                sorted_indices = sorted(list(relevant_indices))
                
                context_parts = []
                last_idx = -999
                for idx in sorted_indices:
                    if idx > last_idx + 1:
                        context_parts.append("\n... [Skipped sections] ...\n")
                    context_parts.append(f"[Section {idx}] {chunk_map[idx]}")
                    last_idx = idx
                    
                context_text = "\n".join(context_parts)
                
                is_belief = mem['type'] == 'BELIEF'
                if is_belief:
                    role_desc = "You are an analyst evaluating if a BELIEF is a valid interpretation of a source document."
                    task_desc = (
                        "Task: Determine if the Belief is a plausible cognitive interpretation, hypothesis, or synthesis derived from the text.\n"
                        "1. Beliefs are subjective or synthetic. They do NOT need to be explicitly stated in the text.\n"
                        "2. Use your cognitive capabilities to judge if the belief is a logical extension or valid impression of the source.\n"
                        "3. REJECT only if the belief is clearly contradicted by the text or completely unrelated."
                    )
                else:
                    role_desc = "You are a fact-checker verifying if a memory is supported by a source document."
                    task_desc = (
                        "Task: Analyze if the Memory Claim is supported by the text.\n"
                        "1. Briefly analyze the relationship between the text and the claim.\n"
                        "2. Allow for reasonable inference, synthesis, and summarization. It does not need to be verbatim.\n"
                        "3. Only reject if the claim clearly contradicts the text or is completely unrelated/hallucinated."
                    )

                prompt = (
                    f"{role_desc}\n"
                    "Note: The Document Excerpts and the Memory Claim might be in DIFFERENT LANGUAGES.\n"
                    "Verify based on meaning, not just keyword matching.\n\n"
                    f"Excerpts from '{filename}':\n{context_text}\n\n"
                    f"Memory Claim: {clean_text}\n\n"
                    f"{task_desc}\n\n"
                    "Output format:\n"
                    "Verdict: VALID or INVALID\n"
                    "Reasoning: <short analysis>"
                )
                
                response = run_local_lm([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=500)
                
                # Check for 400 error in response string (returned by lm.py exception handler)
                if "400" in response and ("Bad Request" in response or "Client Error" in response):
                    self.log(f"⚠️ [Verifier][Worker-{worker_id}] Context too long ({max_chars} chars). Retrying with reduced context...")
                    max_chars = int(max_chars * 0.6)
                    attempts_left -= 1
                    if attempts_left == 0:
                        self.log(f"❌ [Verifier][Worker-{worker_id}] Failed to verify Memory {mem['id']} due to context length after retries.")
                        return None
                else:
                    break
            
            # Extract reasoning for logging
            reasoning = "No reasoning provided."
            match = re.search(r"Reasoning:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()

            # Robust parsing: look for keywords even if formatting is messy
            if "Verdict: INVALID" in response or "Verdict:INVALID" in response:
                return {
                    "proposal": "DELETE",
                    "memory_id": mem['id'],
                    "reason": "invalid_verdict",
                    "origin": "HOD",
                    "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "llm_response": response, "reasoning": reasoning}
                }
            elif "Verdict: VALID" in response or "Verdict:VALID" in response:
                return {
                    "proposal": "VERIFY",
                    "memory_id": mem['id'],
                    "reason": "validated_by_source",
                    "origin": "HOD",
                    "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "source": filename, "reasoning": reasoning}
                }
            else:
                self.log(f"❓ [Verifier][Worker-{worker_id}] INCONCLUSIVE for Memory {mem['id']} against '{filename}'.\n    Content: \"{clean_text}\"\n    Response: {response}")
                
                # Check attempts limit
                attempts = mem.get('verification_attempts', 0) + 1
                if attempts >= max_inconclusive_attempts:
                    return {
                        "proposal": "DELETE",
                        "memory_id": mem['id'],
                        "reason": "inconclusive_limit_reached",
                        "origin": "HOD",
                        "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "attempts": attempts}
                    }
                
                return {
                    "proposal": "INCREMENT_ATTEMPTS",
                    "memory_id": mem['id'],
                    "origin": "HOD"
                }
        except Exception as e:
            self.log(f"⚠️ [Verifier][Worker-{worker_id}] Error verifying memory {mem['id']}: {e}")
            return None

    def _ground_uncited_item(self, mem: Dict, worker_id: str) -> Optional[Dict]:
        """
        Perform Open Grounding (Cognitive Evaluation) for an uncited belief or fact.
        Searches all documents for evidence to support or refute the item.
        """
        max_inconclusive_attempts = int(self.get_settings().get("max_inconclusive_attempts", 3))
        item_type = mem['type']
        self.log(f"🔍 [Verifier][Worker-{worker_id}] Grounding Uncited {item_type} {mem['id']}:\n    \"{mem['text']}\"")
        
        # 1. Search for relevant documents
        if self.embed_fn:
            query_emb = self.embed_fn(mem['text'])
        else:
            query_emb = compute_embedding(mem['text'])
        
        # Detect embedded filename to restrict search scope (Prevent Cross-Document Contamination)
        embedded_filename = None
        file_match = re.search(r"([a-zA-Z0-9\s_\-\(\)]+\.(?:pdf|docx))", mem['text'], re.IGNORECASE)
        if file_match:
            fname = file_match.group(1).strip()
            if self.document_store.get_document_by_filename(fname):
                embedded_filename = fname

        if embedded_filename:
             doc_id = self.document_store.get_document_by_filename(embedded_filename)
             search_results = self.document_store.search_chunks(query_emb, top_k=5, document_id=doc_id)
             self.log(f"🔍 [Verifier][Worker-{worker_id}] Restricted search to detected file: '{embedded_filename}'")
        else:
             search_results = self.document_store.search_chunks(query_emb, top_k=3)

        if not search_results:
            self.log(f"⚠️ [Verifier][Worker-{worker_id}] No relevant documents found for {item_type} {mem['id']}. Skipping.")
            
            # Count as inconclusive attempt
            attempts = mem.get('verification_attempts', 0) + 1
            if attempts >= max_inconclusive_attempts:
                return {
                    "proposal": "DELETE",
                    "memory_id": mem['id'],
                    "reason": "no_relevant_documents_found",
                    "origin": "HOD",
                    "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "attempts": attempts}
                }
            return {
                "proposal": "INCREMENT_ATTEMPTS",
                "memory_id": mem['id'],
                "origin": "HOD"
            }
            
        # 2. Construct Context
        context = f"Claim to Evaluate: {mem['text']}\n\nRelevant Document Evidence:\n"
        found_files = set()
        for res in search_results:
            context += f"- [Doc: {res['filename']}] {res['text'][:400]}...\n"
            found_files.add(res['filename'])
            
        # 3. Evaluate
        prompt = (
            "You are an expert analyst. Evaluate the Claim against the Document Evidence.\n"
            "1. SUPPORTED: If the evidence confirms the claim, output 'Verdict: SUPPORTED'. Provide a short reasoning and cite the specific document.\n"
            "2. REFUTED: If the evidence DIRECTLY contradicts the claim, output 'Verdict: REFUTED'. Provide reasoning and the CORRECT information.\n"
            "   - CRITICAL: Do NOT output REFUTED if the document is simply silent, unrelated, or discusses a different topic. Use UNSUPPORTED.\n"
            "   - CRITICAL: If the claim cites Document A, but the evidence is from Document B, do NOT refute unless Document B explicitly discusses Document A's errors.\n"
            "3. PLAUSIBLE: If the evidence makes it likely but not certain, output 'Verdict: PLAUSIBLE'.\n"
            "4. UNSUPPORTED: If the evidence is unrelated, output 'Verdict: UNSUPPORTED'.\n\n"
            "5. CONFLICT: If the evidence contradicts itself or shows conflicting data points (e.g. Source A says X, Source B says Y), output 'Verdict: CONFLICT'.\n\n"
            "CRITICAL RULES:\n"
            "- To mark as SUPPORTED/PLAUSIBLE, the document must contain specific entities (names, dates, unique terms) matching the claim.\n"
            "- Do NOT match on vague themes (e.g. 'war' in claim vs 'war' in text is NOT enough if the specific characters don't match).\n"
            f"{context}\n\n"
            "Output format:\n"
            "Verdict: <status>\n"
            "Citation: <filename> (Must be EXACTLY one of the filenames listed above, without [Doc:] prefix)\n"
            "Reasoning: <analysis>\n"
            "Correction: <Correct fact if REFUTED, else None>"
        )
        
        response = run_local_lm([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=500)
        
        # 4. Process Verdict
        reasoning = "Analysis complete."
        match = re.search(r"Reasoning:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        if match: 
            # Stop at Correction: if present
            reasoning = match.group(1).split("Correction:")[0].strip()
        
        # Extract citation (common for all verdicts if available)
        citation = None
        match_cit = re.search(r"Citation:\s*(.*)", response, re.IGNORECASE)
        if match_cit:
            raw_cit = match_cit.group(1).strip()
            # Clean up common LLM formatting errors (remove [Doc: prefix and brackets)
            clean_cit = re.sub(r"^\[Doc:\s*", "", raw_cit, flags=re.IGNORECASE).rstrip("]")
            
            # Strict validation: Ensure citation matches a real file we sent
            if clean_cit in found_files:
                citation = clean_cit
            else:
                # Fuzzy fallback: Try to find best substring match in found_files
                for f in found_files:
                    if clean_cit in f or f in clean_cit:
                        citation = f
                        break
        
        if "Verdict: REFUTED" in response:
            # Extract correction
            correction = None
            match_corr = re.search(r"Correction:\s*(.*)", response, re.IGNORECASE)
            if match_corr:
                correction = match_corr.group(1).strip()
                if correction.lower() == "none": correction = None

            new_text = f"{mem['text']} [REFUTED: {reasoning[:500]}]"
            return {
                "proposal": "REFUTE",
                "memory_id": mem['id'],
                "new_text": new_text,
                "correction": correction,
                "reason": "grounding_refutation",
                "origin": "HOD",
                "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "reasoning": reasoning, "source": citation}
            }
            
        elif "Verdict: CONFLICT" in response:
            gap_text = f"Conflict detected regarding: {mem['text']}. Analysis: {reasoning[:500]}"
            return {
                "proposal": "CURIOSITY_GAP",
                "memory_id": mem['id'],
                "text": gap_text,
                "reason": "conflicting_evidence",
                "origin": "HOD",
                "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "reasoning": reasoning}
            }
            
        elif "Verdict: SUPPORTED" in response or "Verdict: PLAUSIBLE" in response:
            if not citation:
                if len(found_files) == 1:
                    citation = list(found_files)[0]
                else:
                    self.log(f"⚠️ [Verifier][Worker-{worker_id}] Verdict SUPPORTED but invalid citation. Skipping to prevent mismatch.")
                    return None
                
            if citation:
                # Update text with citation
                new_text = f"{mem['text']} [Supported by: {citation}]"
                return {
                    "proposal": "VERIFY",
                    "memory_id": mem['id'],
                    "new_text": new_text,
                    "reason": "grounded_in_document",
                    "origin": "HOD",
                    "meta": {"text": mem['text'], "type": item_type, "subject": mem['subject'], "source": citation, "reasoning": reasoning}
                }
        
        self.log(f"⏩ [Verifier][Worker-{worker_id}] {item_type} {mem['id']} remains unverified (Verdict: Unsupported/Inconclusive).\n    Reasoning: {reasoning}")
        
        # Check attempts limit
        attempts = mem.get('verification_attempts', 0) + 1
        if attempts >= max_inconclusive_attempts:
            return {
                "proposal": "DELETE",
                "memory_id": mem['id'],
                "reason": "inconclusive_limit_reached",
                "origin": "HOD",
                "meta": {"text": mem['text'], "type": mem['type'], "subject": mem['subject'], "attempts": attempts}
            }

        return {
            "proposal": "INCREMENT_ATTEMPTS",
            "memory_id": mem['id'],
            "origin": "HOD"
        }

    def get_unverified_count(self) -> int:
        """Get number of unverified memories that require verification."""
        target_types = ("FACT", "BELIEF")
        placeholders = ','.join(['?'] * len(target_types))
        with self.memory_store._connect() as con:
            row = con.execute(f"""
                SELECT COUNT(*) FROM memories 
                WHERE verified = 0 
                AND ((text LIKE '%[Source:%' OR text LIKE '%[Supported by%') OR type IN ('BELIEF', 'FACT'))
                AND type IN ({placeholders})
                AND parent_id IS NULL
                AND deleted = 0
                AND source = 'daydream'
            """, target_types).fetchone()
        return row[0] if row else 0

    def _get_verification_candidates(self) -> List[Dict]:
        """Fetch unverified memories from daydream source."""
        with self.memory_store._connect() as con:
            rows = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at, embedding, verified, verification_attempts
                FROM memories
                WHERE parent_id IS NULL AND deleted = 0
                AND verified = 0
                AND type IN ('FACT', 'BELIEF')
                AND source = 'daydream'
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