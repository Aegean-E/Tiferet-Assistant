import re
import json
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Callable
from ai_core.lm import compute_embedding, run_local_lm, extract_memories_llm, count_tokens, LLMError, DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT
from ai_core.utils import parse_json_array_loose

if TYPE_CHECKING:
    from treeoflife.tiferet import Decider

class ChatHandler:
    def __init__(self, decider: 'Decider'):
        self.decider = decider

    def process_chat_message(self, user_text: str, history: List[Dict], status_callback: Callable[[str], None] = None, image_path: Optional[str] = None, stop_check_fn: Callable[[], bool] = None, stream_callback: Callable[[str], None] = None) -> str:
        """
        Core Chat Logic: RAG -> LLM -> Memory Extraction -> Response.
        Decider now handles the cognitive pipeline for user interactions.
        """
        # Mailbox: Chat is an external interruption that resets the Hod cycle lock
        self.decider.log(f"📬 Decider Mailbox: Received message from User.")
        self.decider.hod_just_ran = False
        self.decider.last_action_was_speak = False

        # Use provided stop check or default
        current_stop_check = stop_check_fn if stop_check_fn else self.decider.stop_check

        # Check for natural language commands
        # Skip NL commands if image is present (prioritize Vision), UNLESS it's a slash command
        if not image_path or user_text.strip().startswith("/"):
            nl_response = self.handle_natural_language_command(user_text, status_callback)
            if nl_response:
                self.decider.log(f"🤖 Decider Command Response: {nl_response}")
                return nl_response

        settings = self.decider.get_settings()

        # 1. Retrieve Context
        context_data = self._retrieve_context(user_text, settings, executor=self.decider.executor)

        # 2. Construct System Prompt
        system_prompt = self._construct_system_prompt(user_text, history, context_data, settings)

        # 3. Call LLM with structured error handling
        try:
            start_time = time.time()
            reply = run_local_lm(
                history,
                system_prompt=system_prompt,
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 0.94),
                max_tokens=settings.get("max_tokens", 800),
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=current_stop_check,
                images=[image_path] if image_path else None,
                stream_callback=stream_callback
            )
        except LLMError as e:
            self.decider.log(f"❌ Chat generation failed: {e}")
            return "⚠️ I encountered an error generating a response. Please check the logs."
        latency = time.time() - start_time

        # Feed Keter (Vibrational Monitoring)
        if self.decider.keter:
            self.decider.keter.track_response_metrics(latency, len(reply))

        # 4. Manifest Persona (Yesod)
        if self.decider.yesod:
            reply = self.decider.yesod.manifest_persona(reply, self.decider.mood)

        # 5. Recursive Theory of Mind (Self-Model of User's Model)
        # Simulate user perception before finalizing (or just for reflection)
        if self.decider.yesod:
            try:
                perception = self.decider.yesod.simulate_user_perception(user_text, reply)
                self.decider.reasoning_store.add(
                    content=f"Recursive ToM Simulation: {perception.get('interpretation')} (Sat: {perception.get('satisfaction')}, Conf: {perception.get('confusion_risk')})",
                    source="recursive_tom",
                    confidence=1.0
                )
                # Future: If confusion_risk > 0.7, trigger refinement loop here
            except Exception as e:
                self.decider.log(f"⚠️ Recursive ToM failed: {e}")

        # Check for LLM error
        if reply.startswith("⚠️"):
            self.decider.log(f"❌ Chat generation failed: {reply}")
            if status_callback: status_callback("Generation Error")
            return "⚠️ I encountered an error generating a response. Please check the logs."

        # Check for Tool Execution in Chat
        if "[EXECUTE:" in reply:
            try:
                match = re.search(r"\[EXECUTE:\s*([A-Z_]+)\s*,\s*(.*?)\]", reply, re.IGNORECASE)
                if match:
                    tool_name = match.group(1).upper()
                    args = match.group(2).strip()
                    result = self.decider.command_executor._execute_tool(tool_name, args)
                    self.decider._track_metric("tool_success", 1.0 if "Error" not in result else 0.0)
                    tool_output = f"\n\n🛠️ Tool Result: {result}"
                    if stream_callback:
                        stream_callback(tool_output)
                    reply += tool_output
            except Exception as e:
                self.decider.log(f"⚠️ Chat tool execution failed: {e}")
                self.decider._track_metric("tool_success", 0.0)

        # 6. Memory Extraction (Side Effect)
        # Run in background thread to unblock UI response
        def background_processing():
            if status_callback: status_callback("Extracting memories...")
            self._extract_and_save_memories(user_text, reply, settings)
            if status_callback: status_callback("Ready")

            # Log interaction
            if hasattr(self.decider.meta_memory_store, 'add_event'):
                user_preview = user_text[:100].replace('\n', ' ')
                if len(user_text) > 100: user_preview += "..."
                reply_preview = reply[:100].replace('\n', ' ')
                if len(reply) > 100: reply_preview += "..."
                self.decider.meta_memory_store.add_event(
                    event_type="DECIDER_CHAT",
                    subject="Assistant",
                    text=f"Chat: '{user_preview}' -> '{reply_preview}'"
                )

            # Update World Model with interaction
            if self.decider.malkuth:
                # Register the chat interaction as an outcome
                self.decider.malkuth.register_outcome("Chat Interaction", "Reply to user", f"User: {user_text}\nAssistant: {reply}")

            # 7. Update Theory of Mind (User Model)
            if self.decider.yesod:
                self.decider.yesod.analyze_user_interaction(user_text, reply)

            # 8. Metacognitive Reflection
            self.decider.decision_maker._reflect_on_decision(user_text, reply)

        if self.decider.executor:
            self.decider.executor.submit(background_processing)
        else:
            threading.Thread(target=background_processing, daemon=True).start()

        self.decider.log(f"🗣️ Assistant Reply: {reply}")

        return reply

    def handle_natural_language_command(self, text: str, status_callback: Callable[[str], None] = None) -> Optional[str]:
        """Check for and execute natural language commands."""
        text = text.lower().strip()

        # Slash Commands
        if text.startswith("/clear_mem"):
            try:
                parts = text.split()
                if len(parts) < 2:
                    return "⚠️ Usage: /clear_mem [ID]"
                mem_id = int(parts[1])
                success = self.decider.memory_store.delete_entry(mem_id)
                if success:
                    self.decider.log(f"🗑️ Manually deleted memory ID {mem_id}")
                    return f"✅ Memory {mem_id} deleted."
                else:
                    return f"⚠️ Memory {mem_id} not found or could not be deleted."
            except ValueError:
                return "⚠️ Invalid ID format."

        # Daydream Loop
        if "run daydream loop" in text or "start daydream loop" in text:
            count = 10
            match = re.search(r'(\d+)\s*times', text)
            if match:
                count = int(match.group(1))

            self.decider.log(f"🤖 Decider enabling Daydream Loop for {count} cycles via natural command.")
            self.decider.command_executor._run_action("start_loop")
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task("daydream", count, "Natural Command Loop")
            return f"🔄 Daydream loop enabled for {count} cycles."

        # Daydream Batch (Specific Count)
        # Matches: "run 5 daydream cycles", "do 3 daydreams", "run 1 daydream cycle"
        batch_match = re.search(r"(?:run|do|start|execute)\s+(\d+)\s+daydream(?:s|ing)?(?: cycles?| loops?)?", text)
        if batch_match:
            count = int(batch_match.group(1))
            # Cap count reasonably
            count = max(1, min(count, 20))

            self.decider.log(f"🤖 Decider enabling Daydream Batch for {count} cycles via natural command.")
            self.decider.daydream_mode = "auto"
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task("daydream", count, "Natural Command Batch")
            return f"☁️ Starting {count} daydream cycles."

        # Learn / Expand Knowledge
        learn_match = re.search(r"(?:expand (?:your )?knowledge(?: about)?|learn(?: about)?|research|study|read up on|educate yourself(?: on| about)?)\s+(.*)", text, re.IGNORECASE)
        if learn_match:
            raw_topic = learn_match.group(1).strip(" .?!")

            # Use LLM to extract the core topic more robustly
            settings = self.decider.get_settings()
            core_topic_prompt = (
                f"Extract the core topic from this phrase: '{raw_topic}'. "
                "Remove any extraneous phrases like 'from your documents', 'research about', 'learn about', etc. "
                "Output ONLY the core topic."
            )
            core_topic = run_local_lm(
                messages=[{"role": "user", "content": core_topic_prompt}],
                system_prompt="You are a precise topic extractor.",
                temperature=0.1,
                max_tokens=50,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=self.decider.stop_check
            ).strip()

            if core_topic and not core_topic.startswith("⚠️"):
                self.decider.goal_manager.create_goal(f"Expand knowledge about {core_topic}")
                self.decider.log(f"🤖 Decider starting Daydream Loop focused on: {core_topic}")
                self.decider.command_executor._run_action("start_loop")
                self.decider.daydream_mode = "read"
                self.decider.daydream_topic = core_topic
                if self.decider.heartbeat:
                    self.decider.heartbeat.force_task("daydream", 5, f"Research: {core_topic}")
                return f"📚 Initiating research protocol for: {core_topic}. I will read relevant documents and generate insights."
            else:
                return f"⚠️ Failed to extract a clear topic from '{raw_topic}'. Please be more specific."

        # Verify All
        if "run verification all" in text or "verify all" in text:
            self.decider.log("🤖 Decider starting Full Verification via natural command.")
            self.decider.command_executor._run_action("verify_all")
            return "🕵️ Full verification triggered."

        # Verify Batch
        if "run verification batch" in text or "verify batch" in text or "run verification" in text:
            self.decider.log("🤖 Decider starting Verification Batch via natural command.")
            self.decider.command_executor._run_action("verify_batch")
            return "🕵️ Verification batch triggered."

        # Verify Beliefs (Internal Consistency/Insight)
        if "verify" in text and "belief" in text:
             self.decider.log("🤖 Decider starting Belief Verification (Grounding).")
             self.decider.command_executor._run_action("verify_batch")
             if self.decider.heartbeat:
                 self.decider.heartbeat.force_task("verify", 3, "Belief Verification")
             return "🕵️ Initiating belief verification."

        # Verify Sources (Facts/Memories against Documents)
        if "verify" in text and ("fact" in text or "memory" in text or "source" in text):
             self.decider.log("🤖 Decider starting Verification Batch via natural command.")
             self.decider.command_executor._run_action("verify_batch")
             return "🕵️ Verification batch triggered."

        # Single Daydream
        if text in ["run daydream", "start daydream", "daydream", "do a daydream"]:
            self.decider.log("🤖 Decider starting single Daydream cycle via natural command.")
            self.decider.command_executor._run_action("start_daydream")
            return "☁️ Daydream triggered."

        # Stop
        if "stop daydream" in text or "stop loop" in text or "stop processing" in text:
            self.decider.log("🤖 Decider stopping processing via natural command.")
            self.decider.command_executor._run_action("stop_daydream")
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task("wait", 0, "Natural Stop Command")
            return "🛑 Processing stopped."

        # Sleep
        if "go to sleep" in text or "enter sleep mode" in text:
            self.decider.command_executor.start_sleep_cycle()
            return "💤 Entering Sleep Mode. Inputs will be ignored while I consolidate memory."

        # Think
        if text.startswith("think about") or text.startswith("analyze") or text.startswith("ponder"):
            topic = text.replace("think about", "").replace("analyze", "").replace("ponder", "").strip()
            self.decider.thought_generator.perform_thinking_chain(topic)
            return f"🧠 Finished thinking about: {topic}"

        # Debate
        if text.startswith("debate") or text.startswith("discuss"):
            topic = text.replace("debate", "").replace("discuss", "").strip()
            if self.decider.dialectics:
                return f"🏛️ Council Result: {self.decider.dialectics.run_debate(topic, reasoning_store=self.decider.reasoning_store)}"

        # Simulate / What If
        if text.startswith("simulate") or text.startswith("what if"):
            premise = text.replace("simulate", "").replace("what if", "").strip()
            if "simulate_counterfactual" in self.decider.actions:
                return f"🌌 Simulation Result: {self.decider.actions['simulate_counterfactual'](premise)}"

        # Tools: Calculator
        if text.startswith("calculate") or text.startswith("solve") or text.startswith("math"):
            expr = re.sub(r'^(calculate|solve|math)\s+', '', text, flags=re.IGNORECASE)
            result = self.decider.command_executor._execute_tool("CALCULATOR", expr)
            return f"🧮 Calculation Result: {result}"

        # Tools: Clock
        if any(phrase in text for phrase in ["what time", "current time", "clock"]):
            result = self.decider.command_executor._execute_tool("CLOCK", "")
            return f"🕒 Current Time: {result}"

        # Tools: Dice
        if "roll" in text and ("dice" in text or "die" in text or "number" in text):
            args = ""
            range_match = re.search(r'(\d+)\s*-\s*(\d+)', text)
            if range_match:
                args = f"{range_match.group(1)}-{range_match.group(2)}"
            else:
                num_match = re.search(r'(\d+)', text)
                if num_match:
                    args = num_match.group(1)
            result = self.decider.command_executor._execute_tool("DICE", args)
            return f"🎲 Dice Roll: {result}"

        # Tools: System Info
        if "system info" in text or "specs" in text or "hardware" in text:
            result = self.decider.command_executor._execute_tool("SYSTEM_INFO", "")
            return f"💻 System Info: {result}"

        # --- Fallback: LLM-based Intent Analysis ---
        # If regex failed but keywords are present, ask the AI what it thinks.
        # Removed common words like "think", "check" to prevent false positives in normal conversation.
        trigger_keywords = ["learn", "study", "research", "summarize", "summary", "verify", "analyze", "ponder", "digest"]

        # Only analyze if keywords exist and it's not a super short greeting
        if any(kw in text for kw in trigger_keywords) and len(text.split()) > 2:
            if status_callback: status_callback("Analyzing intent...")
            self.decider.log(f"🧠 Decider analyzing intent for: '{text}'")

            intent_response = self._analyze_intent(text)
            self.decider.log(f"🧠 Intent detected: {intent_response}")

            if "[LEARN]" in intent_response:
                topic = intent_response.split("]", 1)[1].strip()
                # Clean topic
                clean_topic = re.sub(r"\s+from\s+(?:your\s+)?(?:documents|files|database|memory).*", "", topic, flags=re.IGNORECASE).strip()

                self.decider.goal_manager.create_goal(f"Expand knowledge about {clean_topic}")
                self.decider.log(f"🤖 Decider starting Daydream Loop focused on: {clean_topic}")
                self.decider.command_executor._run_action("start_loop")
                self.decider.daydream_mode = "read"
                self.decider.daydream_topic = clean_topic
                if self.decider.heartbeat:
                    self.decider.heartbeat.force_task("daydream", 5, f"Research: {clean_topic}")
                return f"📚 Initiating research protocol for: {clean_topic}. I will read relevant documents and generate insights."

            elif "[VERIFY]" in intent_response:
                if "belief" in intent_response.lower():
                    self.decider.log("🤖 Decider starting Belief Verification via intent analysis.")
                    self.decider.command_executor._run_action("verify_batch")
                    if self.decider.heartbeat:
                        self.decider.heartbeat.force_task("verify", 3, "Intent: Verify Beliefs")
                    return "🕵️ Initiating belief verification."
                else:
                    self.decider.log("🤖 Decider starting Verification Batch via intent analysis.")
                    self.decider.command_executor._run_action("verify_batch")
                    return "🕵️ Verification batch triggered."

            elif "[THINK]" in intent_response:
                topic = intent_response.split("]", 1)[1].strip()
                self.decider.thought_generator.perform_thinking_chain(topic)
                return f"🧠 Finished thinking about: {topic}"

            elif "[SIMULATE]" in intent_response:
                premise = intent_response.split("]", 1)[1].strip()
                if "simulate_counterfactual" in self.decider.actions:
                    return f"🌌 Simulation Result: {self.decider.actions['simulate_counterfactual'](premise)}"

        return None

    def _retrieve_context(self, user_text: str, settings: Dict, executor=None) -> Dict[str, Any]:
        """
        Retrieves context from various sources (Memory, Docs, Meta-Memory, Binah) in parallel.
        """
        local_executor = None
        submit_fn = executor.submit if executor else None

        if not submit_fn:
            local_executor = ThreadPoolExecutor(max_workers=5)
            submit_fn = local_executor.submit

        try:
            # 1. Start Embedding (Network)
            future_embedding = submit_fn(
                compute_embedding,
                user_text,
                settings.get("base_url"),
                settings.get("embedding_model")
            )

            # 2. Start Memory Retrieval (DB)
            t_retrieval = time.time()
            def fetch_combined():
                # Fetch recent non-daydream items (chat) and general recent items
                chat_mems = self.decider.memory_store.get_recent_filtered(limit=10, exclude_sources=['daydream'])
                recent_mems = self.decider.memory_store.list_recent(limit=5)
                return chat_mems, recent_mems
            future_combined = submit_fn(fetch_combined)
            future_critical = submit_fn(self._get_critical_memories)
            logging.debug(f"⏱️ [Tiferet] Memory retrieval submission took {time.time()-t_retrieval:.3f}s")

            # 3. Start Summary Retrieval (DB)
            def get_summary():
                if hasattr(self.decider.meta_memory_store, 'get_by_event_type'):
                    summaries = self.decider.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1)
                    if summaries:
                        return summaries[0]['text']
                return ""
            future_summary = submit_fn(get_summary)

            # Wait for embedding to proceed with Semantic Search & RAG
            query_embedding = future_embedding.result()

            # 4. Start Semantic Search (FAISS/DB)
            future_semantic = submit_fn(self.decider.memory_store.search, query_embedding, limit=10, target_affect=self.decider.mood)

            # 4.5 Start Meta-Memory Semantic Search (Autobiographical Memory)
            future_meta_semantic = submit_fn(self.decider.meta_memory_store.search, query_embedding, limit=3)

            # 5. Start RAG (FAISS/DB)
            future_rag = None
            if self._should_trigger_rag(user_text):
                t_rag = time.time()
                self.decider.log(f"📚 [RAG] Initiating document search for: '{user_text}'")
                def perform_rag():
                    doc_results = self.decider.document_store.search_chunks(query_embedding, top_k=5)
                    filename_matches = self.decider.document_store.search_filenames(user_text)
                    return doc_results, filename_matches
                future_rag = submit_fn(perform_rag)
                logging.debug(f"⏱️ [Tiferet] RAG submission took {time.time()-t_rag:.3f}s")

            # Gather all results
            t_gather = time.time()
            chat_items, recent_items = future_combined.result()
            critical_items = future_critical.result()
            summary_text = future_summary.result()
            semantic_items = future_semantic.result()
            meta_semantic_items = future_meta_semantic.result()

            # --- Active Association via Binah ---
            associative_items = []
            # Only expand if semantic results are weak or few
            if self.decider.binah and semantic_items and (len(semantic_items) < 3 or semantic_items[0][4] < 0.8):
                # Limit seeds to top 3 to reduce DB queries
                seed_ids = [item[0] for item in semantic_items[:3]]
                assoc_results = self.decider.binah.expand_associative_context(seed_ids, limit=3)
                # Convert to tuple format: (id, type, subject, text, similarity)
                for res in assoc_results:
                    # Use strength as similarity score
                    associative_items.append((res['id'], res['type'], "Association", res['text'], res['strength']))

                if associative_items:
                    self.decider.log(f"🔗 Binah: Expanded context with {len(associative_items)} associated memories.")

            doc_results = []
            filename_matches = []
            if future_rag:
                doc_results, filename_matches = future_rag.result()
            logging.debug(f"⏱️ [Tiferet] Context gathering result wait took {time.time()-t_gather:.3f}s")

            return {
                "chat_items": chat_items,
                "recent_items": recent_items,
                "critical_items": critical_items,
                "summary_text": summary_text,
                "semantic_items": semantic_items,
                "meta_semantic_items": meta_semantic_items,
                "associative_items": associative_items,
                "doc_results": doc_results,
                "filename_matches": filename_matches
            }
        finally:
            if local_executor:
                local_executor.shutdown(wait=False)

    def _construct_system_prompt(self, user_text: str, history: List[Dict], context_data: Dict[str, Any], settings: Dict) -> str:
        """
        Constructs the final system prompt by merging context items within the token budget.
        """
        # Unpack context
        chat_items = context_data.get("chat_items", [])
        recent_items = context_data.get("recent_items", [])
        critical_items = context_data.get("critical_items", [])
        summary_text = context_data.get("summary_text", "")
        semantic_items = context_data.get("semantic_items", [])
        meta_semantic_items = context_data.get("meta_semantic_items", [])
        associative_items = context_data.get("associative_items", [])
        doc_results = context_data.get("doc_results", [])
        filename_matches = context_data.get("filename_matches", [])

        # --- Token Budget Calculation ---
        context_window = int(settings.get("context_window", 4096))
        max_gen_tokens = int(settings.get("max_tokens", 800))
        # Approx 3 chars per token for safety
        total_budget_chars = context_window * 3

        # Reserve space for System Prompt Base, User Text, History, and Generation (approx 2000 chars reserved)
        history_chars = sum(len(m.get('content', '')) for m in history)
        reserved_chars = len(settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)) + len(user_text) + history_chars + (max_gen_tokens * 3) + 500
        available_chars = max(1000, total_budget_chars - reserved_chars)
        self.decider.log(f"💰 Context Budget: {available_chars} chars available for Memory/RAG (Window: {context_window})")

        # Merge and deduplicate
        memory_map = {}
        for item in critical_items:
            memory_map[item[0]] = item
        for item in semantic_items:
            memory_map[item[0]] = (item[0], item[1], item[2], item[3])
        for item in recent_items:
            memory_map[item[0]] = item
        for item in chat_items:
            memory_map[item[0]] = item

        for item in associative_items:
            if item[0] not in memory_map:
                memory_map[item[0]] = (item[0], item[1], item[2], item[3])

        final_memory_items = list(memory_map.values())

        context_blocks = []

        # Layer 1: Session Summary (High-level grounding)
        if summary_text:
            context_blocks.append(f"PREVIOUS SESSION SUMMARY:\n{summary_text}\n\n")

        if final_memory_items:
            user_mems = []
            assistant_identities = []
            assistant_goals = []
            assistant_other = []
            other_mems = []
            autobiographical_mems = []

            for item in final_memory_items:
                _id, _type, subject, mem_text = item[:4]
                if subject and subject.lower() == 'user':
                    user_mems.append(f"- [{_type}] {mem_text}")
                elif subject and subject.lower() == 'assistant':
                    if _type == 'IDENTITY':
                        assistant_identities.append(f"- {mem_text}")
                    elif _type == 'GOAL':
                        assistant_goals.append(f"- {mem_text}")
                    else:
                        assistant_other.append(f"- [{_type}] {mem_text}")
                else:
                    other_mems.append(f"- [{_type}] [{subject}] {mem_text}")

            for m in meta_semantic_items:
                # m is dict
                autobiographical_mems.append(f"- [{m['event_type']}] {m['text']}")

            mem_block = ""
            if user_mems: mem_block += "User Profile (You are talking to):\n" + "\n".join(user_mems) + "\n\n"
            if assistant_identities: mem_block += "Assistant Identity (Who you are):\n" + "\n".join(assistant_identities) + "\n\n"
            if assistant_goals: mem_block += "CURRENT OBJECTIVES (Your internal goals):\n" + "\n".join(assistant_goals) + "\n\n"
            if assistant_other: mem_block += "Assistant Knowledge/State:\n" + "\n".join(assistant_other) + "\n\n"
            if autobiographical_mems: mem_block += "Autobiographical Context (Your History):\n" + "\n".join(autobiographical_mems) + "\n\n"
            if other_mems: mem_block += "Other Context:\n" + "\n".join(other_mems) + "\n\n"
            context_blocks.append(mem_block)

        # 2. RAG: Retrieve Documents
        if doc_results or filename_matches:
            doc_context = "Relevant document information:\n"
            if filename_matches:
                doc_context += "Found documents with matching names:\n" + "\n".join([f"- {fn}" for fn in filename_matches]) + "\n\n"
            if doc_results:
                doc_context += "Relevant excerpts from content:\n"
                for result in doc_results:
                    excerpt = result['text'][:300]
                    doc_context += f"- From '{result['filename']}': {excerpt}...\n"
                doc_context += "\n"
            context_blocks.append(doc_context)

        # Apply Budget
        memory_context = self._enforce_context_budget(context_blocks, available_chars)

        # 3. Construct System Prompt
        base_prompt = settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

        # Get stream from Decider
        stream = getattr(self.decider, 'stream_of_consciousness', [])

        if self.decider.yesod:
            system_prompt = self.decider.yesod.get_dynamic_system_prompt(base_prompt, stream_of_consciousness=stream)
        else:
            system_prompt = base_prompt
            if stream:
                system_prompt += "\n\nSTREAM OF CONSCIOUSNESS:\n" + "\n".join(stream[-5:])

        # GLOBAL WORKSPACE (Spotlight)
        if self.decider.global_workspace:
            gw_context = self.decider.global_workspace.get_context()
            if gw_context and gw_context != "Mind is empty.":
                system_prompt += f"\n\nGLOBAL WORKSPACE (Active Attention):\n{gw_context}"

        # DYNAMIC STRATEGY INJECTION
        # 1. Search for RULE/STRATEGY memories relevant to the user's input
        active_rules = self.decider.memory_store.get_active_by_type("RULE")

        relevant_strategies = []
        user_input_lower = user_text.lower()

        for rule in active_rules:
            # rule structure: (id, subject, text, source) from get_active_by_type
            text = rule[2]
            if "STRATEGY:" in text:
                # Check trigger (simple heuristic: if any word in strategy text matches user input)
                if any(word in user_input_lower for word in text.lower().split() if len(word) > 4):
                    relevant_strategies.append(text)

        if relevant_strategies:
            system_prompt += "\n🧠 LEARNED STRATEGIES (APPLY THESE):\n" + "\n".join(relevant_strategies)

        if memory_context:
            system_prompt = memory_context + system_prompt

        # Log final prompt length
        prompt_tokens = count_tokens(system_prompt)
        self.decider.log(f"📝 Final Prompt: {len(system_prompt)} chars (~{prompt_tokens} tokens)")

        # NEW: Self-Improvement Prompt (Appended after memory context)
        self_improvement_prompt = settings.get("self_improvement_prompt", "")
        if self_improvement_prompt:
            system_prompt += "\n\n" + self_improvement_prompt

        return system_prompt

    def _enforce_context_budget(self, context_blocks: List[str], max_chars: int) -> str:
        """Fit context blocks into the character budget."""
        # Prioritize blocks: The last block is usually the most critical (recent docs/memories)
        # We will fill from the end backwards

        final_blocks = []
        current_len = 0

        for block in reversed(context_blocks):
            if not block: continue
            block_len = len(block)

            if current_len + block_len <= max_chars:
                final_blocks.insert(0, block)
                current_len += block_len
            else:
                remaining = max_chars - current_len
                if remaining > 100:
                    # Keep the END of the block if we have to truncate (usually more relevant)
                    truncated_block = "... [Context Truncated] ...\n" + block[-remaining:]
                    final_blocks.insert(0, truncated_block)
                break

        return "".join(final_blocks)

    def _get_critical_memories(self):
        """Retrieve always-active memories (Identity, Permission, Goals, Rules)."""
        now = time.time()
        if now - self.decider._last_critical_update < 300 and self.decider._critical_memories_cache:
            return self.decider._critical_memories_cache

        critical_types = ["IDENTITY", "PERMISSION", "RULE", "GOAL", "CURIOSITY_GAP"]
        mems = []
        for t in critical_types:
            # get_active_by_type returns (id, subject, text, source, confidence)
            items = self.decider.memory_store.get_active_by_type(t)
            # Sort by ID descending to prioritize recent items
            items.sort(key=lambda x: x[0], reverse=True)

            # Limit high-volume types to prevent context bloat
            if t in ["RULE", "GOAL", "CURIOSITY_GAP"]:
                items = items[:5]

            daydream_count = 0
            daydream_limit = 3  # Reduced from 5

            for item in items:
                mid, subj, text, source = item[0], item[1], item[2], item[3]
                if source == 'daydream':
                    if daydream_count >= daydream_limit:
                        continue
                    daydream_count += 1
                # Format to match list_recent: (id, type, subject, text, source, verified, flags)
                mems.append((mid, t, subj, text, source, 1, None))

        self.decider._critical_memories_cache = mems
        self.decider._last_critical_update = now
        return mems

    def _extract_and_save_memories(self, user_text, assistant_text, settings):
        """Extract memories and run arbiter logic"""
        try:
            # Use a simplified instruction to defer to the System Prompt (which is configurable in settings)
            # This prevents the hardcoded instruction in lm.py from overriding the detailed settings prompt
            custom_instr = "Analyze the conversation. Extract all durable memories (Identity, Facts, Goals, etc.) based on the System Rules. Return JSON."

            candidates, _ = extract_memories_llm(
                user_text=user_text,
                assistant_text=assistant_text,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                embedding_model=settings.get("embedding_model"),
                memory_extractor_prompt=settings.get("memory_extractor_prompt", DEFAULT_MEMORY_EXTRACTOR_PROMPT),
                custom_instruction=custom_instr,
                stop_check_fn=self.decider.stop_check
            )

            # Add source metadata and filter by confidence
            for c in candidates:
                c["source"] = "assistant"
                try:
                    c["confidence"] = float(c.get("confidence", 0.9))
                except (ValueError, TypeError):
                    self.decider.log(f"⚠️ Invalid confidence value '{c.get('confidence')}'. Defaulting to 0.5.")
                    c["confidence"] = 0.5

            # Filter: skip low-confidence
            candidates = [c for c in candidates if c.get("confidence", 0.5) > 0.4]

            if not candidates:
                return

            # Batch add to Reasoning Store
            reasoning_entries = []
            for c in candidates:
                reasoning_entries.append({
                    "content": c["text"],
                    "source": c.get("source", "assistant"),
                    "confidence": c.get("confidence", 0.9)
                })

            if reasoning_entries:
                self.decider.reasoning_store.add_batch(reasoning_entries)
                self.decider.log(f"🧠 Added {len(reasoning_entries)} candidates to Reasoning Store.")

            # Arbiter promotion (batch)
            promoted_ids = self.decider.arbiter.consider_batch(candidates)

            if promoted_ids:
                self.decider.log(f"🧠 Promoted {len(promoted_ids)} memory item(s).")

        except Exception as e:
            self.decider.log(f"Memory extraction error: {e}\nimport traceback; traceback.print_exc()") # Simplified traceback print

    def _should_trigger_rag(self, text: str) -> bool:
        """Determine if we should run RAG based on user input."""
        text = text.strip().lower()

        # More restrictive keywords
        force_keywords = {
            "search for", "find in", "document", "file", "pdf", "docx",
            "read the", "summarize the", "according to", "lookup"
        }
        if any(kw in text for kw in force_keywords): return True

        if "?" in text:
            conversational = ["how are you", "how is it going", "what's up", "who are you", "what is your name", "hi", "hello"]
            if any(c in text for c in conversational): return False

            # Only trigger if it looks like a knowledge-seeking question
            knowledge_triggers = ["what is", "who is", "tell me about", "explain", "how does"]
            return any(k in text for k in knowledge_triggers)

        # Default to False to prevent slowdown on statements
        return False

    def _analyze_intent(self, text: str) -> str:
        """Use LLM to classify user intent for ambiguous commands."""
        settings = self.decider.get_settings()
        prompt = (
            f"Analyze the user's request: '{text}'\n"
            "Classify the intent into one of these categories:\n"
            "1. [LEARN]: User explicitly asks the AI to go research, study, or read up on a NEW topic (triggers background research loop).\n"
            "2. [THINK]: User wants the AI to think step-by-step or analyze a topic deeply (triggers Chain of Thought).\n"
            "3. [VERIFY]: User wants to check facts or sources (triggers Verification).\n"
            "4. [CHAT]: Standard conversation, questions, 'Teach me', 'Explain', or requests for existing knowledge.\n\n"
            "5. [SIMULATE]: User asks a 'What if' or counterfactual question requiring world modeling.\n\n"
            "Output format: [INTENT] Topic (if applicable)\n"
            "Examples:\n"
            "- 'Learn about neurology' -> [LEARN] Neurology\n"
            "- 'Go study the files on space' -> [LEARN] Space\n"
            "- 'Teach me about neurology' -> [CHAT]\n"
            "- 'Explain quantum physics' -> [CHAT]\n"
            "- 'Summarize the meeting notes' -> [CHAT]\n"
            "- 'Think about the meaning of life' -> [THINK] Meaning of life\n"
            "- 'Hello' -> [CHAT]\n"
            "- 'What if the sun disappeared?' -> [SIMULATE] Sun disappeared"
        )

        response = run_local_lm(
            messages=[{"role": "user", "content": "Classify intent."}],
            system_prompt=prompt,
            temperature=0.1, # Low temp for classification
            max_tokens=50,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model"),
            stop_check_fn=self.decider.stop_check
        )
        return response.strip()
