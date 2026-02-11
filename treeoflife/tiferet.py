import re
import ast
import time
import random
import platform
import operator
import difflib
import os
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List
from lm import compute_embedding, run_local_lm, extract_memory_candidates, DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT, _parse_json_array_loose

try:
    import psutil
except ImportError:
    psutil = None

class Decider:
    """
    Autonomous decision module.
    Controls system parameters (Temperature) and operational modes (Daydream, Verification).
    """
    def __init__(
        self,
        get_settings_fn: Callable[[], Dict],
        update_settings_fn: Callable[[Dict], None],
        memory_store,
        document_store,
        reasoning_store,
        arbiter,
        meta_memory_store,
        actions: Dict[str, Callable[[], None]],
        log_fn: Callable[[str], None] = print,
        chat_fn: Optional[Callable[[str, str], None]] = None,
        get_chat_history_fn: Optional[Callable[[], List[Dict]]] = None,
        event_bus: Optional[Any] = None,
        stop_check_fn: Callable[[], bool] = lambda: False,
        hesed=None,
        gevurah=None,
        hod=None,
        netzach=None,
        binah=None,
        dialectics=None,
        keter=None,
        malkuth=None
    ):
        self.get_settings = get_settings_fn
        self.update_settings = update_settings_fn
        self.memory_store = memory_store
        self.document_store = document_store
        self.reasoning_store = reasoning_store
        self.arbiter = arbiter
        self.meta_memory_store = meta_memory_store
        self.actions = actions
        self.log = log_fn
        self.chat_fn = chat_fn
        self.get_chat_history = get_chat_history_fn or (lambda: [])
        self.event_bus = event_bus
        self.stop_check = stop_check_fn
        self.hesed = hesed
        self.gevurah = gevurah
        self.hod_force = hod
        self.netzach_force = netzach
        self.binah = binah
        self.dialectics = dialectics
        self.keter = keter
        self.malkuth = malkuth
        self.cycle_count = 0
        self.hod_just_ran = False
        self.action_taken_in_observation = False
        self.cycles_remaining = 0
        self.current_task = "wait"  # wait, daydream, verify
        self.last_action_was_speak = False
        self.forced_stop_cooldown = False
        self.daydream_mode = "auto" # auto, read, insight
        self.daydream_topic = None
        self.last_tool_usage = 0
        self.consecutive_daydream_batches = 0
        self.wait_start_time = 0
        self.last_daydream_time = 0
        self.last_token_adjustment_time = 0
        self.planning_lock = threading.Lock()
        self.pending_maintenance = [] # List of actions recommended by Hod
        self.latest_netzach_signal = None
        
        # Capture baselines for relative limits
        settings = self.get_settings()
        self.baseline_temp = float(settings.get("default_temperature", 0.7))
        self.baseline_tokens = int(settings.get("default_max_tokens", 800))

    def increase_temperature(self, amount: float = None):
        """Increase temperature by specified amount (percentage) up to 20%."""
        limit_pct = 0.20
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_temperature(1.0 + step)

    def decrease_temperature(self, amount: float = None):
        """Decrease temperature by specified amount (percentage) up to 20%."""
        limit_pct = 0.20
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_temperature(1.0 - step)

    def _adjust_temperature(self, multiplier: float):
        settings = self.get_settings()
        current = float(settings.get("temperature", 0.7))
        new_temp = round(current * multiplier, 2)
        
        limit_pct = 0.20
        
        # Calculate bounds based on BASELINE, not current
        lower_bound = self.baseline_temp * (1.0 - limit_pct)
        upper_bound = self.baseline_temp * (1.0 + limit_pct)
        
        new_temp = max(lower_bound, min(new_temp, upper_bound))
        # Absolute safety floor
        new_temp = max(0.01, new_temp)
        
        settings["temperature"] = new_temp
        self.update_settings(settings)
        self.log(f"🌡️ Decider adjusted temperature to {new_temp} (x{multiplier})")
        
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event(
                event_type="DECIDER_ACTION",
                subject="Assistant",
                text=f"Adjusted temperature to {new_temp}"
            )

    def increase_tokens(self, amount: float = None):
        """Increase max_tokens by specified amount (percentage) up to 20%."""
        limit_pct = 0.20
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_tokens(1.0 + step)

    def decrease_tokens(self, amount: float = None):
        """Decrease max_tokens by specified amount (percentage) up to 20%."""
        limit_pct = 0.20
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_tokens(1.0 - step)

    def _adjust_tokens(self, multiplier: float):
        settings = self.get_settings()
        current = int(settings.get("max_tokens", 800))
        new_tokens = int(current * multiplier)
        
        limit_pct = 0.20
        
        # Calculate bounds based on BASELINE
        lower_bound = int(self.baseline_tokens * (1.0 - limit_pct))
        upper_bound = int(self.baseline_tokens * (1.0 + limit_pct))
        
        new_tokens = max(lower_bound, min(new_tokens, upper_bound))
        # Absolute safety floor
        new_tokens = max(1024, new_tokens)
        
        settings["max_tokens"] = new_tokens
        self.update_settings(settings)
        self.last_token_adjustment_time = time.time()
        self.log(f"📏 Decider adjusted max_tokens to {new_tokens} (x{multiplier})")
        
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event(
                event_type="DECIDER_ACTION",
                subject="Assistant",
                text=f"Adjusted max_tokens to {new_tokens}"
            )

    def start_daydream(self):
        self.log("🤖 Decider starting single Daydream cycle.")
        self.current_task = "daydream"
        self.cycles_remaining = 1
        if time.time() - self.last_daydream_time < 60:
            self.daydream_mode = "insight"
        else:
            self.daydream_mode = "auto"

    def start_verification_batch(self):
        self.log("🤖 Decider starting Verification Batch.")
        self.current_task = "verify"
        self.cycles_remaining = 1

    def verify_all(self):
        self.log("🤖 Decider starting Full Verification.")
        self.current_task = "verify_all"
        self.cycles_remaining = 1

    def start_daydream_loop(self):
        self.log("🤖 Decider enabling Daydream Loop.")
        self._run_action("start_loop")
        self.current_task = "daydream"
        settings = self.get_settings()
        self.cycles_remaining = int(settings.get("daydream_cycle_limit", 10))
        self.last_daydream_time = time.time()

    def stop_daydream(self):
        self.log("🤖 Decider stopping daydream.")
        self._run_action("stop_daydream")

    def report_forced_stop(self):
        """Handle forced stop from UI."""
        self.log("🤖 Decider: Forced stop received. Entering cooldown.")
        self.cycles_remaining = 0
        self.forced_stop_cooldown = True

    def wake_up(self, reason: str = "External Stimulus"):
        """Force the Decider to wake up from a wait state."""
        self.log(f"🤖 Decider: Waking up due to {reason}.")
        self.current_task = "waking_up"
        self.cycles_remaining = 0

    def create_note(self, content: str):
        """Manually create an Assistant Note (NOTE)."""
        if self.malkuth:
            return self.malkuth.create_note(content)
        self.log("⚠️ Malkuth not available for creating note.")

    def edit_note(self, mem_id: int, content: str):
        """Edit an Assistant Note by superseding it."""
        if self.malkuth:
            self.malkuth.edit_note(mem_id, content)

    def _on_reminder_due(self, event):
        """Handles REMINDER_DUE event from Netzach."""
        reminder_text = event.data.get("text")
        if reminder_text:
            self.log(f"⏰ Decider received reminder: {reminder_text}")
            if self.chat_fn:
                self.chat_fn("Assistant", f"⏰ Reminder: {reminder_text}")

    def create_goal(self, content: str):
        """Autonomously create a new GOAL memory."""
        # Deterministic identity for goals
        identity = self.memory_store.compute_identity(content, "GOAL")
        
        mid = self.memory_store.add_entry(
            identity=identity,
            text=content,
            mem_type="GOAL",
            subject="Assistant",
            confidence=1.0,
            source="decider_autonomous"
        )
        self.log(f"🎯 Decider autonomously created GOAL (ID: {mid}): {content}")
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event("GOAL_CREATED", "Assistant", f"Created Goal: {content}")
        return mid

    def run_post_chat_decision_cycle(self):
        """Initiates the decision process after a chat interaction is complete."""
        # This is called after a chat reply has been sent.
        # If a natural language command already set up a task (cycles > 0), don't overwrite it.
        if self.cycles_remaining > 0 and self.current_task != "wait":
            self.log(f"🤖 Decider: Chat complete. Resuming assigned task ({self.current_task}).")
            return

        # The decider should now figure out what to do next.
        self.log("🤖 Decider: Chat complete. Initiating post-chat decision cycle.")
        self._decide_next_batch()

    def run_cycle(self):
        """
        Core execution loop.
        Rules the Daydreamer, Verification, Hod, and Observer.
        """
        # If we are in a wait state, check if we should wake up or just return
        if self.current_task == "wait":
            return

        if self.stop_check():
            return

        self.action_taken_in_observation = False

        # 1. Netzach (Observer) - Always watching in the background
        # "when system is not functioning netzach is always working in behind silently"
        self._run_action("run_observer")

        if self.stop_check():
            return

        # 2. Execute Planned Task
        if self.cycles_remaining > 0:
            if self.current_task == "daydream":
                # Check concurrency setting
                settings = self.get_settings()
                concurrency = int(settings.get("concurrency", 1))
                
                # Determine batch size (don't exceed remaining cycles or concurrency limit)
                batch_size = 1
                if concurrency > 1:
                    batch_size = min(self.cycles_remaining, concurrency)
                
                if batch_size > 1 and "start_daydream_batch" in self.actions:
                    self.actions["start_daydream_batch"](batch_size)
                    self.cycles_remaining -= batch_size
                    # Reset Hod lock for substantive actions
                    self.hod_just_ran = False
                else:
                    self._run_action("start_daydream")
                    self.cycles_remaining -= 1
                
                self.last_daydream_time = time.time()
            elif self.current_task == "verify":
                self._run_action("verify_batch")
                self.cycles_remaining -= 1
            elif self.current_task == "verify_all":
                self._run_action("verify_all")
                self.cycles_remaining -= 1
            
            self.log(f"🤖 Decider: {self.current_task.capitalize()} cycle complete. Remaining: {self.cycles_remaining}")
        
        # 3. If plan finished, decide next steps
        if self.cycles_remaining <= 0:
            self._decide_next_batch()

        # 4. Post-Job Analysis (Hod)
        # Only run Hod if we actually did something substantive (not just waiting)
        if self.current_task != "wait":
            self._run_action("run_hod")
            
        # 5. Authorize Maintenance (Pruning/Refuting) based on Hod's analysis
        self._authorize_maintenance()

    def _decide_next_batch(self):
        """Decide what to do next: Wait, Daydream loop, or Verify loop."""
        # Prevent concurrent planning (e.g. from Chat trigger AND Background loop)
        if not self.planning_lock.acquire(blocking=False):
            self.log("🤖 Decider: Planning already in progress. Skipping.")
            return

        try:
            self.log("🤖 Decider: Planning next batch...")
            settings = self.get_settings()

            # Fetch Active Goals early (Fix for UnboundLocalError in Reflex path)
            all_items = self.memory_store.list_recent(limit=None)
            active_goals = [m for m in all_items if len(m) > 1 and m[1] == 'GOAL']
            active_goals.sort(key=lambda x: x[0], reverse=True)
            active_goals = active_goals[:5]
            
            # Check for Necessity Goals (Self-Generated Objectives)
            necessity_goals = [g for g in active_goals if g[4] == "model_stress"]
            if necessity_goals:
                self.log(f"🚨 Decider detected {len(necessity_goals)} Necessity Goals (Model Stress). Prioritizing.")

            # Capture signal before processing/clearing
            current_netzach_signal = "Unknown"
            if self.latest_netzach_signal:
                current_netzach_signal = self.latest_netzach_signal.get("signal", "Unknown")
            
            # Check Netzach's signal
            self._check_netzach_signal()
            
            # Gather context for decision making
            recent_mems = self.memory_store.list_recent(limit=5)
            chat_mems = self._get_recent_chat_memories(limit=3)
            
            # Merge unique memories (prioritizing recent, but ensuring chat memories are included)
            all_mems = {m[0]: m for m in recent_mems}
            for m in chat_mems:
                all_mems[m[0]] = m
            display_mems = sorted(all_mems.values(), key=lambda x: x[0], reverse=True)[:8]

            recent_meta = self.meta_memory_store.list_recent(limit=5)
            
            # Fetch latest session summary (Layer 1: High-level context)
            latest_summary = None
            last_summary_time = 0
            if hasattr(self.meta_memory_store, 'get_by_event_type'):
                summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1)
                if summaries:
                    latest_summary = summaries[0]
                    last_summary_time = latest_summary.get('created_at', 0)

            chat_hist = self.get_chat_history()[-5:]
            
            # Calculate Hesed/Gevurah
            hesed_score = self.hesed.calculate() if self.hesed else 1.0
            gevurah_score = self.gevurah.calculate() if self.gevurah else 0.0
            self.log(f"⚖️ Cognitive State: Hesed={hesed_score:.2f}, Gevurah={gevurah_score:.2f}")
            
            # Get Signals from Hod and Netzach
            hod_signal = self.hod_force.analyze() if self.hod_force else "Unknown"
            
            # Use the captured signal if the force object is missing (common in this architecture)
            if self.netzach_force and hasattr(self.netzach_force, 'analyze'):
                netzach_signal = self.netzach_force.analyze()
            else:
                netzach_signal = current_netzach_signal

            self.log(f"⚖️ Forces: Hod='{hod_signal}', Netzach='{netzach_signal}'")

            # --- FAST PATH (Reflex) ---
            # Optimization: Skip LLM for obvious states to reduce latency
            reflex_action = None
            reflex_reason = ""

            if necessity_goals:
                # If we have a necessity goal, we MUST act on it.
                reflex_action = "GOAL_ACT"
                reflex_reason = "Necessity Goal (Model Stress) requires resolution."
                target_goal_id = necessity_goals[0][0]
                target_goal_text = necessity_goals[0][3]
            elif gevurah_score > 0.85:
                reflex_action = "VERIFY"
                reflex_reason = "High Gevurah pressure (Constraint > 0.85)"
            elif netzach_signal == "NO_MOMENTUM":
                # Only force daydream if we have NO active goals. 
                # If goals exist, we want the LLM to choose [GOAL_ACT].
                if not active_goals:
                    reflex_action = "DAYDREAM"
                    reflex_reason = "Netzach signal: No Momentum & No Goals"
            elif netzach_signal == "LOW_NOVELTY" and hesed_score > 0.4 and not active_goals:
                reflex_action = "DAYDREAM"
                reflex_reason = "Netzach signal: Low Novelty"
            
            if reflex_action:
                self.log(f"⚡ Decider Reflex Triggered: {reflex_action} due to {reflex_reason}")
                if reflex_action == "VERIFY":
                    self.current_task = "verify"
                    self.cycles_remaining = 2
                elif reflex_action == "DAYDREAM":
                    self.current_task = "daydream"
                    self.cycles_remaining = 1
                    self.daydream_mode = "auto"
                elif reflex_action == "GOAL_ACT":
                    self.log(f"⚡ Decider Reflex: Acting on Necessity Goal {target_goal_id}")
                    self.perform_thinking_chain(f"Resolve Necessity Goal: {target_goal_text}")
                    # FIX: Mark goal as completed to prevent infinite loop
                    if "remove_goal" in self.actions:
                        self.actions["remove_goal"](target_goal_id)
                    self.consecutive_daydream_batches = 0
                return

            recent_docs = self.document_store.list_documents(limit=5)
            
            
            # Fetch Curiosity Gaps (Urgent Questions)
            curiosity_gaps = self.memory_store.get_active_by_type("CURIOSITY_GAP")
            curiosity_gaps = curiosity_gaps[:3] # Limit to 3
            
            # Fetch Strategies (Rules)
            strategies = self.memory_store.get_active_by_type("RULE")
            strategy_section = ""
            if strategies:
                # Filter for strategies (heuristic: starts with STRATEGY or contains it)
                strategy_list = [f"- {s[2]}" for s in strategies if "STRATEGY:" in s[2]]
                if strategy_list:
                    strategy_section = "🧠 LEARNED STRATEGIES:\n" + "\n".join(strategy_list) + "\n\n"

            # Fetch Recent Reasoning (to see tool outputs)
            recent_reasoning = []
            if hasattr(self.reasoning_store, 'list_recent'):
                recent_reasoning = self.reasoning_store.list_recent(limit=5)
            
            context = "CONTEXT:\n"
            if strategy_section:
                context += strategy_section
            context += "System Signals (The Forces):\n"
            context += f"- Gevurah (Constraint): {gevurah_score:.2f} (Pressure)\n"
            context += f"- Hesed (Expansion): {hesed_score:.2f} (Safety)\n"
            context += f"- Hod (Form): {hod_signal}\n"
            context += f"- Netzach (Endurance): {netzach_signal}\n\n"
            if latest_summary:
                context += f"Last Session Summary:\n{latest_summary['text']}\n\n"
            if chat_hist:
                context += "Chat History:\n" + "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in chat_hist]) + "\n"
            if display_mems:
                context += "Recent Memories (Note: REFUTED_BELIEF means the idea is FALSE/REJECTED):\n" + "\n".join([f"- [{m[1]}] [{m[2]}] {m[3][:200]}" for m in display_mems]) + "\n"
            if curiosity_gaps:
                context += "⚠️ CURIOSITY GAPS (CONFLICTS DETECTED - ASK USER):\n" + "\n".join([f"- {g[2]}" for g in curiosity_gaps]) + "\n"
            if active_goals:
                context += "Active Goals:\n" + "\n".join([f"- [ID: {m[0]}] {m[3][:150]}..." for m in active_goals]) + "\n"
            if recent_meta:
                context += "Recent Events:\n" + "\n".join([f"- {m[3][:100]}..." for m in recent_meta]) + "\n"
            if recent_reasoning:
                context += "Recent Thoughts:\n" + "\n".join([f"- {r.get('content', str(r))[:80]}..." if isinstance(r, dict) else f"- {str(r)[:80]}..." for r in recent_reasoning]) + "\n"
            if recent_docs:
                context += "Available Documents:\n" + "\n".join([f"- {d[1]}" for d in recent_docs]) + "\n"
            
            context += f"Current Task: {self.current_task}\n"
            if self.current_task == "wait" and self.wait_start_time > 0:
                wait_duration = time.time() - self.wait_start_time
                context += f"INFO: Currently in WAIT state for {int(wait_duration)} seconds.\n"

            # Safety Truncation (Estimate 1 char ~= 0.25 tokens, keep safe buffer)
            MAX_CONTEXT_CHARS = 10000  # Reduced to ~2500 tokens to be safe

            if len(context) > MAX_CONTEXT_CHARS:
                # NEW: Attempt to compress reasoning before truncating
                if self.actions.get("compress_reasoning"):
                    self.log("⚠️ Context budget exceeded. Attempting semantic compression...")
                    self.actions["compress_reasoning"]()

                # Keep the first 20% (recent/relevant) and last 20% (immediate context)
                # Cut the middle.
                cut_size = int(MAX_CONTEXT_CHARS * 0.2)
                context = (
                    context[:cut_size] 
                    + "\n\n...[CONTEXT TRUNCATED FOR MEMORY SAFETY]...\n\n" 
                    + context[-cut_size:]
                )

            # Strategic Analysis: Force thinking before deciding
            strategy_prompt = (
                "You are the Background Executive. Your goal is to maintain the knowledge base (Verify, Consolidate) and generate insights (Daydream) without disrupting the user.\n"
                "Review the Context above.\n"
                "Harmonize the System Signals to choose a just action:\n"
                "1. Hod says (Form): Is the structure stable or breaking?\n"
                "   - If Breaking/Undefined -> Prioritize [VERIFY] to restore form.\n"
                "2. Netzach says (Endurance): Do we have momentum or are we exhausted?\n"
                "   - If Exhausted -> Prioritize [SUMMARIZE] to clear context or [WAIT] to rest.\n"
                "   - If NO_MOMENTUM -> System is stuck. MUST [GOAL_ACT] or [DAYDREAM] to restart momentum.\n"
                "3. Gevurah says (Constraint): Is pressure too high?\n"
                "   - If > 0.7 -> Stop expansion. [VERIFY] or [GOAL_REMOVE].\n"
                "4. Hesed says (Expansion): Is it safe to explore?\n"
                "   - If > 0.5 -> Safe to [DAYDREAM] or [SEARCH_INTERNET].\n"
                "5. CURIOSITY GAPS: If present, you MUST prioritize asking the user for clarification using [SPEAK].\n"
                "Briefly reason about what should be done next.\n"
                "Output ONLY the reasoning (1-2 sentences)."
            )

            strategy_analysis = run_local_lm(
                messages=[{"role": "user", "content": context + "\nStatus: Analyzing situation..."}],
                system_prompt=strategy_prompt,
                temperature=0.7,
                max_tokens=150,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=self.stop_check
            )

            # Check for interruption during analysis
            if "[Interrupted]" in strategy_analysis:
                self.log("⚠️ Strategic analysis interrupted. Aborting decision cycle.")
                self.current_task = "wait"
                self.cycles_remaining = 0
                return

            self.log(f"🤔 Strategic Thought: {strategy_analysis}")
            if self.chat_fn:
                self.chat_fn("Decider", f"🤔 Thought: {strategy_analysis}")

            if hasattr(self.meta_memory_store, 'add_event'):
                self.meta_memory_store.add_event(
                    event_type="STRATEGIC_THOUGHT",
                    subject="Assistant",
                    text=f"Strategic Analysis: {strategy_analysis}"
                )

            context += f"\nStrategic Analysis: {strategy_analysis}\n"

            # Check if we just finished a heavy task to enforce cooldown
            # just_finished_heavy = self.current_task in ["daydream", "verify", "verify_all"]
            # Disabled to allow continuous operation as requested
            just_finished_heavy = False
            
            # Rate limit Daydreaming: Force variety if we've daydreamed recently
            allow_daydream = True
            if self.consecutive_daydream_batches >= 3:
                allow_daydream = False
                self.log("🤖 Decider: Daydreaming quota reached (3 batches). Forcing other options.")
                
            options_text = "Options:\n1. [WAIT]: Stop processing and wait for user input. Use this if the system is stable and no urgent goals exist.\n"
            
            if not self.forced_stop_cooldown and not just_finished_heavy:
                if allow_daydream:
                    options_text += "2. [DAYDREAM: N, MODE, TOPIC]: Run N cycles (1-5). MODE: 'READ', 'AUTO'. TOPIC: Optional subject. E.g., [DAYDREAM: 3, READ, Neurology]\n"
                    if active_goals:
                        options_text += "   (TIP: Use the TOPIC argument to focus on an active goal!)\n"
                options_text += "3. [VERIFY: N]: Run N batches of verification (1 to 3). Use this if you suspect hallucinations or haven't verified in a while.\n"
                options_text += "4. [VERIFY_ALL]: Run 1 full verification cycle. Use this rarely, only if deep cleaning is needed.\n"
                options_text += "5. [NOTE_CREATE: content]: Create a new Assistant Note. (e.g., [NOTE_CREATE: Remember to check X])\n"
                options_text += "6. [NOTE_EDIT: id, content]: Edit an existing Note by ID.\n"
                options_text += "7. [THINK: specific_topic]: Start a chain of thought (max 30 steps) to analyze a specific topic. Replace 'specific_topic' with the actual subject.\n"
                options_text += "7b. [DEBATE: topic]: Convene 'The Council' (Hesed/Gevurah) to debate a complex topic.\n"
                options_text += "7c. [SIMULATE: premise]: Run a counterfactual simulation based on the World Model. Use for 'What if' scenarios.\n"
                options_text += "8. [EXECUTE: tool_name, args]: Execute a system tool. Available: [CALCULATOR, CLOCK, DICE, SYSTEM_INFO].\n"
                options_text += "8b. [EXECUTE: PHYSICS, 'scenario']: Perform a reality check or Fermi estimation on a physical/biochemical scenario. Use this for sanity checks.\n"
                options_text += "8c. [EXECUTE: CAUSAL, 'treatment, outcome, context']: Perform Bayesian Causal Inference (DoWhy) to estimate p-values. E.g., [EXECUTE: CAUSAL, 'Insulin', 'Weight', 'Metabolic Syndrome'].\n"
                options_text += "9. [GOAL_ACT: goal_id]: Focus strategic thinking on a specific goal. Triggers a thinking chain to progress this goal.\n"
                options_text += "10. [GOAL_REMOVE: goal_id_or_text]: Mark a goal as COMPLETED. Use this when a goal is achieved to archive it.\n"
                options_text += "11. [GOAL_CREATE: text]: Autonomously create a new goal for yourself based on the situation (e.g., 'Research topic X').\n"
                options_text += "11. [LIST_DOCS]: List available documents to choose a topic for daydreaming.\n"
                options_text += "12. [READ_DOC: filename_or_id]: Read the content of a specific document found via LIST_DOCS.\n"
                options_text += "13. [SEARCH_MEM: query]: Actively search your long-term memory for specific information.\n"
                options_text += "16. [SUMMARIZE]: Summarize recent session activity into a meta-memory (Da'at).\n"
                options_text += "17. [WRITE_FILE: filename, content]: Write content to a file in the 'works' folder. E.g., [WRITE_FILE: journal.md, # Daily Log].\n"
                options_text += "18. [SEARCH_INTERNET: query, source]: Request external data. Source: WIKIPEDIA, PUBMED, or ARXIV. E.g., [SEARCH_INTERNET: Quantum Computing, ARXIV].\n"
            else:
                reason = "forced stop" if self.forced_stop_cooldown else "consecutive heavy task prevention"
                self.log(f"🤖 Decider: Daydream/Verify disabled for this turn due to {reason}.")

            if not self.last_action_was_speak:
                options_text += "4. [SPEAK: content]: Write a message to the user. Replace 'content' with the actual text. E.g., [SPEAK: I have updated the database.]\n"

            # Dynamic examples based on allowed actions
            example_outputs = "[WAIT]"
            if allow_daydream and not just_finished_heavy:
                example_outputs += ", [DAYDREAM: 3, READ, Neurology]"
            if not just_finished_heavy:
                example_outputs += ", [THINK: Seizure Types]"

            prompt = (
                "You are the Assistant. You control the cognitive cycle. "
                "The previous task batch is complete. "
                "Review the CONTEXT and your Strategic Analysis.\n\n"
                f"Your Strategic Analysis: '{strategy_analysis}'\n\n"
                "Now, select the SINGLE BEST command from the options below that directly implements your analysis.\n"
                "CRITICAL: If your analysis mentions a specific topic, you MUST use it in your command (e.g., use the TOPIC argument for DAYDREAM or THINK).\n"
                "CRITICAL: If there are unverified items, prefer [VERIFY] over [WAIT].\n"
                "CRITICAL: If there are documents to read and verification is done, prefer [DAYDREAM: 3, READ] over [WAIT].\n"
                "CRITICAL: If your analysis concludes to WAIT, output [WAIT]. If it concludes to ACT (Daydream, Verify, etc.), output that command.\n"
                f"{options_text}\n"
                f"Output ONLY the chosen command token (e.g., {example_outputs})."
            )
            
            response = run_local_lm(
                messages=[{"role": "user", "content": context + "\nStatus: Ready. Decide next step."}],
                system_prompt=prompt,
                temperature=0.5,
                max_tokens=300,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=self.stop_check
            )
            
            response = response.strip()
            response_upper = response.upper()
            self.log(f"🤖 Decider Decision: {response}")
            
            if self.chat_fn:
                self.chat_fn("Decider", f"Decision: {response}")
            
            # Reset cooldown
            if self.forced_stop_cooldown:
                self.forced_stop_cooldown = False

            if "[WAIT]" in response_upper:
                self.current_task = "wait"
                self.cycles_remaining = 0
                self.wait_start_time = time.time()
                # Do not reset consecutive_daydream_batches on WAIT, so we remember to switch tasks after waking up
            elif "[DAYDREAM:" in response_upper:
                if not allow_daydream:
                    self.log("⚠️ Decider tried to Daydream during cooldown/prevention. Forcing WAIT.")
                    self.current_task = "wait"
                    self.cycles_remaining = 0
                else:
                    try:
                        content = response_upper.split(":")[1].strip().replace("]", "")
                        parts = [p.strip() for p in content.split(",")]
                        
                        count = 1
                        mode = "auto"
                        topic = None
                        
                        if len(parts) >= 1 and parts[0].isdigit():
                            count = int(parts[0])
                        if len(parts) >= 2:
                            mode = parts[1].lower()
                        if len(parts) >= 3:
                            # Extract topic from original response to preserve case
                            match = re.search(r"\[DAYDREAM:.*?,.*?,(.*?)\]", response, re.IGNORECASE)
                            if match:
                                topic = match.group(1).strip()
                            else:
                                topic = parts[2] # Fallback
                        
                        self.current_task = "daydream"
                        self.daydream_mode = mode if mode in ["read", "auto"] else "auto"
                        self.daydream_topic = topic
                        self.cycles_remaining = max(1, min(count, 10))
                        self.consecutive_daydream_batches += 1
                    except:
                        self.current_task = "daydream"
                        self.cycles_remaining = 1
                        self.daydream_mode = "auto"
                        self.consecutive_daydream_batches += 1
            elif "[VERIFY_ALL]" in response_upper:
                # --- SAFETY CHECK ---
                stats = self.memory_store.get_memory_stats()
                unverified_count = stats.get('unverified_facts', 0) + stats.get('unverified_beliefs', 0)
                
                if unverified_count == 0:
                    self.log("🛑 Decider attempted VERIFY_ALL, but 0 items pending. Forcing WAIT.")
                    # Forcefully lower pressure so it doesn't loop
                    if self.gevurah: 
                        self.gevurah.last_pressure = 0.0 
                    self.current_task = "wait"
                    self.cycles_remaining = 0
                else:
                    self.current_task = "verify_all"
                    self.cycles_remaining = 1
                    self.consecutive_daydream_batches = 0
            elif "[VERIFY" in response_upper and "ALL" not in response_upper:
                try:
                    if ":" in response_upper:
                        count = int(response_upper.split(":")[1].strip().replace("]", ""))
                    else:
                        count = 3
                    self.current_task = "verify"
                    self.cycles_remaining = max(1, min(count, 5))
                    self.consecutive_daydream_batches = 0
                except:
                    self.current_task = "verify"
                    self.cycles_remaining = 1
                    self.consecutive_daydream_batches = 0
            elif "[SPEAK:" in response_upper:
                try:
                    # Use regex to extract content preserving case
                    match = re.search(r"\[SPEAK:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    msg = ""
                    if match:
                        msg = match.group(1).strip()
                    
                    # Filter out placeholders
                    if msg.upper() in ["MESSAGE", "CONTENT", "TEXT", "MSG", "INSERT TEXT HERE"]:
                        self.log(f"⚠️ Decider generated placeholder '{msg}' for SPEAK. Aborting speak.")
                        self.current_task = "wait"
                    else:
                        if self.chat_fn:
                            self.chat_fn("Decider", msg)
                        self.current_task = "wait"
                        self.cycles_remaining = 0
                        self.last_action_was_speak = True
                except Exception as e:
                    self.log(f"⚠️ Decider failed to speak: {e}")
            elif "[NOTE_CREATE:" in response_upper:
                try:
                    match = re.search(r"\[NOTE_CREATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        args = match.group(1).strip()
                        self.create_note(args)
                    # Don't wait; allow chaining decisions (e.g. create memory -> then daydream)
                    self.current_task = "organizing" 
                    self.cycles_remaining = 0
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to create note: {e}")
            elif "[NOTE_EDIT:" in response_upper:
                try:
                    match = re.search(r"\[NOTE_EDIT:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        args = match.group(1).strip()
                    if "," in args:
                        mid_str, content = args.split(",", 1)
                        self.edit_note(int(mid_str.strip()), content.strip())
                    # Don't wait; allow chaining decisions
                    self.current_task = "organizing"
                    self.cycles_remaining = 0
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to edit note: {e}")
            elif "[THINK:" in response_upper:
                try:
                    match = re.search(r"\[THINK:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    topic = match.group(1).strip() if match else "General"
                    
                    # Filter out placeholders
                    if topic.upper() in ["TOPIC", "SUBJECT", "CONTENT", "TEXT", "INSERT TOPIC", "SPECIFIC_TOPIC"]:
                        self.log(f"⚠️ Decider generated placeholder '{topic}' for THINK. Aborting.")
                        self.current_task = "wait"
                    else:
                        self.perform_thinking_chain(topic)
                        self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to start thinking chain: {e}")
                    self.current_task = "wait"
            elif "[DEBATE:" in response_upper:
                try:
                    match = re.search(r"\[DEBATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    topic = match.group(1).strip() if match else "General"
                    if self.dialectics:
                        result = self.dialectics.run_debate(topic)
                        self.reasoning_store.add(content=f"Council Debate ({topic}):\n{result}", source="council_debate", confidence=1.0)
                        self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to start debate: {e}")
                    self.current_task = "wait"
            elif "[SIMULATE:" in response_upper:
                try:
                    match = re.search(r"\[SIMULATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    premise = match.group(1).strip() if match else "General"
                    if "simulate_counterfactual" in self.actions:
                        result = self.actions["simulate_counterfactual"](premise)
                        self.reasoning_store.add(content=f"Counterfactual Simulation ({premise}):\n{result}", source="daat_simulation", confidence=1.0)
                        self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to start simulation: {e}")
                    self.current_task = "wait"
            elif "[CODE:" in response_upper:
                try:
                    # Extract code content (everything after [CODE: and before the last ])
                    match = re.search(r"\[CODE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    code = match.group(1).strip() if match else ""
                    result = self._execute_tool("CODE", code)
                    self.current_task = "organizing"
                    self.cycles_remaining = 0
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to execute code: {e}")
            elif "[EXECUTE:" in response_upper:
                try:
                    match = re.search(r"\[EXECUTE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    content = match.group(1).strip() if match else ""
                    
                    if "," in content:
                        tool, args = content.split(",", 1)
                        self._execute_tool(tool.strip().upper(), args.strip())
                    else:
                        self._execute_tool(content.strip().upper(), "")
                    self.current_task = "organizing" # Allow chaining (decide next step immediately)
                    self.cycles_remaining = 0
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to execute tool: {e}")
            elif "[GOAL_ACT:" in response_upper:
                try:
                    target_id_str = response_upper.split("[GOAL_ACT:", 1)[1].strip().rstrip("]")
                    target_id = int(target_id_str)
                    
                    # Find goal text
                    all_items = self.memory_store.list_recent(limit=None)
                    goal_text = next((m[3] for m in all_items if m[0] == target_id), None)
                    
                    if goal_text:
                        self.log(f"🤖 Decider acting on Goal {target_id}: {goal_text}")
                        self.perform_thinking_chain(f"Strategic Plan for Goal: {goal_text}")
                    else:
                        self.log(f"⚠️ Goal {target_id} not found.")
                        self.current_task = "wait"
                    
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to act on goal: {e}")
                    self.current_task = "wait"
            elif "[GOAL_REMOVE:" in response_upper:
                try:
                    match = re.search(r"\[GOAL_REMOVE:(.*?)\]?$", response, re.IGNORECASE)
                    target = match.group(1).strip() if match else ""
                    
                    if "remove_goal" in self.actions:
                        result = self.actions["remove_goal"](target)
                        self.log(result)
                        if self.chat_fn:
                            self.chat_fn("Decider", result)
                    else:
                        self.log("⚠️ Action remove_goal not available.")
                    
                    self.current_task = "organizing"
                    self.cycles_remaining = 0
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to remove goal: {e}")
            elif "[GOAL_CREATE:" in response_upper:
                try:
                    match = re.search(r"\[GOAL_CREATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    content = match.group(1).strip() if match else ""
                    
                    self.create_goal(content)
                    self.current_task = "organizing"
                    self.cycles_remaining = 0
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to create goal: {e}")
            elif "[LIST_DOCS]" in response_upper:
                if "list_documents" in self.actions:
                    docs_list = self.actions["list_documents"]()
                    self.reasoning_store.add(content=f"Tool Output [LIST_DOCS]:\n{docs_list}", source="tool_output", confidence=1.0)
                    self.log(f"📚 Documents listed.")
                self.current_task = "organizing"
                self.cycles_remaining = 0
            elif "[READ_DOC:" in response_upper:
                try:
                    match = re.search(r"\[READ_DOC:(.*?)\]?$", response, re.IGNORECASE)
                    target = match.group(1).strip() if match else ""
                    
                    if "read_document" in self.actions:
                        content = self.actions["read_document"](target)
                        self.reasoning_store.add(content=f"Tool Output [READ_DOC]:\n{content}", source="tool_output", confidence=1.0)
                        self.log(f"📄 Read document: {target}")
                except Exception as e:
                    self.log(f"⚠️ Decider failed to read doc: {e}")
                self.current_task = "organizing"
                self.cycles_remaining = 0
            elif "[SEARCH_MEM:" in response_upper:
                try:
                    match = re.search(r"\[SEARCH_MEM:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    query = match.group(1).strip() if match else ""
                    
                    if "search_memory" in self.actions:
                        results = self.actions["search_memory"](query)
                        self.reasoning_store.add(content=f"Tool Output [SEARCH_MEM]:\n{results}", source="tool_output", confidence=1.0)
                        self.log(f"🔍 Searched memory for: {query}")
                except Exception as e:
                    self.log(f"⚠️ Decider failed to search memory: {e}")
                self.current_task = "organizing"
                self.cycles_remaining = 0
            elif "[SUMMARIZE]" in response_upper:
                if "summarize" in self.actions:
                    result = self.actions["summarize"]()
                    if result:
                        self.log(result)
                self.current_task = "organizing"
                self.cycles_remaining = 0
            elif "[WRITE_FILE:" in response_upper:
                try:
                    match = re.search(r"\[WRITE_FILE:\s*(.*?),\s*(.*?)\]", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        filename = match.group(1).strip()
                        content = match.group(2).strip()
                        self._write_file(filename, content)
                    else:
                        self.log("⚠️ Invalid WRITE_FILE format. Use [WRITE_FILE: filename, content]")
                    self.current_task = "organizing"
                    self.cycles_remaining = 0
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to write file: {e}")
            elif "[SEARCH_INTERNET:" in response_upper:
                try:
                    match = re.search(r"\[SEARCH_INTERNET:\s*(.*),\s*([A-Z_]+)\]", response, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip()
                        source = match.group(2).strip().upper()
                        if "search_internet" in self.actions:
                            result = self.actions["search_internet"](query, source)
                            self.reasoning_store.add(content=f"Internet Search [{source}]: {query}\nResult: {result}", source="internet_bridge", confidence=1.0)
                            self.log(f"🌐 Internet Search Result: {result[:100]}...")
                        else:
                            self.log("⚠️ Action search_internet not available.")
                    self.current_task = "organizing"
                    self.cycles_remaining = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to search internet: {e}")
            else:
                # Default fallback
                # FIX: Only default to daydream if allowed and not cooling down
                if allow_daydream and not just_finished_heavy and not self.forced_stop_cooldown:
                    self._run_action("start_daydream")
                    self.current_task = "daydream"
                    self.cycles_remaining = 0 # Just one
                    self.consecutive_daydream_batches += 1
                else:
                    self.log(f"🤖 Decider: Fallback triggered but Daydream is disabled/cooldown. Defaulting to WAIT.")
                    self.current_task = "wait"
                    self.cycles_remaining = 0
        
        except Exception as e:
            self.log(f"❌ Critical Planning Error: {e}")
            self.current_task = "wait"
        finally:
            self.planning_lock.release()

    def _run_action(self, name: str):
        # 1. Capture Pre-Action State (Trigger State)
        start_state = {}
        start_coherence = 0.0
        
        if self.keter:
            # Evaluate Keter to get current baseline (Raw score is better for immediate delta)
            keter_stats = self.keter.evaluate()
            start_coherence = keter_stats.get("raw", keter_stats.get("keter", 0.0))
            
            start_state = {
                "hesed": self.hesed.calculate() if self.hesed else 0,
                "gevurah": self.gevurah.calculate() if self.gevurah else 0,
                "active_goals": len(self.memory_store.get_active_by_type("GOAL")),
                "coherence": start_coherence
            }

        # 2. Execute Action
        if name == "run_hod":
            if self.hod_just_ran:
                self.log("⚠️ Decider: Skipping Hod analysis to prevent loops.")
                return
            
            if name in self.actions:
                self.actions[name]()
                self.hod_just_ran = True
            else:
                self.log(f"⚠️ Decider action '{name}' not available.")
        else:
            if name in self.actions:
                result = self.actions[name]()
                # Reset Hod lock for substantive actions
                if name in ["start_daydream", "verify_batch", "verify_all", "start_loop", "run_observer"]:
                    self.hod_just_ran = False
                
                if name == "run_observer" and result:
                    self.ingest_netzach_signal(result)
            else:
                self.log(f"⚠️ Decider action '{name}' not available.")

        # 3. Measure Result & Record Outcome (Credit Assignment)
        if self.keter and hasattr(self.meta_memory_store, 'add_outcome'):
            # Re-evaluate Keter to see change
            end_stats = self.keter.evaluate()
            end_coherence = end_stats.get("raw", end_stats.get("keter", 0.0))
            delta = end_coherence - start_coherence
            
            self.meta_memory_store.add_outcome(
                action=name,
                trigger_state=start_state,
                result={"coherence_delta": delta, "end_score": end_coherence}
            )

        if hasattr(self.meta_memory_store, 'add_event') and name != "run_observer":
            self.meta_memory_store.add_event(
                event_type="DECIDER_ACTION",
                subject="Assistant",
                text=f"Executed action: {name}"
            )

    def ingest_hod_analysis(self, analysis: Dict):
        """Receive and process analysis from Hod."""
        if not analysis: return
        
        # 1. Log Findings (Persistence)
        findings = analysis.get("findings")
        if findings:
            self.log(f"🔮 Hod Findings: {findings}")
            if self.chat_fn:
                self.chat_fn("Hod", f"🔮 Analysis: {findings}")
            # Decider decides to persist this analysis
            self.reasoning_store.add(
                content=f"Hod Analysis: {findings}",
                source="hod",
                confidence=1.0,
                ttl_seconds=3600
            )

        # 2. Process Observations (Tiferet decides the action)
        observations = analysis.get("observations", [])
        
        for observation in observations:
            otype = observation.get("type")
            
            if otype == "HIGH_ENTROPY":
                self.decrease_temperature() # Tiferet decides magnitude
            elif otype == "CONTEXT_OVERLOAD":
                self.decrease_tokens()
            elif otype == "LOGIC_VIOLATION" or otype == "SELF_CORRECTION_SIGNAL":
                # Map to maintenance action
                if "id" in observation:
                    self.pending_maintenance.append({"type": "REFUTE", "id": observation["id"], "reason": observation.get("context")})
            elif otype == "INVALID_DATA":
                if "id" in observation:
                    self.pending_maintenance.append({"type": "PRUNE", "id": observation["id"], "reason": observation.get("context")})
            elif otype == "CONTEXT_FRAGMENTATION":
                self._run_action("summarize")
            elif otype == "CRITICAL_INFO":
                self.receive_observation(observation.get("context"))

    def ingest_netzach_signal(self, signal: Dict):
        """Receive signal from Netzach."""
        if not signal: return
        self.latest_netzach_signal = signal
        
        # Immediate reaction to high pressure if waiting
        if signal.get("pressure", 0) > 0.7 and self.current_task == "wait":
            self.wake_up("Netzach Pressure")

    def _check_netzach_signal(self):
        """Check latest Netzach signal during planning."""
        if not self.latest_netzach_signal: return
        
        sig = self.latest_netzach_signal.get("signal")
        context = self.latest_netzach_signal.get("context")
        
        if sig == "LOW_NOVELTY":
            self.increase_temperature() # Decider chooses amount
        elif sig == "HIGH_CONSTRAINT":
            self.increase_tokens() # Decider chooses amount
        elif sig == "EXTERNAL_PRESSURE":
            if context and "reason" in context:
                self.receive_observation(context["reason"])
        elif sig == "CONTEXT_PRESSURE":
            self.log("👁️ Netzach requested summary. Decider authorizing Da'at...")
            self._run_action("summarize")
        elif sig == "DISSONANCE":
             self._run_action("run_hod")
        # elif sig == "NO_MOMENTUM":
        #      self.start_daydream() # Removed: Let _decide_next_batch handle this via Goal Act or Daydream
        
        # Clear signal after processing
        self.latest_netzach_signal = None

    def _authorize_maintenance(self):
        """Authorize and dispatch queued maintenance actions (Prune/Refute)."""
        if not self.pending_maintenance: return
        
        self.log(f"🤖 Decider: Authorizing {len(self.pending_maintenance)} maintenance actions.")
        for action in self.pending_maintenance:
            if action["type"] == "PRUNE":
                if "prune_memory" in self.actions:
                    self.actions["prune_memory"](action["id"])
            elif action["type"] == "REFUTE":
                if "refute_memory" in self.actions:
                    self.actions["refute_memory"](action["id"], action.get("reason"))
        
        self.pending_maintenance = []

    def on_stagnation(self, event_data: Any = None):
        """Handle system stagnation event (usually from Netzach)."""
        self.log("🤖 Decider: Stagnation detected. Waking up.")
        self.start_daydream()

    def on_instability(self, event_data: Dict):
        """Handle system instability event (usually from Hod)."""
        self.log(f"🤖 Decider: Instability reported ({event_data.get('reason')}). Switching to verification.")
        self.current_task = "verify"
        self.cycles_remaining = 1

    def receive_observation(self, observation: str):
        """
        Receive an observation/information from Netzach.
        This information MUST result in an action.
        """
        self.log(f"📩 Decider received observation: {observation}")
        
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event(
                event_type="DECIDER_OBSERVATION_RECEIVED",
                subject="Assistant",
                text=f"Received observation from Netzach: {observation}"
            )

        text = observation.lower()
        self.action_taken_in_observation = True
        
        # Map observation content to actions
        if any(w in text for w in ["loop", "cycle", "continue"]):
            self.current_task = "daydream"
            self.cycles_remaining = 3
            self.daydream_mode = "auto"
        elif any(w in text for w in ["stop", "halt", "pause"]) and "daydream" in text:
            self._run_action("stop_daydream")
            self.current_task = "wait"
            self.cycles_remaining = 0
        elif any(w in text for w in ["decrease", "lower", "reduce", "drop"]) and any(w in text for w in ["temp", "temperature"]):
            self.decrease_temperature()
        elif any(w in text for w in ["decrease", "lower", "reduce", "drop"]) and any(w in text for w in ["token", "tokens", "length"]):
            self.decrease_tokens()
        elif any(w in text for w in ["increase", "raise", "boost", "up", "higher"]) and any(w in text for w in ["token", "tokens", "length"]):
            # Debounce token increases (prevent loops)
            if time.time() - self.last_token_adjustment_time > 60:
                self.increase_tokens()
            else:
                self.log("⚠️ Decider ignoring token increase request (cooldown active).")
        elif any(w in text for w in ["verify", "conflict", "contradiction", "error", "inconsistent", "wrong"]):
            self.current_task = "verify"
            self.cycles_remaining = 2
            if "all" in text or "full" in text:
                self.current_task = "verify_all"
                self.cycles_remaining = 1
        elif any(w in text for w in ["hod", "analyze", "analysis", "investigate", "pattern", "reflect", "refuted", "refutation"]):
            self._run_action("run_hod")
        elif "observer" in text or "watch" in text:
            self._run_action("run_observer")
        elif any(w in text for w in ["stagnant", "idle", "bored", "nothing", "quiet", "daydream", "think", "create"]):
            self.current_task = "daydream"
            self.cycles_remaining = 1
            self.daydream_mode = "auto"
        else:
            # Slow Path (Semantic Interpretation) - The "Wisdom" Check
            self.log(f"🤔 Decider: Interpreting complex observation...")
            
            settings = self.get_settings()
            prompt = (
                f"OBSERVATION: \"{observation}\"\n"
                "CONTEXT: You are the AI System Manager.\n"
                "TASK: Translate this observation into a single command.\n"
                "OPTIONS:\n"
                "- [CHANGE_TEMP: X.X] (If exploring/consolidating is needed)\n"
                "- [GOAL: <New Goal>] (If a new direction is suggested)\n"
                "- [IGNORE] (If it's just a status update)\n"
                "- [REFLECT] (If we need to think deeper)\n"
                "OUTPUT: Single command only."
            )

            decision = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a JSON-based control system.",
                max_tokens=50,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )

            if "[CHANGE_TEMP" in decision:
                try:
                    val = float(decision.split(":")[1].strip("] "))
                    s = self.get_settings()
                    s["temperature"] = val
                    self.update_settings(s)
                    self.log(f"🌡️ Decider set temperature to {val}")
                except:
                    self.log("⚠️ Failed to parse temp change.")
            elif "[GOAL" in decision:
                new_goal = decision.split("[GOAL:", 1)[1].strip("] ")
                self.create_goal(new_goal)
                self.log(f"🎯 Decider: Adapted strategy. New Goal: {new_goal}")
            elif "[REFLECT]" in decision:
                self.reasoning_store.add(content="I need to reflect on the system state.", source="decider", confidence=1.0)
                self._run_action("run_hod")
            else:
                self.log(f"⚠️ Decider could not map observation (LLM result: {decision}). No action taken.")

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

    def perform_thinking_chain(self, topic: str):
        """Execute a chain of thought process."""
        self.log(f"🧠 Decider starting chain of thought on: {topic}")
        if self.chat_fn:
            self.chat_fn("Decider", f"🧠 Starting chain of thought: {topic}")
            
        settings = self.get_settings()

        # 1. Gather Context (Memories & Docs) to ground the thinking
        query_embedding = compute_embedding(
            topic, 
            base_url=settings.get("base_url"),
            embedding_model=settings.get("embedding_model")
        )
        
        mem_results = self.memory_store.search(query_embedding, limit=5)
        doc_results = self.document_store.search_chunks(query_embedding, top_k=3)
        
        context_str = ""
        if mem_results:
            context_str += "Relevant Memories:\n" + "\n".join([f"- {m[3]}" for m in mem_results]) + "\n"
        if doc_results:
            context_str += "Relevant Documents:\n" + "\n".join([f"- {d['text'][:300]}..." for d in doc_results]) + "\n"
            
        static_context = f"Topic: {topic}\n"
        if context_str:
            static_context += f"\n{context_str}\n"
        
        recent_thoughts = []
        all_thoughts = []
        consecutive_similar_thoughts = 0
        recent_thought_embeddings = []
        
        for i in range(1, 31):
            if self.stop_check():
                break
                
            # Self-Correction/Reflection Step (Every 5 steps)
            if i > 1 and i % 5 == 0:
                self.log(f"🧠 Decider performing self-reflection on step {i}...")
                
                reflection_history = ""
                start_idx = max(0, len(all_thoughts) - 5)
                for idx, t in enumerate(all_thoughts[start_idx:]):
                    reflection_history += f"Step {start_idx + idx + 1}: {t}\n"

                reflection_prompt = (
                    f"Review the last 5 steps of the Thought Chain:\n"
                    f"{static_context}\n"
                    f"Recent Thoughts:\n{reflection_history}\n"
                    "Critique your reasoning:\n"
                    "1. Are there logical fallacies or hallucinations?\n"
                    "2. Have you drifted from the original Topic?\n"
                    "3. Are you repeating yourself?\n"
                    "If errors exist, output a CORRECTION. If reasoning is sound, output 'VALID'.\n"
                    "Output ONLY the critique or 'VALID'."
                )
                
                critique = run_local_lm(
                    messages=[{"role": "user", "content": "Reflect on reasoning."}],
                    system_prompt=reflection_prompt,
                    temperature=0.3,
                    max_tokens=150,
                    base_url=settings.get("base_url"),
                    chat_model=settings.get("chat_model"),
                    stop_check_fn=self.stop_check
                ).strip()
                
                if "VALID" not in critique.upper():
                    self.log(f"🔧 Self-Correction: {critique}")
                    # Add correction to thoughts so it appears in final summary
                    all_thoughts.append(f"[SELF-CORRECTION] {critique}")

            # Force depth: Don't allow conclusion in first 5 steps
            conclusion_instruction = "If you have reached a final answer or conclusion, start the response with '[CONCLUSION]'."
            if i < 5:
                conclusion_instruction = "Do NOT reach a conclusion yet. Explore the topic deeper. Do NOT use the [CONCLUSION] tag."

            # Dynamic Temperature: If we are getting repetitive, heat up the model
            step_temp = settings.get("temperature", 0.7)
            anti_repetition_msg = ""
            if consecutive_similar_thoughts > 0:
                step_temp = min(1.0, step_temp + (0.15 * consecutive_similar_thoughts))
                self.log(f"🔥 Boosting temperature to {step_temp:.2f} to break potential loop.")
                anti_repetition_msg = "WARNING: You are repeating yourself. Change your approach or perspective immediately."

            prev_thought_context = ""
            if recent_thoughts:
                prev_thought_context = f"PREVIOUS THOUGHT: {recent_thoughts[-1]}\nCONSTRAINT: Your next thought must ADVANCE the reasoning. Do not restate the previous thought."

            # Sliding Window: Only show last 8 thoughts to prevent Context Overflow (400 Error)
            visible_chain_str = ""
            start_idx = max(0, len(all_thoughts) - 8)
            for idx, t in enumerate(all_thoughts[start_idx:]):
                visible_chain_str += f"Step {start_idx + idx + 1}: {t}\n"

            prompt = (
                f"You are the Assistant thinking through a problem step-by-step.\n"
                f"{static_context}\n"
                f"Thought Chain (Recent):\n{visible_chain_str}\n"
                f"{prev_thought_context}\n"
                "Generate the next logical thought step.\n"
                "1. INTEGRATE the Relevant Memories and Documents into your reasoning.\n"
                "2. AVOID repeating ideas from the Thought Chain.\n"
                "3. Keep it concise (1-2 sentences).\n"
                f"{conclusion_instruction}\n"
                f"{anti_repetition_msg}\n"
                "Output ONLY the next thought."
            )
            
            thought = run_local_lm(
                messages=[{"role": "user", "content": "Continue thinking."}],
                system_prompt=prompt,
                temperature=step_temp,
                max_tokens=300,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=self.stop_check
            )
            
            thought = thought.strip()
            
            # Handle premature conclusions
            if "[CONCLUSION]" in thought and i < 5:
                self.log(f"⚠️ Premature conclusion at step {i}. Continuing chain.")
                thought = thought.replace("[CONCLUSION]", "").strip()
            
            if thought.startswith("⚠️"):
                self.log(f"❌ Thinking chain error: {thought}")
                break
            
            # Fuzzy Loop detection
            is_repetitive = False
            max_similarity = 0.0
            
            for past_thought in recent_thoughts:
                ratio = difflib.SequenceMatcher(None, thought, past_thought).ratio()
                if ratio > max_similarity:
                    max_similarity = ratio
                
                # Stricter threshold for long thoughts (0.75 instead of 0.85)
                if ratio > 0.75:
                    is_repetitive = True
                    break
            
            # Semantic Loop Detection (Embeddings)
            if not is_repetitive:
                current_emb = compute_embedding(thought, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
                
                for past_emb in recent_thought_embeddings:
                    # Cosine similarity
                    if len(past_emb) == len(current_emb):
                        sim = np.dot(current_emb, past_emb) / (np.linalg.norm(current_emb) * np.linalg.norm(past_emb) + 1e-9)
                        if sim > 0.95: # Very high threshold for semantic identity
                            is_repetitive = True
                            self.log(f"⚠️ Semantic repetition detected (Sim: {sim:.2f}).")
                            break
                
                recent_thought_embeddings.append(current_emb)
                if len(recent_thought_embeddings) > 10:
                    recent_thought_embeddings.pop(0)

            if is_repetitive:
                self.log("⚠️ Repetitive thought detected. Forcing conclusion.")
                break
            
            # Track "soft" repetition for temperature boosting
            if max_similarity > 0.6:
                consecutive_similar_thoughts += 1
            else:
                consecutive_similar_thoughts = 0

            recent_thoughts.append(thought)
            if len(recent_thoughts) > 10: # Keep more history to detect longer loops
                recent_thoughts.pop(0)
            
            all_thoughts.append(thought)

            # UI/Telegram update
            formatted_msg = f"💭 Thought [{i}/30]: {thought}"
            self.log(formatted_msg)
            if self.chat_fn:
                self.chat_fn("Decider", formatted_msg)
            
            # Store in reasoning
            self.reasoning_store.add(content=f"CoT {i} ({topic}): {thought}", source="decider_cot", confidence=1.0)
            
            # Add to Meta-Memory
            if hasattr(self.meta_memory_store, 'add_event'):
                self.meta_memory_store.add_event("CHAIN_OF_THOUGHT", "Assistant", f"Thought {i}: {thought}")
            
            if "[CONCLUSION]" in thought:
                # Save conclusion as special memory
                clean_conclusion = thought.replace("[CONCLUSION]", "").strip()
                self.create_note(f"Conclusion on {topic}: {clean_conclusion}")
                break
        
        # Post-chain Summarization
        if all_thoughts:
            self.log(f"🧠 Generating summary of {len(all_thoughts)} thoughts...")
            full_chain_text = "\n".join(all_thoughts)
            summary_prompt = (
                f"Synthesize the following chain of thought regarding '{topic}' into a clear, comprehensive summary for the user.\n"
                f"Include key insights and the final conclusion if reached.\n\n"
                f"Thought Chain:\n{full_chain_text}"
            )
            
            summary = run_local_lm(
                messages=[{"role": "user", "content": summary_prompt}],
                system_prompt="You are a helpful assistant summarizing your internal reasoning.",
                temperature=0.5,
                max_tokens=500,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=self.stop_check
            )
            
            if self.chat_fn:
                self.chat_fn("Decider", f"🧠 Thought Chain Summary:\n{summary}")

            # Save summary as note if no formal conclusion was reached (e.g. interrupted by loop)
            # This ensures partial progress is preserved in memory
            if "[CONCLUSION]" not in full_chain_text:
                self.create_note(f"Thought Chain Summary ({topic}): {summary}")

        self.current_task = "wait"
        self.cycles_remaining = 0

    def _execute_tool(self, tool_name: str, args: str):
        """Execute a tool safely and store the result."""
        # Rate limiting (2 seconds between tool calls)
        if time.time() - self.last_tool_usage < 2.0:
            self.log(f"⚠️ Tool rate limit exceeded for {tool_name}")
            return "Error: Tool rate limit exceeded. Please wait."
        self.last_tool_usage = time.time()

        self.log(f"🛠️ Decider executing tool: {tool_name} args: {args}")
        result = ""
        
        if tool_name == "CALCULATOR":
            if len(args) > 50:
                result = "Error: Expression too long (max 50 chars)."
            else:
                result = self._safe_calculate(args)
        elif tool_name == "CLOCK":
            result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif tool_name == "DICE":
            try:
                if "-" in args:
                    mn, mx = map(int, args.split("-"))
                    result = str(random.randint(mn, mx))
                elif args.strip().isdigit():
                    result = str(random.randint(1, int(args.strip())))
                else:
                    result = str(random.randint(1, 6))
            except:
                result = "Error: Invalid dice format. Use 'min-max' or 'max'."
        elif tool_name == "SYSTEM_INFO":
            try:
                uname = platform.uname()
                result = f"OS: {uname.system} {uname.release} ({uname.machine})"
                if psutil:
                    mem = psutil.virtual_memory()
                    load = psutil.getloadavg() if hasattr(psutil, "getloadavg") else [0, 0, 0]
                    result += f" | CPU Load: {load[0]:.1f}% | RAM: {mem.percent}% Used ({mem.available // (1024*1024)}MB Free)"
            except:
                result = "Error retrieving system info."
        elif tool_name == "WRITE_FILE":
            # Parse args: filename, content
            try:
                if "," in args:
                    filename, content = args.split(",", 1)
                    result = self._write_file(filename.strip(), content.strip())
                else:
                    result = "Error: WRITE_FILE requires 'filename, content'"
            except Exception as e:
                result = f"Error writing file: {e}"
        elif tool_name in self.actions:
            # Support for dynamic/injected tools (e.g. SEARCH, WIKI)
            try:
                result = self.actions[tool_name](args)
            except Exception as e:
                result = f"Error executing {tool_name}: {e}"
        else:
            result = f"Tool {tool_name} not found."
            
        self.log(f"🛠️ Tool Result: {result}")
        
        # Store result in reasoning so the AI knows what happened
        self.reasoning_store.add(
            content=f"Tool Execution [{tool_name}]: {args} -> Result: {result}",
            source="tool_output",
            confidence=1.0
        )

        # Add to Meta-Memory
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event("TOOL_EXECUTION", "Assistant", f"Executed {tool_name} ({args}) -> {result}")
        
        if self.chat_fn:
             self.chat_fn("Decider", f"🛠️ Used {tool_name}: {result}")
             
        return result

    def _write_file(self, filename: str, content: str) -> str:
        """
        Delegate file writing to Malkuth.
        """
        if self.malkuth:
            return self.malkuth.write_file(filename, content)
        return "Error: Malkuth not available for file operations."

    def _safe_calculate(self, expression: str) -> str:
        """Safely evaluate a mathematical expression without using eval()."""
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            # ast.Pow: operator.pow, # Disabled to prevent CPU freezing (e.g. 9**999999)
            ast.BitXor: operator.xor,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num): # Python < 3.8 compatibility
                return node.n
            elif isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in operators:
                    raise TypeError(f"Operator {op} not supported")
                return operators[op](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):
                op = type(node.op)
                if op not in operators:
                    raise TypeError(f"Operator {op} not supported")
                return operators[op](eval_node(node.operand))
            else:
                raise TypeError(f"Node type {type(node)} not supported")

        try:
            tree = ast.parse(expression.strip(), mode='eval')
            return str(eval_node(tree.body))
        except Exception as e:
            return f"Calculation Error: {e}"

    def _get_recent_chat_memories(self, limit: int = 20):
        """Retrieve recent memories that are NOT from daydreaming."""
        items = self.memory_store.list_recent(limit=100)
        chat_mems = []
        for item in items:
            # item: (id, type, subject, text, source, verified)
            if len(item) > 4 and item[4] != 'daydream':
                chat_mems.append(item)
                if len(chat_mems) >= limit:
                    break
        return chat_mems

    def _get_critical_memories(self):
        """Retrieve always-active memories (Identity, Permission, Goals, Rules)."""
        critical_types = ["IDENTITY", "PERMISSION", "RULE", "GOAL", "CURIOSITY_GAP"]
        mems = []
        for t in critical_types:
            # get_active_by_type returns (id, subject, text, source)
            items = self.memory_store.get_active_by_type(t)
            # Sort by ID descending to prioritize recent items
            items.sort(key=lambda x: x[0], reverse=True)

            daydream_count = 0
            daydream_limit = 5  # Allow limited daydream memories to support learning

            for (mid, subj, text, source) in items:
                if source == 'daydream':
                    if daydream_count >= daydream_limit:
                        continue
                    daydream_count += 1
                # Format to match list_recent: (id, type, subject, text, source, verified, flags)
                mems.append((mid, t, subj, text, source, 1, None))
        return mems

    def _analyze_intent(self, text: str) -> str:
        """Use LLM to classify user intent for ambiguous commands."""
        settings = self.get_settings()
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
            stop_check_fn=self.stop_check
        )
        return response.strip()

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
                success = self.memory_store.delete_entry(mem_id)
                if success:
                    self.log(f"🗑️ Manually deleted memory ID {mem_id}")
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
            
            self.log(f"🤖 Decider enabling Daydream Loop for {count} cycles via natural command.")
            self._run_action("start_loop")
            self.current_task = "daydream"
            self.cycles_remaining = count
            return f"🔄 Daydream loop enabled for {count} cycles."

        # Daydream Batch (Specific Count)
        # Matches: "run 5 daydream cycles", "do 3 daydreams", "run 1 daydream cycle"
        batch_match = re.search(r"(?:run|do|start|execute)\s+(\d+)\s+daydream(?:s|ing)?(?: cycles?| loops?)?", text)
        if batch_match:
            count = int(batch_match.group(1))
            # Cap count reasonably
            count = max(1, min(count, 20))
            
            self.log(f"🤖 Decider enabling Daydream Batch for {count} cycles via natural command.")
            self.current_task = "daydream"
            self.cycles_remaining = count
            self.daydream_mode = "auto"
            return f"☁️ Starting {count} daydream cycles."

        # Learn / Expand Knowledge
        learn_match = re.search(r"(?:expand (?:your )?knowledge(?: about)?|learn(?: about)?|research|study|read up on|educate yourself(?: on| about)?)\s+(.*)", text, re.IGNORECASE)
        if learn_match:
            raw_topic = learn_match.group(1).strip(" .?!")
            
            # Clean topic: remove "from your documents", "from files", etc.
            clean_topic = re.sub(r"\s+from\s+(?:your\s+)?(?:documents|files|database|memory|docs).*", "", raw_topic, flags=re.IGNORECASE).strip()
            
            if clean_topic:
                self.create_goal(f"Expand knowledge about {clean_topic}")
                self.log(f"🤖 Decider starting Daydream Loop focused on: {clean_topic}")
                self._run_action("start_loop")
                self.current_task = "daydream"
                self.daydream_mode = "read"
                self.daydream_topic = clean_topic
                self.cycles_remaining = 5
                return f"📚 Initiating research protocol for: {clean_topic}. I will read relevant documents and generate insights."

        # Verify All
        if "run verification all" in text or "verify all" in text:
            self.log("🤖 Decider starting Full Verification via natural command.")
            self._run_action("verify_all")
            return "🕵️ Full verification triggered."

        # Verify Batch
        if "run verification batch" in text or "verify batch" in text or "run verification" in text:
            self.log("🤖 Decider starting Verification Batch via natural command.")
            self._run_action("verify_batch")
            return "🕵️ Verification batch triggered."

        # Verify Beliefs (Internal Consistency/Insight)
        if "verify" in text and "belief" in text:
             self.log("🤖 Decider starting Belief Verification (Grounding).")
             self.current_task = "verify"
             self.cycles_remaining = 3
             self._run_action("verify_batch")
             return "🕵️ Initiating belief verification."

        # Verify Sources (Facts/Memories against Documents)
        if "verify" in text and ("fact" in text or "memory" in text or "source" in text):
             self.log("🤖 Decider starting Verification Batch via natural command.")
             self._run_action("verify_batch")
             return "🕵️ Verification batch triggered."

        # Single Daydream
        if text in ["run daydream", "start daydream", "daydream", "do a daydream"]:
            self.log("🤖 Decider starting single Daydream cycle via natural command.")
            self._run_action("start_daydream")
            return "☁️ Daydream triggered."
            
        # Stop
        if "stop daydream" in text or "stop loop" in text or "stop processing" in text:
            self.log("🤖 Decider stopping processing via natural command.")
            self._run_action("stop_daydream")
            self.current_task = "wait"
            self.cycles_remaining = 0
            return "🛑 Processing stopped."
            
        # Think
        if text.startswith("think about") or text.startswith("analyze") or text.startswith("ponder"):
            topic = text.replace("think about", "").replace("analyze", "").replace("ponder", "").strip()
            self.perform_thinking_chain(topic)
            return f"🧠 Finished thinking about: {topic}"
            
        # Debate
        if text.startswith("debate") or text.startswith("discuss"):
            topic = text.replace("debate", "").replace("discuss", "").strip()
            if self.dialectics:
                return f"🏛️ Council Result: {self.dialectics.run_debate(topic)}"
        
        # Simulate / What If
        if text.startswith("simulate") or text.startswith("what if"):
            premise = text.replace("simulate", "").replace("what if", "").strip()
            if "simulate_counterfactual" in self.actions:
                return f"🌌 Simulation Result: {self.actions['simulate_counterfactual'](premise)}"

        # Tools: Calculator
        if text.startswith("calculate") or text.startswith("solve") or text.startswith("math"):
            expr = re.sub(r'^(calculate|solve|math)\s+', '', text, flags=re.IGNORECASE)
            result = self._execute_tool("CALCULATOR", expr)
            return f"🧮 Calculation Result: {result}"
            
        # Tools: Clock
        if any(phrase in text for phrase in ["what time", "current time", "clock"]):
            result = self._execute_tool("CLOCK", "")
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
            result = self._execute_tool("DICE", args)
            return f"🎲 Dice Roll: {result}"
            
        # Tools: System Info
        if "system info" in text or "specs" in text or "hardware" in text:
            result = self._execute_tool("SYSTEM_INFO", "")
            return f"💻 System Info: {result}"

        # --- Fallback: LLM-based Intent Analysis ---
        # If regex failed but keywords are present, ask the AI what it thinks.
        # Removed common words like "think", "check" to prevent false positives in normal conversation.
        trigger_keywords = ["learn", "study", "research", "summarize", "summary", "verify", "analyze", "ponder", "digest"]
        
        # Only analyze if keywords exist and it's not a super short greeting
        if any(kw in text for kw in trigger_keywords) and len(text.split()) > 2:
            if status_callback: status_callback("Analyzing intent...")
            self.log(f"🧠 Decider analyzing intent for: '{text}'")
            
            intent_response = self._analyze_intent(text)
            self.log(f"🧠 Intent detected: {intent_response}")
            
            if "[LEARN]" in intent_response:
                topic = intent_response.split("]", 1)[1].strip()
                # Clean topic
                clean_topic = re.sub(r"\s+from\s+(?:your\s+)?(?:documents|files|database|memory).*", "", topic, flags=re.IGNORECASE).strip()
                
                self.create_goal(f"Expand knowledge about {clean_topic}")
                self.log(f"🤖 Decider starting Daydream Loop focused on: {clean_topic}")
                self._run_action("start_loop")
                self.current_task = "daydream"
                self.daydream_mode = "read"
                self.daydream_topic = clean_topic
                self.cycles_remaining = 5
                return f"📚 Initiating research protocol for: {clean_topic}. I will read relevant documents and generate insights."
                
            elif "[VERIFY]" in intent_response:
                if "belief" in intent_response.lower():
                    self.log("🤖 Decider starting Belief Verification via intent analysis.")
                    self.current_task = "verify"
                    self.cycles_remaining = 3
                    self._run_action("verify_batch")
                    return "🕵️ Initiating belief verification."
                else:
                    self.log("🤖 Decider starting Verification Batch via intent analysis.")
                    self._run_action("verify_batch")
                    return "🕵️ Verification batch triggered."
                
            elif "[THINK]" in intent_response:
                topic = intent_response.split("]", 1)[1].strip()
                self.perform_thinking_chain(topic)
                return f"🧠 Finished thinking about: {topic}"
            
            elif "[SIMULATE]" in intent_response:
                premise = intent_response.split("]", 1)[1].strip()
                if "simulate_counterfactual" in self.actions:
                    return f"🌌 Simulation Result: {self.actions['simulate_counterfactual'](premise)}"

        return None

    def process_chat_message(self, user_text: str, history: List[Dict], status_callback: Callable[[str], None] = None, image_path: Optional[str] = None) -> str:
        """
        Core Chat Logic: RAG -> LLM -> Memory Extraction -> Response.
        Decider now handles the cognitive pipeline for user interactions.
        """
        # Mailbox: Chat is an external interruption that resets the Hod cycle lock
        self.log(f"📬 Decider Mailbox: Received message from User.")
        self.hod_just_ran = False
        self.last_action_was_speak = False

        # Check for natural language commands
        # Skip NL commands if image is present (prioritize Vision), UNLESS it's a slash command
        if not image_path or user_text.strip().startswith("/"):
            nl_response = self.handle_natural_language_command(user_text, status_callback)
            if nl_response:
                self.log(f"🤖 Decider Command Response: {nl_response}")
                return nl_response

        settings = self.get_settings()
        
        # Parallelize Context Retrieval to reduce latency
        with ThreadPoolExecutor() as executor:
            # 1. Start Embedding (Network)
            future_embedding = executor.submit(
                compute_embedding, 
                user_text, 
                settings.get("base_url"), 
                settings.get("embedding_model")
            )
            
            # 2. Start Memory Retrieval (DB)
            future_recent = executor.submit(self.memory_store.list_recent, limit=10)
            future_chat = executor.submit(self._get_recent_chat_memories, limit=20)
            future_critical = executor.submit(self._get_critical_memories)
            
            # 3. Start Summary Retrieval (DB)
            def get_summary():
                if hasattr(self.meta_memory_store, 'get_by_event_type'):
                    summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1)
                    if summaries:
                        return summaries[0]['text']
                return ""
            future_summary = executor.submit(get_summary)
            
            # Wait for embedding to proceed with Semantic Search & RAG
            query_embedding = future_embedding.result()
            
            # 4. Start Semantic Search (FAISS/DB)
            future_semantic = executor.submit(self.memory_store.search, query_embedding, limit=10)
            
            # 5. Start RAG (FAISS/DB)
            future_rag = None
            if self._should_trigger_rag(user_text):
                self.log(f"📚 [RAG] Searching documents for: '{user_text}'")
                def perform_rag():
                    doc_results = self.document_store.search_chunks(query_embedding, top_k=5)
                    filename_matches = self.document_store.search_filenames(user_text)
                    return doc_results, filename_matches
                future_rag = executor.submit(perform_rag)
            
            # Gather all results
            recent_items = future_recent.result()
            chat_items = future_chat.result()
            critical_items = future_critical.result()
            summary_text = future_summary.result()
            semantic_items = future_semantic.result()
            
            # --- Active Association via Binah ---
            associative_items = []
            if self.binah and semantic_items:
                seed_ids = [item[0] for item in semantic_items]
                assoc_results = self.binah.expand_associative_context(seed_ids, limit=5)
                # Convert to tuple format: (id, type, subject, text, similarity)
                for res in assoc_results:
                    # Use strength as similarity score
                    associative_items.append((res['id'], res['type'], "Association", res['text'], res['strength']))
                
                if associative_items:
                    self.log(f"🔗 Binah: Expanded context with {len(associative_items)} associated memories.")
            
            doc_results = []
            filename_matches = []
            if future_rag:
                doc_results, filename_matches = future_rag.result()

        # --- Token Budget Calculation ---
        context_window = int(settings.get("context_window", 4096))
        max_gen_tokens = int(settings.get("max_tokens", 800))
        # Approx 3 chars per token for safety
        total_budget_chars = context_window * 3
        
        # Reserve space for System Prompt Base, User Text, History, and Generation (approx 2000 chars reserved)
        history_chars = sum(len(m.get('content', '')) for m in history)
        reserved_chars = len(settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)) + len(user_text) + history_chars + (max_gen_tokens * 3) + 500
        available_chars = max(1000, total_budget_chars - reserved_chars)
        self.log(f"💰 Context Budget: {available_chars} chars available for Memory/RAG (Window: {context_window})")

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
            
            mem_block = ""
            if user_mems: mem_block += "User Profile (You are talking to):\n" + "\n".join(user_mems) + "\n\n"
            if assistant_identities: mem_block += "Assistant Identity (Who you are):\n" + "\n".join(assistant_identities) + "\n\n"
            if assistant_goals: mem_block += "Active Goals (Execute ONLY if relevant to current topic):\n" + "\n".join(assistant_goals) + "\n\n"
            if assistant_other: mem_block += "Assistant Knowledge/State:\n" + "\n".join(assistant_other) + "\n\n"
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
        system_prompt = settings.get("system_prompt", "")
        if not system_prompt:
            system_prompt = DEFAULT_SYSTEM_PROMPT
            
        # DYNAMIC STRATEGY INJECTION
        # 1. Search for RULE/STRATEGY memories relevant to the user's input
        active_rules = self.memory_store.get_active_by_type("RULE")
        
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

        # NEW: Self-Improvement Prompt (Appended after memory context)
        self_improvement_prompt = settings.get("self_improvement_prompt", "")
        if self_improvement_prompt:
            system_prompt += "\n\n" + self_improvement_prompt

        # 4. Call LLM
        reply = run_local_lm(
            history, 
            system_prompt=system_prompt,
            temperature=settings.get("temperature", 0.7),
            top_p=settings.get("top_p", 0.94),
            max_tokens=settings.get("max_tokens", 800),
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model"),
            stop_check_fn=self.stop_check,
            images=[image_path] if image_path else None
        )
        
        # Check for LLM error
        if reply.startswith("⚠️"):
            self.log(f"❌ Chat generation failed: {reply}")
            if status_callback: status_callback("Generation Error")
            return "⚠️ I encountered an error generating a response. Please check the logs."

        # Check for Tool Execution in Chat
        if "[EXECUTE:" in reply:
            try:
                match = re.search(r"\[EXECUTE:\s*([A-Z_]+)\s*,\s*(.*?)\]", reply, re.IGNORECASE)
                if match:
                    tool_name = match.group(1).upper()
                    args = match.group(2).strip()
                    result = self._execute_tool(tool_name, args)
                    reply += f"\n\n🛠️ Tool Result: {result}"
            except Exception as e:
                self.log(f"⚠️ Chat tool execution failed: {e}")

        # 5. Memory Extraction (Side Effect)
        # Run in background thread to unblock UI response
        def background_processing():
            if status_callback: status_callback("Extracting memories...")
            self._extract_and_save_memories(user_text, reply, settings)
            if status_callback: status_callback("Ready")

            # Log interaction
            if hasattr(self.meta_memory_store, 'add_event'):
                user_preview = user_text[:100].replace('\n', ' ')
                if len(user_text) > 100: user_preview += "..."
                reply_preview = reply[:100].replace('\n', ' ')
                if len(reply) > 100: reply_preview += "..."
                self.meta_memory_store.add_event(
                    event_type="DECIDER_CHAT",
                    subject="Assistant",
                    text=f"Chat: '{user_preview}' -> '{reply_preview}'"
                )
        
        threading.Thread(target=background_processing, daemon=True).start()
        
        self.log(f"🗣️ Assistant Reply: {reply}")

        return reply

    def _should_trigger_rag(self, text: str) -> bool:
        """Determine if we should run RAG based on user input."""
        text = text.strip().lower()
        
        force_keywords = {
            "search", "find", "document", "file", "pdf", "docx", "content", 
            "read", "summarize", "summary", "reference", "source", "lookup",
            "according to", "check", "journal"
        }
        if any(kw in text for kw in force_keywords): return True
        
        if "?" in text:
            conversational = ["how are you", "how is it going", "what's up", "who are you", "what is your name"]
            if any(c in text for c in conversational): return False
            return True
            
        # Default to False to prevent slowdown on statements
        return False

    def _extract_and_save_memories(self, user_text, assistant_text, settings):
        """Extract memories and run arbiter logic"""
        try:
            # Use a simplified instruction to defer to the System Prompt (which is configurable in settings)
            # This prevents the hardcoded instruction in lm.py from overriding the detailed settings prompt
            custom_instr = "Analyze the conversation. Extract all durable memories (Identity, Facts, Goals, etc.) based on the System Rules. Return JSON."

            candidates = extract_memory_candidates(
                user_text=user_text,
                assistant_text=assistant_text,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                embedding_model=settings.get("embedding_model"),
                memory_extractor_prompt=settings.get("memory_extractor_prompt", DEFAULT_MEMORY_EXTRACTOR_PROMPT),
                custom_instruction=custom_instr,
                stop_check_fn=self.stop_check
            )

            # Add source metadata and filter by confidence
            for c in candidates:
                c["source"] = "assistant"
                c["confidence"] = c.get("confidence", 0.9)

            # Filter: skip low-confidence
            candidates = [c for c in candidates if c.get("confidence", 0.5) > 0.4]

            if not candidates: return

            # Reasoning layer
            for c in candidates:
                self.reasoning_store.add(content=c["text"], source=c.get("source", "assistant"), confidence=c.get("confidence", 0.9))

            # Arbiter promotion
            promoted = 0
            for c in candidates:
                r = self.reasoning_store.search(c["text"], top_k=1)
                if not r or len(r) == 0: continue
                
                mid = self.arbiter.consider(
                    text=r[0]["content"],
                    mem_type=c.get("type", "FACT"),
                    subject=c.get("subject", "User"),
                    confidence=c.get("confidence", 0.85),
                    source=r[0].get("source", "reasoning")
                )
                
                if mid is not None:
                    promoted += 1
            
            if promoted:
                self.log(f"🧠 Promoted {promoted} memory item(s).")

        except Exception as e:
            self.log(f"Memory extraction error: {e}")

    def run_autonomous_cycle(self):
        """
        Called by AICore when system is idle but has goals.
        Checks for active goals and executes tools to advance them.
        """
        # 1. Check for Active Goals
        stats = self.memory_store.get_memory_stats()
        if stats.get('active_goals', 0) == 0:
            return None # Nothing to do

        # 2. Pick a goal, prioritizing LEAF goals (actionable) over ROOT goals (planning)
        with self.memory_store._connect() as con:
            # Fetch more candidates to ensure we find leaf goals
            goals = con.execute("SELECT id, text, parent_id FROM memories WHERE type='GOAL' AND completed=0 ORDER BY id DESC LIMIT 20").fetchall()
        
        if not goals: return None
        
        # Separate into Leaf (has parent) and Root (no parent)
        leaf_goals = [g for g in goals if g[2] is not None]
        root_goals = [g for g in goals if g[2] is None]
        
        target_goal = None
        
        # 80% chance to pick a leaf goal if available (Action bias)
        if leaf_goals and random.random() < 0.8:
            target_goal = random.choice(leaf_goals)
        elif root_goals:
            target_goal = random.choice(root_goals)
        elif leaf_goals:
            target_goal = random.choice(leaf_goals)
            
        if not target_goal: return None
        
        goal = target_goal
        goal_id, goal_text, parent_id = goal

        # 2.5. The Architect: Check if goal needs decomposition
        # Only decompose ROOT goals (parent_id is None) to prevent infinite recursion
        if parent_id is None:
            # We check if it already has children
            with self.memory_store._connect() as con:
                has_children = con.execute("SELECT 1 FROM memories WHERE parent_id = ? LIMIT 1", (goal_id,)).fetchone()
            
            if not has_children:
                if self._decompose_goal(goal_text, goal_id):
                    return None # Decomposition happened, wait for next cycle to pick up sub-goals

        # 3. SELF-PROMPT: "We have an active goal..."
        # We simulate a 'System' message to trigger the LLM to use a tool
        system_injection = f"[SYSTEM_TRIGGER]: You are idle. Active Goal: '{goal_text}'. Execute a tool (SEARCH/WIKI) to advance this."
        
        self.log(f"🤖 Decider: Autonomously pursuing goal: {goal_text}")
        
        # 4. Run the thought loop (This triggers [EXECUTE: ...])
        response = run_local_lm(
            messages=[{"role": "system", "content": system_injection}],
            system_prompt=self.get_settings().get("system_prompt")
        )
        
        return response

    def _decompose_goal(self, goal_text, goal_id):
        """Breaks a big goal into small steps."""
        # Only decompose if it looks complex (heuristic: length > 20 chars)
        if len(goal_text) < 20: return False

        # HTN (Hierarchical Task Network) Decomposition
        prompt = (
            f"Goal: '{goal_text}'\n"
            "Decompose this goal into a Hierarchical Task Network (HTN) tree structure.\n"
            "Return a JSON array of objects, where each object represents a high-level sub-task:\n"
            "[\n"
            "  {\"step\": \"Step description\", \"success_criteria\": \"Specific verifiable condition\"},\n"
            "  ...\n"
            "]\n"
            "Limit to 3-5 high-level steps."
        )
        settings = self.get_settings()
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a strategic planner.",
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        steps = _parse_json_array_loose(response)
        
        if steps:
            self.log(f"🏗️ The Architect: Breaking goal '{goal_text}' into {len(steps)} steps.")
            for step in steps:
                if isinstance(step, dict) and "step" in step:
                    text = f"{step['step']} [Criteria: {step.get('success_criteria', 'None')}]"
                    self.memory_store.add_entry(
                        identity=self.memory_store.compute_identity(text, "GOAL"),
                        text=text,
                        mem_type="GOAL",
                        subject="Assistant",
                        confidence=1.0,
                        source="architect_decomposition",
                        parent_id=goal_id 
                    )
                elif isinstance(step, str):
                    # Fallback for simple string list
                    self.memory_store.add_entry(
                        identity=self.memory_store.compute_identity(step, "GOAL"),
                        text=step,
                        mem_type="GOAL",
                        subject="Assistant",
                        confidence=1.0,
                        source="architect_decomposition",
                        parent_id=goal_id 
                    )
            return True
        return False