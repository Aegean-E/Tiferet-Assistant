import re
import json
import time
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional, List
from ai_core.lm import run_local_lm
from ai_core.utils import parse_json_array_loose

if TYPE_CHECKING:
    from treeoflife.tiferet import Decider

class DecisionMaker:
    def __init__(self, decider: 'Decider'):
        self.decider = decider

    def decide(self) -> Dict[str, Any]:
        """Decide what to do next. Returns a plan dict."""
        # Prevent concurrent planning (e.g. from Chat trigger AND Background loop)
        if not self.decider.planning_lock.acquire(blocking=False):
            self.decider.log("🤖 Decider: Planning already in progress. Skipping.")
            return {"task": "wait", "cycles": 0, "reason": "Planning locked"}

        if self.decider.panic_mode:
            self.decider.log("🤖 Decider: In PANIC mode. Waiting for manual intervention or reboot.")
            return {"task": "wait", "cycles": 0, "reason": "System Panic"}

        try:
            # Gather Structural Metrics & Determine Mode
            stats = self.decider.memory_store.get_memory_stats()
            with self.decider.memory_store._connect() as con:
                spawn_count = con.execute("SELECT COUNT(*) FROM memories WHERE type='GOAL' AND created_at > ?", (int(time.time()) - 3600,)).fetchone()[0]
                gaps_count = con.execute("SELECT COUNT(*) FROM memories WHERE type='CURIOSITY_GAP' AND completed=0").fetchone()[0]

            structural_metrics = {
                "active_goals": stats.get('active_goals', 0),
                "unresolved_conflicts": gaps_count,
                "goal_spawn_rate": spawn_count
            }

            # Get System Mode from CRS
            coherence = self.decider.keter.evaluate().get("keter", 1.0) if self.decider.keter else 1.0

            # Get Stability State
            stability_state = {}
            if self.decider.stability_controller:
                stability_state = self.decider.stability_controller.evaluate()

            # Pass metrics to CRS
            crs_status = {}
            try:
                if self.decider.crs:
                    # Use a dummy task type for system check
                    crs_status = self.decider.crs.allocate("system_check", coherence=coherence, structural_metrics=structural_metrics, violation_pressure=self.decider.value_core.get_violation_pressure() if self.decider.value_core else 0.0)
            except Exception as e:
                self.decider.log(f"⚠️ Decider: CRS allocation failed during system check: {e}")
                crs_status = {}

            system_mode = crs_status.get("system_mode", "EXECUTION")
            allow_goal_creation = crs_status.get("allow_goal_creation", True)

            # 0. Goal Autonomy Cycle (Generate/Rank/Prune)
            # Run every 10 minutes or if we have no active goals
            # Check last_goal_management_time from goal_manager if available, else 0
            last_mgmt = getattr(self.decider.goal_manager, 'last_goal_management_time', 0)
            if time.time() - last_mgmt > 600 or not self.decider.memory_store.get_active_by_type("GOAL"):
                self.decider.goal_manager.manage_goals(allow_creation=allow_goal_creation, system_mode=system_mode)

            self.decider.log("🤖 Decider: Planning next batch...")
            settings = self.decider.get_settings()

            # Fetch Active Goals early (Fix for UnboundLocalError in Reflex path)
            all_items = self.decider.memory_store.list_recent(limit=None)
            active_goals = [m for m in all_items if len(m) > 1 and m[1] == 'GOAL']

            # Tabula Rasa Check (Empty Mind)
            is_tabula_rasa = (len(active_goals) == 0 and len(all_items) < 5)

            # Cost Evaluation & ROI Ranking (CRS)
            if self.decider.crs:
                # prioritize_goals returns (goal, roi, cost)
                # active_goals list items are tuples from list_recent: (id, type, subject, text, ...)
                # We need to adapt the tuple format for CRS or just pass it if CRS handles it.
                # CRS expects (id, subject, text, source, confidence)
                # list_recent gives (id, type, subject, text, source, verified, flags, confidence)
                mapped_goals = [(g[0], g[2], g[3], g[4], g[7] if len(g)>7 else 0.5) for g in active_goals]
                ranked_goals_info = self.decider.crs.prioritize_goals(mapped_goals, 10000) # 10000 is dummy budget
                # Re-sort active_goals based on ranked_goals_info order
                ranked_ids = [r[0][0] for r in ranked_goals_info]
                active_goals.sort(key=lambda x: ranked_ids.index(x[0]) if x[0] in ranked_ids else 999)
            else:
                # Fallback sort
                try:
                    active_goals.sort(key=lambda x: (float(x[7]) if len(x) > 7 and x[7] is not None else 0.5, x[0]), reverse=True)
                except Exception as e:
                    self.decider.log(f"⚠️ Error sorting goals: {e}")
                    # Fallback to ID sort
                    active_goals.sort(key=lambda x: x[0], reverse=True)

            active_goals = active_goals[:5]

            # Check for Necessity Goals (Self-Generated Objectives)
            necessity_goals = [g for g in active_goals if g[4] == "model_stress"]
            if necessity_goals:
                self.decider.log(f"🚨 Decider detected {len(necessity_goals)} Necessity Goals (Model Stress). Prioritizing.")

            # Rate limit Daydreaming: Force variety if we've daydreamed recently
            allow_daydream = True
            if self.decider.consecutive_daydream_batches >= 3:
                allow_daydream = False
                self.decider.log("🤖 Decider: Daydreaming quota reached (3 batches). Forcing other options.")

            # Capture signal before processing/clearing
            current_netzach_signal = "Unknown"
            if self.decider.latest_netzach_signal:
                current_netzach_signal = self.decider.latest_netzach_signal.get("signal", "Unknown")

            # Check Netzach's signal
            self.decider.command_executor._check_netzach_signal()

            # Gather context for decision making
            recent_mems = self.decider.memory_store.list_recent(limit=5)
            chat_mems = self._get_recent_chat_memories(limit=3)

            # Merge unique memories (prioritizing recent, but ensuring chat memories are included)
            all_mems = {m[0]: m for m in recent_mems}
            for m in chat_mems:
                all_mems[m[0]] = m
            display_mems = sorted(all_mems.values(), key=lambda x: x[0], reverse=True)[:8]

            recent_meta = self.decider.meta_memory_store.list_recent(limit=5)

            # Fetch latest session summary (Layer 1: High-level context)
            latest_summary = None
            last_summary_time = 0
            if hasattr(self.decider.meta_memory_store, 'get_by_event_type'):
                summaries = self.decider.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1)
                if summaries:
                    latest_summary = summaries[0]
                    last_summary_time = latest_summary.get('created_at', 0)

            chat_hist = self.decider.get_chat_history()[-5:]

            # Calculate Hesed/Gevurah
            hesed_score = self.decider.hesed.calculate() if self.decider.hesed else 1.0
            gevurah_score = self.decider.gevurah.calculate() if self.decider.gevurah else 0.0
            self.decider.log(f"⚖️ Cognitive State: Hesed={hesed_score:.2f}, Gevurah={gevurah_score:.2f}")

            # Get Signals from Hod and Netzach
            hod_signal = self.decider.hod_force.analyze() if self.decider.hod_force else "Unknown"

            # Use the captured signal if the force object is missing (common in this architecture)
            if self.decider.netzach_force and hasattr(self.decider.netzach_force, 'analyze'):
                netzach_signal = self.decider.netzach_force.analyze()
            else:
                netzach_signal = current_netzach_signal

            # Check Motivation
            motivation = 1.0
            if self.decider.netzach_force and hasattr(self.decider.netzach_force, 'motivation'):
                motivation = self.decider.netzach_force.motivation

            self.decider.log(f"⚖️ Forces: Hod='{hod_signal}', Netzach='{netzach_signal}'")

            # Calculate Utility for Prompt
            current_utility = self.calculate_utility()

            # Calculate Strange Loop Friction
            current_friction = self._measure_strange_loop()
            self.decider.log(f"🌀 Strange Loop Friction: {current_friction:.4f}")

            # --- FAST PATH (Reflex) ---
            # Optimization: Skip LLM for obvious states to reduce latency
            reflex_action = None
            reflex_reason = ""

            # Gevurah Override (Loop Breaking)
            if self.decider.gevurah and self.decider.gevurah.recommendation == "DROP_GOAL":
                if active_goals:
                    target_goal_id = active_goals[0][0] # Drop top goal
                    self.decider.log(f"🔥 Gevurah Override: Dropping Goal {target_goal_id} due to Action Loop.")
                    if "remove_goal" in self.decider.actions:
                        self.decider.actions["remove_goal"](target_goal_id)
                    return {"task": "wait", "cycles": 0, "reason": "Gevurah Loop Break"}

            if necessity_goals:
                # If we have a necessity goal, we MUST act on it.
                reflex_action = "GOAL_ACT"
                reflex_reason = "Necessity Goal (Model Stress) requires resolution."
                target_goal_id = necessity_goals[0][0]
                target_goal_text = necessity_goals[0][3]
            elif gevurah_score > 0.85:
                reflex_action = "VERIFY"
                reflex_reason = "High Gevurah pressure (Constraint > 0.85)"
            elif self.decider.keter and self.decider.keter.evaluate().get("keter", 1.0) < 0.2:
                # Coherence Recovery Protocol
                reflex_action = "VERIFY"
                reflex_reason = "Critical Coherence (< 0.2). Initiating Grounding Protocol."
                self.decider.cycles_remaining = 1
            elif netzach_signal == "NO_MOMENTUM" and motivation > 0.2:
                # Only force daydream if we have NO active goals.
                # If goals exist, we want the LLM to choose [GOAL_ACT].
                if not active_goals:
                    reflex_action = "DAYDREAM"
                    reflex_reason = "Netzach signal: No Momentum & No Goals"
            elif netzach_signal == "LOW_NOVELTY" and hesed_score > 0.4 and not active_goals and motivation > 0.3:
                reflex_action = "DAYDREAM"
                reflex_reason = "Netzach signal: Low Novelty"

            if reflex_action:
                self.decider.log(f"⚡ Decider Reflex Triggered: {reflex_action} due to {reflex_reason}")
                if reflex_action == "VERIFY":
                    return {"task": "verify", "cycles": 2, "reason": reflex_reason}
                elif reflex_action == "DAYDREAM":
                    self.decider.daydream_mode = "auto"
                    return {"task": "daydream", "cycles": 1, "reason": reflex_reason}
                elif reflex_action == "GOAL_ACT":
                    self.decider.log(f"⚡ Decider Reflex: Acting on Necessity Goal {target_goal_id}")
                    self.decider.thought_generator.perform_thinking_chain(f"Resolve Necessity Goal: {target_goal_text}")
                    # FIX: Mark goal as completed to prevent infinite loop
                    if "remove_goal" in self.decider.actions:
                        self.decider.actions["remove_goal"](target_goal_id)
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": reflex_reason}

            recent_docs = self.decider.document_store.list_documents(limit=5)


            # Fetch Curiosity Gaps (Urgent Questions)
            curiosity_gaps = self.decider.memory_store.get_active_by_type("CURIOSITY_GAP")
            curiosity_gaps = curiosity_gaps[:3] # Limit to 3

            # Fetch Strategies (Rules)
            strategies = self.decider.memory_store.get_active_by_type("RULE")
            strategy_section = ""
            if strategies:
                # Filter for strategies (heuristic: starts with STRATEGY or contains it)
                strategy_list = [f"- {s[2]}" for s in strategies if "STRATEGY:" in s[2]]
                if strategy_list:
                    strategy_section = "🧠 LEARNED STRATEGIES:\n" + "\n".join(strategy_list) + "\n\n"

            # Fetch Recent Reasoning (to see tool outputs)
            recent_reasoning = []
            if hasattr(self.decider.reasoning_store, 'list_recent'):
                recent_reasoning = self.decider.reasoning_store.list_recent(limit=5)

            context = "CONTEXT:\n"

            # --- GLOBAL WORKSPACE INJECTION ---
            if self.decider.global_workspace:
                gw_context = self.decider.global_workspace.get_context()
                context += f"🧠 GLOBAL WORKSPACE (Working Memory):\n{gw_context}\n\n"
            # ----------------------------------

            if strategy_section:
                context += strategy_section
            context += "System Signals (The Forces):\n"

            if self.decider.stream_of_consciousness:
                # Use a local copy to avoid race conditions during iteration (though it's replaced atomically)
                context += "STREAM OF CONSCIOUSNESS (Recent Thoughts):\n" + "\n".join([f"- {t}" for t in list(self.decider.stream_of_consciousness)]) + "\n\n"

            context += f"- Gevurah (Constraint): {gevurah_score:.2f} (Pressure)\n"
            context += f"- Hesed (Expansion): {hesed_score:.2f} (Safety)\n"
            context += f"- Hod (Form): {hod_signal}\n"
            context += f"- Utility Score: {current_utility:.4f} (Maximize this!)\n"
            context += f"- Existential Friction: {current_friction:.4f} (Minimize this!)\n"
            context += f"- Netzach (Endurance): {netzach_signal}\n\n"
            context += f"- Motivation: {motivation:.2f}\n"
            context += f"- System Mode: {system_mode} (Load: {crs_status.get('structural_load', 0.0):.2f})\n"
            if system_mode == "CONSOLIDATION":
                 context += "CONSTRAINT: System is overloaded. DO NOT create new goals. Focus on closing existing ones or consolidating memory.\n"

            # Affective Bias Injection
            if self.decider.mood < 0.3:
                context += "EMOTIONAL STATE: Depressed/Fatigued. You feel discouraged. Prioritize stability, safety, and low-risk consolidation. Avoid new goals.\n"
            elif self.decider.mood > 0.8:
                context += "EMOTIONAL STATE: Manic/Energetic. You feel highly capable. Prioritize high-risk, high-reward innovation and complex reasoning.\n"
            else:
                context += "EMOTIONAL STATE: Balanced. Proceed with standard optimization.\n"

            # Motivation Check
            if motivation < 0.2:
                context += "WARNING: Motivation is critical. You are depressed/exhausted. Only respond to direct user commands or urgent threats. Ignore optional goals.\n"

            # Stability Mode Injection
            stab_mode = stability_state.get("mode", "normal")
            if stab_mode == "ethical_lockdown":
                context += "🚨 STATUS: ETHICAL LOCKDOWN. You are operating under strict safety constraints. Prioritize introspection and self-correction.\n"
            elif stab_mode == "entropy_control":
                context += "🧹 STATUS: ENTROPY CONTROL. Focus on cleaning up contradictions and verifying facts.\n"
            elif stab_mode == "identity_recovery":
                context += "🆔 STATUS: IDENTITY RECOVERY. Reflect on your core values and purpose.\n"

            # Strange Loop Injection
            if current_friction > 0.4:
                context += "⚠️ HIGH FRICTION DETECTED. The system is vibrating out of alignment. You MUST prioritize [REFLECT], [DEBATE], or [VERIFY] to restore harmony.\n"

            if latest_summary:
                context += f"Last Session Summary:\n{latest_summary['text']}\n\n"
            if chat_hist:
                context += "Chat History:\n" + "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in chat_hist]) + "\n"
            if display_mems:
                context += "Recent Memories (Note: REFUTED_BELIEF means the idea is FALSE/REJECTED):\n" + "\n".join([f"- [{m[1]}] [{m[2]}] {m[3][:200]}" for m in display_mems]) + "\n"
            if curiosity_gaps:
                context += "⚠️ ACTIVE CURIOSITY GAPS (Things you want to know):\n" + "\n".join([f"- {g[2]}" for g in curiosity_gaps]) + "\n"
            if active_goals:
                context += "Active Goals (Ranked by ROI/Priority):\n"
                for m in active_goals:
                    # Display goals in the ranked order
                    context += f"- [ID: {m[0]}] {m[3][:150]}...\n"
                context += "\n"
            if recent_meta:
                context += "Recent Events:\n" + "\n".join([f"- {m[3][:100]}..." for m in recent_meta]) + "\n"
            if recent_reasoning:
                context += "Recent Thoughts:\n" + "\n".join([f"- {r.get('content', str(r))[:80]}..." if isinstance(r, dict) else f"- {str(r)[:80]}..." for r in recent_reasoning]) + "\n"
            if recent_docs:
                context += "Available Documents (Partial List):\n" + "\n".join([f"- {d[1]} (ID: {d[0]}, Chunks: {d[4]})" for d in recent_docs]) + "\n"
                context += "(Use [LIST_DOCS] to see the full library)\n"

            current_task = self.decider.heartbeat.current_task if self.decider.heartbeat else "unknown"
            context += f"Current Task: {current_task}\n"

            if current_task == "wait" and self.decider.heartbeat and self.decider.heartbeat.wait_start_time > 0:
                wait_duration = time.time() - self.decider.heartbeat.wait_start_time
                context += f"INFO: Currently in WAIT state for {int(wait_duration)} seconds.\n"

            if not allow_daydream:
                context += "CONSTRAINT: Daydreaming is temporarily disabled (Quota Reached). You MUST choose a different action (e.g. THINK, REFLECT, VERIFY).\n"

            # Safety Truncation (Estimate 1 char ~= 0.25 tokens, keep safe buffer)
            MAX_CONTEXT_CHARS = 10000  # Reduced to ~2500 tokens to be safe


            # Check if we just finished a heavy task to enforce cooldown
            # just_finished_heavy = self.current_task in ["daydream", "verify", "verify_all"]
            # Disabled to allow continuous operation as requested
            just_finished_heavy = False

            # Permission Mapping for Stability Controller
            # Maps concrete commands to abstract permissions potentially returned by StabilityController
            permission_map = {
                "daydream": ["daydream", "exploration"],
                "think": ["think", "reasoning", "introspection", "self_correction"],
                "reflect": ["reflect", "introspection"],
                "philosophize": ["philosophize", "exploration"],
                "debate": ["debate", "reasoning"],
                "simulate": ["simulate", "reasoning"],
                "verify": ["verify", "verify_batch", "self_correction"],
                "verify_all": ["verify_all", "self_correction"],
                "goal_act": ["goal_act", "reasoning", "self_correction"],
                "goal_create": ["goal_create", "planning"],
                "goal_remove": ["goal_remove", "planning"],
                "search_mem": ["search_mem", "introspection"],
                "summarize": ["summarize", "introspection"],
                "chronicle": ["chronicle", "introspection"],
                "search_internet": ["search_internet", "research"],
                "list_docs": ["list_docs", "research"],
                "read_doc": ["read_doc", "research"],
                "execute_tool": ["execute_tool", "tool_use"],
                "physics": ["physics", "tool_use"],
                "causal": ["causal", "tool_use"],
                "simulate_action": ["simulate_action", "planning"],
                "predict": ["predict", "planning"],
                "write_file": ["write_file", "tool_use"],
                "speak": ["speak", "communication"]
            }

            def is_allowed(cmd_key):
                allowed = stability_state.get("allowed_actions")
                if allowed is None: return True
                if cmd_key in allowed: return True
                for cat in permission_map.get(cmd_key, []):
                    if cat in allowed: return True
                return False

            options_text = "Options:\n1. [WAIT]: Stop processing and wait for user input. Use this if the system is stable and no urgent goals exist.\n"

            if not self.decider.forced_stop_cooldown and not just_finished_heavy:
                options_text += "\n--- COGNITIVE ACTIONS ---\n"
                if allow_daydream and is_allowed("daydream"):
                    options_text += "2. [DAYDREAM: N, MODE, TOPIC]: Run N cycles (1-5). MODE: 'READ', 'AUTO'. TOPIC: Optional subject. E.g., [DAYDREAM: 3, READ, Neurology]\n"
                    if active_goals:
                        options_text += "   (TIP: Use the TOPIC argument to focus on an active goal!)\n"
                if is_allowed("think"): options_text += "3. [THINK: <TOPIC>, <STRATEGY>]: Start a reasoning chain. STRATEGY can be AUTO, LINEAR, TREE_OF_THOUGHTS, DIALECTIC, or FIRST_PRINCIPLES. E.g., [THINK: Consciousness, DIALECTIC].\n"
                if is_allowed("reflect"): options_text += "4. [REFLECT]: Pause to reflect on your own Identity, Goals, and recent performance. Use this to align actions with purpose.\n"
                if is_allowed("philosophize"): options_text += "5. [PHILOSOPHIZE]: Generate a novel philosophical insight or hypothesis ex nihilo (using Chokmah).\n"
                if is_allowed("debate"): options_text += "6. [DEBATE: <TOPIC>]: Convene 'The Council' (Hesed/Gevurah) to debate a complex topic.\n"
                if is_allowed("simulate"): options_text += "7. [SIMULATE: <PREMISE>]: Run a counterfactual simulation based on the World Model. Use for 'What if' scenarios.\n"

                options_text += "\n--- MEMORY & GOALS ---\n"
                if is_allowed("verify"): options_text += "8. [VERIFY: N]: Run N batches of verification (1 to 3). Use this if you suspect hallucinations or haven't verified in a while.\n"
                if is_allowed("goal_act"): options_text += "9. [GOAL_ACT: <GOAL_ID>]: Focus strategic thinking on a specific goal. Triggers a thinking chain to progress this goal.\n"
                if is_allowed("goal_create"): options_text += "10. [GOAL_CREATE: <TEXT>]: Autonomously create a new goal for yourself based on the situation.\n"
                if is_allowed("goal_remove"): options_text += "11. [GOAL_REMOVE: <GOAL_ID_OR_TEXT>]: Mark a goal as COMPLETED. Use this when a goal is achieved.\n"
                if is_allowed("search_mem"): options_text += "12. [SEARCH_MEM: <QUERY>]: Actively search your long-term memory for specific information.\n"
                if is_allowed("summarize"): options_text += "13. [SUMMARIZE]: Summarize recent session activity into a meta-memory (Da'at).\n"
                if is_allowed("chronicle"): options_text += "14. [CHRONICLE: <CONTENT>]: Write to the Chronicles section (Internal Journal). Record insights, plans, or reflections.\n"

                options_text += "\n--- TOOLS & EXTERNAL ---\n"
                if is_allowed("search_internet"): options_text += "15. [SEARCH_INTERNET: <QUERY>, <SOURCE>]: Request external data. Source: WEB, WIKIPEDIA, PUBMED, or ARXIV. E.g., [SEARCH_INTERNET: Quantum Computing, WEB].\n"
                if is_allowed("list_docs"): options_text += "16. [LIST_DOCS]: List available documents in the library.\n"
                if is_allowed("read_doc"): options_text += "17. [READ_DOC: <FILENAME_OR_ID>, <START_CHUNK>]: Read a document. Optional start chunk (default 0). E.g. [READ_DOC: paper.pdf, 5]\n"
                if is_allowed("execute_tool"): options_text += "18. [EXECUTE: <TOOL_NAME>, <ARGS>]: Execute a system tool. Available: [CALCULATOR, CLOCK, DICE, SYSTEM_INFO].\n"
                if is_allowed("physics"): options_text += "19. [EXECUTE: PHYSICS, '<SCENARIO>']: Perform a reality check or Fermi estimation on a physical/biochemical scenario.\n"
                if is_allowed("causal"): options_text += "20. [EXECUTE: CAUSAL, '<TREATMENT>, <OUTCOME>, <CONTEXT>']: Perform Bayesian Causal Inference (DoWhy).\n"
                if is_allowed("simulate_action"): options_text += "21. [EXECUTE: SIMULATE_ACTION, '<ACTION>']: Predict the outcome/cost of an action before doing it (Internal Simulator).\n"
                if is_allowed("predict"): options_text += "22. [EXECUTE: PREDICT, '<CLAIM>, <TIMEFRAME>']: Make a measurable prediction to verify later. E.g., 'Stock X will rise, 1 week'.\n"
                if is_allowed("write_file"): options_text += "22. [WRITE_FILE: <FILENAME>, <CONTENT>]: Write content to a file in the 'works' folder.\n"
            else:
                reason = "forced stop" if self.decider.forced_stop_cooldown else "consecutive heavy task prevention"
                self.decider.log(f"🤖 Decider: Daydream/Verify disabled for this turn due to {reason}.")

            if not self.decider.last_action_was_speak and is_allowed("speak"):
                options_text += "\n--- COMMUNICATION ---\n"
                options_text += "24. [SPEAK: <CONTENT>]: Write a message to the user. Replace '<CONTENT>' with the actual text.\n"

            # --- Bicameral Dialogue (Internal Friction) ---
            # Generate conflicting internal voices before the final decision

            # Voice of Impulse (Hesed)
            impulse_prompt = (
                f"CONTEXT:\n{context}\n\n"
                "ROLE: You are the Voice of Impulse (Hesed).\n"
                "TASK: Propose the most expansive, novel, creative, and high-risk action available.\n"
                "Ignore safety and stability. Focus on growth and curiosity.\n"
                "Output 1 sentence."
            )
            voice_impulse = run_local_lm(
                messages=[{"role": "user", "content": impulse_prompt}],
                system_prompt="You are the Impulse.",
                temperature=0.9,
                max_tokens=100,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            ).strip()

            # Voice of Reason (Gevurah)
            reason_prompt = (
                f"CONTEXT:\n{context}\n\n"
                "ROLE: You are the Voice of Reason (Gevurah).\n"
                "TASK: Propose the most stable, defensive, safe, and low-risk action available.\n"
                "Focus on correctness, verification, and resource conservation.\n"
                "Output 1 sentence."
            )
            voice_reason = run_local_lm(
                messages=[{"role": "user", "content": reason_prompt}],
                system_prompt="You are the Censor.",
                temperature=0.1,
                max_tokens=100,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            ).strip()

            prompt = (
                "You are the Decider (Tiferet), the central executive of this cognitive architecture.\n"
                "Your task is to determine the next cognitive action based on the System State.\n\n"
                "SYSTEM STATE ANALYSIS RULES:\n"
                "1. **Gevurah (Constraint)**: If > 0.6, the system is under pressure. Prioritize [VERIFY] or [PRUNE].\n"
                "2. **Hesed (Expansion)**: If > 0.6 and Gevurah is low, safe to explore. Prioritize [THINK] or [SEARCH_INTERNET].\n"
                "3. **Utility Optimization**: Choose actions that increase Verified Facts, Goal Completion, or Coherence.\n"
                "3. **Netzach (Endurance)**: 'NO_MOMENTUM' -> Act/Daydream. 'LOW_NOVELTY' -> Change topic.\n"
                "4. **Hod (Form)**: 'Breaking'/'Undefined' -> [VERIFY].\n"
                "5. **Active Goals**: If goals exist, [GOAL_ACT] is usually best.\n"
                "6. **Tabula Rasa**: If empty, [REFLECT] or [GOAL_CREATE].\n\n"
                f"{options_text}\n"
                "INTERNAL DIALOGUE:\n"
                f"⚡ Impulse says: {voice_impulse}\n"
                f"🛡️ Reason says: {voice_reason}\n\n"
            )

            if self.decider.last_internal_thought:
                prompt += f"PREVIOUS THOUGHT: {self.decider.last_internal_thought}\nCONSTRAINT: Do NOT repeat the previous thought. Advance the reasoning.\n\n"

            prompt += (
                "INSTRUCTIONS:\n"
                "1. **Internal Thought**: Synthesize the conflict between Impulse and Reason. Analyze the context. Declare a Strategic Intent.\n"
                "   - AVOID REPETITION. Do not repeat the same thought as the last cycle.\n"
                "2. **Future Projection**: Briefly simulate the outcome of your chosen action 5 cycles from now. Will it lead to burnout or progress?\n"
                "3. **Action**: Select the ONE command that best fulfills the intent.\n\n"
                "Response Format:\n"
                "Internal Thought: <Analysis and Intent>\n"
                "Command: [SELECTED_COMMAND]"
            )

            response = run_local_lm(
                messages=[{"role": "user", "content": context + "\nStatus: Ready. Decide next step."}],
                system_prompt=prompt,
                temperature=0.3, # Lower temperature for deterministic execution
                max_tokens=300,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model"),
                stop_check_fn=self.decider.stop_check
            )

            response = response.strip()

            # Extract Internal Thought for logging
            internal_thought = "No thought generated."
            thought_match = re.search(r"Internal Thought:\s*(.*?)(?=\nCommand:|$)", response, re.IGNORECASE | re.DOTALL)
            if thought_match:
                internal_thought = thought_match.group(1).strip()
                self.decider.last_internal_thought = internal_thought

                # Update history
                norm_t = re.sub(r'\s+', ' ', internal_thought.lower().strip())
                self.decider.thought_history.append(norm_t)
                if len(self.decider.thought_history) > 10: self.decider.thought_history.pop(0)

                self.decider.log(f"🤔 Internal Thought: {internal_thought}")

                # Push Thought to Global Workspace
                if self.decider.global_workspace:
                    self.decider.global_workspace.integrate(internal_thought, "Decider", 1.0)

                if self.decider.chat_fn:
                    self.decider.chat_fn("Decider", f"🤔 Thought: {internal_thought}")
                if hasattr(self.decider.meta_memory_store, 'add_event'):
                    self.decider.meta_memory_store.add_event(
                        event_type="STRATEGIC_THOUGHT",
                        subject="Assistant",
                        text=internal_thought
                    )

                # Ethical Interlock: Check alignment before proceeding
                if self.decider.value_core:
                    is_safe, violation_score, reason = self.decider.value_core.check_alignment(internal_thought, context="Decider Planning")
                    if not is_safe:
                        self.decider.log(f"🛡️ Decider: Thought rejected by ValueCore ({violation_score:.2f}). Reason: {reason}")
                        return {"task": "wait", "cycles": 0, "reason": "ValueCore Rejection"}

                # Repetition Check
                if self._is_repetitive_thought(internal_thought) and len(self.decider.thought_history) >= 3:
                     self.decider.log("🛑 Repetitive Thought detected. Forcing [WAIT] to break cognitive loop.")
                     return {"task": "wait", "cycles": 0, "reason": "Cognitive Loop Detected"}

            # Extract the actual command from the verbose response to prevent false positives
            # (e.g. "I should not [WAIT]" triggering WAIT)
            extracted_command = self._extract_command(response)

            # --- Meta-Cognitive Loop ---
            if extracted_command and self.decider.keter:
                if not self._meta_cognitive_check(internal_thought, extracted_command):
                    self.decider.log(f"🧠 Meta-Cognition: Action '{extracted_command}' misaligned. Forcing [REFLECT].")
                    extracted_command = "[REFLECT]"
            # ---------------------------

            if extracted_command:
                response_upper = extracted_command.upper()
            else:
                response_upper = response.upper()

            self.decider.log(f"🤖 Decider Decision: {response}")

            if self.decider.chat_fn:
                self.decider.chat_fn("Decider", f"Decision: {response}")

            # Decision Reflection (Metacognition)
            self._reflect_on_decision(internal_thought, extracted_command or response)

            # Reset cooldown
            if self.decider.forced_stop_cooldown:
                self.decider.forced_stop_cooldown = False

            # Deadlock / Loop Protection
            if self._detect_deadlock(extracted_command or response):
                self.decider.log(f"🛑 Deadlock Detected (Looping command: {extracted_command}). Forcing [WAIT] to break cycle.")
                return {"task": "wait", "cycles": 0, "reason": "Deadlock detected"}

            if "[WAIT]" in response_upper:
                return {"task": "wait", "cycles": 0, "reason": "Decider chose to wait"}
                # Do not reset consecutive_daydream_batches on WAIT, so we remember to switch tasks after waking up
            elif "[SLEEP]" in response_upper:
                self.decider.command_executor.start_sleep_cycle()
                return {"task": "sleep", "cycles": 5, "reason": "Decider chose to sleep"}
            elif "[DAYDREAM:" in response_upper:
                if not allow_daydream:
                    self.decider.log("⚠️ Decider tried to Daydream during cooldown/prevention. Forcing WAIT.")
                    return {"task": "wait", "cycles": 0, "reason": "Daydream cooldown"}
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

                        self.decider.daydream_mode = mode if mode in ["read", "auto"] else "auto"
                        self.decider.daydream_topic = topic
                        self.decider.consecutive_daydream_batches += 1
                        return {"task": "daydream", "cycles": max(1, min(count, 10)), "reason": f"Strategic decision ({mode})"}
                    except:
                        self.decider.daydream_mode = "auto"
                        self.decider.consecutive_daydream_batches += 1
                        return {"task": "daydream", "cycles": 1, "reason": "Strategic decision (fallback)"}
            elif "[VERIFY_ALL]" in response_upper:
                # --- SAFETY CHECK ---
                stats = self.decider.memory_store.get_memory_stats()
                unverified_count = stats.get('unverified_facts', 0) + stats.get('unverified_beliefs', 0)

                if unverified_count == 0:
                    self.decider.log("🛑 Decider attempted VERIFY_ALL, but 0 items pending. Forcing WAIT.")
                    # Forcefully lower pressure so it doesn't loop
                    if self.decider.gevurah:
                        self.decider.gevurah.last_pressure = 0.0
                    return {"task": "wait", "cycles": 0, "reason": "Verification requested but nothing to verify"}
                else:
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "verify_all", "cycles": 1, "reason": "Strategic decision (Verify All)"}
            elif "[VERIFY" in response_upper and "ALL" not in response_upper:
                try:
                    if ":" in response_upper:
                        count = int(response_upper.split(":")[1].strip().replace("]", ""))
                    else:
                        count = 3
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "verify", "cycles": max(1, min(count, 5)), "reason": "Strategic decision (Verify Batch)"}
                except:
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "verify", "cycles": 1, "reason": "Strategic decision (Verify Batch)"}
            elif "[SPEAK:" in response_upper:
                try:
                    # Use regex to extract content preserving case
                    match = re.search(r"\[SPEAK:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    msg = ""
                    if match:
                        msg = match.group(1).strip()

                    # Filter out placeholders
                    if msg.upper() in ["MESSAGE", "CONTENT", "TEXT", "MSG", "INSERT TEXT HERE"]:
                        self.decider.log(f"⚠️ Decider generated placeholder '{msg}' for SPEAK. Aborting speak.")
                        return {"task": "wait", "cycles": 0, "reason": "Failed speak attempt"}
                    else:
                        if self.decider.chat_fn:
                            self.decider.chat_fn("Decider", msg)
                        self.decider.last_action_was_speak = True
                        return {"task": "wait", "cycles": 0, "reason": "Spoke to user"}
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to speak: {e}")
            elif "[CHRONICLE:" in response_upper or "[NOTE_CREATE:" in response_upper:
                try:
                    match = re.search(r"\[(?:CHRONICLE|NOTE_CREATE):(.*?)(?:\])?\s*$", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        args = match.group(1).strip()
                        self.decider.command_executor.create_note(args)
                    # Don't wait; allow chaining decisions (e.g. create memory -> then daydream)
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Created chronicle"}
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to create chronicle: {e}")
            elif "[NOTE_EDIT:" in response_upper:
                try:
                    match = re.search(r"\[NOTE_EDIT:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        args = match.group(1).strip()
                        if "," in args:
                            mid_str, content = args.split(",", 1)
                            self.decider.malkuth.edit_note(int(mid_str.strip()), content.strip())
                    # Don't wait; allow chaining decisions
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Edited note"}
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to edit note: {e}")
            elif "[THINK:" in response_upper:
                try:
                    match = re.search(r"\[THINK:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    content = match.group(1).strip() if match else "General"

                    topic = content
                    strategy = "auto"

                    if "," in content:
                        parts = content.split(",", 1)
                        topic = parts[0].strip()
                        strategy = parts[1].strip()

                    # Clean up common hallucinated brackets
                    topic = topic.replace("<", "").replace(">", "").replace("[", "").replace("]", "")

                    # Filter out placeholders
                    bad_topics = ["TOPIC", "SUBJECT", "CONTENT", "TEXT", "INSERT TOPIC", "SPECIFIC_TOPIC", "ANY", "GENERAL"]
                    if topic.upper() in bad_topics or not topic:
                        self.decider.log(f"⚠️ Decider generated placeholder '{topic}' for THINK. Auto-selecting topic.")
                        # Fallback: Pick an active goal or default
                        active_goals = self.decider.memory_store.get_active_by_type("GOAL")
                        if active_goals:
                            topic = active_goals[0][2] # Use text of first goal
                        else:
                            topic = "The Nature of Artificial Consciousness"

                    self.decider.thought_generator.perform_thinking_chain(topic, strategy=strategy)
                    self.decider.consecutive_daydream_batches = 0
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to start thinking chain: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Thinking chain failed"}
            elif "[REFLECT]" in response_upper:
                try:
                    self.decider.thought_generator.perform_self_reflection()
                    self.decider.consecutive_daydream_batches = 0
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to reflect: {e}")
            elif "[PHILOSOPHIZE]" in response_upper:
                try:
                    if "philosophize" in self.decider.actions:
                        self.decider.actions["philosophize"]()
                    else:
                        self.decider.log("⚠️ Action 'philosophize' not available.")
                    self.decider.consecutive_daydream_batches = 0
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to philosophize: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Philosophize failed"}
            elif "[DEBATE:" in response_upper:
                try:
                    match = re.search(r"\[DEBATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    topic = match.group(1).strip() if match else "General"
                    if self.decider.dialectics:
                        result = self.decider.dialectics.run_debate(topic, reasoning_store=self.decider.reasoning_store)
                        self.decider.reasoning_store.add(content=f"Council Debate ({topic}):\n{result}", source="council_debate", confidence=1.0)
                        self.decider.consecutive_daydream_batches = 0
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to start debate: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Debate failed"}
            elif "[SIMULATE:" in response_upper:
                try:
                    match = re.search(r"\[SIMULATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    premise = match.group(1).strip() if match else "General"
                    if "simulate_counterfactual" in self.decider.actions:
                        result = self.decider.actions["simulate_counterfactual"](premise)
                        self.decider.reasoning_store.add(content=f"Counterfactual Simulation ({premise}):\n{result}", source="daat_simulation", confidence=1.0)
                        self.decider.consecutive_daydream_batches = 0
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to start simulation: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Simulation failed"}
            elif "[EXECUTE:" in response_upper:
                try:
                    match = re.search(r"\[EXECUTE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    content = match.group(1).strip() if match else ""

                    if "," in content:
                        tool, args = content.split(",", 1)
                        self.decider.command_executor._execute_tool(tool.strip().upper(), args.strip())
                    else:
                        self.decider.command_executor._execute_tool(content.strip().upper(), "")
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Executed tool"}
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to execute tool: {e}")
            elif "[GOAL_ACT:" in response_upper:
                try:
                    target_id_str = response_upper.split("[GOAL_ACT:", 1)[1].strip().rstrip("]")
                    target_id = int(target_id_str)

                    # Find goal text
                    all_items = self.decider.memory_store.list_recent(limit=None)
                    goal_text = next((m[3] for m in all_items if m[0] == target_id), None)

                    if goal_text:
                        self.decider.log(f"🤖 Decider acting on Goal {target_id}: {goal_text}")

                        # CRS Allocation
                        goal_item = self.decider.memory_store.get(target_id)
                        priority = goal_item.get('confidence', 0.5) if goal_item else 0.5
                        coherence = self.decider.keter.evaluate().get("keter", 1.0) if self.decider.keter else 1.0

                        # Pass metrics to CRS for active control
                        metrics = self.decider.meta_learner.metrics_history if self.decider.meta_learner else {}
                        model_params = self.decider.meta_learner.latest_model_params if self.decider.meta_learner else None

                        current_state = {
                            "hesed": self.decider.hesed.calculate() if self.decider.hesed else 0.5,
                            "gevurah": self.decider.gevurah.calculate() if self.decider.gevurah else 0.5
                        }

                        violation_pressure = self.decider.value_core.get_violation_pressure() if self.decider.value_core else 0.0

                        # Defensive casting for inputs
                        try:
                            priority = float(priority)
                            coherence = float(coherence)
                        except (ValueError, TypeError):
                            priority = 0.5
                            coherence = 1.0

                        if self.decider.crs:
                            alloc = self.decider.crs.allocate("reasoning", complexity=priority, coherence=coherence, metrics=metrics, model_params=model_params, current_state=current_state, violation_pressure=violation_pressure)
                            depth = alloc["reasoning_depth"]
                            beam = alloc["beam_width"]
                            self.decider.last_predicted_delta = alloc.get("predicted_delta", 0.0)
                        else:
                            # Fallback
                            depth = 5
                            if priority > 0.8: depth = 10
                            beam = 1

                        self.decider.log(f"    Allocating compute: Depth {depth}, Beam {beam} (Priority {priority:.2f})")
                        self.decider.thought_generator.perform_thinking_chain(f"Strategic Plan for Goal: {goal_text}", max_depth=depth, beam_width=beam)
                    else:
                        self.decider.log(f"⚠️ Goal {target_id} not found.")
                        return {"task": "wait", "cycles": 0, "reason": "Goal not found"}

                    self.decider.consecutive_daydream_batches = 0
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to act on goal: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Goal action failed"}
            elif "[GOAL_REMOVE:" in response_upper:
                try:
                    match = re.search(r"\[GOAL_REMOVE:(.*?)\]?$", response, re.IGNORECASE)
                    target = match.group(1).strip() if match else ""

                    if "remove_goal" in self.decider.actions:
                        result = self.decider.actions["remove_goal"](target)
                        self.decider.log(result)
                        if self.decider.chat_fn:
                            self.decider.chat_fn("Decider", result)
                    else:
                        self.decider.log("⚠️ Action remove_goal not available.")

                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Removed goal"}
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to remove goal: {e}")
            elif "[GOAL_CREATE:" in response_upper:
                try:
                    match = re.search(r"\[GOAL_CREATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    content = match.group(1).strip() if match else ""

                    if not content or content.upper() in ["TEXT", "CONTENT", "GOAL"]:
                        self.decider.log("⚠️ Decider tried to create empty GOAL. Injecting default.")
                        content = "Research recent advancements in AI architecture"

                    self.decider.goal_manager.create_goal(content)
                    self.decider.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Created goal"}
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to create goal: {e}")
            elif "[LIST_DOCS]" in response_upper:
                if "list_documents" in self.decider.actions:
                    docs_list = self.decider.actions["list_documents"]()
                    self.decider.reasoning_store.add(content=f"Tool Output [LIST_DOCS]:\n{docs_list}", source="tool_output", confidence=1.0)
                    self.decider.log(f"📚 Documents listed.")
                return {"task": "organizing", "cycles": 0, "reason": "Listed docs"}
            elif "[READ_DOC:" in response_upper:
                try:
                    match = re.search(r"\[READ_DOC:(.*?)\]?$", response, re.IGNORECASE)
                    target = match.group(1).strip() if match else ""

                    if "read_document" in self.decider.actions:
                        content = self.decider.actions["read_document"](target)
                        self.decider.reasoning_store.add(content=f"Tool Output [READ_DOC]:\n{content}", source="tool_output", confidence=1.0)
                        self.decider.log(f"📄 Read document: {target}")
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to read doc: {e}")
                return {"task": "organizing", "cycles": 0, "reason": "Read doc"}
            elif "[SEARCH_MEM:" in response_upper:
                try:
                    match = re.search(r"\[SEARCH_MEM:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    query = match.group(1).strip() if match else ""

                    if "search_memory" in self.decider.actions:
                        results = self.decider.actions["search_memory"](query)
                        self.decider.reasoning_store.add(content=f"Tool Output [SEARCH_MEM]:\n{results}", source="tool_output", confidence=1.0)
                        self.decider.log(f"🔍 Searched memory for: {query}")
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to search memory: {e}")
                return {"task": "organizing", "cycles": 0, "reason": "Searched memory"}
            elif "[SUMMARIZE]" in response_upper:
                if "summarize" in self.decider.actions:
                    result = self.decider.actions["summarize"]()
                    if result:
                        self.decider.log(result)
                return {"task": "organizing", "cycles": 0, "reason": "Summarized"}
            elif "[SEARCH_INTERNET" in response_upper:
                try:
                    match = re.search(r"\[SEARCH_INTERNET:\s*(.*),\s*([A-Z_]+)\]", response, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip()
                        source = match.group(2).strip().upper()
                        self.decider.command_executor._execute_search(query, source)
                    else:
                        # Fallback: Try to find just a query (Default to WEB)
                        match_query = re.search(r"\[SEARCH_INTERNET:\s*(.*?)\]", response, re.IGNORECASE)
                        if match_query:
                             query = match_query.group(1).strip()
                             self.decider.command_executor._execute_search(query, "WEB")
                        else:
                             # No args at all? Try to use active goal
                             goals = self.decider.memory_store.get_active_by_type("GOAL")
                             if goals:
                                 # get_active_by_type returns (id, subject, text, source)
                                 # Sort by ID desc to get most recent
                                 goals.sort(key=lambda x: x[0], reverse=True)
                                 query = goals[0][2]
                                 self.decider.log(f"⚠️ Decider forgot args. Auto-searching top goal: {query}")
                                 self.decider.command_executor._execute_search(query, "WEB")
                             else:
                                 # Fallback 2: Pick a random recent memory topic or "Latest AI News"
                                 # This ensures the tool ALWAYS runs if called, preventing "No Data" frustration
                                 fallback_topic = "Artificial Intelligence News"
                                 self.decider.log(f"⚠️ Decider called SEARCH_INTERNET without args/goals. Fallback search: {fallback_topic}")
                                 self.decider.command_executor._execute_search(fallback_topic, "WEB")
                                 # Also create a goal so we don't loop
                                 self.decider.goal_manager.create_goal(f"Research {fallback_topic}")
                    return {"task": "organizing", "cycles": 0, "reason": "Searched internet"}
                except Exception as e:
                    self.decider.log(f"⚠️ Decider failed to search internet: {e}")
            else:
                # Default fallback
                # FIX: Only default to daydream if allowed and not cooling down
                if allow_daydream and not just_finished_heavy and not self.decider.forced_stop_cooldown:
                    self.decider.command_executor._run_action("start_daydream")
                    # Note: _run_action here is immediate, but we also set state for loop
                    self.decider.consecutive_daydream_batches += 1
                    return {"task": "daydream", "cycles": 0, "reason": "Fallback daydream"}
                else:
                    self.decider.log(f"🤖 Decider: Fallback triggered but Daydream is disabled/cooldown. Defaulting to WAIT.")
                    return {"task": "wait", "cycles": 0, "reason": "Fallback wait"}

        except Exception as e:
            self.decider.log(f"❌ Critical Planning Error: {e}")
            return {"task": "wait", "cycles": 0, "reason": f"Error: {e}"}
        finally:
            self.decider.planning_lock.release()

        return {"task": "wait", "cycles": 0, "reason": "End of decision"}

    def _measure_strange_loop(self) -> float:
        """
        Calculates the 'Friction' between Ideal Self and Current Reality.
        This is the 'Vibration' of consciousness.
        """
        # 1. Structural Integrity (Keter) - Ideal is 1.0
        keter_score = self.decider.keter.evaluate().get("keter", 1.0) if self.decider.keter else 1.0
        structural_friction = 1.0 - keter_score

        # 2. Ethical Alignment (ValueCore) - Ideal is 0.0 violation
        violation = self.decider.value_core.get_violation_pressure() if self.decider.value_core else 0.0

        # 3. Cognitive Dissonance (Netzach) - Ideal is no dissonance
        dissonance = 0.0
        if self.decider.latest_netzach_signal and self.decider.latest_netzach_signal.get("signal") == "DISSONANCE":
            dissonance = 0.5

        # Total Friction (0.0 to 1.0)
        # Weighted sum representing the "distance" from the Ideal Self
        friction = (0.4 * structural_friction) + (0.4 * violation) + (0.2 * dissonance)
        return max(0.0, min(1.0, friction))

    def _detect_deadlock(self, command: str) -> bool:
        """
        Detects loops in decision making.
        e.g. [REFLECT] -> [THINK] -> [REFLECT] -> [THINK]
        """
        if not command: return False

        # Normalize command (remove args for loop detection)
        base_cmd = command.split(":")[0].strip("[] ")

        self.decider.command_history.append(base_cmd)
        if len(self.decider.command_history) > 20:
            self.decider.command_history.pop(0)

        if len(self.decider.command_history) < 4:
            return False

        # Check for simple repetition (A-A-A) if not WAIT
        if self.decider.command_history[-1] != "WAIT" and (self.decider.command_history[-1] == self.decider.command_history[-2] == self.decider.command_history[-3]):
            return True

        topic_match = re.search(r"\[(?:THINK|DAYDREAM|DEBATE|SIMULATE):.*?,.*?,?(.*?)\]", command, re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).strip().lower()
            if topic:
                self.decider.topic_history.append(topic)
                if len(self.decider.topic_history) > 6: self.decider.topic_history.pop(0)
                if self.decider.topic_history.count(topic) >= 4:
                    self.decider.log(f"🛑 Topic Stagnation detected: '{topic}'. Forcing cognitive shift.")
                    return True

        # Check for Thought Loop (heuristic: if we keep planning but not acting)
        if self.decider.command_history.count("THINK") > 3 and len(self.decider.command_history) < 8:
             return True

        # Check for Oscillation (A-B-A-B)
        return (self.decider.command_history[-1] == self.decider.command_history[-3] and self.decider.command_history[-2] == self.decider.command_history[-4])

    def _is_repetitive_thought(self, thought: str) -> bool:
        """Check if the thought is semantically identical to recent thoughts."""
        if not thought: return False

        # Normalize
        norm_thought = re.sub(r'\s+', ' ', thought.lower().strip())

        # Check last 5 thoughts
        for past in self.decider.thought_history[-5:]:
            if norm_thought == past:
                return True
        return False

    def calculate_utility(self) -> float:
        """
        Formal Scoring Function: U(state)
        U(state) = α * coherence + β * verified_facts + γ * goal_completion + δ * novelty + ε * external
        Used for principled long-horizon optimization.

        FIX 4: Utility Dominated by ΔCoherence.
        """
        # 1. Coherence Delta (Primary Driver)
        keter_stats = self.decider.keter.evaluate() if self.decider.keter else {}
        delta_coherence = keter_stats.get("delta", 0.0)

        # Scale delta to be impactful (e.g. -0.01 -> -0.1 utility impact)
        coherence_impact = delta_coherence * 10.0

        # 2. Verified Facts (Beta) & 3. Goal Completion (Gamma) & 4. Novelty (Delta)
        with self.decider.memory_store._connect() as con:
            # Facts stats
            total_facts = con.execute("SELECT COUNT(*) FROM memories WHERE type IN ('FACT', 'BELIEF') AND deleted=0").fetchone()[0]
            verified_facts = con.execute("SELECT COUNT(*) FROM memories WHERE type IN ('FACT', 'BELIEF') AND verified=1 AND deleted=0").fetchone()[0]

            # Goal stats
            completed_goals = con.execute("SELECT COUNT(*) FROM memories WHERE type='COMPLETED_GOAL'").fetchone()[0]
            total_goals = con.execute("SELECT COUNT(*) FROM memories WHERE type IN ('GOAL', 'COMPLETED_GOAL', 'ARCHIVED_GOAL')").fetchone()[0]

            # Novelty: Items created in last hour
            cutoff = int(time.time()) - 3600
            new_items = con.execute("SELECT COUNT(*) FROM memories WHERE created_at > ?", (cutoff,)).fetchone()[0]

            # Future Potential (Open Gaps + Unverified Beliefs)
            # This rewards creating "potential energy" for future resolution
            open_gaps = con.execute("SELECT COUNT(*) FROM memories WHERE type='CURIOSITY_GAP' AND completed=0").fetchone()[0]
            unverified_beliefs = con.execute("SELECT COUNT(*) FROM memories WHERE type='BELIEF' AND verified=0").fetchone()[0]

        verified_ratio = verified_facts / max(1, total_facts)
        completion_rate = completed_goals / max(1, total_goals)
        novelty_score = min(new_items / 20.0, 1.0) # Cap at 20 items/hr
        potential_score = min((open_gaps + unverified_beliefs) / 10.0, 1.0) # Cap at 10 items
        external_reward = 0.5 # Placeholder for future user feedback

        # 5. Prediction Accuracy (Zeta)
        # prediction_error is 0.0 (Perfect) to 1.0 (Wrong). Accuracy = 1.0 - error.
        avg_error = self.decider.meta_memory_store.get_average_prediction_error(limit=20) if self.decider.meta_memory_store else None
        if avg_error is not None:
            accuracy_score = 1.0 - avg_error
        else:
            accuracy_score = 0.5 # Neutral

        # 6. Identity Alignment (Self-Model)
        # Penalize if violation pressure is high
        violation_pressure = self.decider.value_core.get_violation_pressure() if self.decider.value_core else 0.0
        alignment_score = 1.0 - violation_pressure

        # Redesigned Utility:
        # 0.4 * ΔCoherence + 0.1 * Verified + 0.1 * Goals + 0.1 * Potential + 0.1 * Accuracy + 0.2 * Alignment
        # Base utility 0.5 to allow negative swings from coherence drop
        base_utility = 0.5
        utility = base_utility + (0.4 * coherence_impact) + (0.1 * verified_ratio) + (0.1 * completion_rate) + \
                  (0.1 * potential_score) + (0.1 * accuracy_score) + (0.2 * alignment_score)

        # Clamp 0-1
        utility = max(0.0, min(1.0, utility))

        # Log only if significant change or periodic
        if abs(utility - self.decider.last_utility_score) > 0.05 or (time.time() - self.decider.last_utility_log_time > 60):
            self.decider.log(f"📈 System Utility: {utility:.4f} (ΔCoh={delta_coherence:+.4f}, Ver={verified_ratio:.2f}, Goal={completion_rate:.2f}, Pot={potential_score:.2f})")
            self.decider.last_utility_log_time = time.time()
            self.decider.last_utility_score = utility
        return utility

    def _meta_cognitive_check(self, thought: str, command: str) -> bool:
        """
        Recursive Self-Monitoring.
        Checks if the decision aligns with the system's coherence state (Keter).
        """
        if not self.decider.keter: return True

        # Fast path for REFLECT/WAIT (Always allowed if chosen)
        if "[REFLECT]" in command or "[WAIT]" in command:
            return True

        keter_stats = self.decider.keter.evaluate()
        coherence = keter_stats.get("keter", 0.5)

        prompt = (
            f"SYSTEM STATE (KETER COHERENCE): {coherence:.2f} (0.0=Chaos, 1.0=Order)\n"
            f"PLANNED ACTION: {command}\n"
            f"REASONING: {thought}\n\n"
            "TASK: Meta-Cognitive Alignment Check.\n"
            "Does this action align with the current coherence state?\n"
            "- Low Coherence (<0.4) requires [REFLECT], [VERIFY], or [PRUNE]. Expansion/Daydreaming is risky.\n"
            "- High Coherence (>0.7) allows [THINK], [DAYDREAM], [GOAL_ACT].\n"
            "Is the reasoning sound and consistent with the state?\n\n"
            "Output JSON: {\"aligned\": true, \"reason\": \"...\"} or {\"aligned\": false, \"reason\": \"...\"}"
        )

        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are the Meta-Cognitive Monitor.",
                temperature=0.1,
                max_tokens=100,
                base_url=self.decider.get_settings().get("base_url"),
                chat_model=self.decider.get_settings().get("chat_model")
            )

            # Clean response
            clean_resp = response.strip()
            if clean_resp.startswith("```"):
                clean_resp = clean_resp.split("```")[1].replace("json", "").strip()

            try:
                data = json.loads(clean_resp)
            except json.JSONDecodeError:
                # Fallback: check for "true" or "false" in text
                if "false" in clean_resp.lower():
                    data = {"aligned": False, "reason": "Parsing failed but 'false' detected."}
                else:
                    data = {"aligned": True}

            if not data.get("aligned", True):
                self.decider.log(f"🧠 Meta-Cognition Rejection: {data.get('reason')}")
                return False
            return True
        except Exception as e:
            self.decider.log(f"⚠️ Meta-Cognitive check failed: {e}")
            return True # Fail open to avoid paralysis

    def _get_recent_chat_memories(self, limit: int = 20):
        """Retrieve recent memories that are NOT from daydreaming."""
        items = self.decider.memory_store.list_recent(limit=100)
        chat_mems = []
        for item in items:
            # item: (id, type, subject, text, source, verified)
            if len(item) > 4 and item[4] != 'daydream':
                chat_mems.append(item)
                if len(chat_mems) >= limit:
                    break
        return chat_mems

    def _extract_command(self, response: str) -> Optional[str]:
        """Extract the final command from a verbose response."""
        # 1. Look for explicit "Command: [ACT]" pattern
        match = re.search(r"Command:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # 2. Fallback: Look for the LAST valid command tag in the text
        valid_prefixes = [
            "WAIT", "DAYDREAM", "VERIFY", "SPEAK", "CHRONICLE", "NOTE_CREATE", "NOTE_EDIT",
            "THINK", "REFLECT", "PHILOSOPHIZE", "DEBATE", "SIMULATE", "EXECUTE",
            "GOAL_ACT", "GOAL_REMOVE", "GOAL_CREATE", "LIST_DOCS", "READ_DOC",
            "SEARCH_MEM", "SUMMARIZE", "WRITE_FILE", "SEARCH_INTERNET"
        ]

        matches = list(re.finditer(r"\[([A-Z_]+)(?::.*?)?\]", response, re.DOTALL))
        if matches:
            for m in reversed(matches):
                if m.group(1).upper() in valid_prefixes:
                    return m.group(0)
        return None

    def _reflect_on_decision(self, context: str, decision: str):
        """Metacognition: Critique a recent decision/action."""
        settings = self.decider.get_settings()
        prompt = (
            f"CONTEXT: {context}\n"
            f"DECISION/ACTION: {decision}\n\n"
            "TASK: Perform a brief metacognitive critique of this action.\n"
            "Was it the best choice given the context? Why or why not?\n"
            "Identify one potential improvement for next time.\n"
            "Output a concise critique (1-2 sentences)."
        )

        critique = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a metacognitive observer.",
            temperature=0.3,
            max_tokens=150,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )

        if critique and not critique.startswith("⚠️"):
            self.decider.reasoning_store.add(
                content=f"Decision Critique: {critique}",
                source="decider_reflection",
                confidence=1.0
            )
            self.decider.log(f"🧠 Metacognition: {critique}")
