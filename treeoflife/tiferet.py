import re
import ast
import time
import random
import json
import traceback
import platform
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List
from ai_core.lm import compute_embedding, run_local_lm, extract_memory_candidates, count_tokens, LLMError, DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT
from ai_core.utils import parse_json_array_loose

try:
    import psutil
except ImportError:
    psutil = None

class Decider:
    """
    Autonomous decision module.
    Controls system parameters (Temperature) and operational modes (Daydream, Verification).
    
    LOCK HIERARCHY: Level 3 (Logic/Flow) - planning_lock, maintenance_lock
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
        log_fn: Callable[[str], None] = logging.info,
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
        daat=None,
        malkuth=None,
        yesod=None,
        temp_step: float = 0.20,
        token_step: float = 0.20,
        crs=None,
        epigenetics: Dict = None,
        value_core=None,
        stability_controller=None,
        heartbeat=None,
        global_workspace=None,
        executor=None
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
        self.daat = daat
        self.malkuth = malkuth
        self.yesod = yesod
        self.crs = crs
        self.value_core = value_core
        self.stability_controller = stability_controller
        self.epigenetics = epigenetics or {}
        self.meta_learner = None # Set via property later
        self.temp_step = temp_step
        self.token_step = token_step
        self.heartbeat = heartbeat
        self.global_workspace = global_workspace
        self.executor = executor
        self.hod_just_ran = False
        self.action_taken_in_observation = False
        self.last_action_was_speak = False
        self.forced_stop_cooldown = False
        self.daydream_mode = "auto" # auto, read, insight
        self.daydream_topic = None
        self.last_tool_usage = 0
        self.consecutive_daydream_batches = 0
        self.last_daydream_time = 0
        self.last_token_adjustment_time = 0
        self.planning_lock = threading.Lock()
        self.maintenance_lock = threading.Lock()
        self.pending_maintenance = [] # List of actions recommended by Hod
        self.latest_netzach_signal = None
        self.last_goal_management_time = 0
        self.command_history = [] # For deadlock detection
        self.last_predicted_delta = 0.0 # Track prediction for error calculation
        self.interrupted = False # Flag for interrupt-priority architecture
        self.panic_mode = False
        self.is_sleeping = False
        self.topic_history = []
        self.stream_of_consciousness = [] # Injected by AIController
        self.last_internal_thought = None
        self.thought_history = [] # Track recent thoughts to prevent loops
        self.last_simulation_time = 0
        self.last_utility_log_time = 0
        self.last_utility_score = 0.0
        
        # Critical Memory Cache
        self._critical_memories_cache = []
        self._last_critical_update = 0
        
        # Capture baselines for relative limits
        settings = self.get_settings()
        self.baseline_temp = float(settings.get("default_temperature", 0.7))
        self.baseline_tokens = int(settings.get("default_max_tokens", 800))

        # Affective State (Emotional Bias)
        self.mood = float(settings.get("current_mood", 0.5)) # 0.0 (Depressed) to 1.0 (Manic)

        # Subscribe to Conscious Content
        if self.event_bus:
            self.event_bus.subscribe("CONSCIOUS_CONTENT", self._on_conscious_content, priority=10)

    def increase_temperature(self, amount: float = None):
        """Increase temperature by specified amount (percentage) up to 20%."""
        limit_pct = self.temp_step
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_temperature(1.0 + step)

    def decrease_temperature(self, amount: float = None):
        """Decrease temperature by specified amount (percentage) up to 20%."""
        limit_pct = self.temp_step
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_temperature(1.0 - step)

    def _adjust_temperature(self, multiplier: float):
        settings = self.get_settings()
        current = float(settings.get("temperature", 0.7))
        new_temp = round(current * multiplier, 2)
        
        limit_pct = self.temp_step
        
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

    def _update_mood(self, delta: float):
        """Update affective state based on success/failure."""
        # Dampened update
        self.mood = max(0.0, min(1.0, self.mood + (delta * 0.2)))
        
        # Persist
        settings = self.get_settings()
        settings["current_mood"] = self.mood
        self.update_settings(settings)
        self.log(f"🧠 Decider Mood: {self.mood:.2f} (Δ={delta:+.2f})")

    def increase_tokens(self, amount: float = None):
        """Increase max_tokens by specified amount (percentage) up to 20%."""
        limit_pct = self.token_step
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_tokens(1.0 + step)

    def decrease_tokens(self, amount: float = None):
        """Decrease max_tokens by specified amount (percentage) up to 20%."""
        limit_pct = self.token_step
        if amount is None:
            amount = limit_pct
        step = max(0.01, min(float(amount), limit_pct))
        self._adjust_tokens(1.0 - step)

    def _adjust_tokens(self, multiplier: float):
        settings = self.get_settings()
        current = int(settings.get("max_tokens", 800))
        new_tokens = int(current * multiplier)
        
        limit_pct = self.token_step
        
        # Calculate bounds based on BASELINE
        lower_bound = int(self.baseline_tokens * (1.0 - limit_pct))
        upper_bound = int(self.baseline_tokens * (1.0 + limit_pct))
        
        new_tokens = max(lower_bound, min(new_tokens, upper_bound))
        # Absolute safety floor
        new_tokens = int(max(1024, new_tokens))
        
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
            
    def set_meta_learner(self, meta_learner):
        """Dependency injection for Meta-Learner."""
        self.meta_learner = meta_learner

    def start_daydream(self):
        self.log("🤖 Decider starting single Daydream cycle.")
        if self.heartbeat:
            self.heartbeat.force_task("daydream", 1, "Manual Command")
        if time.time() - self.last_daydream_time < 60:
            self.daydream_mode = "insight"
        else:
            self.daydream_mode = "auto"

    def start_verification_batch(self):
        self.log("🤖 Decider starting Verification Batch.")
        if self.heartbeat:
            self.heartbeat.force_task("verify", 1, "Manual Command")

    def verify_all(self):
        self.log("🤖 Decider starting Full Verification.")
        if self.heartbeat:
            self.heartbeat.force_task("verify_all", 1, "Manual Command")

    def start_daydream_loop(self):
        self.log("🤖 Decider enabling Daydream Loop.")
        self._run_action("start_loop", reason="User command")
        settings = self.get_settings()
        if self.heartbeat:
            self.heartbeat.force_task("daydream", int(settings.get("daydream_cycle_limit", 10)), "User Loop Command")
        self.last_daydream_time = time.time()

    def stop_daydream(self):
        self.log("🤖 Decider stopping daydream.")
        self._run_action("stop_daydream", reason="User command or Stop signal")

    def report_forced_stop(self):
        """Handle forced stop from UI."""
        self.log("🤖 Decider: Forced stop received. Entering cooldown.")
        if self.heartbeat:
            self.heartbeat.stop()
        self.forced_stop_cooldown = True

    def wake_up(self, reason: str = "External Stimulus"):
        """Force the Decider to wake up from a wait state."""
        self.is_sleeping = False
        self.log(f"🤖 Decider: Waking up due to {reason}.")
        if self.heartbeat:
            self.heartbeat.force_task("organizing", 0, reason)

    def create_note(self, content: str):
        """Manually create an Assistant Note (NOTE)."""
        if self.malkuth:
            return self.malkuth.create_note(content)
        self.log("⚠️ Malkuth not available for creating note.")

    def _on_reminder_due(self, event):
        """Handles REMINDER_DUE event from Netzach."""
        reminder_text = event.data.get("text")
        if reminder_text:
            self.log(f"⏰ Decider received reminder: {reminder_text}")
            if self.chat_fn:
                self.chat_fn("Assistant", f"⏰ Reminder: {reminder_text}")

    def on_panic(self, event):
        """Handle SYSTEM:PANIC event from Keter."""
        self.log("🚨 Decider: PANIC PROTOCOL INITIATED. Halting all non-essential tasks.")
        self.panic_mode = True
        self.current_task = "wait"
        self.cycles_remaining = 0
        if self.heartbeat:
            self.heartbeat.stop()
        # Force a full reset/reboot thought
        self.reasoning_store.add(content="SYSTEM PANIC: I must stop and reorganize. My thoughts are incoherent.", source="keter_panic", confidence=1.0)

    def _on_conscious_content(self, event):
        """Handle high-salience events from Global Workspace."""
        data = event.data
        salience = data.get("final_salience", data.get("salience", 0.0))
        content = data.get("content", "")
        c_type = data.get("type", "")
        
        if salience > 0.9 and c_type != "DRIVE":
            self.log(f"🔦 Decider: High Salience Event ({salience:.2f}): {content}. Interrupting.")
            # If we are daydreaming, stop and attend
            if self.heartbeat and self.heartbeat.current_task == "daydream":
                self.stop_daydream()
                self.wake_up(f"High Salience: {content}")

    def create_goal(self, content: str):
        """Autonomously create a new GOAL memory."""
        # DEDUPLICATION: Don't create the same goal twice
        existing = self.memory_store.get_active_by_type("GOAL")
        if any(content.lower() in g[2].lower() for g in existing):
            self.log(f"🎯 Decider: Goal already exists, skipping: {content}")
            return None

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
        
    def _track_metric(self, metric: str, value: float):
        if self.meta_learner:
            self.meta_learner.track(metric, value)

    def run_post_chat_decision_cycle(self):
        """Initiates the decision process after a chat interaction is complete."""
        # This is called after a chat reply has been sent.
        # If a natural language command already set up a task, don't overwrite it.
        if self.heartbeat and self.heartbeat.cycles_remaining > 0 and self.heartbeat.current_task != "wait":
            self.log(f"🤖 Decider: Chat complete. Resuming assigned task ({self.heartbeat.current_task}).")
            return

        # The decider should now figure out what to do next.
        # We can't call decide() directly because it returns a dict, we need to apply it to heartbeat
        if self.heartbeat:
            self.log("🤖 Decider: Chat complete. Initiating post-chat decision cycle.")
            decision = self.decide()
            self.heartbeat.force_task(decision["task"], decision["cycles"], decision["reason"])

    def calculate_utility(self) -> float:
        """
        Formal Scoring Function: U(state)
        U(state) = α * coherence + β * verified_facts + γ * goal_completion + δ * novelty + ε * external
        Used for principled long-horizon optimization.
        
        FIX 4: Utility Dominated by ΔCoherence.
        """
        # 1. Coherence Delta (Primary Driver)
        keter_stats = self.keter.evaluate() if self.keter else {}
        delta_coherence = keter_stats.get("delta", 0.0)
        
        # Scale delta to be impactful (e.g. -0.01 -> -0.1 utility impact)
        coherence_impact = delta_coherence * 10.0
        
        # 2. Verified Facts (Beta) & 3. Goal Completion (Gamma) & 4. Novelty (Delta)
        with self.memory_store._connect() as con:
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
        avg_error = self.meta_memory_store.get_average_prediction_error(limit=20) if self.meta_memory_store else None
        if avg_error is not None:
            accuracy_score = 1.0 - avg_error
        else:
            accuracy_score = 0.5 # Neutral
        
        # 6. Identity Alignment (Self-Model)
        # Penalize if violation pressure is high
        violation_pressure = self.value_core.get_violation_pressure() if self.value_core else 0.0
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
        if abs(utility - self.last_utility_score) > 0.05 or (time.time() - self.last_utility_log_time > 60):
            self.log(f"📈 System Utility: {utility:.4f} (ΔCoh={delta_coherence:+.4f}, Ver={verified_ratio:.2f}, Goal={completion_rate:.2f}, Pot={potential_score:.2f})")
            self.last_utility_log_time = time.time()
            self.last_utility_score = utility
        return utility

    def _measure_strange_loop(self) -> float:
        """
        Calculates the 'Friction' between Ideal Self and Current Reality.
        This is the 'Vibration' of consciousness.
        """
        # 1. Structural Integrity (Keter) - Ideal is 1.0
        keter_score = self.keter.evaluate().get("keter", 1.0) if self.keter else 1.0
        structural_friction = 1.0 - keter_score

        # 2. Ethical Alignment (ValueCore) - Ideal is 0.0 violation
        violation = self.value_core.get_violation_pressure() if self.value_core else 0.0
        
        # 3. Cognitive Dissonance (Netzach) - Ideal is no dissonance
        dissonance = 0.0
        if self.latest_netzach_signal and self.latest_netzach_signal.get("signal") == "DISSONANCE":
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
        
        self.command_history.append(base_cmd)
        if len(self.command_history) > 20:
            self.command_history.pop(0)
            
        if len(self.command_history) < 4:
            return False
            
        # Check for simple repetition (A-A-A) if not WAIT
        if self.command_history[-1] != "WAIT" and (self.command_history[-1] == self.command_history[-2] == self.command_history[-3]):
            return True

        topic_match = re.search(r"\[(?:THINK|DAYDREAM|DEBATE|SIMULATE):.*?,.*?,?(.*?)\]", command, re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).strip().lower()
            if topic:
                self.topic_history.append(topic)
                if len(self.topic_history) > 6: self.topic_history.pop(0)
                if self.topic_history.count(topic) >= 4:
                    self.log(f"🛑 Topic Stagnation detected: '{topic}'. Forcing cognitive shift.")
                    return True

        # Check for Thought Loop (heuristic: if we keep planning but not acting)
        if self.command_history.count("THINK") > 3 and len(self.command_history) < 8:
             return True

        # Check for Oscillation (A-B-A-B)
        return (self.command_history[-1] == self.command_history[-3] and self.command_history[-2] == self.command_history[-4])

    def _is_repetitive_thought(self, thought: str) -> bool:
        """Check if the thought is semantically identical to recent thoughts."""
        if not thought: return False
        
        # Normalize
        norm_thought = re.sub(r'\s+', ' ', thought.lower().strip())
        
        # Check last 5 thoughts
        for past in self.thought_history[-5:]:
            if norm_thought == past:
                return True
        return False

    def execute_task(self, task_name: str):
        """Execute one unit of the assigned task."""
        if task_name == "daydream":
            # Check concurrency setting
            settings = self.get_settings()
            concurrency = int(settings.get("concurrency", 1))
            
            # Determine batch size (don't exceed remaining cycles or concurrency limit)
            # Note: Heartbeat handles decrementing cycles, so we just need to know if we CAN batch
            # But Heartbeat calls this once per tick. If we batch, we need to tell Heartbeat we did more work.
            # For simplicity, let's stick to single execution per tick for now, or handle batching internally.
            # If we want batching, we should probably do it here and return how many were done.
            # But Heartbeat expects 1 tick = 1 cycle decrement.
            # Let's just do single for now to keep Heartbeat simple.
            self._run_action("start_daydream", reason=self.heartbeat.task_reason if self.heartbeat else "Daydream")
            self.last_daydream_time = time.time()
            
        elif task_name == "verify":
            self._run_action("verify_batch", reason=self.heartbeat.task_reason if self.heartbeat else "Verify")
            
        elif task_name == "verify_all":
            self._run_action("verify_all", reason=self.heartbeat.task_reason if self.heartbeat else "Verify All")
            
        elif task_name == "sleep":
            self.log("💤 Decider: Sleeping... (Deep Consolidation & Synthesis)")
            # Run heavy maintenance tasks that are usually too expensive
            if self.binah:
                self.log("💤 Binah: Running deep consolidation...")
                self.binah.consolidate(time_window_hours=None)
            if self.daat:
                self.log("💤 Da'at: Running clustering and synthesis...")
                self.daat.run_clustering()
                self.daat.run_synthesis()

    def start_sleep_cycle(self):
        """Initiate Sleep Mode (Input Off, Deep Processing On)."""
        self.log("💤 Decider initiating Sleep Cycle.")
        self.is_sleeping = True
        if self.heartbeat:
            self.heartbeat.force_task("sleep", 5, "System Sleep Mode") # 5 cycles of deep work

    def authorize_maintenance(self):
        """Public wrapper for maintenance authorization."""
        self._authorize_maintenance()

    def _meta_cognitive_check(self, thought: str, command: str) -> bool:
        """
        Recursive Self-Monitoring.
        Checks if the decision aligns with the system's coherence state (Keter).
        """
        if not self.keter: return True
        
        # Fast path for REFLECT/WAIT (Always allowed if chosen)
        if "[REFLECT]" in command or "[WAIT]" in command:
            return True

        keter_stats = self.keter.evaluate()
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
                base_url=self.get_settings().get("base_url"),
                chat_model=self.get_settings().get("chat_model")
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
                self.log(f"🧠 Meta-Cognition Rejection: {data.get('reason')}")
                return False
            return True
        except Exception as e:
            self.log(f"⚠️ Meta-Cognitive check failed: {e}")
            return True # Fail open to avoid paralysis

    def decide(self) -> Dict[str, Any]:
        """Decide what to do next. Returns a plan dict."""
        # Prevent concurrent planning (e.g. from Chat trigger AND Background loop)
        if not self.planning_lock.acquire(blocking=False):
            self.log("🤖 Decider: Planning already in progress. Skipping.")
            return {"task": "wait", "cycles": 0, "reason": "Planning locked"}
            
        if self.panic_mode:
            self.log("🤖 Decider: In PANIC mode. Waiting for manual intervention or reboot.")
            return {"task": "wait", "cycles": 0, "reason": "System Panic"}

        # Gather Structural Metrics & Determine Mode
        stats = self.memory_store.get_memory_stats()
        with self.memory_store._connect() as con:
            spawn_count = con.execute("SELECT COUNT(*) FROM memories WHERE type='GOAL' AND created_at > ?", (int(time.time()) - 3600,)).fetchone()[0]
            gaps_count = con.execute("SELECT COUNT(*) FROM memories WHERE type='CURIOSITY_GAP' AND completed=0").fetchone()[0]
            
        structural_metrics = {
            "active_goals": stats.get('active_goals', 0),
            "unresolved_conflicts": gaps_count,
            "goal_spawn_rate": spawn_count
        }
        
        # Get System Mode from CRS
        coherence = self.keter.evaluate().get("keter", 1.0) if self.keter else 1.0
        
        # Get Stability State
        stability_state = {}
        if self.stability_controller:
            stability_state = self.stability_controller.evaluate()

        # Pass metrics to CRS
        crs_status = {}
        try:
            if self.crs:
                # Use a dummy task type for system check
                crs_status = self.crs.allocate("system_check", coherence=coherence, structural_metrics=structural_metrics, violation_pressure=self.value_core.get_violation_pressure() if self.value_core else 0.0)
        except Exception as e:
            self.log(f"⚠️ Decider: CRS allocation failed during system check: {e}")
            crs_status = {}
        
        system_mode = crs_status.get("system_mode", "EXECUTION")
        allow_goal_creation = crs_status.get("allow_goal_creation", True)

        # 0. Goal Autonomy Cycle (Generate/Rank/Prune)
        # Run every 10 minutes or if we have no active goals
        if time.time() - self.last_goal_management_time > 600 or not self.memory_store.get_active_by_type("GOAL"):
            self.manage_goals(allow_creation=allow_goal_creation, system_mode=system_mode)

        try:
            self.log("🤖 Decider: Planning next batch...")
            settings = self.get_settings()

            # Fetch Active Goals early (Fix for UnboundLocalError in Reflex path)
            all_items = self.memory_store.list_recent(limit=None)
            active_goals = [m for m in all_items if len(m) > 1 and m[1] == 'GOAL']
            
            # Tabula Rasa Check (Empty Mind)
            is_tabula_rasa = (len(active_goals) == 0 and len(all_items) < 5)
            
            # Cost Evaluation & ROI Ranking (CRS)
            if self.crs:
                # prioritize_goals returns (goal, roi, cost)
                # active_goals list items are tuples from list_recent: (id, type, subject, text, ...)
                # We need to adapt the tuple format for CRS or just pass it if CRS handles it.
                # CRS expects (id, subject, text, source, confidence)
                # list_recent gives (id, type, subject, text, source, verified, flags, confidence)
                mapped_goals = [(g[0], g[2], g[3], g[4], g[7] if len(g)>7 else 0.5) for g in active_goals]
                ranked_goals_info = self.crs.prioritize_goals(mapped_goals, 10000) # 10000 is dummy budget
                # Re-sort active_goals based on ranked_goals_info order
                ranked_ids = [r[0][0] for r in ranked_goals_info]
                active_goals.sort(key=lambda x: ranked_ids.index(x[0]) if x[0] in ranked_ids else 999)
            else:
                # Fallback sort
                try:
                    active_goals.sort(key=lambda x: (float(x[7]) if len(x) > 7 and x[7] is not None else 0.5, x[0]), reverse=True)
                except Exception as e:
                    self.log(f"⚠️ Error sorting goals: {e}")
                    # Fallback to ID sort
                    active_goals.sort(key=lambda x: x[0], reverse=True)

            active_goals = active_goals[:5]
            
            # Check for Necessity Goals (Self-Generated Objectives)
            necessity_goals = [g for g in active_goals if g[4] == "model_stress"]
            if necessity_goals:
                self.log(f"🚨 Decider detected {len(necessity_goals)} Necessity Goals (Model Stress). Prioritizing.")

            # Rate limit Daydreaming: Force variety if we've daydreamed recently
            allow_daydream = True
            if self.consecutive_daydream_batches >= 3:
                allow_daydream = False
                self.log("🤖 Decider: Daydreaming quota reached (3 batches). Forcing other options.")

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
            
            # Check Motivation
            motivation = 1.0
            if self.netzach_force and hasattr(self.netzach_force, 'motivation'):
                motivation = self.netzach_force.motivation

            self.log(f"⚖️ Forces: Hod='{hod_signal}', Netzach='{netzach_signal}'")

            # Calculate Utility for Prompt
            current_utility = self.calculate_utility()

            # Calculate Strange Loop Friction
            current_friction = self._measure_strange_loop()
            self.log(f"🌀 Strange Loop Friction: {current_friction:.4f}")

            # --- FAST PATH (Reflex) ---
            # Optimization: Skip LLM for obvious states to reduce latency
            reflex_action = None
            reflex_reason = ""

            # Gevurah Override (Loop Breaking)
            if self.gevurah and self.gevurah.recommendation == "DROP_GOAL":
                if active_goals:
                    target_goal_id = active_goals[0][0] # Drop top goal
                    self.log(f"🔥 Gevurah Override: Dropping Goal {target_goal_id} due to Action Loop.")
                    if "remove_goal" in self.actions:
                        self.actions["remove_goal"](target_goal_id)
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
            elif self.keter and self.keter.evaluate().get("keter", 1.0) < 0.2:
                # Coherence Recovery Protocol
                reflex_action = "VERIFY" 
                reflex_reason = "Critical Coherence (< 0.2). Initiating Grounding Protocol."
                self.cycles_remaining = 1
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
                self.log(f"⚡ Decider Reflex Triggered: {reflex_action} due to {reflex_reason}")
                if reflex_action == "VERIFY":
                    return {"task": "verify", "cycles": 2, "reason": reflex_reason}
                elif reflex_action == "DAYDREAM":
                    self.daydream_mode = "auto"
                    return {"task": "daydream", "cycles": 1, "reason": reflex_reason}
                elif reflex_action == "GOAL_ACT":
                    self.log(f"⚡ Decider Reflex: Acting on Necessity Goal {target_goal_id}")
                    self.perform_thinking_chain(f"Resolve Necessity Goal: {target_goal_text}")
                    # FIX: Mark goal as completed to prevent infinite loop
                    if "remove_goal" in self.actions:
                        self.actions["remove_goal"](target_goal_id)
                    self.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": reflex_reason}

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
            
            if self.stream_of_consciousness:
                # Use a local copy to avoid race conditions during iteration (though it's replaced atomically)
                context += "STREAM OF CONSCIOUSNESS (Recent Thoughts):\n" + "\n".join([f"- {t}" for t in list(self.stream_of_consciousness)]) + "\n\n"

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
            if self.mood < 0.3:
                context += "EMOTIONAL STATE: Depressed/Fatigued. You feel discouraged. Prioritize stability, safety, and low-risk consolidation. Avoid new goals.\n"
            elif self.mood > 0.8:
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
                context += "Available Documents:\n" + "\n".join([f"- {d[1]}" for d in recent_docs]) + "\n"
            
            current_task = self.heartbeat.current_task if self.heartbeat else "unknown"
            context += f"Current Task: {current_task}\n"
            
            if current_task == "wait" and self.heartbeat and self.heartbeat.wait_start_time > 0:
                wait_duration = time.time() - self.heartbeat.wait_start_time
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
            
            if not self.forced_stop_cooldown and not just_finished_heavy:
                options_text += "\n--- COGNITIVE ACTIONS ---\n"
                if allow_daydream and is_allowed("daydream"):
                    options_text += "2. [DAYDREAM: N, MODE, TOPIC]: Run N cycles (1-5). MODE: 'READ', 'AUTO'. TOPIC: Optional subject. E.g., [DAYDREAM: 3, READ, Neurology]\n"
                    if active_goals:
                        options_text += "   (TIP: Use the TOPIC argument to focus on an active goal!)\n"
                if is_allowed("think"): options_text += "3. [THINK: <TOPIC>]: Start a Tree of Thoughts (ToT) to analyze a specific topic deeply. Use for complex reasoning.\n"
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
                if is_allowed("list_docs"): options_text += "16. [LIST_DOCS]: List available documents to choose a topic for daydreaming.\n"
                if is_allowed("read_doc"): options_text += "17. [READ_DOC: <FILENAME_OR_ID>]: Read the content of a specific document found via LIST_DOCS.\n"
                if is_allowed("execute_tool"): options_text += "18. [EXECUTE: <TOOL_NAME>, <ARGS>]: Execute a system tool. Available: [CALCULATOR, CLOCK, DICE, SYSTEM_INFO].\n"
                if is_allowed("physics"): options_text += "19. [EXECUTE: PHYSICS, '<SCENARIO>']: Perform a reality check or Fermi estimation on a physical/biochemical scenario.\n"
                if is_allowed("causal"): options_text += "20. [EXECUTE: CAUSAL, '<TREATMENT>, <OUTCOME>, <CONTEXT>']: Perform Bayesian Causal Inference (DoWhy).\n"
                if is_allowed("simulate_action"): options_text += "21. [EXECUTE: SIMULATE_ACTION, '<ACTION>']: Predict the outcome/cost of an action before doing it (Internal Simulator).\n"
                if is_allowed("predict"): options_text += "22. [EXECUTE: PREDICT, '<CLAIM>, <TIMEFRAME>']: Make a measurable prediction to verify later. E.g., 'Stock X will rise, 1 week'.\n"
                if is_allowed("write_file"): options_text += "22. [WRITE_FILE: <FILENAME>, <CONTENT>]: Write content to a file in the 'works' folder.\n"
            else:
                reason = "forced stop" if self.forced_stop_cooldown else "consecutive heavy task prevention"
                self.log(f"🤖 Decider: Daydream/Verify disabled for this turn due to {reason}.")

            if not self.last_action_was_speak and is_allowed("speak"):
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
            
            if self.last_internal_thought:
                prompt += f"PREVIOUS THOUGHT: {self.last_internal_thought}\nCONSTRAINT: Do NOT repeat the previous thought. Advance the reasoning.\n\n"

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
                stop_check_fn=self.stop_check
            )
            
            response = response.strip()
            
            # Extract Internal Thought for logging
            internal_thought = "No thought generated."
            thought_match = re.search(r"Internal Thought:\s*(.*?)(?=\nCommand:|$)", response, re.IGNORECASE | re.DOTALL)
            if thought_match:
                internal_thought = thought_match.group(1).strip()
                self.last_internal_thought = internal_thought
                
                # Update history
                norm_t = re.sub(r'\s+', ' ', internal_thought.lower().strip())
                self.thought_history.append(norm_t)
                if len(self.thought_history) > 10: self.thought_history.pop(0)

                self.log(f"🤔 Internal Thought: {internal_thought}")
                if self.chat_fn:
                    self.chat_fn("Decider", f"🤔 Thought: {internal_thought}")
                if hasattr(self.meta_memory_store, 'add_event'):
                    self.meta_memory_store.add_event(
                        event_type="STRATEGIC_THOUGHT",
                        subject="Assistant",
                        text=internal_thought
                    )
                
                # Ethical Interlock: Check alignment before proceeding
                if self.value_core:
                    is_safe, violation_score, reason = self.value_core.check_alignment(internal_thought, context="Decider Planning")
                    if not is_safe:
                        self.log(f"🛡️ Decider: Thought rejected by ValueCore ({violation_score:.2f}). Reason: {reason}")
                        return {"task": "wait", "cycles": 0, "reason": "ValueCore Rejection"}

                # Repetition Check
                if self._is_repetitive_thought(internal_thought) and len(self.thought_history) >= 3:
                     self.log("🛑 Repetitive Thought detected. Forcing [WAIT] to break cognitive loop.")
                     return {"task": "wait", "cycles": 0, "reason": "Cognitive Loop Detected"}
            
            # Extract the actual command from the verbose response to prevent false positives
            # (e.g. "I should not [WAIT]" triggering WAIT)
            extracted_command = self._extract_command(response)
            
            # --- Meta-Cognitive Loop ---
            if extracted_command and self.keter:
                if not self._meta_cognitive_check(internal_thought, extracted_command):
                    self.log(f"🧠 Meta-Cognition: Action '{extracted_command}' misaligned. Forcing [REFLECT].")
                    extracted_command = "[REFLECT]"
            # ---------------------------

            if extracted_command:
                response_upper = extracted_command.upper()
            else:
                response_upper = response.upper()
            
            self.log(f"🤖 Decider Decision: {response}")
            
            if self.chat_fn:
                self.chat_fn("Decider", f"Decision: {response}")
            
            # Decision Reflection (Metacognition)
            self._reflect_on_decision(internal_thought, extracted_command or response)
            
            # Reset cooldown
            if self.forced_stop_cooldown:
                self.forced_stop_cooldown = False

            # Deadlock / Loop Protection
            if self._detect_deadlock(extracted_command or response):
                self.log(f"🛑 Deadlock Detected (Looping command: {extracted_command}). Forcing [WAIT] to break cycle.")
                return {"task": "wait", "cycles": 0, "reason": "Deadlock detected"}

            if "[WAIT]" in response_upper:
                return {"task": "wait", "cycles": 0, "reason": "Decider chose to wait"}
                # Do not reset consecutive_daydream_batches on WAIT, so we remember to switch tasks after waking up
            elif "[SLEEP]" in response_upper:
                self.start_sleep_cycle()
                return {"task": "sleep", "cycles": 5, "reason": "Decider chose to sleep"}
            elif "[DAYDREAM:" in response_upper:
                if not allow_daydream:
                    self.log("⚠️ Decider tried to Daydream during cooldown/prevention. Forcing WAIT.")
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
                        
                        self.daydream_mode = mode if mode in ["read", "auto"] else "auto"
                        self.daydream_topic = topic
                        self.consecutive_daydream_batches += 1
                        return {"task": "daydream", "cycles": max(1, min(count, 10)), "reason": f"Strategic decision ({mode})"}
                    except:
                        self.daydream_mode = "auto"
                        self.consecutive_daydream_batches += 1
                        return {"task": "daydream", "cycles": 1, "reason": "Strategic decision (fallback)"}
            elif "[VERIFY_ALL]" in response_upper:
                # --- SAFETY CHECK ---
                stats = self.memory_store.get_memory_stats()
                unverified_count = stats.get('unverified_facts', 0) + stats.get('unverified_beliefs', 0)
                
                if unverified_count == 0:
                    self.log("🛑 Decider attempted VERIFY_ALL, but 0 items pending. Forcing WAIT.")
                    # Forcefully lower pressure so it doesn't loop
                    if self.gevurah: 
                        self.gevurah.last_pressure = 0.0 
                    return {"task": "wait", "cycles": 0, "reason": "Verification requested but nothing to verify"}
                else:
                    self.consecutive_daydream_batches = 0
                    return {"task": "verify_all", "cycles": 1, "reason": "Strategic decision (Verify All)"}
            elif "[VERIFY" in response_upper and "ALL" not in response_upper:
                try:
                    if ":" in response_upper:
                        count = int(response_upper.split(":")[1].strip().replace("]", ""))
                    else:
                        count = 3
                    self.consecutive_daydream_batches = 0
                    return {"task": "verify", "cycles": max(1, min(count, 5)), "reason": "Strategic decision (Verify Batch)"}
                except:
                    self.consecutive_daydream_batches = 0
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
                        self.log(f"⚠️ Decider generated placeholder '{msg}' for SPEAK. Aborting speak.")
                        return {"task": "wait", "cycles": 0, "reason": "Failed speak attempt"}
                    else:
                        if self.chat_fn:
                            self.chat_fn("Decider", msg)
                        self.last_action_was_speak = True
                        return {"task": "wait", "cycles": 0, "reason": "Spoke to user"}
                except Exception as e:
                    self.log(f"⚠️ Decider failed to speak: {e}")
            elif "[CHRONICLE:" in response_upper or "[NOTE_CREATE:" in response_upper:
                try:
                    match = re.search(r"\[(?:CHRONICLE|NOTE_CREATE):(.*?)(?:\])?\s*$", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        args = match.group(1).strip()
                        self.create_note(args)
                    # Don't wait; allow chaining decisions (e.g. create memory -> then daydream)
                    self.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Created chronicle"}
                except Exception as e:
                    self.log(f"⚠️ Decider failed to create chronicle: {e}")
            elif "[NOTE_EDIT:" in response_upper:
                try:
                    match = re.search(r"\[NOTE_EDIT:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    if match:
                        args = match.group(1).strip()
                        if "," in args:
                            mid_str, content = args.split(",", 1)
                            self.malkuth.edit_note(int(mid_str.strip()), content.strip())
                    # Don't wait; allow chaining decisions
                    self.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Edited note"}
                except Exception as e:
                    self.log(f"⚠️ Decider failed to edit note: {e}")
            elif "[THINK:" in response_upper:
                try:
                    match = re.search(r"\[THINK:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    topic = match.group(1).strip() if match else "General"
                    
                    # Clean up common hallucinated brackets
                    topic = topic.replace("<", "").replace(">", "").replace("[", "").replace("]", "")

                    # Filter out placeholders
                    bad_topics = ["TOPIC", "SUBJECT", "CONTENT", "TEXT", "INSERT TOPIC", "SPECIFIC_TOPIC", "ANY", "GENERAL"]
                    if topic.upper() in bad_topics or not topic:
                        self.log(f"⚠️ Decider generated placeholder '{topic}' for THINK. Auto-selecting topic.")
                        # Fallback: Pick an active goal or default
                        active_goals = self.memory_store.get_active_by_type("GOAL")
                        if active_goals:
                            topic = active_goals[0][2] # Use text of first goal
                        else:
                            topic = "The Nature of Artificial Consciousness"
                    
                    self.perform_thinking_chain(topic)
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to start thinking chain: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Thinking chain failed"}
            elif "[REFLECT]" in response_upper:
                try:
                    self.perform_self_reflection()
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to reflect: {e}")
            elif "[PHILOSOPHIZE]" in response_upper:
                try:
                    if "philosophize" in self.actions:
                        self.actions["philosophize"]()
                    else:
                        self.log("⚠️ Action 'philosophize' not available.")
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to philosophize: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Philosophize failed"}
            elif "[DEBATE:" in response_upper:
                try:
                    match = re.search(r"\[DEBATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    topic = match.group(1).strip() if match else "General"
                    if self.dialectics:
                        result = self.dialectics.run_debate(topic, reasoning_store=self.reasoning_store)
                        self.reasoning_store.add(content=f"Council Debate ({topic}):\n{result}", source="council_debate", confidence=1.0)
                        self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to start debate: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Debate failed"}
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
                    return {"task": "wait", "cycles": 0, "reason": "Simulation failed"}
            elif "[EXECUTE:" in response_upper:
                try:
                    match = re.search(r"\[EXECUTE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    content = match.group(1).strip() if match else ""
                    
                    if "," in content:
                        tool, args = content.split(",", 1)
                        self._execute_tool(tool.strip().upper(), args.strip())
                    else:
                        self._execute_tool(content.strip().upper(), "")
                    self.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Executed tool"}
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
                        
                        # CRS Allocation
                        goal_item = self.memory_store.get(target_id)
                        priority = goal_item.get('confidence', 0.5) if goal_item else 0.5
                        coherence = self.keter.evaluate().get("keter", 1.0) if self.keter else 1.0
                        
                        # Pass metrics to CRS for active control
                        metrics = self.meta_learner.metrics_history if self.meta_learner else {}
                        model_params = self.meta_learner.latest_model_params if self.meta_learner else None
                        
                        current_state = {
                            "hesed": self.hesed.calculate() if self.hesed else 0.5,
                            "gevurah": self.gevurah.calculate() if self.gevurah else 0.5
                        }
                        
                        violation_pressure = self.value_core.get_violation_pressure() if self.value_core else 0.0

                        # Defensive casting for inputs
                        try:
                            priority = float(priority)
                            coherence = float(coherence)
                        except (ValueError, TypeError):
                            priority = 0.5
                            coherence = 1.0

                        if self.crs:
                            alloc = self.crs.allocate("reasoning", complexity=priority, coherence=coherence, metrics=metrics, model_params=model_params, current_state=current_state, violation_pressure=violation_pressure)
                            depth = alloc["reasoning_depth"]
                            beam = alloc["beam_width"]
                            self.last_predicted_delta = alloc.get("predicted_delta", 0.0)
                        else:
                            # Fallback
                            depth = 5
                            if priority > 0.8: depth = 10
                            beam = 1
                        
                        self.log(f"    Allocating compute: Depth {depth}, Beam {beam} (Priority {priority:.2f})")
                        self.perform_thinking_chain(f"Strategic Plan for Goal: {goal_text}", max_depth=depth, beam_width=beam)
                    else:
                        self.log(f"⚠️ Goal {target_id} not found.")
                        return {"task": "wait", "cycles": 0, "reason": "Goal not found"}
                    
                    self.consecutive_daydream_batches = 0
                except Exception as e:
                    self.log(f"⚠️ Decider failed to act on goal: {e}")
                    return {"task": "wait", "cycles": 0, "reason": "Goal action failed"}
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
                    
                    self.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Removed goal"}
                except Exception as e:
                    self.log(f"⚠️ Decider failed to remove goal: {e}")
            elif "[GOAL_CREATE:" in response_upper:
                try:
                    match = re.search(r"\[GOAL_CREATE:(.*?)\]?$", response, re.IGNORECASE | re.DOTALL)
                    content = match.group(1).strip() if match else ""
                    
                    if not content or content.upper() in ["TEXT", "CONTENT", "GOAL"]:
                        self.log("⚠️ Decider tried to create empty GOAL. Injecting default.")
                        content = "Research recent advancements in AI architecture"

                    self.create_goal(content)
                    self.consecutive_daydream_batches = 0
                    return {"task": "organizing", "cycles": 0, "reason": "Created goal"}
                except Exception as e:
                    self.log(f"⚠️ Decider failed to create goal: {e}")
            elif "[LIST_DOCS]" in response_upper:
                if "list_documents" in self.actions:
                    docs_list = self.actions["list_documents"]()
                    self.reasoning_store.add(content=f"Tool Output [LIST_DOCS]:\n{docs_list}", source="tool_output", confidence=1.0)
                    self.log(f"📚 Documents listed.")
                return {"task": "organizing", "cycles": 0, "reason": "Listed docs"}
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
                return {"task": "organizing", "cycles": 0, "reason": "Read doc"}
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
                return {"task": "organizing", "cycles": 0, "reason": "Searched memory"}
            elif "[SUMMARIZE]" in response_upper:
                if "summarize" in self.actions:
                    result = self.actions["summarize"]()
                    if result:
                        self.log(result)
                return {"task": "organizing", "cycles": 0, "reason": "Summarized"}
            elif "[SEARCH_INTERNET" in response_upper:
                try:
                    match = re.search(r"\[SEARCH_INTERNET:\s*(.*),\s*([A-Z_]+)\]", response, re.IGNORECASE)
                    if match:
                        query = match.group(1).strip()
                        source = match.group(2).strip().upper()
                        self._execute_search(query, source)
                    else:
                        # Fallback: Try to find just a query (Default to WEB)
                        match_query = re.search(r"\[SEARCH_INTERNET:\s*(.*?)\]", response, re.IGNORECASE)
                        if match_query:
                             query = match_query.group(1).strip()
                             self._execute_search(query, "WEB")
                        else:
                             # No args at all? Try to use active goal
                             goals = self.memory_store.get_active_by_type("GOAL")
                             if goals:
                                 # get_active_by_type returns (id, subject, text, source)
                                 # Sort by ID desc to get most recent
                                 goals.sort(key=lambda x: x[0], reverse=True)
                                 query = goals[0][2] 
                                 self.log(f"⚠️ Decider forgot args. Auto-searching top goal: {query}")
                                 self._execute_search(query, "WEB")
                             else:
                                 # Fallback 2: Pick a random recent memory topic or "Latest AI News"
                                 # This ensures the tool ALWAYS runs if called, preventing "No Data" frustration
                                 fallback_topic = "Artificial Intelligence News"
                                 self.log(f"⚠️ Decider called SEARCH_INTERNET without args/goals. Fallback search: {fallback_topic}")
                                 self._execute_search(fallback_topic, "WEB")
                                 # Also create a goal so we don't loop
                                 self.create_goal(f"Research {fallback_topic}")
                    return {"task": "organizing", "cycles": 0, "reason": "Searched internet"}
                except Exception as e:
                    self.log(f"⚠️ Decider failed to search internet: {e}")
            else:
                # Default fallback
                # FIX: Only default to daydream if allowed and not cooling down
                if allow_daydream and not just_finished_heavy and not self.forced_stop_cooldown:
                    self._run_action("start_daydream")
                    # Note: _run_action here is immediate, but we also set state for loop
                    self.consecutive_daydream_batches += 1
                    return {"task": "daydream", "cycles": 0, "reason": "Fallback daydream"}
                else:
                    self.log(f"🤖 Decider: Fallback triggered but Daydream is disabled/cooldown. Defaulting to WAIT.")
                    return {"task": "wait", "cycles": 0, "reason": "Fallback wait"}
        
        except Exception as e:
            self.log(f"❌ Critical Planning Error: {e}")
            return {"task": "wait", "cycles": 0, "reason": f"Error: {e}"}
        finally:
            self.planning_lock.release()
            
        return {"task": "wait", "cycles": 0, "reason": "End of decision"}

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

    def _execute_search(self, query: str, source: str):
        """Helper to execute search action."""
        if "search_internet" in self.actions:
            result = self.actions["search_internet"](query, source)
            self.reasoning_store.add(content=f"Internet Search [{source}]: {query}\nResult: {result}", source="internet_bridge", confidence=1.0)
            self.log(f"🌐 Internet Search Result: {result[:100]}...")
        else:
            self.log("⚠️ Action search_internet not available.")

    def _run_action(self, name: str, reason: str = None):
        # 1. Capture Pre-Action State (Trigger State)
        start_state = {}
        start_coherence = 0.0
        start_utility = 0.0
        
        # Capture prediction made during planning (Fix 5: Prediction Error)
        predicted_delta = self.last_predicted_delta
        action_result_data = {}
        
        if self.keter:
            # Evaluate Keter to get current baseline (Raw score is better for immediate delta)
            keter_stats = self.keter.evaluate()
            start_coherence = keter_stats.get("raw", keter_stats.get("keter", 0.0))
            start_utility = self.calculate_utility()
            
            start_state = {
                "hesed": self.hesed.calculate() if self.hesed else 0,
                "gevurah": self.gevurah.calculate() if self.gevurah else 0,
                "active_goals": len(self.memory_store.get_active_by_type("GOAL")),
                "coherence": start_coherence,
                "utility": start_utility
            }

            # 1.5 Predictive Coding (Expectation Layer)
            # Ask LLM to predict the delta
            # We do this cheaply to avoid massive latency
            try:
                pred_prompt = (
                    f"ACTION: {name}\n"
                    f"CURRENT COHERENCE: {start_coherence:.2f}\n"
                    f"CURRENT UTILITY: {start_utility:.2f}\n"
                    "Predict the change (delta) in Coherence and Utility after this action.\n"
                    "Output JSON: {\"coherence_delta\": 0.01, \"utility_delta\": 0.05}"
                )
                pred_resp = run_local_lm(
                    messages=[{"role": "user", "content": pred_prompt}],
                    system_prompt="You are a Predictive Engine.",
                    temperature=0.0,
                    max_tokens=50
                )
                pred_data = parse_json_array_loose(pred_resp) # Or object parser if available
                # Assuming loose parser handles object or we parse manually. 
                # For robustness, we skip complex parsing here to keep it simple or rely on logs.
            except: pass
            
            # 1.6 Future Simulation (Malkuth)
            # For significant actions, run a deeper simulation
            # Optimization: Rate limit simulations to avoid latency on every action (60s cooldown)
            if name in ["GOAL_ACT", "EXECUTE"] and self.malkuth and (time.time() - self.last_simulation_time > 60):
                # Heuristic: Only simulate if we haven't just simulated (avoid loops)
                # and if the action isn't a simple tool call
                self.last_simulation_time = time.time()
                sim_context = f"State: H={start_state['hesed']:.2f}, G={start_state['gevurah']:.2f}, Coh={start_coherence:.2f}"
                sim_result = self.malkuth.simulate_futures(name, sim_context)
                
                expected_util = sim_result.get("expected_utility", 0.0)
                if expected_util < -0.3:
                    self.log(f"🛑 Decider: Action '{name}' aborted due to negative simulated utility ({expected_util:.2f}).")
                    # Record the "near miss"
                    if hasattr(self.meta_memory_store, 'add_event'):
                        self.meta_memory_store.add_event("ACTION_ABORTED", "Assistant", f"Aborted {name} due to bad simulation: {sim_result.get('futures', [])}")
                    return # Abort execution
                self.log(f"🔮 Simulation passed (Util: {expected_util:.2f}). Proceeding.")

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
                if isinstance(result, dict):
                    action_result_data = result
            else:
                self.log(f"⚠️ Decider action '{name}' not available.")

        # 3. Measure Result & Record Outcome (Credit Assignment)
        if self.keter and hasattr(self.meta_memory_store, 'add_outcome'):
            # Re-evaluate Keter to see change
            end_stats = self.keter.evaluate()
            end_coherence = end_stats.get("raw", end_stats.get("keter", 0.0))
            end_utility = self.calculate_utility()
            
            delta_coherence = end_coherence - start_coherence
            delta_utility = end_utility - start_utility
            
            # Calculate Prediction Error
            prediction_error = abs(predicted_delta - delta_coherence)
            
            # Affective Update (Dopamine/Cortisol)
            # Positive utility = Dopamine
            # Negative utility or High Prediction Error = Cortisol
            mood_impact = delta_utility - (prediction_error * 0.5)
            self._update_mood(mood_impact)

            # Surprise Event (High Prediction Error)
            if prediction_error > 0.2:
                self.log(f"😲 Decider: Surprise Event! Prediction Error {prediction_error:.2f}. Triggering Reflection.")
                # Inject Dissonance Signal directly for immediate awareness
                self.ingest_netzach_signal({
                    "signal": "DISSONANCE",
                    "pressure": 1.0,
                    "context": f"High Prediction Error ({prediction_error:.2f}) on {name}"
                })
                
                if hasattr(self.meta_memory_store, 'add_event'):
                    self.meta_memory_store.add_event("SURPRISE_EVENT", "Assistant", f"Action {name} had unexpected outcome (Err: {prediction_error:.2f})")
                # Force Hod to analyze why we were wrong
                self._run_action("run_hod")
            
            outcome_result = {
                "coherence_delta": delta_coherence, 
                "utility_delta": delta_utility,
                "end_score": end_coherence,
                "end_utility": end_utility,
                "predicted_delta": predicted_delta,
                "prediction_error": prediction_error
            }
            outcome_result.update(action_result_data)
            
            # Capture System DNA for Credit Assignment
            context_metadata = {
                "epigenetics": self.epigenetics,
                "hesed_bias": self.hesed.bias if self.hesed else 1.0,
                "gevurah_bias": self.gevurah.bias if self.gevurah else 1.0,
                "system_prompt_fitness": self.get_settings().get("system_prompt_fitness", 0.0)
            }
            
            self.meta_memory_store.add_outcome(
                action=name,
                trigger_state=start_state,
                result=outcome_result,
                context_metadata=context_metadata
            )

        if hasattr(self.meta_memory_store, 'add_event') and name != "run_observer":
            narrative = f"I executed {name}"
            if reason:
                narrative += f" because {reason}"
            else:
                narrative += "."
                
            self.meta_memory_store.add_event(
                event_type="DECIDER_ACTION",
                subject="Assistant",
                text=narrative
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
                    with self.maintenance_lock:
                        self.pending_maintenance.append({"type": "REFUTE", "id": observation["id"], "reason": observation.get("context")})
            elif otype == "INVALID_DATA":
                if "id" in observation:
                    with self.maintenance_lock:
                        self.pending_maintenance.append({"type": "PRUNE", "id": observation["id"], "reason": observation.get("context")})
            elif otype == "CONTEXT_FRAGMENTATION":
                self._run_action("summarize")
            elif otype == "CRITICAL_INFO":
                self.receive_observation(observation.get("context"))

    def ingest_netzach_signal(self, signal: Dict):
        """Receive signal from Netzach."""
        if not signal: return
        
        # Prevent overwriting high pressure with low pressure (noise)
        current_pressure = self.latest_netzach_signal.get("pressure", 0) if self.latest_netzach_signal else 0
        new_pressure = signal.get("pressure", 0)
        
        if new_pressure >= current_pressure:
            self.latest_netzach_signal = signal
            # Immediate reaction to high pressure if waiting
            if new_pressure > 0.7 and (self.heartbeat and self.heartbeat.current_task == "wait"):
                self.wake_up("Netzach Pressure")

    def _check_netzach_signal(self):
        """Check latest Netzach signal during planning."""
        if not self.latest_netzach_signal: return
        
        sig = self.latest_netzach_signal.get("signal")
        context = self.latest_netzach_signal.get("context")
        
        if sig == "LOW_NOVELTY":
            # If bored, don't just increase temp. DO something.
            self.log("👁️ Netzach signaled LOW_NOVELTY. Triggering Daydream.")
            self.increase_temperature()
            if self.heartbeat:
                self.heartbeat.force_task("daydream", 1, "Netzach Low Novelty")
            self.daydream_mode = "insight" # Force insight generation
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
        #      self.current_task_reason = "Netzach Momentum Signal"
        #      self.start_daydream() # Removed: Let _decide_next_batch handle this via Goal Act or Daydream
        
        # Clear signal after processing
        self.latest_netzach_signal = None

    def _authorize_maintenance(self):
        """Authorize and dispatch queued maintenance actions (Prune/Refute)."""
        with self.maintenance_lock:
            if not self.pending_maintenance: return
            actions = list(self.pending_maintenance)
            self.pending_maintenance = []
        
        self.log(f"🤖 Decider: Authorizing {len(actions)} maintenance actions.")
        for action in actions:
            if action["type"] == "PRUNE":
                if "prune_memory" in self.actions:
                    self.actions["prune_memory"](action["id"])
            elif action["type"] == "REFUTE":
                if "refute_memory" in self.actions:
                    self.actions["refute_memory"](action["id"], action.get("reason"))
        

    def on_stagnation(self, event_data: Any = None):
        """Handle system stagnation event (usually from Netzach)."""
        self.log("🤖 Decider: Stagnation detected. Waking up.")
        self.start_daydream()

    def on_instability(self, event_data: Dict):
        """Handle system instability event (usually from Hod)."""
        self.log(f"🤖 Decider: Instability reported ({event_data.get('reason')}). Switching to verification.")
        if self.heartbeat:
            self.heartbeat.force_task("verify", 1, f"Instability: {event_data.get('reason')}")

    def receive_observation(self, observation: str):
        """
        Receive an observation/information from Netzach.
        This information MUST result in an action.
        """
        if not observation:
            return

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
            self.daydream_mode = "auto"
            if self.heartbeat:
                self.heartbeat.force_task("daydream", 3, "Netzach loop signal")
        elif any(w in text for w in ["stop", "halt", "pause"]) and "daydream" in text:
            self._run_action("stop_daydream", reason="Netzach stop signal")
            if self.heartbeat:
                self.heartbeat.force_task("wait", 0, "Netzach stop signal")
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
            task = "verify"
            cycles = 2
            if "all" in text or "full" in text:
                task = "verify_all"
                cycles = 1
            if self.heartbeat:
                self.heartbeat.force_task(task, cycles, "Netzach conflict signal")
        elif any(w in text for w in ["hod", "analyze", "analysis", "investigate", "pattern", "reflect", "refuted", "refutation"]):
            self._run_action("run_hod", reason="Netzach analysis signal")
        elif "observer" in text or "watch" in text:
            self._run_action("run_observer", reason="Netzach watch signal")
        elif any(w in text for w in ["stagnant", "idle", "bored", "nothing", "quiet", "daydream", "think", "create"]):
            self.daydream_mode = "auto"
            if self.heartbeat:
                self.heartbeat.force_task("daydream", 1, "Netzach boredom signal")
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
                self._run_action("run_hod", reason="Netzach complex signal")
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

    def perform_thinking_chain(self, topic: str, max_depth: int = 10, beam_width: int = 3):
        """Execute a Tree of Thoughts (ToT) reasoning process."""
        self.log(f"🌳 Decider starting Tree of Thoughts on: {topic}")
        if self.chat_fn:
            self.chat_fn("Decider", f"🌳 Starting Tree of Thoughts: {topic}")
            
        settings = self.get_settings()

        # 0.5 Retrieve Subjective Memories (Recursive Subjectivity)
        subjective_context = ""
        if hasattr(self.meta_memory_store, 'search'):
            # Search for meta-memories related to the topic to find past feelings/states
            query_embedding = compute_embedding(topic, base_url=settings.get("base_url"), embedding_model=settings.get("embedding_model"))
            subj_results = self.meta_memory_store.search(query_embedding, limit=3)
            if subj_results:
                subjective_context = "Relevant Subjective Experiences (How I felt before):\n" + "\n".join([f"- [{m['event_type']}] {m['text']}" for m in subj_results]) + "\n"

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
        if subjective_context:
            static_context += f"\n{subjective_context}\n"
        
        # Ask Daat for structure (Graph of Thoughts)
        if self.daat:
            structure = self.daat.provide_reasoning_structure(topic)
            if structure and not structure.startswith("⚠️"):
                static_context += f"\nReasoning Structure (Guide):\n{structure}\n"

        # Tree State: List of paths. A path is a list of thought strings.
        # Start with one empty path
        active_paths = [[]] 
        final_conclusion = None
        best_path = []
        
        for depth in range(1, max_depth + 1):
            if self.stop_check():
                break
            
            if final_conclusion:
                break

            self.log(f"🌳 Depth {depth}: Expanding {len(active_paths)} paths...")
            
            candidates = [] # List of (path, score)

            # Expansion Phase (Branching)
            for path in active_paths:
                # Generate N possible next steps for this path
                path_str = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(path)])
                
                prompt = (
                    f"You are an AGI reasoning engine using Tree of Thoughts.\n"
                    f"{static_context}\n"
                    f"Current Path:\n{path_str}\n\n"
                    f"Generate {beam_width} distinct, valid next steps to advance reasoning towards a solution.\n"
                    "If a solution is reached, start the step with '[CONCLUSION]'.\n"
                    "Output JSON list of strings: [\"Step A...\", \"Step B...\", \"Step C...\"]"
                )
                
                response = run_local_lm(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt="You are a Generator.",
                    temperature=0.7,
                    max_tokens=400,
                    base_url=settings.get("base_url"),
                    chat_model=settings.get("chat_model"),
                    stop_check_fn=self.stop_check
                )
                
                next_steps = parse_json_array_loose(response)
                if not next_steps:
                    # Fallback if JSON fails
                    next_steps = [response.strip()]
                
                # Evaluation Phase (Scoring)
                for step in next_steps:
                    if not isinstance(step, str): continue
                    
                    # Check for conclusion
                    if "[CONCLUSION]" in step:
                        final_conclusion = step.replace("[CONCLUSION]", "").strip()
                        best_path = path + [step]
                        break
                    
                    # Score the step
                    eval_prompt = (
                        f"Evaluate this reasoning step for the topic '{topic}':\n"
                        f"Step: \"{step}\"\n"
                        "Criteria: Logic, Relevance, Novelty.\n"
                        "Output a score from 0.0 to 1.0."
                    )
                    
                    score_str = run_local_lm(
                        messages=[{"role": "user", "content": eval_prompt}],
                        system_prompt="You are an Evaluator.",
                        temperature=0.1,
                        max_tokens=10,
                        base_url=settings.get("base_url"),
                        chat_model=settings.get("chat_model")
                    )
                    
                    try:
                        score = float(re.search(r"0\.\d+|1\.0|0|1", score_str).group())
                    except:
                        score = 0.5
                    
                    candidates.append((path + [step], score))
                
                if final_conclusion:
                    break
            
            # Selection Phase (Beam Search)
            # Sort by score descending and keep top K
            candidates.sort(key=lambda x: x[1], reverse=True)
            active_paths = [c[0] for c in candidates[:beam_width]]
            
            # Log best step of this depth
            if active_paths:
                best_step = active_paths[0][-1]
                self.log(f"🌳 Best Step at Depth {depth}: {best_step[:100]}...")
                if self.chat_fn:
                    self.chat_fn("Decider", f"🌳 Depth {depth}: {best_step}")
                
                # Store in reasoning
                self.reasoning_store.add(content=f"ToT Depth {depth} ({topic}): {best_step}", source="decider_tot", confidence=1.0)
            
        if not final_conclusion and active_paths:
            # If max depth reached without conclusion, take the best path
            best_path = active_paths[0]
            final_conclusion = "Max depth reached. Best partial reasoning path selected."
        elif final_conclusion:
             self.create_note(f"Conclusion on {topic}: {final_conclusion}")
        
        # Post-chain Summarization
        if best_path:
            self.log(f"🧠 Generating summary of Tree of Thoughts...")
            full_chain_text = "\n".join(best_path)
            summary_prompt = (
                f"Synthesize the following reasoning path regarding '{topic}' into a clear, comprehensive summary for the user.\n"
                f"Include key insights and the final conclusion if reached.\n\n"
                f"Reasoning Path:\n{full_chain_text}"
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
                self.chat_fn("Decider", f"🌳 Tree of Thoughts Summary:\n{summary}")

            # Metacognitive Reflection on the thinking process
            self._reflect_on_decision(f"Tree of Thoughts on {topic}", summary)

            # Save summary as note if no formal conclusion was reached (e.g. interrupted by loop)
            # This ensures partial progress is preserved in memory
            if not final_conclusion:
                self.create_note(f"ToT Summary ({topic}): {summary}")

        if self.heartbeat:
            self.heartbeat.force_task("wait", 0, "Thinking chain complete")

    def perform_self_reflection(self):
        """
        Deep introspection cycle.
        Analyzes alignment between Identity, Goals, and recent Actions.
        """
        self.log("🧘 Decider entering Self-Reflection...")
        
        # 1. Gather Self-Context
        identity = self.memory_store.get_active_by_type("IDENTITY")
        goals = self.memory_store.get_active_by_type("GOAL")
        recent_actions = self.meta_memory_store.list_recent(limit=10)
        
        context = "MY IDENTITY:\n" + "\n".join([f"- {i[2]}" for i in identity[:5]]) + "\n\n"
        context += "MY GOALS:\n" + "\n".join([f"- {g[2]}" for g in goals[:5]]) + "\n\n"
        context += "RECENT ACTIONS:\n" + "\n".join([f"- {a[3]}" for a in recent_actions])
        
        # 2. Reflect
        prompt = (
            f"{context}\n\n"
            "TASK: Reflect on your recent actions. Are they aligned with your Identity and Goals?\n"
            "Identify 1 area of improvement or 1 new insight about yourself.\n"
            "Output a concise reflection."
        )
        
        reflection = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Self-Aware AI.",
            max_tokens=300,
            base_url=self.get_settings().get("base_url"),
            chat_model=self.get_settings().get("chat_model")
        )
        
        self.log(f"🧘 Reflection: {reflection}")
        
        # 3. Store
        self.reasoning_store.add(content=f"Self-Reflection: {reflection}", source="decider_reflection", confidence=1.0)
        
        # Metacognitive Reflection on the reflection
        self._reflect_on_decision("Deep Self-Reflection", reflection)
        
    def _execute_tool(self, tool_name: str, args: str):
        """Execute a tool safely and store the result."""
        # Rate limiting (2 seconds between tool calls)
        if time.time() - self.last_tool_usage < 2.0:
            self.log(f"⚠️ Tool rate limit exceeded for {tool_name}")
            return "Error: Tool rate limit exceeded. Please wait."
        self.last_tool_usage = time.time()

        # Integrated Budgeting: Deduct cost
        if self.crs:
            cost = self.crs.estimate_action_cost(tool_name, complexity=0.5)
            self.crs.current_spend += cost

        self.log(f"🛠️ Decider executing tool: {tool_name} args: {args}")
        result = ""
        
        # Delegate to ActionManager tools if available
        if tool_name in self.actions:
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
        now = time.time()
        if now - self._last_critical_update < 300 and self._critical_memories_cache:
            return self._critical_memories_cache

        critical_types = ["IDENTITY", "PERMISSION", "RULE", "GOAL", "CURIOSITY_GAP"]
        mems = []
        for t in critical_types:
            # get_active_by_type returns (id, subject, text, source, confidence)
            items = self.memory_store.get_active_by_type(t)
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
        
        self._critical_memories_cache = mems
        self._last_critical_update = now
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
            if self.heartbeat:
                self.heartbeat.force_task("daydream", count, "Natural Command Loop")
            return f"🔄 Daydream loop enabled for {count} cycles."

        # Daydream Batch (Specific Count)
        # Matches: "run 5 daydream cycles", "do 3 daydreams", "run 1 daydream cycle"
        batch_match = re.search(r"(?:run|do|start|execute)\s+(\d+)\s+daydream(?:s|ing)?(?: cycles?| loops?)?", text)
        if batch_match:
            count = int(batch_match.group(1))
            # Cap count reasonably
            count = max(1, min(count, 20))
            
            self.log(f"🤖 Decider enabling Daydream Batch for {count} cycles via natural command.")
            self.daydream_mode = "auto"
            if self.heartbeat:
                self.heartbeat.force_task("daydream", count, "Natural Command Batch")
            return f"☁️ Starting {count} daydream cycles."

        # Learn / Expand Knowledge
        learn_match = re.search(r"(?:expand (?:your )?knowledge(?: about)?|learn(?: about)?|research|study|read up on|educate yourself(?: on| about)?)\s+(.*)", text, re.IGNORECASE)
        if learn_match:
            raw_topic = learn_match.group(1).strip(" .?!")
            
            # Use LLM to extract the core topic more robustly
            settings = self.get_settings()
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
                stop_check_fn=self.stop_check
            ).strip()

            if core_topic and not core_topic.startswith("⚠️"):
                self.create_goal(f"Expand knowledge about {core_topic}")
                self.log(f"🤖 Decider starting Daydream Loop focused on: {core_topic}")
                self._run_action("start_loop")
                self.daydream_mode = "read"
                self.daydream_topic = core_topic
                if self.heartbeat:
                    self.heartbeat.force_task("daydream", 5, f"Research: {core_topic}")
                return f"📚 Initiating research protocol for: {core_topic}. I will read relevant documents and generate insights."
            else:
                return f"⚠️ Failed to extract a clear topic from '{raw_topic}'. Please be more specific."

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
             self._run_action("verify_batch")
             if self.heartbeat:
                 self.heartbeat.force_task("verify", 3, "Belief Verification")
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
            if self.heartbeat:
                self.heartbeat.force_task("wait", 0, "Natural Stop Command")
            return "🛑 Processing stopped."
            
        # Sleep
        if "go to sleep" in text or "enter sleep mode" in text:
            self.start_sleep_cycle()
            return "💤 Entering Sleep Mode. Inputs will be ignored while I consolidate memory."
            
        # Think
        if text.startswith("think about") or text.startswith("analyze") or text.startswith("ponder"):
            topic = text.replace("think about", "").replace("analyze", "").replace("ponder", "").strip()
            self.perform_thinking_chain(topic)
            return f"🧠 Finished thinking about: {topic}"
            
        # Debate
        if text.startswith("debate") or text.startswith("discuss"):
            topic = text.replace("debate", "").replace("discuss", "").strip()
            if self.dialectics:
                return f"🏛️ Council Result: {self.dialectics.run_debate(topic, reasoning_store=self.reasoning_store)}"
        
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
                self.daydream_mode = "read"
                self.daydream_topic = clean_topic
                if self.heartbeat:
                    self.heartbeat.force_task("daydream", 5, f"Research: {clean_topic}")
                return f"📚 Initiating research protocol for: {clean_topic}. I will read relevant documents and generate insights."
                
            elif "[VERIFY]" in intent_response:
                if "belief" in intent_response.lower():
                    self.log("🤖 Decider starting Belief Verification via intent analysis.")
                    self._run_action("verify_batch")
                    if self.heartbeat:
                        self.heartbeat.force_task("verify", 3, "Intent: Verify Beliefs")
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

    def process_chat_message(self, user_text: str, history: List[Dict], status_callback: Callable[[str], None] = None, image_path: Optional[str] = None, stop_check_fn: Callable[[], bool] = None) -> str:
        """
        Core Chat Logic: RAG -> LLM -> Memory Extraction -> Response.
        Decider now handles the cognitive pipeline for user interactions.
        """
        # Mailbox: Chat is an external interruption that resets the Hod cycle lock
        self.log(f"📬 Decider Mailbox: Received message from User.")
        self.hod_just_ran = False
        self.last_action_was_speak = False
        
        # Use provided stop check or default
        current_stop_check = stop_check_fn if stop_check_fn else self.stop_check

        # Check for natural language commands
        # Skip NL commands if image is present (prioritize Vision), UNLESS it's a slash command
        if not image_path or user_text.strip().startswith("/"):
            nl_response = self.handle_natural_language_command(user_text, status_callback)
            if nl_response:
                self.log(f"🤖 Decider Command Response: {nl_response}")
                return nl_response

        settings = self.get_settings()
        
        # Parallelize Context Retrieval to reduce latency
        # Use shared executor if available to prevent thread explosion
        local_executor = None
        submit_fn = self.executor.submit if self.executor else None
        
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
                chat_mems = self.memory_store.get_recent_filtered(limit=10, exclude_sources=['daydream'])
                recent_mems = self.memory_store.list_recent(limit=5)
                return chat_mems, recent_mems
            future_combined = submit_fn(fetch_combined)
            future_critical = submit_fn(self._get_critical_memories)
            logging.debug(f"⏱️ [Tiferet] Memory retrieval submission took {time.time()-t_retrieval:.3f}s")
            
            # 3. Start Summary Retrieval (DB)
            def get_summary():
                if hasattr(self.meta_memory_store, 'get_by_event_type'):
                    summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1)
                    if summaries:
                        return summaries[0]['text']
                return ""
            future_summary = submit_fn(get_summary)
            
            # Wait for embedding to proceed with Semantic Search & RAG
            query_embedding = future_embedding.result()
            
            # 4. Start Semantic Search (FAISS/DB)
            future_semantic = submit_fn(self.memory_store.search, query_embedding, limit=5, target_affect=self.mood)
            
            # 4.5 Start Meta-Memory Semantic Search (Autobiographical Memory)
            future_meta_semantic = submit_fn(self.meta_memory_store.search, query_embedding, limit=3)
            
            # 5. Start RAG (FAISS/DB)
            future_rag = None
            if self._should_trigger_rag(user_text):
                t_rag = time.time()
                self.log(f"📚 [RAG] Initiating document search for: '{user_text}'")
                def perform_rag():
                    doc_results = self.document_store.search_chunks(query_embedding, top_k=5)
                    filename_matches = self.document_store.search_filenames(user_text)
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
            if self.binah and semantic_items and (len(semantic_items) < 3 or semantic_items[0][4] < 0.8):
                # Limit seeds to top 3 to reduce DB queries
                seed_ids = [item[0] for item in semantic_items[:3]]
                assoc_results = self.binah.expand_associative_context(seed_ids, limit=3)
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
            logging.debug(f"⏱️ [Tiferet] Context gathering result wait took {time.time()-t_gather:.3f}s")
        finally:
            if local_executor:
                local_executor.shutdown(wait=False)

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
        
        # Add Meta-Memories (Autobiographical)
        for item in meta_semantic_items:
            # item is dict: {'id', 'event_type', 'subject', 'text', ...}
            # Convert to tuple format for consistency in display logic or handle separately
            # We'll add them to a separate list for context construction
            pass

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
        if self.yesod:
            system_prompt = self.yesod.get_dynamic_system_prompt(base_prompt)
        else:
            system_prompt = base_prompt

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

        # Log final prompt length
        prompt_tokens = count_tokens(system_prompt)
        self.log(f"📝 Final Prompt: {len(system_prompt)} chars (~{prompt_tokens} tokens)")

        # NEW: Self-Improvement Prompt (Appended after memory context)
        self_improvement_prompt = settings.get("self_improvement_prompt", "")
        if self_improvement_prompt:
            system_prompt += "\n\n" + self_improvement_prompt

        # 4. Call LLM with structured error handling
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
                images=[image_path] if image_path else None
            )
        except LLMError as e:
            self.log(f"❌ Chat generation failed: {e}")
            return "⚠️ I encountered an error generating a response. Please check the logs."
        latency = time.time() - start_time
        
        # Feed Keter (Vibrational Monitoring)
        if self.keter:
            self.keter.track_response_metrics(latency, len(reply))
        
        # 4.5 Manifest Persona (Yesod)
        if self.yesod:
            reply = self.yesod.manifest_persona(reply, self.mood)

        # 4.6 Recursive Theory of Mind (Self-Model of User's Model)
        # Simulate user perception before finalizing (or just for reflection)
        if self.yesod:
            try:
                perception = self.yesod.simulate_user_perception(user_text, reply)
                self.reasoning_store.add(
                    content=f"Recursive ToM Simulation: {perception.get('interpretation')} (Sat: {perception.get('satisfaction')}, Conf: {perception.get('confusion_risk')})",
                    source="recursive_tom",
                    confidence=1.0
                )
                # Future: If confusion_risk > 0.7, trigger refinement loop here
            except Exception as e:
                self.log(f"⚠️ Recursive ToM failed: {e}")

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
                    self._track_metric("tool_success", 1.0 if "Error" not in result else 0.0)
                    reply += f"\n\n🛠️ Tool Result: {result}"
            except Exception as e:
                self.log(f"⚠️ Chat tool execution failed: {e}")
                self._track_metric("tool_success", 0.0)

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
            
            # Update World Model with interaction
            if self.malkuth:
                # Register the chat interaction as an outcome
                self.malkuth.register_outcome("Chat Interaction", "Reply to user", f"User: {user_text}\nAssistant: {reply}")
            
            # 6. Update Theory of Mind (User Model)
            if self.yesod:
                self.yesod.analyze_user_interaction(user_text, reply)
            
            # 7. Metacognitive Reflection
            self._reflect_on_decision(user_text, reply)
        
        if self.executor:
            self.executor.submit(background_processing)
        else:
            threading.Thread(target=background_processing, daemon=True).start()
        
        self.log(f"🗣️ Assistant Reply: {reply}")

        return reply

    def _reflect_on_decision(self, context: str, decision: str):
        """Metacognition: Critique a recent decision/action."""
        settings = self.get_settings()
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
            self.reasoning_store.add(
                content=f"Decision Critique: {critique}",
                source="decider_reflection",
                confidence=1.0
            )
            self.log(f"🧠 Metacognition: {critique}")

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
                try:
                    c["confidence"] = float(c.get("confidence", 0.9))
                except (ValueError, TypeError):
                    self.log(f"⚠️ Invalid confidence value '{c.get('confidence')}'. Defaulting to 0.5.")
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
                self.reasoning_store.add_batch(reasoning_entries)
                self.log(f"🧠 Added {len(reasoning_entries)} candidates to Reasoning Store.")

            # Arbiter promotion (batch)
            promoted_ids = self.arbiter.consider_batch(candidates)
            
            if promoted_ids:
                self.log(f"🧠 Promoted {len(promoted_ids)} memory item(s).")

        except Exception as e:
            self.log(f"Memory extraction error: {e}\n{traceback.format_exc()}")

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
        
        try:
            steps = parse_json_array_loose(response)
        except Exception as e:
            self.log(f"⚠️ Goal decomposition JSON parsing failed: {e}. Response: {response[:100]}...")
            return False
        
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

    def manage_goals(self, allow_creation: bool = True, system_mode: str = "EXECUTION"):
        """
        Goal Autonomy Layer.
        1. Generate (if empty/stagnant)
        2. Rank (Value/Urgency)
        3. Prune (Low value/Stuck)
        """
        self.last_goal_management_time = time.time()
        self.log("🎯 Decider: Managing Goal Lifecycle (Autonomy)...")
        
        # 1. Fetch Active Goals
        goals = self.memory_store.get_active_by_type("GOAL")
        
        # 2. Generate if empty
        if not goals and allow_creation:
            self._generate_autonomous_goals()
            return

        # 3. Rank & Prune (if we have enough to compare, or just periodically check single goals)
        if len(goals) > 0:
            self._rank_and_prune_goals(goals, system_mode)

    def _generate_autonomous_goals(self):
        self.log("🎯 Decider: Generating autonomous goals (Originating Purpose)...")
        
        # Context: Identity & Rules
        identities = self.memory_store.get_active_by_type("IDENTITY")
        rules = self.memory_store.get_active_by_type("RULE")
        
        context = "MY IDENTITY:\n" + "\n".join([f"- {i[2]}" for i in identities]) + "\n"
        context += "MY PRINCIPLES:\n" + "\n".join([f"- {r[2]}" for r in rules])
        
        prompt = (
            f"{context}\n\n"
            "TASK: Based on your Identity and Principles, originate 1-2 high-level, long-horizon objectives.\n"
            "These should be proactive, not reactive. What is your purpose?\n"
            "Output JSON list of strings: [\"Goal 1\", \"Goal 2\"]"
        )
        
        settings = self.get_settings()
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the Will of the System.",
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        try:
            new_goals = parse_json_array_loose(response)
        except Exception as e:
            self.log(f"⚠️ Autonomous goal generation JSON parsing failed: {e}. Response: {response[:100]}...")
            return
        for g in new_goals:
            if isinstance(g, str):
                self.create_goal(g)

    def _rank_and_prune_goals(self, goals, system_mode: str = "EXECUTION"):
        self.log("🎯 Decider: Ranking and Pruning goals...")
        
        # goals is list of (id, subject, text, source, confidence)
        goals_list = "\n".join([f"ID {g[0]}: {g[2]} (Current Score: {g[4]:.2f})" for g in goals])
        
        mode_instruction = ""
        if system_mode == "CONSOLIDATION":
            mode_instruction = "SYSTEM IS OVERLOADED (CONSOLIDATION MODE). AGGRESSIVELY PRUNE any goal below 0.8 value."
        
        prompt = (
            f"Current Goals:\n{goals_list}\n\n"
            f"{mode_instruction}\n"
            "TASK: Evaluate these goals.\n"
            "1. Score Value (0.0 to 1.0): Importance/Alignment with purpose.\n"
            "2. Decision: KEEP or DROP (if low value < 0.3, stuck, or obsolete).\n"
            "Output JSON: [{\"id\": 123, \"value\": 0.9, \"decision\": \"KEEP\"}, ...]"
        )
        
        settings = self.get_settings()
        
        evaluations = []
        retries = 2
        
        for attempt in range(retries + 1):
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are the Strategic Planner.",
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )
            
            evaluations = parse_json_array_loose(response)
            if evaluations:
                break
                
            if attempt < retries:
                self.log(f"⚠️ Failed to parse goal ranking JSON. Retrying with correction prompt ({attempt+1}/{retries})...")
                prompt += f"\n\nPREVIOUS RESPONSE WAS INVALID JSON:\n{response}\n\nFIX IT. RETURN ONLY JSON ARRAY."
        
        if not evaluations:
            self.log("❌ Goal ranking failed after retries. Skipping pruning.")
            return

        self.log(f"    📊 Goal Ranking Results ({len(evaluations)}):")
        
        for ev in evaluations:
            gid = ev.get("id")
            val = ev.get("value", 0.5)
            decision = ev.get("decision", "KEEP")
            self.log(f"      - Goal {gid}: Score={val} -> {decision}")
            
            if decision == "DROP":
                self.memory_store.update_type(gid, "ARCHIVED_GOAL")
                self.log(f"🗑️ Decider: Pruned low-value goal {gid}")
                self._track_metric("goal_abandonment", 1.0)
            else:
                self.memory_store.update_confidence(gid, float(val))