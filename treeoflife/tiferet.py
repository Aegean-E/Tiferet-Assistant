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
from ai_core.lm import compute_embedding, run_local_lm, extract_memories_llm, count_tokens, LLMError, DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT
from ai_core.utils import parse_json_array_loose

# Import components
from treeoflife.tiferet_components.decision_maker import DecisionMaker
from treeoflife.tiferet_components.command_executor import CommandExecutor
from treeoflife.tiferet_components.chat_handler import ChatHandler
from treeoflife.tiferet_components.thought_generator import ThoughtGenerator
from treeoflife.tiferet_components.goal_manager import GoalManager

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
        executor=None,
        self_model=None
    ):
        self.get_settings = get_settings_fn
        self.update_settings = update_settings_fn
        self.self_model = self_model
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

        # State variables
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
        # self.last_goal_management_time is now in GoalManager
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

        # Initialize Components
        self.decision_maker = DecisionMaker(self)
        self.command_executor = CommandExecutor(self)
        self.chat_handler = ChatHandler(self)
        self.thought_generator = ThoughtGenerator(self)
        self.goal_manager = GoalManager(self)

        # Subscribe to Conscious Content
        if self.event_bus:
            self.event_bus.subscribe("CONSCIOUS_CONTENT", self._on_conscious_content, priority=10)

    # --- Properties and State Adjustments ---

    def get_system_prompt(self) -> str:
        """
        Retrieve the dynamic system prompt, injecting Identity and State.
        """
        base_prompt = self.get_settings().get("system_prompt", DEFAULT_SYSTEM_PROMPT)

        if self.yesod:
            return self.yesod.get_dynamic_system_prompt(base_prompt)
        elif self.self_model:
            # Fallback: simple injection if Yesod is missing
            try:
                projection = self.self_model.project_self()
                return f"{base_prompt}\n\nCURRENT SELF REPORT:\n{projection}"
            except Exception as e:
                self.log(f"⚠️ Failed to project self: {e}")
                return base_prompt
        return base_prompt

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

    def _track_metric(self, metric: str, value: float):
        if self.meta_learner:
            self.meta_learner.track(metric, value)

    # --- Delegated Methods ---

    def start_daydream(self):
        self.command_executor.start_daydream()

    def start_verification_batch(self):
        self.command_executor.start_verification_batch()

    def verify_all(self):
        self.command_executor.verify_all()

    def start_daydream_loop(self):
        self.command_executor.start_daydream_loop()

    def stop_daydream(self):
        self.command_executor.stop_daydream()

    def report_forced_stop(self):
        self.command_executor.report_forced_stop()

    def wake_up(self, reason: str = "External Stimulus"):
        self.command_executor.wake_up(reason)

    def create_note(self, content: str):
        return self.command_executor.create_note(content)

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
        return self.goal_manager.create_goal(content)

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

    def execute_task(self, task_name: str):
        self.command_executor.execute_task(task_name)

    def start_sleep_cycle(self):
        self.command_executor.start_sleep_cycle()

    def authorize_maintenance(self):
        self.command_executor.authorize_maintenance()

    def decide(self) -> Dict[str, Any]:
        if self.keter:
            keter_stats = self.keter.evaluate()
            coherence = keter_stats.get('keter', 1.0)
            if coherence < 0.3:
                self.log(f"🚨 Decider: Critical Coherence (< 0.3). Forcing Sleep/Recovery.")
                return {"task": "sleep", "reason": "Low Coherence Recovery", "cycles": 5}
        return self.decision_maker.decide()

    def _run_action(self, name: str, reason: str = None):
        """Delegated for internal access compatibility."""
        self.command_executor._run_action(name, reason)

    def ingest_hod_analysis(self, analysis: Dict):
        self.command_executor.ingest_hod_analysis(analysis)

    def ingest_netzach_signal(self, signal: Dict):
        self.command_executor.ingest_netzach_signal(signal)

    def on_stagnation(self, event_data: Any = None):
        self.command_executor.on_stagnation(event_data)

    def on_instability(self, event_data: Dict):
        self.command_executor.on_instability(event_data)

    def receive_observation(self, observation: str):
        self.command_executor.receive_observation(observation)

    def perform_thinking_chain(self, topic: str, max_depth: int = 10, beam_width: int = 3):
        self.thought_generator.perform_thinking_chain(topic, max_depth, beam_width)

    def perform_self_reflection(self):
        self.thought_generator.perform_self_reflection()

    def handle_natural_language_command(self, text: str, status_callback: Callable[[str], None] = None) -> Optional[str]:
        return self.chat_handler.handle_natural_language_command(text, status_callback)

    def process_chat_message(self, user_text: str, history: List[Dict], status_callback: Callable[[str], None] = None, image_path: Optional[str] = None, stop_check_fn: Callable[[], bool] = None, stream_callback: Callable[[str], None] = None) -> str:
        return self.chat_handler.process_chat_message(user_text, history, status_callback, image_path, stop_check_fn, stream_callback)

    def run_autonomous_cycle(self):
        return self.goal_manager.run_autonomous_cycle()

    def manage_goals(self, allow_creation: bool = True, system_mode: str = "EXECUTION"):
        self.goal_manager.manage_goals(allow_creation, system_mode)
