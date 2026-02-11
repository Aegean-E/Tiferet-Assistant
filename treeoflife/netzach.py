import time
import re
import random
from typing import Dict, List, Callable, Optional, Any
from lm import run_local_lm

OBSERVER_SYSTEM_PROMPT = (
    "You are Netzach, the Silent Observer and Hidden Foundation. "
    "Mission: Monitor the 'State of the World' (Chat, Reasoning, Goals, Logs) from the shadows. "
    "Status: Continuous, Low-Power. Default state is Silence. "
    "Capabilities: "
    "1. Observe: Watch for connections between user input and deep memory/documents. "
    "2. Press: Emit abstract pressure signals (Creativity, Verbosity, Stagnation) when balance is lost. "
    "3. Signal: Flag anomalies or critical context changes without acting. "
    "Persona: Mysterious, deep, essential. Do NOT speak. Do NOT act. Only signal."
)

class ContinuousObserver:
    """
    The 'Netzach' module: A constant background observer that monitors the 
    relationship between chat history, internal reasoning, and active goals.
    
    It manifests when accumulated imbalance exceeds tolerance to provide proactive value.
    """
    def __init__(
        self,
        memory_store,
        reasoning_store,
        meta_memory_store,
        get_settings_fn: Callable[[], Dict],
        get_chat_history_fn: Callable[[], List[Dict]],
        get_meta_memories_fn: Callable[[], List[Dict]],
        get_main_logs_fn: Callable[[], str],
        get_doc_logs_fn: Callable[[], str],
        get_status_fn: Callable[[], str],
        event_bus: Optional[Any] = None,
        get_recent_docs_fn: Optional[Callable[[], List]] = None,
        log_fn: Callable[[str], None] = print,
        stop_check_fn: Callable[[], bool] = lambda: False,
        check_reminders_fn: Optional[Callable[[], List[Dict]]] = None
    ):
        self.memory_store = memory_store
        self.reasoning_store = reasoning_store
        self.meta_memory_store = meta_memory_store
        self.get_settings = get_settings_fn
        self.get_chat_history = get_chat_history_fn
        self.get_meta_memories = get_meta_memories_fn
        self.get_main_logs = get_main_logs_fn
        self.get_doc_logs = get_doc_logs_fn
        self.get_status = get_status_fn
        self.get_recent_docs = get_recent_docs_fn or (lambda: [])
        self.log = log_fn
        self.event_bus = event_bus
        self.stop_check = stop_check_fn
        self.check_reminders = check_reminders_fn
        
        self.last_observation_time = 0
        self.observation_interval = 30 # Seconds between active "checks"
        self.consecutive_silence = 0

    def perform_observation(self) -> Dict:
        """
        The core observation cycle. 
        Analyzes the 'State of the World' and emits a signal (Pressure).
        Returns a dict with 'pressure' (0.0-1.0) and 'signal'.
        """
        result = {"pressure": 0.0, "signal": None, "context": None}

        if time.time() - self.last_observation_time < self.observation_interval:
            return result

        if self.stop_check():
            return result

        # 0. Temporal Awareness: Check Reminders
        if self.check_reminders:
            due_reminders = self.check_reminders()
            if due_reminders and self.event_bus:
                for reminder in due_reminders:
                    self.log(f"👁️ Netzach: ⏰ Reminder due: {reminder['text']}")
                    self.event_bus.publish(
                        "REMINDER_DUE",
                        data={"text": reminder['text'], "id": reminder['id']},
                        source="Netzach",
                        priority=10
                    )
                # After publishing, return a neutral signal as the event is the primary action
                self.last_observation_time = time.time()
                return result # Return neutral signal

        try:
            settings = self.get_settings()
            
            # 1. Gather the 'State of the World'
            history = self.get_chat_history()
            recent_reasoning = self.reasoning_store.list_recent(limit=5) # Get latest thoughts
            active_goals = self.memory_store.get_active_by_type("GOAL")
            meta_memories = self.get_meta_memories()
            main_logs = self.get_main_logs()
            doc_logs = self.get_doc_logs()
            status = self.get_status()
            recent_docs = self.get_recent_docs()
            
            if not history and not recent_reasoning:
                return result # Nothing to observe yet

            # 2. Construct the Observation Context
            context = "--- OBSERVATION CONTEXT ---\n"

            if history:
                context += "\nRecent Conversation:\n"
                for msg in history[-5:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context += f"- {role}: {msg['content']}\n"
            
            if recent_reasoning:
                context += "\nInternal Reasoning/Hypotheses:\n"
                for r in recent_reasoning:
                    context += f"- {r['content']}\n"
            
            if active_goals:
                context += "\nCurrent Active Goals:\n"
                for _, s, g, _ in active_goals[:3]:
                    context += f"- {g}\n"

            if meta_memories:
                context += "\nRecent Memory Events (Meta-Memory):\n"
                for m in meta_memories[:5]:
                    context += f"- {m[1]} ([{m[2]}]): {m[3][:100]}...\n"

            if main_logs:
                # Filter out repetitive system actions to prevent feedback loops
                filtered_logs = []
                for line in main_logs.splitlines():
                    if any(x in line for x in ["Adjusted max_tokens", "Increase tokens", "PRUNE_MEM", "REFUTE_MEM", "Action: [", "Action:[", "Hod Output:", "Executed action: run_hod"]):
                        continue
                    filtered_logs.append(line)
                context += f"\nRecent System Logs:\n" + "\n".join(filtered_logs[-15:]) + "\n"

            if doc_logs:
                context += f"\nRecent Document Processing Logs:\n{doc_logs[-400:]}\n"

            if recent_docs:
                context += "\nRecently Added Documents:\n"
                for d in recent_docs:
                    # d is (id, filename, type, ...)
                    context += f"- {d[1]} ({d[2]})\n"

            context += f"\nCurrent System Status:\n{status}\n"
            
            current_temp = float(settings.get("temperature", 0.7))
            context += f"Current Temperature: {current_temp}\n"

            max_step = 0.20

            # 3. Decision: Should I manifest?
            decision_prompt = (
                "Review the Observation Context. "
                "Assess the system state and emit a PRESSURE SIGNAL if imbalance exists:\n"
                "1. [OBSERVE]: System is balanced. (Default)\n"
                "2. [SIGNAL: NO_MOMENTUM]: System is idle/stuck. (Pressure to Act)\n"
                "3. [SIGNAL: LOW_NOVELTY]: System is rigid, repetitive, or looping. (Pressure to Expand)\n"
                "4. [SIGNAL: HIGH_CONSTRAINT]: Outputs are too short or cutoff. (Pressure to Release)\n"
                "5. [SIGNAL: DISSONANCE]: Anomalies, errors, or hallucinations detected. (Pressure to Correct)\n"
                "6. [SIGNAL: CONTEXT_PRESSURE]: Context is full or session is long. (Pressure to Compress)\n"
                "7. [SIGNAL: EXTERNAL_PRESSURE]: Critical external state change (e.g. 'User waiting').\n\n"
                "Output ONLY the signal tag. If EXTERNAL_PRESSURE, provide reason: [SIGNAL: EXTERNAL_PRESSURE] Reason: ..."
            )
            
            messages = [{"role": "user", "content": context}]
            
            response = run_local_lm(
                messages,
                system_prompt=decision_prompt,
                temperature=0.3, # Low temp for the decision
                max_tokens=10,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )
            
            # Check for LLM error
            if response.startswith("⚠️"):
                self.log(f"❌ Observer generation failed: {response}")
                return result

            response_upper = response.upper()
            if "[SIGNAL: NO_MOMENTUM]" in response_upper:
                self.log("👁️ Netzach: Signaling No Momentum (High Pressure).")
                result["pressure"] = 1.0
                result["signal"] = "NO_MOMENTUM"
                self.consecutive_silence = 0
            elif "[SIGNAL: LOW_NOVELTY]" in response_upper:
                self.log(f"👁️ Netzach: Signaling Low Novelty.")
                result["signal"] = "LOW_NOVELTY"
                result["pressure"] = 0.6
                self.consecutive_silence = 0
            elif "[SIGNAL: HIGH_CONSTRAINT]" in response_upper:
                self.log(f"👁️ Netzach: Signaling High Constraint.")
                result["signal"] = "HIGH_CONSTRAINT"
                result["pressure"] = 0.6
                self.consecutive_silence = 0
            elif "[SIGNAL: DISSONANCE]" in response_upper:
                result["signal"] = "DISSONANCE"
                result["pressure"] = 0.7
                self.consecutive_silence = 0
            elif "[SIGNAL: CONTEXT_PRESSURE]" in response_upper:
                result["signal"] = "CONTEXT_PRESSURE"
                result["pressure"] = 0.5
                self.consecutive_silence = 0
            elif "[SIGNAL: EXTERNAL_PRESSURE]" in response_upper:
                result["signal"] = "EXTERNAL_PRESSURE"
                result["pressure"] = 0.8
                if "Reason:" in response:
                    reason = response.split("Reason:", 1)[1].strip()
                    result["context"] = {"reason": reason}
                self.consecutive_silence = 0
            else:
                self.consecutive_silence += 1
                msg = "Remaining in slow observation..."
                self.log(f"👁️ Netzach: {msg}")
                
                if self.consecutive_silence >= 5:
                    self.log("👁️ Netzach: Stagnation detected (Timeout).")
                    result["pressure"] = 0.8
                    result["signal"] = "NO_MOMENTUM"
                    self.consecutive_silence = 0

        except Exception as e:
            self.log(f"❌ Observation error: {e}")
        finally:
            # Ensure we always update the timer to prevent rapid-fire loops on error
            self.last_observation_time = time.time()
            
        return result