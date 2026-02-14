import time
import logging
from typing import Callable, Optional, Dict, Any

class Heartbeat:
    """
    The Pulse of the System.
    Manages the cognitive cycle, scheduling, and task duration.
    Decouples execution timing from decision logic.
    """
    def __init__(self, core, log_fn: Callable[[str], None] = logging.info):
        self.core = core
        self.log = log_fn
        
        # Scheduler State
        self.current_task = "wait"
        self.cycles_remaining = 0
        self.task_reason = "System Startup"
        self.wait_start_time = 0
        
        # External Control
        self.stop_requested = False

    def tick(self):
        """
        Single heartbeat cycle.
        1. Observe (Netzach)
        2. Execute Task (if active)
        3. Decide (if idle) (Tiferet)
        4. Reflect (Hod)
        """
        if self.stop_requested:
            return

        # 1. Netzach (Observer)
        # Always runs to detect anomalies or user state changes
        signal = None
        if self.core.netzach_force:
            try:
                signal = self.core.netzach_force.observe()
                if self.core.decider and signal:
                    self.core.decider.ingest_netzach_signal(signal)
            except Exception as e:
                self.log(f"⚠️ Heartbeat: Netzach error: {e}")

        if self.stop_requested:
            return

        # 2. Execute Planned Task
        if self.cycles_remaining > 0:
            try:
                self._execute_current_task()
                
                # Task Completion Trigger
                if self.cycles_remaining == 0 and self.current_task != "wait":
                    if self.core.hod_force:
                        self.core.hod_force.reflect(f"Finished {self.current_task}")
                    
                    # 5. Maintenance
                    if self.core.decider:
                        self.core.decider.authorize_maintenance()
            except Exception as e:
                self.log(f"⚠️ Heartbeat: Execution error: {e}")
                self.cycles_remaining = 0 # Abort task
        
        # 3. If plan finished (or idle), decide next steps
        if self.cycles_remaining <= 0:
            # If we were waiting, check autonomy first (Intrinsic Motivation)
            if self.current_task == "wait":
                try:
                    self.core.run_autonomous_agency_check(signal)
                except Exception as e:
                    self.log(f"⚠️ Heartbeat: Autonomy error: {e}")

            try:
                self._solicit_decision()
            except Exception as e:
                self.log(f"⚠️ Heartbeat: Decision error: {e}")
                self.current_task = "wait"
                self.cycles_remaining = 5 # Backoff

    def _execute_current_task(self):
        if not self.core.decider: return
        self.core.decider.execute_task(self.current_task)
        self.cycles_remaining -= 1
        self.log(f"💓 Heartbeat: {self.current_task.capitalize()} cycle complete. Remaining: {self.cycles_remaining}")

    def _solicit_decision(self):
        if not self.core.decider: return
        
        # Ask Tiferet what to do
        decision = self.core.decider.decide()
        
        self.current_task = decision.get("task", "wait")
        self.cycles_remaining = decision.get("cycles", 0)
        self.task_reason = decision.get("reason", "Decider choice")
        
        if self.current_task == "wait":
            self.wait_start_time = time.time()
            # Enforce minimum wait to prevent hot loop if Decider returns 0 cycles
            if self.cycles_remaining <= 0:
                self.cycles_remaining = 1
        else:
            self.log(f"💓 Heartbeat: New Task -> {self.current_task} ({self.cycles_remaining} cycles). Reason: {self.task_reason}")

    def force_task(self, task: str, cycles: int, reason: str):
        """External override (e.g. from Chat or UI)."""
        self.current_task = task
        self.cycles_remaining = cycles
        self.task_reason = reason
        self.log(f"💓 Heartbeat: Forced Task -> {task} ({cycles} cycles). Reason: {reason}")

    def stop(self):
        self.stop_requested = True
        self.cycles_remaining = 0
        self.current_task = "wait"
        self.log("💓 Heartbeat: Stopped.")

    def resume(self):
        self.stop_requested = False
        self.log("💓 Heartbeat: Resumed.")