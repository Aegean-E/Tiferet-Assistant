import time
import json
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from ai_core.lm import run_local_lm
from ai_core.utils import parse_json_array_loose

if TYPE_CHECKING:
    from treeoflife.tiferet import Decider

class CommandExecutor:
    def __init__(self, decider: 'Decider'):
        self.decider = decider

    def execute_task(self, task_name: str):
        """Execute one unit of the assigned task."""
        if task_name == "daydream":
            # Check concurrency setting
            settings = self.decider.get_settings()
            concurrency = int(settings.get("concurrency", 1))

            # Determine batch size (don't exceed remaining cycles or concurrency limit)
            # Note: Heartbeat handles decrementing cycles, so we just need to know if we CAN batch
            # But Heartbeat calls this once per tick. If we batch, we need to tell Heartbeat we did more work.
            # For simplicity, let's stick to single execution per tick for now, or handle batching internally.
            # If we want batching, we should probably do it here and return how many were done.
            # But Heartbeat expects 1 tick = 1 cycle decrement.
            # Let's just do single for now to keep Heartbeat simple.
            self._run_action("start_daydream", reason=self.decider.heartbeat.task_reason if self.decider.heartbeat else "Daydream")
            self.decider.last_daydream_time = time.time()

        elif task_name == "verify":
            self._run_action("verify_batch", reason=self.decider.heartbeat.task_reason if self.decider.heartbeat else "Verify")

        elif task_name == "verify_all":
            self._run_action("verify_all", reason=self.decider.heartbeat.task_reason if self.decider.heartbeat else "Verify All")

        elif task_name == "sleep":
            self.decider.log("💤 Decider: Sleeping... (Deep Consolidation & Synthesis)")
            # Run heavy maintenance tasks that are usually too expensive
            if self.decider.binah:
                self.decider.log("💤 Binah: Running deep consolidation...")
                self.decider.binah.consolidate(time_window_hours=None)
            if self.decider.daat:
                self.decider.log("💤 Da'at: Running clustering and synthesis...")
                self.decider.daat.run_clustering()
                self.decider.daat.run_synthesis()

    def _run_action(self, name: str, reason: str = None):
        # 1. Capture Pre-Action State (Trigger State)
        start_state = {}
        start_coherence = 0.0
        start_utility = 0.0

        # Capture prediction made during planning (Fix 5: Prediction Error)
        predicted_delta = self.decider.last_predicted_delta
        action_result_data = {}

        if self.decider.keter:
            # Evaluate Keter to get current baseline (Raw score is better for immediate delta)
            keter_stats = self.decider.keter.evaluate()
            start_coherence = keter_stats.get("raw", keter_stats.get("keter", 0.0))
            start_utility = self.decider.decision_maker.calculate_utility()

            start_state = {
                "hesed": self.decider.hesed.calculate() if self.decider.hesed else 0,
                "gevurah": self.decider.gevurah.calculate() if self.decider.gevurah else 0,
                "active_goals": len(self.decider.memory_store.get_active_by_type("GOAL")),
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
            if name in ["GOAL_ACT", "EXECUTE"] and self.decider.malkuth and (time.time() - self.decider.last_simulation_time > 60):
                # Heuristic: Only simulate if we haven't just simulated (avoid loops)
                # and if the action isn't a simple tool call
                self.decider.last_simulation_time = time.time()
                sim_context = f"State: H={start_state['hesed']:.2f}, G={start_state['gevurah']:.2f}, Coh={start_coherence:.2f}"
                sim_result = self.decider.malkuth.simulate_futures(name, sim_context)

                expected_util = sim_result.get("expected_utility", 0.0)
                if expected_util < -0.3:
                    self.decider.log(f"🛑 Decider: Action '{name}' aborted due to negative simulated utility ({expected_util:.2f}).")
                    # Record the "near miss"
                    if hasattr(self.decider.meta_memory_store, 'add_event'):
                        self.decider.meta_memory_store.add_event("ACTION_ABORTED", "Assistant", f"Aborted {name} due to bad simulation: {sim_result.get('futures', [])}")
                    return # Abort execution
                self.decider.log(f"🔮 Simulation passed (Util: {expected_util:.2f}). Proceeding.")

        # 2. Execute Action
        if name == "run_hod":
            if self.decider.hod_just_ran:
                self.decider.log("⚠️ Decider: Skipping Hod analysis to prevent loops.")
                return

            if name in self.decider.actions:
                self.decider.actions[name]()
                self.decider.hod_just_ran = True
            else:
                self.decider.log(f"⚠️ Decider action '{name}' not available.")
        else:
            if name in self.decider.actions:
                result = self.decider.actions[name]()
                # Reset Hod lock for substantive actions
                if name in ["start_daydream", "verify_batch", "verify_all", "start_loop", "run_observer"]:
                    self.decider.hod_just_ran = False

                if name == "run_observer" and result:
                    self.ingest_netzach_signal(result)
                if isinstance(result, dict):
                    action_result_data = result
            else:
                self.decider.log(f"⚠️ Decider action '{name}' not available.")

        # 3. Measure Result & Record Outcome (Credit Assignment)
        if self.decider.keter and hasattr(self.decider.meta_memory_store, 'add_outcome'):
            # Re-evaluate Keter to see change
            end_stats = self.decider.keter.evaluate()
            end_coherence = end_stats.get("raw", end_stats.get("keter", 0.0))
            end_utility = self.decider.decision_maker.calculate_utility()

            delta_coherence = end_coherence - start_coherence
            delta_utility = end_utility - start_utility

            # Calculate Prediction Error
            prediction_error = abs(predicted_delta - delta_coherence)

            # Affective Update (Dopamine/Cortisol)
            # Positive utility = Dopamine
            # Negative utility or High Prediction Error = Cortisol
            mood_impact = delta_utility - (prediction_error * 0.5)
            self.decider._update_mood(mood_impact)

            # Surprise Event (High Prediction Error)
            if prediction_error > 0.2:
                self.decider.log(f"😲 Decider: Surprise Event! Prediction Error {prediction_error:.2f}. Triggering Reflection.")
                # Inject Dissonance Signal directly for immediate awareness
                self.ingest_netzach_signal({
                    "signal": "DISSONANCE",
                    "pressure": 1.0,
                    "context": f"High Prediction Error ({prediction_error:.2f}) on {name}"
                })

                if hasattr(self.decider.meta_memory_store, 'add_event'):
                    self.decider.meta_memory_store.add_event("SURPRISE_EVENT", "Assistant", f"Action {name} had unexpected outcome (Err: {prediction_error:.2f})")
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
                "epigenetics": self.decider.epigenetics,
                "hesed_bias": self.decider.hesed.bias if self.decider.hesed else 1.0,
                "gevurah_bias": self.decider.gevurah.bias if self.decider.gevurah else 1.0,
                "system_prompt_fitness": self.decider.get_settings().get("system_prompt_fitness", 0.0)
            }

            self.decider.meta_memory_store.add_outcome(
                action=name,
                trigger_state=start_state,
                result=outcome_result,
                context_metadata=context_metadata
            )

        if hasattr(self.decider.meta_memory_store, 'add_event') and name != "run_observer":
            narrative = f"I executed {name}"
            if reason:
                narrative += f" because {reason}"
            else:
                narrative += "."

            self.decider.meta_memory_store.add_event(
                event_type="DECIDER_ACTION",
                subject="Assistant",
                text=narrative
            )

    def _execute_tool(self, tool_name: str, args: str):
        """Execute a tool safely and store the result."""
        # Rate limiting (2 seconds between tool calls)
        if time.time() - self.decider.last_tool_usage < 2.0:
            self.decider.log(f"⚠️ Tool rate limit exceeded for {tool_name}")
            return "Error: Tool rate limit exceeded. Please wait."
        self.decider.last_tool_usage = time.time()

        # Integrated Budgeting: Deduct cost
        if self.decider.crs:
            cost = self.decider.crs.estimate_action_cost(tool_name, complexity=0.5)
            self.decider.crs.current_spend += cost

        self.decider.log(f"🛠️ Decider executing tool: {tool_name} args: {args}")
        result = ""

        # Delegate to ActionManager tools if available
        if tool_name in self.decider.actions:
             try:
                 result = self.decider.actions[tool_name](args)
             except Exception as e:
                 result = f"Error executing {tool_name}: {e}"
        else:
            result = f"Tool {tool_name} not found."

        self.decider.log(f"🛠️ Tool Result: {result}")

        # Store result in reasoning so the AI knows what happened
        self.decider.reasoning_store.add(
            content=f"Tool Execution [{tool_name}]: {args} -> Result: {result}",
            source="tool_output",
            confidence=1.0
        )

        # Add to Meta-Memory
        if hasattr(self.decider.meta_memory_store, 'add_event'):
            self.decider.meta_memory_store.add_event("TOOL_EXECUTION", "Assistant", f"Executed {tool_name} ({args}) -> {result}")

        if self.decider.chat_fn:
             self.decider.chat_fn("Decider", f"🛠️ Used {tool_name}: {result}")

        return result

    def _execute_search(self, query: str, source: str):
        """Helper to execute search action."""
        if "search_internet" in self.decider.actions:
            result = self.decider.actions["search_internet"](query, source)
            self.decider.reasoning_store.add(content=f"Internet Search [{source}]: {query}\nResult: {result}", source="internet_bridge", confidence=1.0)
            self.decider.log(f"🌐 Internet Search Result: {result[:100]}...")
        else:
            self.decider.log("⚠️ Action search_internet not available.")

    def receive_observation(self, observation: str):
        """
        Receive an observation/information from Netzach.
        This information MUST result in an action.
        """
        if not observation:
            return

        self.decider.log(f"📩 Decider received observation: {observation}")

        if hasattr(self.decider.meta_memory_store, 'add_event'):
            self.decider.meta_memory_store.add_event(
                event_type="DECIDER_OBSERVATION_RECEIVED",
                subject="Assistant",
                text=f"Received observation from Netzach: {observation}"
            )

        text = observation.lower()
        self.decider.action_taken_in_observation = True

        # Map observation content to actions
        if any(w in text for w in ["loop", "cycle", "continue"]):
            self.decider.daydream_mode = "auto"
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task("daydream", 3, "Netzach loop signal")
        elif any(w in text for w in ["stop", "halt", "pause"]) and "daydream" in text:
            self._run_action("stop_daydream", reason="Netzach stop signal")
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task("wait", 0, "Netzach stop signal")
        elif any(w in text for w in ["decrease", "lower", "reduce", "drop"]) and any(w in text for w in ["temp", "temperature"]):
            self.decider.decrease_temperature()
        elif any(w in text for w in ["decrease", "lower", "reduce", "drop"]) and any(w in text for w in ["token", "tokens", "length"]):
            self.decider.decrease_tokens()
        elif any(w in text for w in ["increase", "raise", "boost", "up", "higher"]) and any(w in text for w in ["token", "tokens", "length"]):
            # Debounce token increases (prevent loops)
            if time.time() - self.decider.last_token_adjustment_time > 60:
                self.decider.increase_tokens()
            else:
                self.decider.log("⚠️ Decider ignoring token increase request (cooldown active).")
        elif any(w in text for w in ["verify", "conflict", "contradiction", "error", "inconsistent", "wrong"]):
            task = "verify"
            cycles = 2
            if "all" in text or "full" in text:
                task = "verify_all"
                cycles = 1
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task(task, cycles, "Netzach conflict signal")
        elif any(w in text for w in ["hod", "analyze", "analysis", "investigate", "pattern", "reflect", "refuted", "refutation"]):
            self._run_action("run_hod", reason="Netzach analysis signal")
        elif "observer" in text or "watch" in text:
            self._run_action("run_observer", reason="Netzach watch signal")
        elif any(w in text for w in ["stagnant", "idle", "bored", "nothing", "quiet", "daydream", "think", "create"]):
            self.decider.daydream_mode = "auto"
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task("daydream", 1, "Netzach boredom signal")
        else:
            # Slow Path (Semantic Interpretation) - The "Wisdom" Check
            self.decider.log(f"🤔 Decider: Interpreting complex observation...")

            settings = self.decider.get_settings()
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
                    s = self.decider.get_settings()
                    s["temperature"] = val
                    self.decider.update_settings(s)
                    self.decider.log(f"🌡️ Decider set temperature to {val}")
                except:
                    self.decider.log("⚠️ Failed to parse temp change.")
            elif "[GOAL" in decision:
                new_goal = decision.split("[GOAL:", 1)[1].strip("] ")
                self.decider.goal_manager.create_goal(new_goal)
                self.decider.log(f"🎯 Decider: Adapted strategy. New Goal: {new_goal}")
            elif "[REFLECT]" in decision:
                self.decider.reasoning_store.add(content="I need to reflect on the system state.", source="decider", confidence=1.0)
                self._run_action("run_hod", reason="Netzach complex signal")
            else:
                self.decider.log(f"⚠️ Decider could not map observation (LLM result: {decision}). No action taken.")

    def authorize_maintenance(self):
        """Public wrapper for maintenance authorization."""
        self._authorize_maintenance()

    def _authorize_maintenance(self):
        """Authorize and dispatch queued maintenance actions (Prune/Refute)."""
        with self.decider.maintenance_lock:
            if not self.decider.pending_maintenance: return
            actions = list(self.decider.pending_maintenance)
            self.decider.pending_maintenance = []

        self.decider.log(f"🤖 Decider: Authorizing {len(actions)} maintenance actions.")
        for action in actions:
            if action["type"] == "PRUNE":
                if "prune_memory" in self.decider.actions:
                    self.decider.actions["prune_memory"](action["id"])
            elif action["type"] == "REFUTE":
                if "refute_memory" in self.decider.actions:
                    self.decider.actions["refute_memory"](action["id"], action.get("reason"))

    def ingest_hod_analysis(self, analysis: Dict):
        """Receive and process analysis from Hod."""
        if not analysis: return

        # 1. Log Findings (Persistence)
        findings = analysis.get("findings")
        if findings:
            self.decider.log(f"🔮 Hod Findings: {findings}")
            if self.decider.chat_fn:
                self.decider.chat_fn("Hod", f"🔮 Analysis: {findings}")
            # Decider decides to persist this analysis
            self.decider.reasoning_store.add(
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
                self.decider.decrease_temperature() # Tiferet decides magnitude
            elif otype == "CONTEXT_OVERLOAD":
                self.decider.decrease_tokens()
            elif otype == "LOGIC_VIOLATION" or otype == "SELF_CORRECTION_SIGNAL":
                # Map to maintenance action
                if "id" in observation:
                    with self.decider.maintenance_lock:
                        self.decider.pending_maintenance.append({"type": "REFUTE", "id": observation["id"], "reason": observation.get("context")})
            elif otype == "INVALID_DATA":
                if "id" in observation:
                    with self.decider.maintenance_lock:
                        self.decider.pending_maintenance.append({"type": "PRUNE", "id": observation["id"], "reason": observation.get("context")})
            elif otype == "CONTEXT_FRAGMENTATION":
                self._run_action("summarize")
            elif otype == "CRITICAL_INFO":
                self.receive_observation(observation.get("context"))

    def ingest_netzach_signal(self, signal: Dict):
        """Receive signal from Netzach."""
        if not signal: return

        # Prevent overwriting high pressure with low pressure (noise)
        current_pressure = self.decider.latest_netzach_signal.get("pressure", 0) if self.decider.latest_netzach_signal else 0
        new_pressure = signal.get("pressure", 0)

        if new_pressure >= current_pressure:
            self.decider.latest_netzach_signal = signal
            # Immediate reaction to high pressure if waiting
            if new_pressure > 0.7 and (self.decider.heartbeat and self.decider.heartbeat.current_task == "wait"):
                self.wake_up("Netzach Pressure")

    def _check_netzach_signal(self):
        """Check latest Netzach signal during planning."""
        if not self.decider.latest_netzach_signal: return

        sig = self.decider.latest_netzach_signal.get("signal")
        context = self.decider.latest_netzach_signal.get("context")

        if sig == "LOW_NOVELTY":
            # If bored, don't just increase temp. DO something.
            self.decider.log("👁️ Netzach signaled LOW_NOVELTY. Triggering Daydream.")
            self.decider.increase_temperature()
            if self.decider.heartbeat:
                self.decider.heartbeat.force_task("daydream", 1, "Netzach Low Novelty")
            self.decider.daydream_mode = "insight" # Force insight generation
        elif sig == "HIGH_CONSTRAINT":
            self.decider.increase_tokens() # Decider chooses amount
        elif sig == "EXTERNAL_PRESSURE":
            if context and "reason" in context:
                self.receive_observation(context["reason"])
        elif sig == "CONTEXT_PRESSURE":
            self.decider.log("👁️ Netzach requested summary. Decider authorizing Da'at...")
            self._run_action("summarize")
        elif sig == "DISSONANCE":
             self._run_action("run_hod")
        # elif sig == "NO_MOMENTUM":
        #      self.current_task_reason = "Netzach Momentum Signal"
        #      self.start_daydream() # Removed: Let _decide_next_batch handle this via Goal Act or Daydream

        # Clear signal after processing
        self.decider.latest_netzach_signal = None

    def on_stagnation(self, event_data: Any = None):
        """Handle system stagnation event (usually from Netzach)."""
        self.decider.log("🤖 Decider: Stagnation detected. Waking up.")
        self.start_daydream()

    def on_instability(self, event_data: Dict):
        """Handle system instability event (usually from Hod)."""
        self.decider.log(f"🤖 Decider: Instability reported ({event_data.get('reason')}). Switching to verification.")
        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("verify", 1, f"Instability: {event_data.get('reason')}")

    def start_daydream(self):
        self.decider.log("🤖 Decider starting single Daydream cycle.")
        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("daydream", 1, "Manual Command")
        if time.time() - self.decider.last_daydream_time < 60:
            self.decider.daydream_mode = "insight"
        else:
            self.decider.daydream_mode = "auto"

    def start_verification_batch(self):
        self.decider.log("🤖 Decider starting Verification Batch.")
        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("verify", 1, "Manual Command")

    def verify_all(self):
        self.decider.log("🤖 Decider starting Full Verification.")
        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("verify_all", 1, "Manual Command")

    def start_daydream_loop(self):
        self.decider.log("🤖 Decider enabling Daydream Loop.")
        self._run_action("start_loop", reason="User command")
        settings = self.decider.get_settings()
        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("daydream", int(settings.get("daydream_cycle_limit", 10)), "User Loop Command")
        self.decider.last_daydream_time = time.time()

    def stop_daydream(self):
        self.decider.log("🤖 Decider stopping daydream.")
        self._run_action("stop_daydream", reason="User command or Stop signal")

    def report_forced_stop(self):
        """Handle forced stop from UI."""
        self.decider.log("🤖 Decider: Forced stop received. Entering cooldown.")
        if self.decider.heartbeat:
            self.decider.heartbeat.stop()
        self.decider.forced_stop_cooldown = True

    def wake_up(self, reason: str = "External Stimulus"):
        """Force the Decider to wake up from a wait state."""
        self.decider.is_sleeping = False
        self.decider.log(f"🤖 Decider: Waking up due to {reason}.")
        if self.decider.heartbeat:
            self.decider.heartbeat.force_task("organizing", 0, reason)

    def create_note(self, content: str):
        """Manually create an Assistant Note (NOTE)."""
        if self.decider.malkuth:
            return self.decider.malkuth.create_note(content)
        self.decider.log("⚠️ Malkuth not available for creating note.")
