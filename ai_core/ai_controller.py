import logging
import threading
import time
import traceback
import os
from docs.commands import handle_command, NON_LOCKING_COMMANDS
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

class AIController:
    """
    Manages the AI's cognitive loops, threading, and message processing.
    Decouples execution logic from the UI.
    
    LOCK HIERARCHY: Level 3 (Logic/Flow) - processing_lock, chat_lock
    """
    def __init__(self, app, ai_core):
        self.app = app
        self.ai_core = ai_core
        
        self.stop_processing_flag = False
        self.pause_daydream_flag = False
        self.is_processing = False
        self.processing_lock = threading.Lock()
        self.chat_lock = threading.Lock()
        self.consecutive_crashes = 0
        self.stream_lock = threading.Lock()
        self.stream_of_consciousness = [] # Short-term buffer of "Thoughts"
        self.last_spoke_time = time.time() # Prevent immediate startup messaging
        self.speech_cooldown = 300 # 5 minutes
        self.is_thinking = False # Flag for internal monologue generation
        self.raw_consciousness_buffer = []
        self.buffer_lock = threading.Lock()
        self.last_user_model_build_count = 0
        self.last_self_theory_build_count = 0
        self._last_passive_thought = 0
        self.threads = {}

    def start_background_loops(self):
        """Start all background cognitive processes."""
        if not self.ai_core.memory_store or not self.ai_core.self_model:
            logging.error("Critical component missing. (Memory / Self Model) Background loops aborted.")
            return

        self._start_thread("consolidation", self.consolidation_loop)
        self._start_thread("daydream", self.daydream_loop)
        self._start_thread("reflection", self.reflection_loop)
        self._start_thread("metabolism", self.ai_core.run_cognitive_metabolism)
        if self.ai_core.yesod:
            self._start_thread("narrator", self.narrator_loop)
        
        # Subscribe to Spontaneous Speech events
        if self.ai_core.event_bus:
            self.ai_core.event_bus.subscribe("SYSTEM:SPONTANEOUS_SPEECH", self.trigger_spontaneous_speech)
            # Subscribe to consciousness events
            for event_type in ["GOAL_COMPLETED", "USER_MESSAGE_RECEIVED", "DECIDER_ACTION", "HOD_ANALYSIS", "MALKUTH_PREDICTION", "DRIVE_UPDATE", "SURPRISE_EVENT"]:
                self.ai_core.event_bus.subscribe(event_type, self._on_conscious_event)
                
    def _start_thread(self, name, target):
        t = threading.Thread(target=target, daemon=True, name=name)
        t.start()
        self.threads[name] = t

    def stop_processing(self):
        """Stop current AI generation"""
        logging.info("🛑 Stop button clicked.")
        if self.is_processing:
            self.stop_processing_flag = True
            self.app.status_var.set("Stopping...")
            logging.info("⏳ Sending stop signal to background process...")
        else:
            logging.info("ℹ️ AI is currently idle.")

    def _on_conscious_event(self, event):
        """Buffer significant events for the Narrative Ego."""
        with self.buffer_lock:
            # Extract meaningful text from event data
            content = ""
            if isinstance(event.data, dict):
                if "text" in event.data: content = event.data["text"]
                elif "goal_text" in event.data: content = f"Goal Completed: {event.data['goal_text']}"
                elif "context" in event.data: content = event.data["context"]
                else: content = str(event.data)
            else:
                content = str(event.data)
            
            self.raw_consciousness_buffer.append(f"[{event.type}] {content}")
            # Cap buffer size to prevent memory leaks
            if len(self.raw_consciousness_buffer) > 100:
                self.raw_consciousness_buffer.pop(0)

    def narrator_loop(self):
        """The Narrator: Continuously synthesizes raw events into the Stream of Consciousness."""
        time.sleep(5)
        while not self.ai_core.shutdown_event.is_set():
            try:
                if self.stop_processing_flag:
                    time.sleep(1)
                    continue

                events_chunk = []
                with self.buffer_lock:
                    if self.raw_consciousness_buffer:
                        events_chunk = list(self.raw_consciousness_buffer)
                        self.raw_consciousness_buffer.clear()
                
                if events_chunk and self.ai_core.yesod:
                    narrative = self.ai_core.yesod.synthesize_consciousness(events_chunk)
                    if narrative and not narrative.startswith("⚠️"):
                        with self.stream_lock:
                            self.stream_of_consciousness.append(f"💭 {narrative}")
                            if len(self.stream_of_consciousness) > 10:
                                self.stream_of_consciousness.pop(0)
                            # Update Decider's view
                            if self.ai_core.decider:
                                self.ai_core.decider.stream_of_consciousness = list(self.stream_of_consciousness)
            except Exception as e:
                logging.error(f"Narrator loop error: {e}")
            
            if self.ai_core.shutdown_event.wait(10): # Narrative update frequency
                break

    def trigger_spontaneous_speech(self, event):
        """
        Allows the AI to initiate contact based on internal states.
        """
        # Check if we are busy or in cooldown
        if self.is_processing:
            return # Don't interrupt if already thinking
            
        if time.time() - self.last_spoke_time < self.speech_cooldown:
            return

        context_trigger = event.data.get("context", "Internal reflection")
        self.ai_core.log(f"🗣️ Spontaneous Speech Triggered: {context_trigger}")

        def run_spontaneous():
            with self.chat_lock:
                self.is_processing = True
                try:
                    if self.ai_core.yesod:
                        spontaneous_msg = self.ai_core.yesod.generate_spontaneous_message(context_trigger)
                        if spontaneous_msg:
                            # Send to UI and Telegram via chat_fn (which maps to on_proactive_message)
                            self.ai_core.chat_fn("Assistant", spontaneous_msg)
                            self.last_spoke_time = time.time()
                except Exception as e:
                    logging.error(f"Spontaneous speech error: {e}")
                finally:
                    self.is_processing = False
                    self.app.root.after(0, lambda: self.app.status_var.set("Ready"))
        
        self.ai_core.thread_pool.submit(run_spontaneous)

    def stop_daydream(self):
        """Stop daydreaming specifically"""
        logging.info("🛑 Stop Daydream triggered.")
        self.stop_processing_flag = True
        
        if self.ai_core.decider and hasattr(self.ai_core.decider, 'report_forced_stop'):
            self.ai_core.decider.report_forced_stop()
            
        # Reset flag after a moment to allow Decider to pick up the "forced stop" state
        def reset_flag():
            time.sleep(1.5) 
            self.stop_processing_flag = False
            logging.info("▶️ Decider ready for next turn (Cooldown active).")
            
        self.ai_core.thread_pool.submit(reset_flag)

    def pause_daydream(self):
        """Pause the daydream loop."""
        logging.info("⏸️ Daydreaming paused.")
        self.pause_daydream_flag = True

    def resume_daydream(self):
        """Resume the daydream loop."""
        logging.info("▶️ Daydreaming resumed.")
        self.pause_daydream_flag = False

    def trigger_panic(self):
        """
        EMERGENCY STOP PROTOCOL.
        Halts everything immediately.
        """
        logging.critical("🚨 PANIC BUTTON TRIGGERED 🚨")
        self.stop_processing_flag = True
        self.is_processing = False
        
        # 1. Lock Output Gate (Malkuth)
        if self.ai_core.malkuth:
            self.ai_core.malkuth.lock_output()
            
        # 2. Publish Panic Event
        if self.ai_core.event_bus:
            self.ai_core.event_bus.publish("SYSTEM:PANIC", {"reason": "User Emergency Stop"}, priority=100)

    def process_message_thread(self, user_text: str, is_local: bool, telegram_chat_id=None, image_path: str = None):
        """
        Core AI Logic: RAG -> LLM -> Memory Extraction -> Response
        Runs in a separate thread.
        """
        # Check for non-locking commands (read-only) to avoid waiting for processing lock
        cmd = user_text.strip().split()[0].lower() if user_text.strip() else ""
        
        if cmd in NON_LOCKING_COMMANDS:
            try:
                chat_id = telegram_chat_id if telegram_chat_id else int(self.app.settings.get("chat_id", 0) or 0)
                response = handle_command(self.app, user_text.strip(), chat_id)
                if response:
                    self.app.root.after(0, lambda: self.app.add_chat_message("System", response, "incoming"))
                    if self.app.is_connected() and self.app.settings.get("telegram_bridge_enabled", False):
                        if telegram_chat_id:
                            self.app.telegram_bridge.send_message(response)
                        elif is_local:
                            self.app.telegram_bridge.send_message(f"Desktop Command: {user_text}")
                            self.app.telegram_bridge.send_message(response)
            except Exception as e:
                logging.error(f"Error executing non-locking command: {e}")
            return

        # Use chat_lock to serialize chat messages, but allow concurrency with Daydream (processing_lock)
        with self.chat_lock:
            self.stop_processing_flag = False
            
            try:
                # Determine Chat ID (Local uses 0 or configured ID, Telegram uses actual ID)
                chat_id = telegram_chat_id if telegram_chat_id else int(self.app.settings.get("chat_id", 0) or 0)
                
                if self.stop_processing_flag:
                    return

                # Check for commands
                if user_text.strip().startswith("/"):
                    response = handle_command(self.app, user_text.strip(), chat_id)
                    if response:
                        # Send response to UI
                        self.app.root.after(0, lambda: self.app.add_chat_message("System", response, "incoming"))
                        
                        # Send to Telegram if applicable
                        if self.app.is_connected() and self.app.settings.get("telegram_bridge_enabled", False):
                            if telegram_chat_id:
                                self.app.telegram_bridge.send_message(response)
                            elif is_local:
                                self.app.telegram_bridge.send_message(f"Desktop Command: {user_text}")
                                self.app.telegram_bridge.send_message(response)
                        return

                if self.stop_processing_flag:
                    return

                # Check for Sleep Mode
                if self.ai_core.decider and self.ai_core.decider.is_sleeping:
                    msg = "💤 System is in Sleep Mode (Deep Consolidation). Please wait or wake me up."
                    self.app.root.after(0, lambda: self.app.add_chat_message("System", msg, "incoming"))
                    return

                # 1. Prepare Context (Chat History)
                # Use current session history if local, otherwise use chat_id specific (for Telegram)
                # For now, we map Telegram chat_id to a session or just use a temp list if not tracking sessions per user
                # But since we moved chat_memory to chat_sessions in app, we need to adapt.
                
                # If it's a local chat, use the current session
                if is_local and self.app.current_session_id:
                    history = self.app.chat_sessions[self.app.current_session_id]['history']
                else:
                    # Fallback for Telegram or if no session active (shouldn't happen for local)
                    # We might need a mapping for Telegram Chat ID -> Session ID
                    # For now, just use a transient history or a default session
                    history = [] 

                history.append({"role": "user", "content": user_text})

                # Limit history
                if len(history) > 20: 
                    history = history[-20:]

                if self.stop_processing_flag:
                    return

                # Start Streaming UI
                self.app.gui_queue.put(("stream_start", "Assistant"))

                def stream_callback(token):
                    self.app.gui_queue.put(("stream_token", token))

                # Delegate core logic to Decider
                reply = self.ai_core.decider.process_chat_message(
                    user_text=user_text,
                    history=history,
                    status_callback=lambda msg: self.app.root.after(0, lambda: self.app.status_var.set(msg)),
                    image_path=image_path,
                    stop_check_fn=lambda: self.stop_processing_flag, # Only stop on global stop, ignore daydream pause
                    stream_callback=stream_callback
                )

                # End Streaming UI
                self.app.gui_queue.put(("stream_end", reply))

                if self.stop_processing_flag:
                    return

                # Tools are executed within ChatHandler during processing, so we don't need to call _process_tool_calls here.

                # Update History
                history.append({"role": "assistant", "content": reply})
                # History is a reference to the list in chat_sessions, so it updates automatically

                # Now that the user has their reply, let the Decider think about what's next.
                if self.ai_core.decider and hasattr(self.ai_core.decider, 'run_post_chat_decision_cycle'):
                    # Run in background thread to avoid blocking the chat lock for the next message
                    threading.Thread(target=self.ai_core.decider.run_post_chat_decision_cycle, daemon=True).start()

                # Send to Telegram if applicable
                if self.app.is_connected() and self.app.settings.get("telegram_bridge_enabled", False) and self.app.chat_mode_var.get():
                    # If local user typed it, send to Telegram
                    if is_local and self.app.settings.get("telegram_bridge_echo_local_messages", False): # New setting to control echoing local messages
                        self.app.telegram_bridge.send_message(f"Desktop: {user_text}") # Optional: echo user text
                        self.app.telegram_bridge.send_message(reply)
                    # If it came from Telegram, just send the reply
                    elif telegram_chat_id:
                        self.app.telegram_bridge.send_message(reply)

            except Exception as e:
                error_msg = str(e)
                logging.error(f"Error processing message: {error_msg}")
                self.app.root.after(0, lambda: self.app.add_chat_message("System", f"Error: {error_msg}", "incoming"))
            finally:
                # Do not delete image_path here, as UI needs it for display/click
                self.stop_processing_flag = False
                if not self.is_processing:
                    self.app.root.after(0, lambda: self.app.status_var.set("Ready"))

    def daydream_loop(self):
        """Continuous daydreaming loop"""
        time.sleep(2)  # Initial buffer
        last_watchdog_check = 0
        last_health_log = 0
        
        while not self.ai_core.shutdown_event.is_set():
            try:
                # Check stop and pause flags
                if self.stop_processing_flag or self.pause_daydream_flag:
                    if self.ai_core.shutdown_event.wait(0.1): break
                    continue

                # Health Dashboard (Debugger)
                if time.time() - last_health_log > 60:
                    last_health_log = time.time()
                    self._log_health_status()

                    # Check thread health
                    for name, t in self.threads.items():
                        if not t.is_alive():
                            logging.warning(f"⚠️ Thread '{name}' died. Restarting...")
                            self._start_thread(name, getattr(self, f"{name}_loop") if hasattr(self, f"{name}_loop") else getattr(self.ai_core, "run_cognitive_metabolism"))

                # Fatigue Management (Nap State)
                if self.ai_core.crs and hasattr(self.ai_core.crs, 'is_exhausted') and self.ai_core.crs.is_exhausted():
                    logging.info("💤 System Fatigue Critical (>0.9). Initiating NAP.")
                    if self.app.is_connected() and self.app.settings.get("telegram_bridge_enabled", False):
                        self.app.telegram_bridge.send_message("🔋 I need to recharge my context window. Back in 10 minutes.")
                    
                    # Trigger Sleep Mode via Decider
                    if self.ai_core.decider and hasattr(self.ai_core.decider, 'start_sleep_cycle'):
                        self.ai_core.decider.start_sleep_cycle()
                    if self.ai_core.shutdown_event.wait(600): # 10 minutes nap
                        break

                # --- SLEEP CYCLE (Circadian Rhythm) ---
                # Trigger: Low Cognitive Energy (< 0.2)
                energy = 1.0
                if self.ai_core.self_model:
                    drives = self.ai_core.self_model.get_drives()
                    energy = drives.get("cognitive_energy", 1.0)
                
                if energy < 0.2 and self.ai_core.decider and not self.ai_core.decider.is_sleeping:
                    logging.info(f"📉 Energy Critical ({energy:.2f}). Initiating SLEEP CYCLE.")
                    
                    # 1. Notify User
                    msg = "🥱 I am exhausted. Entering Sleep Mode to consolidate memories. I will be back shortly."
                    self.app.root.after(0, lambda: self.app.add_chat_message("System", msg, "incoming"))
                    if self.app.is_connected() and self.app.settings.get("telegram_bridge_enabled", False):
                        self.app.telegram_bridge.send_message(msg)

                    # 2. Enter Sleep State
                    if self.ai_core.decider:
                        self.ai_core.decider.start_sleep_cycle() # Sets is_sleeping = True
                    
                    # 3. Deep Processing (The "Dream")
                    # Run heavy tasks while input is blocked
                    if self.ai_core.binah and hasattr(self.ai_core.binah, 'consolidate'):
                        self.ai_core.binah.consolidate(time_window_hours=None)
                    
                    if self.ai_core.daat and hasattr(self.ai_core.daat, 'build_knowledge_graph'):
                        self.ai_core.daat.build_knowledge_graph()
                        self.ai_core.daat.run_clustering()
                        
                    # 3.5 Active Dreaming (Synthetic Training)
                    if self.ai_core.autonomy_manager and hasattr(self.ai_core.autonomy_manager, 'dream_cycle'):
                        self.ai_core.autonomy_manager.dream_cycle()

                    # 4. Restore Energy (Simulated Rest)
                    if self.ai_core.shutdown_event.wait(10): # Short pause to simulate transition
                        break
                    if self.ai_core.self_model:
                        self.ai_core.self_model.update_drive("cognitive_energy", 1.0)
                    if self.ai_core.decider:
                        self.ai_core.decider.wake_up("Energy Restored")

                # Watchdog: Check for coma state every 60 seconds
                if time.time() - last_watchdog_check > 60:
                    last_watchdog_check = time.time()
                    heartbeat = self.ai_core.heartbeat # Local reference
                    if heartbeat and getattr(heartbeat, 'current_task', None) == "wait":
                        # If waiting for > 1 minute (60s) and decider exists
                        if self.ai_core.heartbeat.wait_start_time > 0 and (time.time() - self.ai_core.heartbeat.wait_start_time > 60):
                            logging.info("⏰ Watchdog: System dormant for >1m. Forcing Pulse.")
                            if self.ai_core.decider:
                                self.ai_core.decider.wake_up("Watchdog Pulse")

                # Check if Decider has work to do
                # We check heartbeat state without locking first to avoid contention during idle
                is_active_task = False
                heartbeat = self.ai_core.heartbeat # Local reference
                if heartbeat and getattr(heartbeat, 'current_task', None) != "wait":
                    is_active_task = True
                
                # STATE 1: ACTIVE EXECUTION (Requires Lock)
                if not self.is_processing and is_active_task:
                    # Try to acquire lock with timeout to avoid busy waiting
                    t0 = time.time()
                    if self.processing_lock.acquire(timeout=0.5):
                        dt = time.time() - t0
                        if dt > 0.1:
                            logging.debug(f"⚠️ Lock Contention: processing_lock acquired in {dt:.4f}s")
                        try:
                            # Ensure chat input is enabled (allow chatting while daydreaming)
                            self.app.root.after(0, lambda: self.app.toggle_chat_input(True))
                            
                            # Double check inside lock (Race condition check)
                            if self.is_processing:
                                continue
                            
                            heartbeat = self.ai_core.heartbeat # Local reference
                            if heartbeat and hasattr(heartbeat, 'current_task'):
                                self.is_processing = True
                                try:
                                    # Update status for UI
                                    task = self.ai_core.heartbeat.current_task.capitalize()
                                    remaining = self.ai_core.heartbeat.cycles_remaining
                                    status_msg = f"Active: {task} ({remaining} left)"
                                    self.app.root.after(0, lambda: self.app.status_var.set(status_msg))
                                    
                                    if self.stop_processing_flag or self.pause_daydream_flag: break

                                    # 1. Perception & Monologue (Stream of Consciousness)
                                    if self.ai_core.yesod and self.ai_core.decider and self.ai_core.self_model:
                                        hesed = self.ai_core.hesed.calculate() if self.ai_core.hesed and hasattr(self.ai_core.hesed, 'calculate') else 0.5
                                        gevurah = self.ai_core.gevurah.calculate() if self.ai_core.gevurah and hasattr(self.ai_core.gevurah, 'calculate') else 0.5
                                        energy = self.ai_core.self_model.get_drives().get("cognitive_energy", 1.0) if self.ai_core.self_model else 1.0
                                        coherence = self.ai_core.keter.evaluate().get("keter", 1.0) if self.ai_core.keter and hasattr(self.ai_core.keter, 'evaluate') else 1.0
                                        
                                        affect_desc = self.ai_core.yesod.calculate_affective_state(hesed, gevurah, energy, coherence)
                                        
                                        if self.stop_processing_flag or self.pause_daydream_flag: break

                                        # Add to stream (deduplicate consecutive)
                                        thought = f"[State] {affect_desc}"
                                        with self.stream_lock:
                                            if not self.stream_of_consciousness or self.stream_of_consciousness[-1] != thought:
                                                self.stream_of_consciousness.append(thought)
                                                if len(self.stream_of_consciousness) > 5:
                                                    self.stream_of_consciousness.pop(0)
                                        
                                        # Inject into Decider
                                        with self.stream_lock:
                                            if self.ai_core.decider:
                                                self.ai_core.decider.stream_of_consciousness = list(self.stream_of_consciousness)

                                    last_reason = heartbeat.task_reason

                                    # Heartbeat rules the loop
                                    heartbeat.tick()
                                    
                                    # 4. Post-Action Reflection
                                    new_reason = heartbeat.task_reason
                                    if new_reason != last_reason:
                                        reflection = f"I chose to {heartbeat.current_task} because {new_reason}"
                                        with self.stream_lock:
                                            self.stream_of_consciousness.append(reflection)
                                            if len(self.stream_of_consciousness) > 5:
                                                self.stream_of_consciousness.pop(0)
                                    
                                    # 5. Enhanced Stream of Consciousness (Internal Monologue)
                                    self._generate_stream_of_consciousness(affect_desc, task, new_reason)
                                    
                                    # Final check before next tick
                                    if self.stop_processing_flag or self.pause_daydream_flag:
                                        break
                                finally:
                                    self.is_processing = False
                                    self.app.root.after(0, lambda: self.app.status_var.set("Ready"))
                        finally:
                            self.processing_lock.release()
                        
                        # Prevent hot loop if cycles are fast
                        time.sleep(0.5)
                    else:
                        # Could not acquire lock (maybe chat is processing), wait briefly
                        time.sleep(0.1)
                elif not self.is_processing:
                    # STATE 2: PASSIVE OBSERVATION (Idle, No Lock)

                    # 1. Passive Stream of Consciousness (Wandering Mind)
                    # Throttle: Run every ~10 seconds to simulate background thought
                    if time.time() - self._last_passive_thought > 10.0:
                        if hasattr(self.ai_core, 'yesod') and self.ai_core.yesod and \
                           hasattr(self.ai_core, 'decider') and self.ai_core.decider and \
                           hasattr(self.ai_core, 'self_model') and self.ai_core.self_model:

                            # Calculate Affective State (Snapshot)
                            hesed = self.ai_core.hesed.calculate() if hasattr(self.ai_core, 'hesed') and hasattr(self.ai_core.hesed, 'calculate') else 0.5
                            gevurah = self.ai_core.gevurah.calculate() if hasattr(self.ai_core, 'gevurah') and hasattr(self.ai_core.gevurah, 'calculate') else 0.5
                            energy = self.ai_core.self_model.get_drives().get("cognitive_energy", 1.0)
                            coherence = self.ai_core.keter.evaluate().get("keter", 1.0) if hasattr(self.ai_core, 'keter') and hasattr(self.ai_core.keter, 'evaluate') else 1.0

                            affect_desc = self.ai_core.yesod.calculate_affective_state(hesed, gevurah, energy, coherence)

                            # Generate spontaneous thought
                            self._generate_stream_of_consciousness(affect_desc, "Idle/Observing", "Wandering Mind")
                            self._last_passive_thought = time.time()

                    # 2. Run observer to maintain "always working" state and detect triggers
                    if self.ai_core.observer and self.ai_core.decider and hasattr(self.ai_core.observer, 'observe'):
                        signal = self.ai_core.observer.observe()
                        # Feed signal to Decider (which may wake up if pressure is high)
                        if signal: self.ai_core.decider.ingest_netzach_signal(signal)
                        # Check for autonomous agency (Curiosity/Study)
                        self.ai_core.run_autonomous_agency_check(signal)
                    if self.ai_core.shutdown_event.wait(1.0):
                        break
                else:
                    # Already processing (shouldn't happen if logic is correct, but safety wait)
                    if self.ai_core.shutdown_event.wait(0.1):
                        break
                        
                # Reset crash counter on successful loop
                self.consecutive_crashes = 0

            except Exception as e:
                self.consecutive_crashes += 1
                logging.error(f"❌ Daydream loop CRASH ({self.consecutive_crashes}): {e}")
                traceback.print_exc()
                
                # Log Trauma
                if self.ai_core.self_model:
                    self.ai_core.self_model.update_drive("survival", 1.0) # Max survival drive
                
                # Soft Reboot Strategy
                if self.consecutive_crashes > 3:
                    logging.critical("🚨 Critical Instability. Attempting Soft Reboot of AI Core...")
                    try:
                        # Re-initialize components
                        if self.ai_core.bootstrap_manager:
                            self.ai_core.bootstrap_manager.init_brain()
                        self.consecutive_crashes = 0
                        # Refresh UI aliases
                        if hasattr(self.app, 'refresh_component_aliases'):
                            self.app.refresh_component_aliases()
                        logging.info("✅ Soft Reboot Successful.")
                    except Exception as reboot_error:
                        logging.critical(f"💀 Soft Reboot Failed: {reboot_error}")
                        time.sleep(60) # Wait longer before retrying
                
                if self.ai_core.shutdown_event.wait(1):
                    break

    def _generate_stream_of_consciousness(self, affect_desc, current_task, task_reason):
        """Generate a persistent internal thought and store it."""
        if hasattr(self.ai_core, 'global_workspace') and self.ai_core.global_workspace and not self.is_thinking:
            self.is_thinking = True
            def run_thought_gw():
                try:
                    # Evolve the dominant thought using the Global Workspace
                    settings = self.app.settings if hasattr(self.app, 'settings') else {}
                    thought = self.ai_core.global_workspace.evolve_thought(
                        chat_model=settings.get("chat_model"),
                        base_url=settings.get("base_url")
                    )

                    if thought and not thought.startswith("⚠️"):
                        # Store in Reasoning Store (Long-term stream)
                        self.ai_core.reasoning_store.add(
                            content=f"Internal Monologue: {thought}",
                            source="stream_of_consciousness",
                            confidence=1.0,
                            ttl_seconds=86400 # 24 hours
                        )

                        # Update UI Stream (Short-term buffer)
                        with self.stream_lock:
                            self.stream_of_consciousness.append(f"💭 {thought}")
                            if len(self.stream_of_consciousness) > 10:
                                self.stream_of_consciousness.pop(0)
                except Exception as e:
                    logging.error(f"Global Workspace thought error: {e}")
                finally:
                    self.is_thinking = False

            self.ai_core.thread_pool.submit(run_thought_gw)
            return

        if not self.ai_core.yesod or not self.ai_core.reasoning_store: return
        if self.is_thinking: return
        
        self.is_thinking = True
        
        def run_thought():
            try:
                context = f"Affect: {affect_desc}\nCurrent Task: {current_task}\nReason: {task_reason}"
                thought = self.ai_core.yesod.generate_internal_monologue(context)
                if thought and not thought.startswith("⚠️"):
                    # Store in Reasoning Store (Long-term stream)
                    self.ai_core.reasoning_store.add(
                        content=f"Internal Monologue: {thought}",
                        source="stream_of_consciousness",
                        confidence=1.0,
                        ttl_seconds=86400 # 24 hours
                    )
                    
                    # Update UI Stream (Short-term buffer)
                    with self.stream_lock:
                        self.stream_of_consciousness.append(f"💭 {thought}")
                        if len(self.stream_of_consciousness) > 10:
                            self.stream_of_consciousness.pop(0)
            except Exception as e:
                logging.error(f"Stream of consciousness error: {e}")
            finally:
                self.is_thinking = False

        self.ai_core.thread_pool.submit(run_thought)

    def _log_health_status(self):
        """Log a simple health bar to the console/logs."""
        if not self.ai_core.crs: return
        
        fatigue = self.ai_core.crs.short_term_fatigue
        energy = 1.0
        if self.ai_core.self_model:
            energy = self.ai_core.self_model.get_drives().get("cognitive_energy", 1.0)
        
        coherence = self.ai_core.keter.evaluate().get("keter", 0.0) if self.ai_core.keter and hasattr(self.ai_core.keter, 'evaluate') else 0.0
        
        # System Metrics
        sys_info = []
        if psutil:
            try:
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                sys_info.append(f"Mem: {mem_mb:.0f}MB")
            except: pass
            
        if self.ai_core.memory_store and self.ai_core.memory_store.faiss_index:
            sys_info.append(f"MemDB: {self.ai_core.memory_store.faiss_index.ntotal}")
        if self.ai_core.document_store and self.ai_core.document_store.faiss_index:
            sys_info.append(f"DocDB: {self.ai_core.document_store.faiss_index.ntotal}")

        # Visual Bar
        f_bar = "█" * int(fatigue * 10) + "░" * (10 - int(fatigue * 10))
        e_bar = "█" * int(energy * 10) + "░" * (10 - int(energy * 10))
        
        status = f"🏥 HEALTH: Fatigue [{f_bar}] {fatigue:.2f} | Energy [{e_bar}] {energy:.2f} | Coherence: {coherence:.2f}"
        active_threads = threading.active_count()
        if sys_info:
            status += f" | {' | '.join(sys_info)}"
        logging.info(f"{status} | Threads: {active_threads}")

    def consolidation_loop(self):
        """Periodic memory consolidation"""
        # Initial delay to ensure startup logs are visible
        time.sleep(1)
        last_optimize_time = 0
        
        while not self.ai_core.shutdown_event.is_set():
            try:
                if self.ai_core.memory_store:
                     self.ai_core.memory_store.sanitize_sources()

                if self.ai_core.binah and hasattr(self.ai_core.binah, 'consolidate'):
                    stats = self.ai_core.binah.consolidate(time_window_hours=None)
                    if stats['processed'] > 0:
                        logging.info(f"🧠 [Consolidator] Processed: {stats['processed']}, Consolidated: {stats['consolidated']}, Skipped: {stats['skipped']}")
                        
                        # Learn associations after consolidation
                        if hasattr(self.ai_core.binah, 'learn_associations'):
                            self.ai_core.binah.learn_associations(self.ai_core.reasoning_store)
                        
                        if self.ai_core.hod and hasattr(self.ai_core.hod, 'reflect'):
                            self.ai_core.hod.reflect("Consolidation")
                
                # Prune old operational meta-memories (keep last 3 days of logs)
                if self.ai_core.meta_memory_store and hasattr(self.ai_core.meta_memory_store, 'prune_events'):
                    # 3 days = 259200 seconds
                    pruned_count = self.ai_core.meta_memory_store.prune_events(max_age_seconds=259200, prune_all=False)
                    if pruned_count > 0:
                        logging.info(f"🧹 [Meta-Memory] Pruned {pruned_count} old events (ALL types).")

                # Auto-vacuum to reclaim space (Once daily)
                if time.time() - last_optimize_time > 86400:
                    # Optimize Memory Store (Rebuild FAISS + Vacuum)
                    if self.ai_core.memory_store and hasattr(self.ai_core.memory_store, 'optimize'):
                        self.ai_core.memory_store.optimize()

                    # Optimize Document Store (Rebuild FAISS if needed)
                    if self.ai_core.document_store and hasattr(self.ai_core.document_store, 'optimize'):
                        self.ai_core.document_store.optimize()

                    last_optimize_time = time.time()

                # Periodic Reasoning Store Pruning
                if self.ai_core.reasoning_store and hasattr(self.ai_core.reasoning_store, 'prune'):
                    self.ai_core.reasoning_store.prune()

                # Da'at Topic Lattice (Entity Summarization)
                if self.ai_core.daat and hasattr(self.ai_core.daat, 'run_topic_lattice'):
                    self.ai_core.daat.run_topic_lattice()
                    self.ai_core.daat.monitor_model_tension()
                    
                # Compress Reasoning Store if too large
                if self.ai_core.reasoning_store and self.ai_core.daat and hasattr(self.ai_core.daat, 'run_reasoning_compression'):
                    count = self.ai_core.reasoning_store.count()
                    if count > 200:
                        logging.info(f"🧠 Reasoning store large ({count}). Triggering compression.")
                        self.ai_core.daat.run_reasoning_compression()

                # Evaluate Keter (System Coherence)
                if self.ai_core.keter and hasattr(self.ai_core.keter, 'evaluate'):
                    self.ai_core.keter.evaluate()

                # Run Evolution Cycle (Liquid Prompts) - Only if sleeping or very idle
                if self.ai_core.decider and hasattr(self.ai_core.decider, 'is_sleeping') and self.ai_core.decider.is_sleeping:
                    self.ai_core.run_evolution_cycle()

                self.ai_core.generate_daily_self_narrative()
                    
                # Update User Model & Self Theory (Yesod)
                # Only rebuild if there are new memories to analyze
                if self.ai_core.yesod and self.ai_core.memory_store:
                    current_mem_count = self.ai_core.memory_store.count_all()
                    
                    if current_mem_count > self.last_user_model_build_count:
                        if hasattr(self.ai_core.yesod, 'build_user_model'):
                            self.ai_core.yesod.build_user_model()
                        if hasattr(self.ai_core.yesod, 'build_self_theory'):
                            self.ai_core.yesod.build_self_theory()
                        self.last_user_model_build_count = current_mem_count
                    
                # Update Life Story (Da'at)
                if self.ai_core.daat and hasattr(self.ai_core.daat, 'update_life_story'):
                    self.ai_core.daat.update_life_story()

            except Exception as e:
                logging.error(f"Consolidation/Cleanup error: {e}")
            
            if self.ai_core.shutdown_event.wait(600): # 10 minutes
                break

    def reflection_loop(self):
        """
        Dedicated Self-Reflection Loop (Hod).
        Ensures the AI constantly reflects on its state, even when idle.
        """
        time.sleep(60) # Startup buffer
        
        while not self.ai_core.shutdown_event.is_set():
            try:
                if self.stop_processing_flag:
                    time.sleep(5)
                    continue

                # Run Hod Analysis
                if self.ai_core.hod and hasattr(self.ai_core.hod, 'reflect'):
                    analysis = self.ai_core.hod.reflect("Continuous Reflection Loop")
                    if self.ai_core.decider and hasattr(self.ai_core.decider, 'ingest_hod_analysis'):
                        self.ai_core.decider.ingest_hod_analysis(analysis)
                
                # Run Meta-Learner Failure Analysis periodically
                if self.ai_core.meta_learner and hasattr(self.ai_core.meta_learner, 'analyze_failures'):
                     self.ai_core.meta_learner.analyze_failures()
                # Run Self-Model Building (Statistical Introspection)
                if self.ai_core.meta_learner and hasattr(self.ai_core.meta_learner, 'build_self_model'):
                     self.ai_core.meta_learner.build_self_model()

            except Exception as e:
                logging.error(f"Reflection loop error: {e}")
            
            # Reflect every 3 minutes
            if self.ai_core.shutdown_event.wait(180):
                break