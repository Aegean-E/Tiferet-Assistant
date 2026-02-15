"""
AI Desktop Assistant
A standalone desktop application with integrated chat, document management, and Telegram bridge
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import time
import json
import os
import re
from datetime import datetime
from collections import deque
import subprocess
import shutil
from typing import Dict, List, Optional

# Audio recording
try:
    import pyaudio
    import wave
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Import AI Core
from ai_core.ai_core import AICore
from ai_core.ai_controller import AIController
from ai_core.lm import transcribe_audio, configure_lm
from docs.default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT, DAYDREAM_EXTRACTOR_PROMPT

from bridges.telegram_api import TelegramBridge

from ui import DesktopAssistantUI
from docs.commands import handle_command as process_command_logic, NON_LOCKING_COMMANDS

CURRENT_SETTINGS_VERSION = 1.1

class UILogHandler(logging.Handler):
    """Redirects logging records to the UI's log method."""
    def __init__(self, app):
        super().__init__()
        self.app = app

    def emit(self, record):
        msg = self.format(record)
        self.app.log_to_main(msg + '\n')

class DesktopAssistantApp(DesktopAssistantUI):

    def __init__(self, root):
        self.root = root
        self.root.title("AI Desktop Assistant")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Track the settings file path first
        self.settings_file_path = "./settings.json"
        self.settings_lock = threading.RLock()

        # Initialize logging buffers early (migration might log)
        self.log_buffer = deque(maxlen=5000)
        self.debug_log_buffer = deque(maxlen=1000)

        # State
        self.settings = self.load_settings()
        self.telegram_bridge = None
        self.observer = None
        self.connected = False
        self.is_showing_placeholder = False  # Track placeholder state
        self.decider = None
        self.hod = None
        self.keter = None
        self.start_time = time.time()
        self.controller = None
        
        self.telegram_status_sent = False  # Track if status has been sent to avoid spam
        
        # Initialize chat mode based on settings
        initial_mode = self.settings.get("ai_mode", "Daydream")
        self.chat_mode_var = tk.StringVar(value=initial_mode)
        self.daydream_cycle_count = 0
        self.pending_confirmation_command = None
        self.is_recording = False
        self.audio_frames = []

        # Initialize ttkbootstrap style with loaded theme
        theme_map = {
            "Cosmo": "cosmo",
            "Cyborg": "cyborg",
            "Darkly": "darkly"
        }
        theme_to_apply = theme_map.get(self.settings.get("theme", "Darkly"), self.settings.get("theme", "darkly"))
        self.style = ttk.Style(theme=theme_to_apply)

        # Initialize bridge toggle
        self.telegram_bridge_enabled = tk.BooleanVar()

        self.setup_ui()
        self.load_settings_into_ui()
        
        # Setup Logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Add UI Handler
        ui_handler = UILogHandler(self)
        ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        self.logger.addHandler(ui_handler)
        self.redirect_logging()
        
        # Redirect stdout/stderr to logging
        # self.redirect_logging() # Disabled in favor of standard logging

        # Initialize Brain (Memory & Documents) - Moved after UI setup to capture logs
        self.chat_sessions = {} # {session_id: {'name': str, 'history': []}}
        self.current_session_id = None
        self.create_new_session("Default Session")
        
        self.init_ai_core()

        # Initialize Controller
        self.controller = AIController(self, self.ai_core)
        
        # Refresh documents list now that DB is initialized
        self.refresh_documents()
        self.refresh_database_view()
        
        # Refresh chat list
        self.refresh_chat_list()

        # Start background processes (Consolidation)
        self.controller.start_background_loops()

        # Initialize connection state based on settings
        if (self.settings.get("telegram_bridge_enabled", False) and
            self.settings.get("bot_token") and
            self.settings.get("chat_id")):
            self.telegram_bridge_enabled.set(True)
            # Attempt to connect if credentials are provided and bridge is enabled
            self.bot_token_var.set(self.settings.get("bot_token"))
            self.chat_id_var.set(self.settings.get("chat_id"))
            # Connect automatically if settings are valid
            self.connect()
        else:
            self.telegram_bridge_enabled.set(False)
            # Ensure we're disconnected
            self.disconnect()
            
        # Start Settings Watcher
        self.start_settings_watcher()
        
    def create_new_session(self, name="New Chat"):
        """Create a new chat session."""
        session_id = int(time.time() * 1000)
        self.chat_sessions[session_id] = {'name': name, 'history': []}
        self.current_session_id = session_id
        if hasattr(self, 'refresh_chat_list'):
            self.refresh_chat_list()
        if hasattr(self, 'clear_chat_display'):
            self.clear_chat_display()
        return session_id

    def switch_session(self, session_id):
        """Switch to a specific chat session."""
        if session_id in self.chat_sessions:
            self.current_session_id = session_id
            if hasattr(self, 'refresh_chat_list'):
                self.refresh_chat_list()
            if hasattr(self, 'load_chat_history_ui'):
                self.load_chat_history_ui(self.chat_sessions[session_id]['history'])

    def delete_session(self, session_id):
        """Delete a chat session."""
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
            
            # If we deleted the current session, switch to another or create new
            if self.current_session_id == session_id:
                if self.chat_sessions:
                    new_id = list(self.chat_sessions.keys())[0]
                    self.switch_session(new_id)
                else:
                    self.create_new_session("Default Session")
            else:
                if hasattr(self, 'refresh_chat_list'):
                    self.refresh_chat_list()

    def get_current_chat_history(self) -> List[Dict]:
        """Helper for Daydreamer/Observer to see the current conversation"""
        if self.current_session_id and self.current_session_id in self.chat_sessions:
            return self.chat_sessions[self.current_session_id]['history']
        return []

    def on_close(self):
        """Handle application shutdown."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.disconnect() # Stop telegram polling
            if hasattr(self, 'ai_core'):
                self.ai_core.shutdown()
            if hasattr(self, 'telegram_bridge') and self.telegram_bridge:
                self.telegram_bridge.close()
            if hasattr(self, 'ai_core') and self.ai_core.internet_bridge:
                self.ai_core.internet_bridge.close()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.root.destroy()
            # Force exit to kill any lingering non-daemon threads (e.g. ThreadPool workers)
            os._exit(0)

    def handle_ui_refresh(self, target=None):
        """Callback for AICore to request UI updates"""
        if target == 'db':
            self.root.after(0, self.refresh_database_view)
        elif target == 'docs':
            self.root.after(0, self.refresh_documents)
        else:
            self.root.after(0, self.refresh_database_view)
            self.root.after(0, self.refresh_documents)

    def init_ai_core(self):
        """Initialize the AI Core and alias components"""
        try:
            # Wrapper to bridge legacy log_fn calls to logging module
            def log_with_newline(msg):
                if msg:
                    logging.info(msg)

            self.ai_core = AICore(
                settings_provider=self.get_settings_safe,
                log_fn=log_with_newline,
                chat_fn=self.on_proactive_message,
                status_callback=lambda msg: self.root.after(0, lambda: self.status_var.set(msg)),
                telegram_status_callback=self.send_telegram_status,
                ui_refresh_callback=self.handle_ui_refresh,
                get_chat_history_fn=self.get_current_chat_history,
                get_logs_fn=self.get_recent_main_logs,
                get_doc_logs_fn=self.get_recent_doc_logs,
                get_status_text_fn=self.get_current_status_text,
                update_settings_fn=self.update_settings_from_decider,
                stop_check_fn=lambda: (self.controller.stop_processing_flag or self.controller.pause_daydream_flag) if self.controller else False,
                enable_loop_fn=lambda: setattr(self.controller, 'stop_processing_flag', False) if self.controller else None,
                stop_daydream_fn=self.stop_processing, # Will delegate to controller
                sync_journal_fn=self.sync_journal
            )
            self.refresh_component_aliases()
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize AI Core:\n{e}")

    def refresh_component_aliases(self):
        """Refresh local aliases for AI Core components (used after init or reboot)."""
        if not self.ai_core: return
        
        self.memory_store = self.ai_core.memory_store
        self.meta_memory_store = self.ai_core.meta_memory_store
        self.document_store = self.ai_core.document_store
        self.reasoning_store = self.ai_core.reasoning_store
        self.arbiter = self.ai_core.arbiter
        self.binah = self.ai_core.binah
        self.event_bus = self.ai_core.event_bus
        self.keter = self.ai_core.keter
        self.hesed = self.ai_core.hesed
        self.gevurah = self.ai_core.gevurah
        self.decider = self.ai_core.decider
        self.observer = self.ai_core.observer
        self.hod = self.ai_core.hod
        self.daat = self.ai_core.daat
        self.document_processor = self.ai_core.document_processor
        self.internet_bridge = self.ai_core.internet_bridge
        self.heartbeat = self.ai_core.heartbeat

        if self.ai_core.event_bus:
            # Subscribe to AI_SPEAK
            self.ai_core.event_bus.subscribe("AI_SPEAK", self.on_ai_speak, priority=10)

    def get_settings_safe(self) -> Dict:
        """Thread-safe retrieval of settings."""
        with self.settings_lock:
            return self.settings.copy()

    def start_settings_watcher(self):
        """Start a background thread to watch for settings file changes."""
        def watcher_loop():
            last_mtime = 0
            if os.path.exists(self.settings_file_path):
                try:
                    last_mtime = os.stat(self.settings_file_path).st_mtime
                except OSError:
                    pass
            
            while True:
                time.sleep(10)
                try:
                    if os.path.exists(self.settings_file_path):
                        mtime = os.stat(self.settings_file_path).st_mtime
                        if mtime != last_mtime:
                            last_mtime = mtime
                            # Debounce slightly
                            time.sleep(0.1)
                            logging.info("🔄 Settings file changed on disk. Reloading...")
                            self.reload_settings_from_disk()
                except Exception as e:
                    logging.error(f"Settings watcher error: {e}")
                
                if not self.root.winfo_exists(): # Stop if window closed
                    break
        
        threading.Thread(target=watcher_loop, daemon=True).start()

    def reload_settings_from_disk(self):
        """Reload settings from disk and update UI/Core."""
        with self.settings_lock:
            try:
                with open(self.settings_file_path, 'r', encoding='utf-8') as f:
                    new_settings = json.load(f)
                self.settings.update(new_settings)
                
                configure_lm(self.settings)

                # Trigger updates
                if hasattr(self, 'ai_core'):
                    self.ai_core.reload_models()
                
                # Update UI on main thread
                self.root.after(0, self.load_settings_into_ui)
                
            except Exception as e:
                logging.error(f"Failed to reload settings: {e}")

    def update_settings_from_decider(self, new_settings: Dict):
        """Callback for Decider to update settings and UI"""
        with self.settings_lock:
            self.settings.update(new_settings)
            configure_lm(self.settings)
            self.save_settings()
            
        # Update UI on main thread
        if hasattr(self, 'temperature_var'):
            self.root.after(0, lambda: self.temperature_var.set(new_settings.get("temperature", 0.7)))
        if hasattr(self, 'max_tokens_var'):
            self.root.after(0, lambda: self.max_tokens_var.set(new_settings.get("max_tokens", 800)))


    def get_recent_main_logs(self) -> str:
        """Get last 15 lines of main logs for Netzach"""
        if hasattr(self, 'log_buffer'):
            # Convert deque to list to allow slicing, then join
            buffer_list = [str(x) for x in self.log_buffer if x is not None]
            return "\n".join(buffer_list[-15:])
        return ""

    def get_recent_doc_logs(self) -> str:
        """Get last 10 lines of document logs for Netzach"""
        if hasattr(self, 'debug_log_buffer'):
            # Deques do not support slicing. Convert to list first.
            return "".join(list(self.debug_log_buffer)[-10:])
        return ""

    def get_current_status_text(self) -> str:
        """Get current status bar text for Netzach"""
        return self.status_var.get()

    def load_settings(self) -> Dict:
        """Load settings from file"""
        settings_file = self.settings_file_path
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            # Auto-repair: Fix memory extractor prompt if it contains the old "echoes" rule
            current_prompt = settings.get("memory_extractor_prompt", "")
            if "echoes of the other party" in current_prompt or "SELF-CONTAINED" not in current_prompt or "Assistant's suggestions" not in current_prompt:
                logging.info("🔧 Auto-repairing memory extractor prompt in settings...")
                settings["memory_extractor_prompt"] = DEFAULT_MEMORY_EXTRACTOR_PROMPT
                self.save_settings_to_file(settings)
            
            # Ensure defaults exist for Decider baselines to prevent drift
            if "default_temperature" not in settings:
                settings["default_temperature"] = 0.7
            if "default_max_tokens" not in settings:
                settings["default_max_tokens"] = 800
            
            # Version Migration
            version = settings.get("version", 1.0)
            if version < CURRENT_SETTINGS_VERSION:
                settings = self.migrate_settings(settings, version)
            
            configure_lm(settings)
            return settings
        else:
            # File missing: Create defaults and save
            logging.warning("⚠️ Settings file not found. Creating default settings.json...")
            defaults = {
                "version": CURRENT_SETTINGS_VERSION,
                "bot_token": "",
                "chat_id": "",
                "theme": "darkly",
                "window_size": "1200x800",
                "telegram_bridge_enabled": False,
                "base_url": "http://127.0.0.1:1234/v1",
                "chat_model": "qwen2.5-vl-7b-instruct-abliterated",
                "embedding_model": "text-embedding-nomic-embed-text-v1.5",
                "temperature": 0.7,
                "top_p": 0.94,
                "max_tokens": 800,
                "daydream_cycle_limit": 10,
                "max_inconclusive_attempts": 3,
                "max_retrieval_failures": 3,
                "concurrency": 3,
                "context_window": 4096,
                "ai_mode": "Chat",
                "temperature_step": 0.2,
                "system_prompt_fitness": 0.82125,
                "current_mood": 0.5,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "memory_extractor_prompt": DEFAULT_MEMORY_EXTRACTOR_PROMPT,
                "daydream_extractor_prompt": DAYDREAM_EXTRACTOR_PROMPT,
                "permissions": {
                    "SEARCH": True, "WIKI": True, "FIND_PAPER": True, 
                    "CALCULATOR": True, "CLOCK": True, "SYSTEM_INFO": True,
                    "PHYSICS": True, "SIMULATE_PHYSICS": True, "CAUSAL": True, "SIMULATE_ACTION": True,
                    "PREDICT": True, "READ_CHUNK": True, "DESCRIBE_IMAGE": True, "WRITE_FILE": True
                }
            }
            self.settings = defaults
            self.save_settings()
            configure_lm(defaults)
            return defaults

    def migrate_settings(self, settings: Dict, old_version: float) -> Dict:
        """Handle settings format changes between versions."""
        self.log_to_main(f"🔄 Migrating settings from v{old_version} to v{CURRENT_SETTINGS_VERSION}...\n")
        
        if old_version < 1.1:
            # Add new fields for v1.1
            if "plugin_config" not in settings:
                settings["plugin_config"] = {}
            if "faiss_save_threshold" not in settings:
                settings["faiss_save_threshold"] = 50
            
        settings["version"] = CURRENT_SETTINGS_VERSION
        self.save_settings_to_file(settings)
        return settings

    def save_settings(self):
        """Save settings to file"""
        with self.settings_lock:
            settings_file = self.settings_file_path
            temp_file = settings_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, settings_file)
            if hasattr(self, 'ai_core'):
                self.ai_core.reload_models()
            
    def save_settings_to_file(self, settings_dict):
        """Helper to write settings dict to disk"""
        temp_file = self.settings_file_path + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, self.settings_file_path)

    def on_proactive_message(self, sender, msg):
        """Handle proactive messages from Daydreamer or Observer (Netzach)"""
        # Filter out ingestion reports from the Netzach/Internal Thoughts window
        if "Successfully ingested" in msg or "Failed to ingest" in msg or "Background Ingestion" in msg:
            return

        # 1. Always log to AI Interactions (Netzach) window for transparency
        self.root.after(0, lambda: self.add_netzach_message(f"{sender}: {msg}"))

        # 2. Determine if it should appear in the Main Chat
        # Only explicit messages (SPEAK) should appear in main chat.
        # Thoughts, decisions, and daydreaming are internal.
        should_show_in_chat = False
        internal_markers = ["🤔", "💭", "🤖", "🛠️", "📩", "⚠️", "✅", "Decision:", "Thought:", "🌳", "🔮", "• [", "☁️"]
        
        # Allow Assistant messages (Spontaneous speech, Insights)
        if sender.startswith("Assistant"):
            should_show_in_chat = True
        
        # Allow Decider/Hod/Daydream ONLY if they don't contain internal markers (e.g. [SPEAK])
        elif sender in ["Decider", "Hod", "Daydream"]:
             should_show_in_chat = True

        # Final Safety Filter: If message contains internal markers, hide it (unless it's a user command response which usually doesn't come here)
        if any(marker in msg for marker in internal_markers):
            should_show_in_chat = False
        
        if should_show_in_chat:
            # Show in local UI
            self.root.after(0, lambda: self.add_chat_message("Assistant", msg, "incoming"))
            
            # Forward to Telegram
            if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
                 self.telegram_bridge.send_message(msg)

            # Update Current Session Memory
            if self.current_session_id:
                history = self.chat_sessions[self.current_session_id]['history']
                history.append({"role": "assistant", "content": msg})
                # Limit handled in process_message_thread usually, but good to keep safe here too

    def on_ai_speak(self, event):
        """Handle proactive AI speech events"""
        message = event.data
        self.root.after(0, lambda: self.on_proactive_message("Assistant (Insight)", message))

    def stop_processing(self):
        """Stop current AI generation"""
        logging.info("🛑 Stop button clicked.")
        if self.controller:
            self.controller.stop_processing()

    def stop_daydream(self):
        """Stop daydreaming specifically"""
        if self.controller:
            self.controller.stop_daydream()

    def trigger_panic(self):
        """Trigger emergency panic protocol"""
        if self.controller:
            self.controller.trigger_panic()

    def start_daydream(self):
        """Manually trigger a daydream cycle"""
        if self.controller and self.controller.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy processing a task.")
            return
            
        if self.decider:
            self.decider.start_daydream()
            # The background loop will pick this up
        else:
            messagebox.showerror("Error", "Decider not initialized.")

    def verify_memory_sources(self):
        """Manually trigger memory source verification"""
        # Alias: "Binah" | Function: Reasoning and Logic
        if self.controller and self.controller.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy (e.g. Daydreaming). Please click 'Stop' or enable 'Chat Mode' first.")
            return
            
        if not hasattr(self, 'hod'):
            return

        def verify_thread():
            logging.info("🧹 [Manual Verifier] Starting quick batch verification...")
            with self.controller.processing_lock:
                self.controller.is_processing = True
                self.root.after(0, lambda: self.status_var.set("Verifying memory sources..."))
                
                try:
                    # Use a smaller batch for quick verification
                    if self.controller.stop_processing_flag:
                        return

                    concurrency = int(self.settings.get("concurrency", 4))
                    removed = self.hod.verify_sources(batch_size=50, concurrency=concurrency, stop_check_fn=lambda: self.controller.stop_processing_flag)
                    msg = f"Verification complete. Removed {removed} hallucinated memories."
                    logging.info(f"🧹 [Manual Verifier] {msg}")
                    self.root.after(0, lambda: messagebox.showinfo("Verification Result", msg))
                    self.root.after(0, self.refresh_database_view)
                    if self.hod:
                        self.hod.reflect("Verification Batch")
                except Exception as e:
                    logging.error(f"Verification error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Verification failed: {e}"))
                finally:
                    self.controller.is_processing = False
                    self.root.after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=verify_thread, daemon=True).start()

    def verify_all_memory_sources(self):
        """Loop verification until all memories are verified"""
        if self.controller and self.controller.is_processing:
            messagebox.showinfo("Busy", "AI is currently busy. Please click 'Stop' first.")
            return
            
        if not hasattr(self, 'hod'):
            return

        def verify_all_thread():
            logging.info("🧹 [Manual Verifier] Starting FULL verification loop...")
            with self.controller.processing_lock:
                self.controller.is_processing = True
                self.root.after(0, lambda: self.status_var.set("Verifying ALL sources..."))
                
                total_removed = 0
                last_remaining = -1
                stuck_count = 0
                
                try:
                    while True:
                        if self.controller.stop_processing_flag:
                            logging.info("🛑 Verification loop stopped by user.")
                            break
                        
                        # Check if anything left to verify
                        remaining = self.hod.get_unverified_count()
                        if remaining == 0:
                            logging.info("✅ All cited memories verified.")
                            break
                        
                        # Loop protection
                        if remaining == last_remaining:
                            stuck_count += 1
                            if stuck_count >= 5:
                                logging.warning(f"⚠️ Verification loop stuck on {remaining} memories. Aborting.")
                                break
                        else:
                            stuck_count = 0
                            last_remaining = remaining
                        
                        self.root.after(0, lambda: self.status_var.set(f"Verifying... ({remaining} left)"))
                        
                        # Verify a batch
                        concurrency = int(self.settings.get("concurrency", 4))
                        removed = self.hod.verify_sources(batch_size=10000, concurrency=concurrency, stop_check_fn=lambda: self.controller.stop_processing_flag)
                        total_removed += len(removed)
                        
                        # Refresh UI to show progress
                        self.root.after(0, self.refresh_database_view)
                        
                    msg = f"Full verification complete. Removed {total_removed} memories."
                    logging.info(f"🧹 [Manual Verifier] {msg}")
                    self.root.after(0, lambda: messagebox.showinfo("Verification Result", msg))
                    if self.hod:
                        self.hod.reflect("Full Verification")
                except Exception as e:
                    logging.error(f"Verification error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Verification failed: {e}"))
                finally:
                    self.controller.is_processing = False
                    self.root.after(0, lambda: self.status_var.set("Ready"))

        threading.Thread(target=verify_all_thread, daemon=True).start()

    def toggle_connection(self):
        """Toggle connection to Telegram"""
        # Toggle the setting
        with self.settings_lock:
            new_state = not self.telegram_bridge_enabled.get()
            self.telegram_bridge_enabled.set(new_state)
            # Save the new state
            self.settings["telegram_bridge_enabled"] = new_state
            self.save_settings()

        # Update connection based on new state
        if new_state:
            # Only connect if both credentials are provided
            bot_token = self.bot_token_var.get().strip()
            chat_id_str = self.chat_id_var.get().strip()
            if bot_token and chat_id_str:
                try:
                    int(chat_id_str)  # Validate chat ID is numeric
                    self.connect()
                except ValueError:
                    messagebox.showerror("Connection Error", "Chat ID must be a valid number")
                    self.telegram_bridge_enabled.set(False)
                    with self.settings_lock:
                        self.settings["telegram_bridge_enabled"] = False
                        self.save_settings()
            else:
                messagebox.showerror("Connection Error", "Please enter both Bot Token and Chat ID in Settings")
                self.telegram_bridge_enabled.set(False)
                with self.settings_lock:
                    self.settings["telegram_bridge_enabled"] = False
                    self.save_settings()
        else:
            self.disconnect()

    def connect(self):
        """Connect to Telegram (Non-blocking)"""
        if self.is_connected():
            return  # Already connected

        bot_token = self.bot_token_var.get().strip()
        chat_id_str = self.chat_id_var.get().strip()

        if not bot_token or not chat_id_str:
            return

        def connection_worker():
            try:
                chat_id = int(chat_id_str)
                bridge = TelegramBridge(bot_token, chat_id, log_fn=self.log_to_main)

                # Test connection (Blocks thread, but not UI)
                if bridge.send_message("✅ Connected to Desktop Assistant"):
                    self.telegram_bridge = bridge
                    self.connected = True
                    
                    # Update UI on main thread
                    self.root.after(0, lambda: self.connect_button.config(text="Connected", bootstyle="success"))
                    self.root.after(0, lambda: self.status_var.set("Connected to Telegram"))

                    # Start message polling
                    threading.Thread(
                        target=self.telegram_bridge.listen,
                        kwargs={
                            "on_text": self.handle_telegram_text,
                            "on_document": lambda m: threading.Thread(target=self.handle_telegram_document, args=(m,), daemon=True).start(),
                            "on_photo": self.handle_telegram_photo,
                            "on_voice": lambda m: threading.Thread(target=self.handle_telegram_voice, args=(m,), daemon=True).start(),
                            "running_check": lambda: self.is_connected() and self.settings.get("telegram_bridge_enabled", False),
                            "start_timestamp": self.start_time
                        },
                        daemon=True
                    ).start()
                else:
                    raise Exception("Failed to send test message")

            except Exception as e:
                logging.error(f"Telegram connection error: {e}")
                if self.settings.get("telegram_bridge_enabled", False):
                    self.root.after(0, lambda: messagebox.showerror("Connection Error", f"Failed to connect: {e}"))
                self.root.after(0, self.disconnect)

        threading.Thread(target=connection_worker, daemon=True).start()

    def disconnect(self):
        """Disconnect from Telegram"""
        self.connected = False
        self.telegram_bridge = None
        self.connect_button.config(text="Connect", bootstyle="secondary")
        self.status_var.set("Disconnected from Telegram")

    def send_telegram_status(self, message: str):
        """Send a status update to Telegram if connected"""
        if self.is_connected() and self.settings.get("telegram_bridge_enabled", False):
             # Suppress repetitive status messages until user interacts
             if self.telegram_status_sent:
                 return

             if self.telegram_bridge.send_message(message):
                 if "finished" in message.lower():
                     self.telegram_status_sent = True

    def is_connected(self):
        """Check if connected to Telegram"""
        return self.connected and self.telegram_bridge is not None

    def handle_command(self, text: str, chat_id: int) -> Optional[str]:
        """Process slash commands and return response if matched"""
        return process_command_logic(self, text, chat_id)

    def send_message(self, event=None):
        """Send message to both local chat and Telegram"""
        message = self.message_entry.get().strip()
        if not message:
            return

        # Add to local chat UI immediately
        self.add_chat_message("You", message, "outgoing")
        self.message_entry.delete(0, tk.END)
        
        # Update session name if it's the first message and name is default
        if self.current_session_id:
            if self.chat_sessions[self.current_session_id]['name'] in ["New Chat", "Default Session"]:
                self.chat_sessions[self.current_session_id]['name'] = message[:30]
                self.refresh_chat_list()

    def send_image(self):
        """Select and send an image to the AI"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return

        # Get optional caption from entry
        caption = self.message_entry.get().strip()
        if not caption:
            caption = "Analyze this image."
        
        # Clear entry
        self.message_entry.delete(0, tk.END)

        # Create a temp copy to avoid deleting the user's original file
        temp_dir = "./data/temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}_{filename}")
        shutil.copy2(file_path, temp_path)

        # Add to UI
        self.add_chat_message("You", f"{filename}\n{caption}", "outgoing", image_path=temp_path, trigger_processing=False)

        # Process in background
        threading.Thread(
            target=self.process_message_thread,
            args=(caption, True, None, temp_path),
            daemon=True
        ).start()

    def toggle_recording(self):
        """Toggle voice recording state."""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "PyAudio not installed. Cannot record.")
            return

        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.voice_btn.config(text="⏹️", bootstyle="danger")
        self.audio_frames = []
        
        # Start blinking effect
        self._blink_recording_indicator()
        
        def record_loop():
            try:
                chunk = 1024
                format = pyaudio.paInt16
                channels = 1
                rate = 44100
                
                p = pyaudio.PyAudio()
                stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
                
                while self.is_recording:
                    data = stream.read(chunk)
                    self.audio_frames.append(data)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                # Save to file
                temp_wav = "./data/temp_voice_input.wav"
                wf = wave.open(temp_wav, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(format))
                wf.setframerate(rate)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
                
                # Transcribe
                self.root.after(0, lambda: self.status_var.set("Transcribing voice..."))
                text = transcribe_audio(temp_wav)
                
                # Insert into entry
                self.root.after(0, lambda: self.message_entry.insert(tk.END, text + " "))
                self.root.after(0, lambda: self.status_var.set("Ready"))
                
                # Cleanup
                try: os.remove(temp_wav)
                except: pass
            except Exception as e:
                logging.error(f"Voice recording error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Recording Error", f"Failed to record audio: {e}"))
                self.root.after(0, self.stop_recording)

        threading.Thread(target=record_loop, daemon=True).start()

    def _blink_recording_indicator(self):
        """Blink the recording button to indicate activity."""
        if not self.is_recording:
            return
        
        current_text = self.voice_btn.cget("text")
        # Toggle between Stop icon and Red Circle
        new_text = "🔴" if current_text == "⏹️" else "⏹️"
        self.voice_btn.config(text=new_text)
        
        self.root.after(500, self._blink_recording_indicator)

    def stop_recording(self):
        self.is_recording = False
        self.voice_btn.config(text="🎤", bootstyle="default", style="Big.Link.TButton")

    def handle_feedback(self, message_text, rating):
        """
        Handle user feedback (Upvote/Downvote).
        rating: 1.0 (Good) or -1.0 (Bad)
        """
        logging.info(f"👍 User Feedback: {rating} for message: {message_text[:30]}...")
        
        if self.ai_core.meta_learner:
            # Run in background
            self.ai_core.thread_pool.submit(
                self.ai_core.meta_learner.learn_from_feedback,
                message_text,
                rating
            )
            
        self.status_var.set("Feedback recorded. Thank you!")

    def _ingest_document(self, file_path: str, upload_source: str = "desktop_gui", original_filename: str = None) -> Dict:
        """
        Ingests a local document file into the system.
        Returns dict with status ('success', 'duplicate') and details.
        """
        filename = original_filename if original_filename else os.path.basename(file_path)
        file_hash = self.document_store.compute_file_hash(file_path)

        if self.document_store.document_exists(file_hash):
            return {'status': 'duplicate', 'filename': filename}

        chunks, page_count, file_type = self.document_processor.process_document(file_path)

        self.document_store.add_document(
            file_hash=file_hash,
            filename=filename,
            file_type=file_type,
            file_size=os.path.getsize(file_path),
            page_count=page_count,
            chunks=chunks,
            upload_source=upload_source
        )

        return {
            'status': 'success',
            'filename': filename,
            'chunks_count': len(chunks),
            'page_count': page_count
        }

    def handle_telegram_document(self, msg: Dict):
        """Handle document upload from Telegram"""
        try:
            file_info = msg["document"]
            file_id = file_info["file_id"]
            file_name = file_info.get("file_name", "unknown_file")
            chat_id = msg["chat_id"]

            # Check supported types
            if not file_name.lower().endswith(('.pdf', '.docx')):
                self.telegram_bridge.send_message(f"⚠️ Unsupported file type: {file_name}. Please send PDF or DOCX.")
                return

            self.telegram_bridge.send_message(f"📄 Received {file_name}, processing...")

            # Get file path from Telegram
            file_data = self.telegram_bridge.get_file_info(file_id)
            telegram_file_path = file_data["file_path"]

            # Download
            local_dir = "./data/uploaded_docs"
            os.makedirs(local_dir, exist_ok=True)
            local_file_path = os.path.join(local_dir, file_name)
            
            self.telegram_bridge.download_file(telegram_file_path, local_file_path)

            # Ingest using common logic
            result = self._ingest_document(local_file_path, upload_source="telegram", original_filename=file_name)

            if result['status'] == 'duplicate':
                self.telegram_bridge.send_message(f"⚠️ Document '{file_name}' already exists in database. Skipping...")
            elif result['status'] == 'success':
                self.telegram_bridge.send_message(f"✅ Successfully added '{file_name}' to database ({result['chunks_count']} chunks).")
                self.root.after(0, self.refresh_documents)

        except Exception as e:
            logging.error(f"Error handling Telegram document: {e}")
            if self.telegram_bridge:
                self.telegram_bridge.send_message(f"❌ Error processing document: {str(e)}")
        finally:
            if 'local_file_path' in locals() and os.path.exists(local_file_path):
                os.remove(local_file_path)

    def handle_disrupt_command(self, chat_id):
        """Handle /disrupt command from Telegram to stop processing immediately"""
        logging.info("🛑 Disrupt command received from Telegram.")
        if self.telegram_bridge:
            self.telegram_bridge.send_message("🛑 Disrupting current process...")
        
        if self.controller:
            self.controller.stop_processing_flag = True
        
        if self.ai_core and self.ai_core.decider:
            self.ai_core.decider.report_forced_stop()
            
        def reset_flag():
            time.sleep(1.5) 
            if self.controller:
                self.controller.stop_processing_flag = False
            logging.info("▶️ Decider ready for next turn (Cooldown active).")
            if self.telegram_bridge:
                self.telegram_bridge.send_message("▶️ Process disrupted. Decider is in cooldown.")
            
        threading.Thread(target=reset_flag, daemon=True).start()

    def handle_telegram_text(self, msg: Dict):
        """Handle text message from Telegram"""
        # Reset status suppression on interaction
        self.telegram_status_sent = False

        # Check for disrupt command OR implicit disrupt on any message
        text_content = msg.get("text", "").strip().lower()
        is_explicit_disrupt = text_content == "/disrupt"
        
        if is_explicit_disrupt:
            self.handle_disrupt_command(msg["chat_id"])
            return

        # Show in UI
        self.root.after(0, lambda m=msg: self.add_chat_message(m["from"], m["text"], "incoming"))
        # Process logic
        threading.Thread(
            target=self.process_message_thread, 
            args=(msg["text"], False, msg["chat_id"]), # Use actual chat_id from msg
            daemon=True
        ).start()

    def handle_telegram_photo(self, msg: Dict):
        """Handle photo from Telegram"""
        try:
            file_id = msg["photo"]["file_id"]
            caption = msg.get("caption", "") or "Analyze this image."
            
            # Download to temp
            temp_path = f"./data/temp_img_{file_id}.jpg"
            file_data = self.telegram_bridge.get_file_info(file_id)
            self.telegram_bridge.download_file(file_data["file_path"], temp_path)
            
            self.root.after(0, lambda m=msg, c=caption, p=temp_path: self.add_chat_message(m["from"], c, "incoming", image_path=p))
            
            threading.Thread(
                target=self.process_message_thread,
                args=(caption, False, msg["chat_id"], temp_path), # Use actual chat_id from msg
                daemon=True
            ).start()
        except Exception as e:
            logging.error(f"Error handling photo: {e}")

    def handle_telegram_voice(self, msg: Dict):
        """Handle voice message from Telegram"""
        try:
            file_id = msg["voice"]["file_id"]
            chat_id = msg["chat_id"]
            
            self.root.after(0, lambda: self.status_var.set("🎙️ Receiving voice message..."))
            
            file_data = self.telegram_bridge.get_file_info(file_id)
            telegram_file_path = file_data["file_path"]
            
            temp_dir = "./data/temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            local_file_path = os.path.join(temp_dir, f"voice_{file_id}.ogg")
            
            self.telegram_bridge.download_file(telegram_file_path, local_file_path)
            
            self.root.after(0, lambda: self.status_var.set("📝 Transcribing voice..."))
            text = transcribe_audio(local_file_path)
            
            if text and not text.startswith("[Error"):
                self.root.after(0, lambda: self.add_chat_message(msg["from"], f"🎙️ {text}", "incoming"))
                threading.Thread(
                    target=self.process_message_thread,
                    args=(text, False, chat_id),
                    daemon=True
                ).start()
            else:
                self.telegram_bridge.send_message(f"⚠️ Sorry, I couldn't transcribe that voice message: {text}")
                
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                
        except Exception as e:
            logging.error(f"Error handling Telegram voice: {e}")
            if self.telegram_bridge:
                self.telegram_bridge.send_message(f"❌ Error processing voice message: {str(e)}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))

    def upload_documents(self):
        """Upload documents via GUI"""
        file_paths = filedialog.askopenfilenames(
            title="Select PDF or DOCX files",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("DOCX files", "*.docx"),
                ("All supported", "*.pdf *.docx")
            ]
        )

        if not file_paths:
            return

        def upload_thread():
            success_count = 0
            total_files = len(file_paths)

            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Starting upload of {total_files} document(s)")

            for i, file_path in enumerate(file_paths):
                # Check for stop signal
                if self.controller and self.controller.stop_check():
                    self.log_debug_message("Upload interrupted by stop signal.")
                    break

                try:
                    filename = os.path.basename(file_path)
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Processing ({i+1}/{total_files}): {filename}")
                        self.log_debug_message(f"Extracting text from: {filename}")

                    result = self._ingest_document(file_path, upload_source="desktop_gui", original_filename=filename)

                    if result['status'] == 'duplicate':
                        if hasattr(self, 'debug_log'):
                            self.log_debug_message(f"Skipping duplicate: {filename}")
                        continue

                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Successfully extracted {result['chunks_count']} chunks from {filename} ({result['page_count']} pages)")
                        self.log_debug_message(f"Successfully added: {filename}")

                    success_count += 1

                except Exception as e:
                    if hasattr(self, 'debug_log'):
                        self.log_debug_message(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                    logging.error(f"Error processing {file_path}: {e}")

            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Upload complete: {success_count}/{total_files} documents processed successfully")

            # Update UI in main thread
            self.root.after(0, lambda: self.refresh_documents())
            self.root.after(0, lambda: self.status_var.set(f"Uploaded {success_count} documents"))

        self.ai_core.thread_pool.submit(upload_thread)

    def process_message_thread(self, user_text: str, is_local: bool, telegram_chat_id=None, image_path: str = None):
        """Delegate to controller"""
        if self.controller:
            self.controller.process_message_thread(user_text, is_local, telegram_chat_id, image_path)

    def sync_journal(self):
        """
        Compile all Assistant Notes into a journal file and ingest it into the Document Store.
        This allows the AI to 'read' its own journal via RAG.
        """
        try:
            # 1. Fetch Notes
            items = self.memory_store.list_recent(limit=None)
            # Filter for NOTE type
            notes = [item for item in items if item[1] == "NOTE"]

            # Also fetch Self-Knowledge (Rules about self)
            self_knowledge = [item for item in items if item[4] == "meta_learner_self_model"]

            # Fetch Self-Narratives (Autobiography) from Meta-Memory
            narratives = []
            if self.meta_memory_store:
                narratives = self.meta_memory_store.get_by_event_type("SELF_NARRATIVE", limit=100)

            # Sort chronologically (list_recent is DESC, so reverse)
            notes.reverse()
            self_knowledge.reverse()
            narratives.sort(key=lambda x: x['created_at'])

            if not notes and not narratives and not self_knowledge:
                messagebox.showinfo("Journal", "No journal entries, narratives, or self-knowledge to sync.")
                return

            # 2. Create File Content
            content = "ASSISTANT JOURNAL\n=================\n\n"

            # Add Narratives (The Story)
            for n in narratives:
                date_str = datetime.fromtimestamp(n['created_at']).strftime("%Y-%m-%d %H:%M")
                content += f"[{date_str}] [SELF-NARRATIVE]\n{n['text']}\n\n"

            # Add Notes (The Thoughts)
            for note in notes:
                # note: (id, type, subject, text, ...)
                content += f"Entry [ID:{note[0]}]:\n{note[3]}\n\n" + ("-"*30) + "\n\n"

            # Add Self-Knowledge (The Rules)
            if self_knowledge:
                content += "SELF-KNOWLEDGE (LEARNED RULES)\n==============================\n"
                for sk in self_knowledge:
                    content += f"- {sk[3]}\n"

            # 3. Write to Docs Folder
            docs_dir = "./data/uploaded_docs"
            os.makedirs(docs_dir, exist_ok=True)
            filename = "assistant_journal.txt"
            file_path = os.path.join(docs_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # 4. Ingest into Document Store
            # Remove old version if exists to ensure clean update
            old_id = self.document_store.get_document_by_filename(filename)
            if old_id:
                self.document_store.delete_document(old_id)

            # Process and Add
            file_hash = self.document_store.compute_file_hash(file_path)
            chunks, page_count, file_type = self.document_processor.process_document(file_path)

            self.document_store.add_document(
                file_hash=file_hash,
                filename=filename,
                file_type=file_type,
                file_size=os.path.getsize(file_path),
                page_count=page_count,
                chunks=chunks,
                upload_source="journal_sync"
            )

            self.refresh_documents()
            messagebox.showinfo("Journal Sync", f"Journal synced to documents ({len(notes)} entries).")

        except Exception as e:
            messagebox.showerror("Journal Sync Error", f"Failed to sync journal: {e}")

def main():
    root = tk.Tk()
    app = DesktopAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()