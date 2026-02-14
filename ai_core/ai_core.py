import logging
import sqlite3
import threading
import time
import os
from typing import Callable, Dict, List, Optional
import shutil
import concurrent.futures
from types import SimpleNamespace
from typing import Dict, Callable, Optional, List, Any

from .lm import compute_embedding, run_local_lm

from .core_autonomy import AutonomyManager
from .core_actions import ActionManager
from .core_bootstrap import BootstrapManager
from .core_self_model import SelfModel, IdentityManager
from .core_stability import StabilityController
from .heartbeat import Heartbeat
from .core_drives import DriveSystem

class AICore:
    """
    Core AI Engine.
    Manages the lifecycle and interaction of all cognitive components.
    Decoupled from UI implementation details.

    LOCK HIERARCHY:
    To prevent deadlocks, locks must be acquired in the following order (High -> Low):

    Level 3 (Logic/Flow):
      - processing_lock (AIController)
      - chat_lock (AIController) -> Acquire BEFORE planning_lock
      - chat_lock (AIController)
      - planning_lock (Decider)
      - maintenance_lock (Decider)
      - rl_lock (AutonomyManager)

    Level 2 (Resource/IO):
      - write_lock (MemoryStore, MetaMemoryStore, ReasoningStore)
      - index_lock (DocumentStore)
      - faiss_lock (MemoryStore)
      - lock (SelfModel, InternetBridge)

    Level 1 (Leaf/Data):
      - stream_lock (AIController)
      - buffer_lock (AIController, Thalamus)
      - _lock (EventBus)
      - _LM_CACHE_LOCK (lm.py)

    Rule: A thread holding a Level N lock can only acquire locks of Level < N.
    """
    def __init__(
        self,
        settings_provider: Callable[[], Dict],
        log_fn: Callable[[str], None],
        chat_fn: Callable[[str, str], None],
        status_callback: Callable[[str], None],
        telegram_status_callback: Callable[[str], None],
        ui_refresh_callback: Callable[[Optional[str]], None],
        get_chat_history_fn: Callable[[], List[Dict]],
        get_logs_fn: Callable[[], str],
        get_doc_logs_fn: Callable[[], str],
        get_status_text_fn: Callable[[], str],
        update_settings_fn: Callable[[Dict], None],
        stop_check_fn: Callable[[], bool],
        enable_loop_fn: Callable[[], None],
        stop_daydream_fn: Callable[[], None],
        sync_journal_fn: Callable[[], None]
    ):
        self.get_settings = settings_provider
        self.log = log_fn
        self.chat_fn = chat_fn
        self.status_callback = status_callback
        self.telegram_status_callback = telegram_status_callback
        self.ui_refresh_callback = ui_refresh_callback
        self.get_chat_history = get_chat_history_fn
        self.get_logs = get_logs_fn
        self.get_doc_logs = get_doc_logs_fn
        self.get_status_text = get_status_text_fn
        self.update_settings = update_settings_fn
        self.stop_check = stop_check_fn
        self.enable_loop_fn = enable_loop_fn
        self.stop_daydream_fn = stop_daydream_fn
        self.sync_journal_fn = sync_journal_fn
        
        # Shared Thread Pool for cognitive tasks to prevent thread explosion
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="AICoreWorker")
        
        self.shutdown_event = threading.Event()

        # Components
        self.memory_store = None
        self.meta_memory_store = None
        self.document_store = None
        self.reasoning_store = None
        self.arbiter = None
        self.binah = None
        self.event_bus = None
        self.chokmah = None
        self.keter = None
        self.hesed = None
        self.gevurah = None
        self.hod_force = None
        self.netzach_force = None
        self.decider = None
        self.observer = None
        self.hod = None
        self.daat = None
        self.malkuth = None
        self.document_processor = None
        self.internet_bridge = None
        self.meta_learner = None
        self.dialectics = None
        self.value_core = None
        self.crs = None
        self.epigenetics = {}
        self.self_model = None
        self.stability_controller = None
        self.heartbeat = None
        self.drive_system = None
        self.last_narrative_update = 0
        self.last_backup_time = 0
        self.global_workspace = None

        self.current_embedding_model = self.get_settings().get("embedding_model")
        self.identity_manager = IdentityManager(self)
        self.autonomy_manager = AutonomyManager(self)
        self.action_manager = ActionManager(self)
        self.bootstrap_manager = BootstrapManager(self)

        self.bootstrap_manager.init_brain()
        
        # Initialize Drive System
        self.drive_system = DriveSystem(self)

        # Load plugins
        self.action_manager.load_plugins()

    def get_embedding_fn(self):
        return lambda text: compute_embedding(
            text, 
            base_url=self.get_settings().get("base_url"), 
            embedding_model=self.get_settings().get("embedding_model")
        )

    def reload_models(self):
        """Check for model changes and re-index if needed."""
        settings = self.get_settings()
        new_emb_model = settings.get("embedding_model")
        
        if new_emb_model != self.current_embedding_model:
            self.log(f"🔄 Embedding model changed: {self.current_embedding_model} -> {new_emb_model}")
            self.current_embedding_model = new_emb_model
            
            def reindex_worker():
                self.log("🔄 Starting background re-indexing...")
                from .lm import clear_embedding_cache
                clear_embedding_cache()
                
                # Re-index
                embed_fn = self.get_embedding_fn()
                
                if self.memory_store: self.memory_store.reindex_embeddings(embed_fn)
                if self.meta_memory_store: self.meta_memory_store.reindex_embeddings(embed_fn)
                if self.reasoning_store: self.reasoning_store.reindex_embeddings() # Uses self.embed_fn which calls compute_embedding
                if self.document_store: self.document_store.reindex_embeddings(embed_fn)
                self.log("✅ Background re-indexing complete.")

            self.thread_pool.submit(reindex_worker)

    # Proxy methods to maintain API compatibility
    def _safe_search(self, query, source):
        return self.action_manager.safe_search(query, source)

    def _process_tool_calls(self, text: str) -> str:
        return self.action_manager.process_tool_calls(text)

    def _on_goal_completed(self, event):
        self.autonomy_manager._on_goal_completed(event)

    def get_event_logs(self, limit: int = 100):
        """Retrieve recent event bus activity."""
        return self.event_bus.get_history(limit) if self.event_bus else []

    def run_evolution_cycle(self):
        self.autonomy_manager.run_evolution_cycle()

    def restore_subjective_continuity(self):
        self.identity_manager.restore_subjective_continuity()

    def generate_daily_self_narrative(self):
        self.identity_manager.generate_daily_self_narrative()

    def run_autonomous_agency_check(self, observation):
        self.autonomy_manager.run_autonomous_agency_check(observation)

    def on_user_interaction(self):
        """Called when user interacts with the system."""
        if self.self_model:
            self.self_model.update_last_interaction()
        if self.drive_system:
            self.drive_system.satisfy_drive("loneliness", 1.0) # Full satisfaction

    def backup_databases(self):
        """Create a timestamped backup of all SQLite databases."""
        backup_dir = "./data/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        db_files = [
            "memory.sqlite3",
            "meta_memory.sqlite3",
            "reasoning.sqlite3",
            "documents_faiss.sqlite3"
        ]
        
        self.log(f"💾 AICore: Starting database backup to {backup_dir}...")
        for db_file in db_files:
            src = os.path.join("./data", db_file)
            if os.path.exists(src):
                dst = os.path.join(backup_dir, f"{os.path.splitext(db_file)[0]}_{timestamp}.sqlite3")
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    self.log(f"⚠️ Failed to backup {db_file}: {e}")
        
        self.last_backup_time = time.time()
        self.log("✅ AICore: Backup complete.")

    def run_cognitive_metabolism(self):
        """
        Cognitive Metabolism Loop.
        Maintains homeostasis: Energy, Entropy, Attention, Continuity.
        """
        while not self.shutdown_event.is_set():
            if self.stop_check():
                time.sleep(1)
                continue
                
            try:
                self._maintain_homeostasis()
                self._monitor_integrity()
            except Exception as e:
                self.log(f"⚠️ Metabolism loop error: {e}")
            
            if self.shutdown_event.wait(5): # Heartbeat with interrupt
                break

    def shutdown(self):
        """Gracefully shut down all components."""
        self.log("🛑 AICore: Initiating graceful shutdown...")
        self.shutdown_event.set()
        
        if self.event_bus:
            self.event_bus.stop()
            
        if hasattr(self, 'thalamus') and self.thalamus:
            self.thalamus.stop()
            
        if self.self_model:
            self.self_model.stop()
            
        if self.memory_store:
            self.memory_store.save_index()
            
        self.thread_pool.shutdown(wait=False, cancel_futures=True)
        self.log("✅ AICore: Shutdown complete.")

    def _connect(self) -> sqlite3.Connection:
        """Helper to get a thread-local SQLite connection."""
        # This is a placeholder. In a real scenario, you'd use a connection pool
        # or threading.local to manage connections per thread.
        return sqlite3.connect(":memory:") # Example: in-memory DB for testing

    def _maintain_homeostasis(self):
        """Update internal metabolic state."""
        if not self.self_model or not self.memory_store: return

        # Update Circadian Phase
        self.self_model.update_circadian_phase()

        # Update Global Workspace (Attention)
        if self.global_workspace:
            self.global_workspace.update()

        # Delegate to DriveSystem
        if self.drive_system:
            # Pass latest observation if available (from Netzach)
            obs = self.netzach_force.last_result if self.netzach_force else None
            self.drive_system.update(obs)
            
        drives = self.self_model.get_drives()
        current_loneliness = drives.get("loneliness", 0.0)
        new_energy = drives.get("cognitive_energy", 1.0)
        entropy = drives.get("entropy_score", 0.0)
        
        # Trigger Spontaneous Speech if lonely and energy is decent
        if current_loneliness > 0.6 and new_energy > 0.4:
            if self.event_bus:
                self.event_bus.publish("SYSTEM:SPONTANEOUS_SPEECH", {
                    "context": "high_loneliness",
                    "value": current_loneliness,
                    "reason": "System seeking interaction to verify user model or share internal state."
                })
        
        # 2. SYNCHRONIZE WITH CRS (Mind-Body Connection)
        if self.crs:
            self.crs.update_metabolic_state(new_energy, entropy)
            if entropy > 0.6:
                self.log(f"📉 Metabolism: High Entropy ({entropy:.2f}) forcing CRS Fatigue.")
        
        # 3. PERIODIC BACKUP (Every 24 hours)
        if time.time() - self.last_backup_time > 86400:
            self.backup_databases()

        # 4. PERIODIC FAISS REPAIR (Every 30 minutes)
        if time.time() - getattr(self, '_last_faiss_repair', 0) > 1800:
            if self.memory_store:
                self.thread_pool.submit(self.memory_store._sync_faiss_index)
            self._last_faiss_repair = time.time()

        # Triggers
        if entropy > 0.7:
            self.log(f"🔥 High Entropy ({entropy:.2f}). Triggering cleanup.")
            if self.decider: self.decider.start_verification_batch() # Force verification
            
        if new_energy < 0.2:
            self.log(f"🔋 Low Energy ({new_energy:.2f}). Restricting complex tasks.")
            # Could signal CRS to lock high-cost tools
            
        # Periodic Narrative Update (Every hour)
        if time.time() - self.last_narrative_update > 3600:
            if self.autonomy_manager:
                self.thread_pool.submit(self.autonomy_manager.generate_narrative_ego)
                self.last_narrative_update = time.time()

    def _monitor_integrity(self):
        """
        Self-Preservation Layer.
        Detects value drift or corruption.
        """
        if not self.self_model:
            return

        # Check if Core Values are intact
        values = self.self_model.get_values()
        if not values or len(values) < 3:
            self.log("🚨 CRITICAL: Core Values corrupted or missing! Integrity compromised.")