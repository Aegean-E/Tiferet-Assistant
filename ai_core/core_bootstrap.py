import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import AI components
from docs.document_store_faiss import FaissDocumentStore
from docs.document_processor import DocumentProcessor
from bridges.internet_bridge import InternetBridge
from .event_bus import EventBus
from treeoflife.netzach import ContinuousObserver
from memory.memory import MemoryStore
from memory.meta_memory import MetaMemoryStore
from memory.reasoning import ReasoningStore
from memory.memory_arbiter import MemoryArbiter
from treeoflife.binah import Binah
from treeoflife.chokmah import Chokmah
from treeoflife.tiferet import Decider
from treeoflife.hod import Hod as HodAgent
from treeoflife.daat import Daat
from treeoflife.keter import Keter
from treeoflife.hesed_gevurah import Hesed, Gevurah
from treeoflife.malkuth import Malkuth
from treeoflife.yesod import Yesod
from .meta_learner import MetaLearner
from treeoflife.dialectics import Dialectics
from .value_core import ValueCore
from .crs import CognitiveResourceController
from .core_self_model import SelfModel
from .core_stability import StabilityController
from .heartbeat import Heartbeat
from .core_spotlight import GlobalWorkspace
from .core_thalamus import Thalamus

class BootstrapManager:
    """
    Handles the initialization and wiring of all AI components.
    """
    def __init__(self, ai_core):
        self.core = ai_core

    def init_brain(self):
        """Initialize the AI memory and reasoning components"""
        try:
            # Cleanup existing components if re-initializing (Self-Healing)
            if getattr(self.core, 'event_bus', None):
                self.core.event_bus.stop()
            if getattr(self.core, 'thalamus', None):
                self.core.thalamus.stop()
            if getattr(self.core, 'self_model', None):
                self.core.self_model.stop()

            # Initialize Event Bus
            self.core.event_bus = EventBus(log_fn=self.core.log)
            
            # Initialize Self Model (Persistent Identity & Drives)
            self.core.self_model = SelfModel()
            
            # Load Epigenetics from Self Model
            self.core.epigenetics = self.core.self_model.get_epigenetics()

            # Alias: "Da'at" | Function: Knowledge
            self.core.memory_store = MemoryStore(db_path="./data/memory.sqlite3", config=self.core.get_settings())
            self.core.memory_store.self_model = self.core.self_model

            # Initialize Value Core (Invariant Layer)
            self.core.value_core = ValueCore(
                get_settings_fn=self.core.get_settings,
                log_fn=self.core.log,
                memory_store=self.core.memory_store,
                self_model=self.core.self_model,
                embed_fn=self.core.get_embedding_fn()
            )

            self.core.meta_memory_store = MetaMemoryStore(
                db_path="./data/meta_memory.sqlite3",
                embed_fn=self.core.get_embedding_fn()
            )
            self.core.document_store = FaissDocumentStore(
                db_path="./data/documents_faiss.sqlite3",
                embed_fn=self.core.get_embedding_fn()
            )
            self.core.reasoning_store = ReasoningStore(embed_fn=self.core.get_embedding_fn(), log_fn=self.core.log)
            self.core.arbiter = MemoryArbiter(
                self.core.memory_store, 
                meta_memory_store=self.core.meta_memory_store, 
                embed_fn=self.core.get_embedding_fn(), 
                log_fn=self.core.log,
                event_bus=self.core.event_bus,
                config=self.core.epigenetics
            )
            
            # Alias: "Binah" | Function: Reasoning, Logic, Structure
            # Validate consolidation thresholds (must be dict)
            cons_thresh = self.core.epigenetics.get("consolidation_threshold")
            if not isinstance(cons_thresh, dict):
                # Fallback to settings if epigenetics is invalid/missing/float
                cons_thresh = self.core.get_settings().get("consolidation_thresholds")
                if not isinstance(cons_thresh, dict):
                    cons_thresh = None # Use Binah defaults

            self.core.binah = Binah(
                self.core.memory_store, 
                meta_memory_store=self.core.meta_memory_store, 
                consolidation_thresholds=cons_thresh,
                get_settings_fn=self.core.get_settings,
                embed_fn=self.core.get_embedding_fn(),
                log_fn=self.core.log
            )
            
            # Initialize Da'at (Knowledge Integrator)
            self.core.daat = Daat(
                memory_store=self.core.memory_store,
                meta_memory_store=self.core.meta_memory_store,
                reasoning_store=self.core.reasoning_store,
                document_store=self.core.document_store,
                get_settings_fn=self.core.get_settings,
                embed_fn=self.core.get_embedding_fn(),
                log_fn=self.core.log
            )

            # Initialize Chokmah (Pure Generator)
            # Alias: "Chokhmah" | Function: Raw, Creative Output
            self.core.chokmah = Chokmah(
                memory_store=self.core.memory_store,
                document_store=self.core.document_store,
                get_settings_fn=self.core.get_settings,
                log_fn=self.core.log,
                stop_check_fn=self.core.stop_check,
                get_gap_topic_fn=lambda: self.core.daat.get_sparse_topic() if self.core.daat else None,
                embed_fn=self.core.get_embedding_fn(),
                event_bus=self.core.event_bus
            )

            # Initialize Keter (The Crown)
            self.core.keter = Keter(
                memory_store=self.core.memory_store,
                meta_memory_store=self.core.meta_memory_store,
                reasoning_store=self.core.reasoning_store,
                event_bus=self.core.event_bus,
                log_fn=self.core.log,
                smoothing=self.core.epigenetics.get("keter_smoothing_alpha", 0.95)
            )

            # Initialize Hesed (Expansion) and Gevurah (Constraint)
            self.core.hesed = Hesed(self.core.memory_store, self.core.keter, log_fn=self.core.log, bias=self.core.epigenetics.get("hesed_expansion_bias", 1.0))
            self.core.gevurah = Gevurah(
                self.core.memory_store, 
                self.core.keter, 
                log_fn=self.core.log, 
                bias=self.core.epigenetics.get("gevurah_constraint_bias", 1.0),
                meta_memory_store=self.core.meta_memory_store
            )

            # Initialize Malkuth (Causal Engine)
            self.core.malkuth = Malkuth(
                memory_store=self.core.memory_store,
                meta_memory_store=self.core.meta_memory_store,
                log_fn=self.core.log,
                event_bus=self.core.event_bus
            )

            # Initialize Yesod (The Foundation/Persona)
            self.core.yesod = Yesod(
                memory_store=self.core.memory_store,
                meta_memory_store=self.core.meta_memory_store,
                get_settings_fn=self.core.get_settings,
                log_fn=self.core.log,
                stop_check_fn=self.core.stop_check
            )

            # Initialize Cognitive Resource Controller (CRS)
            self.core.crs = CognitiveResourceController(
                log_fn=self.core.log,
                skill_map=self.core.self_model.data.get("skills", {}),
                self_model=self.core.self_model
            )

            # Initialize Stability Controller (Global Governor)
            self.core.stability_controller = StabilityController(self.core)

            # Initialize Meta-Learner (Self-Improvement)
            self.core.meta_learner = MetaLearner(
                memory_store=self.core.memory_store,
                meta_memory_store=self.core.meta_memory_store,
                get_settings_fn=self.core.get_settings,
                update_settings_fn=self.core.update_settings,
                log_fn=self.core.log,
                reasoning_store=self.core.reasoning_store,
                self_model=self.core.self_model,
                value_core=self.core.value_core,
                stability_controller=self.core.stability_controller,
                event_bus=self.core.event_bus,
                autonomy_manager=self.core.autonomy_manager
            )

            # Initialize Dialectics (The Council)
            self.core.dialectics = Dialectics(
                get_settings_fn=self.core.get_settings, 
                log_fn=self.core.log,
                memory_store=self.core.memory_store,
                hesed=self.core.hesed,
                gevurah=self.core.gevurah,
                event_bus=self.core.event_bus
            )

            # Initialize Utilities
            self.core.document_processor = DocumentProcessor(embed_fn=self.core.get_embedding_fn(), log_fn=self.core.log)
            self.core.internet_bridge = InternetBridge(
                get_settings_fn=self.core.get_settings,
                log_fn=self.core.log
            )
            
            # Inject Internet Bridge into Chokmah for Proactive Research
            if self.core.chokmah:
                self.core.chokmah.internet_bridge = self.core.internet_bridge

            # Initialize Heartbeat (Scheduler)
            self.core.heartbeat = Heartbeat(self.core, log_fn=self.core.log)

            # Initialize Global Workspace (Attention)
            self.core.global_workspace = GlobalWorkspace(self.core)

            # Initialize Thalamus (Sensory Gating)
            self.core.thalamus = Thalamus(self.core)
            self.core.thalamus.start()

            # --- Wrappers for Decider Actions ---
            # These wrappers capture self.core to allow Decider to invoke system components
            
            def integrate_thought_packet(packet):
                if not packet: return
                thought = packet.get("thought", "")
                candidates = packet.get("candidates", [])
                
                self.core.event_bus.publish("DAYDREAM_THOUGHT", thought, source="Chokmah")
                self.core.reasoning_store.add(content=f"Daydream Stream: {thought}", source="daydream_raw", confidence=1.0, ttl_seconds=3600)
                
                if self.core.chat_fn:
                    display_thought = f"☁️ {thought}"
                    if candidates:
                        lines = []
                        for item in candidates:
                            lines.append(f"• [{item.get('type', '?')}] {item.get('text', '')}")
                        display_thought = "\n".join(lines)
                    self.core.chat_fn("Daydream", display_thought)
                
                promoted = 0
                for c in candidates:
                    mid = self.core.arbiter.consider(
                        text=c["text"],
                        mem_type=c.get("type", "FACT"),
                        subject=c.get("subject", "User"),
                        confidence=self.core.epigenetics.get("arbiter_confidence_threshold", 0.85),
                        source="daydream"
                    )
                    if mid is not None:
                        promoted += 1
                        self.core.log(f"✅ [Integrator] Memory saved with ID: {mid}")
                
                if promoted:
                    self.core.log(f"🧠 Promoted {promoted} memory item(s) from daydream.")

            def start_daydream_wrapper():
                self.core.telegram_status_callback("☁️ Model is processing memories (Daydreaming)...")
                try:
                    mode = "auto"
                    topic = None
                    if self.core.decider:
                        mode = self.core.decider.daydream_mode
                        topic = getattr(self.core.decider, 'daydream_topic', None)
                    
                    packet = self.core.chokmah.emanate(impulse=mode, topic=topic)
                    integrate_thought_packet(packet)
                finally:
                    self.core.telegram_status_callback("✅ Processing finished.")

            def start_daydream_batch_wrapper(count):
                self.core.telegram_status_callback(f"☁️ Model is processing memories (Daydreaming Batch x{count})...")
                try:
                    mode = "auto"
                    topic = None
                    if self.core.decider:
                        mode = self.core.decider.daydream_mode
                        topic = getattr(self.core.decider, 'daydream_topic', None)
                    
                    with ThreadPoolExecutor(max_workers=count) as executor:
                        futures = [executor.submit(self.core.chokmah.emanate, impulse=mode, topic=topic) for _ in range(count)]
                        for future in as_completed(futures):
                            try:
                                packet = future.result()
                                integrate_thought_packet(packet)
                            except Exception as e:
                                self.core.log(f"❌ Batch daydream error: {e}")
                finally:
                    self.core.telegram_status_callback("✅ Processing finished.")

            def philosophize_wrapper():
                self.core.telegram_status_callback("☁️ Model is philosophizing...")
                try:
                    thought = self.core.chokmah._generate_philosophy(self.core.get_settings())
                    integrate_thought_packet({"thought": thought, "candidates": [], "source_file": None})
                finally:
                    self.core.telegram_status_callback("✅ Processing finished.")

            def verify_batch_wrapper():
                self.core.telegram_status_callback("⚙️ Model is processing memories (Verification Batch)...")
                try:
                    if self.core.hod:
                        self.core.status_callback("Verifying sources (Decider)...")
                        concurrency = int(self.core.get_settings().get("concurrency", 4))
                        proposals = self.core.hod.verify_sources(batch_size=50, concurrency=concurrency, stop_check_fn=self.core.stop_check)
                    
                        removed = 0
                        verified = 0
                        refuted = 0
                    
                        for p in proposals:
                            action = p.get("proposal")
                            mid = p.get("memory_id")
                        
                            if action == "DELETE":
                                if self.core.memory_store.soft_delete_entry(mid):
                                    removed += 1
                                    self.core.log(f"🗑️ [Verifier] Pruned Memory {mid}: {p.get('reason')}")
                            elif action == "VERIFY":
                                self.core.memory_store.mark_verified(mid)
                                if "new_text" in p:
                                    self.core.memory_store.update_text(mid, p["new_text"])
                                verified += 1
                                self.core.log(f"✅ [Verifier] Verified Memory {mid}")
                            elif action == "REFUTE":
                                if "new_text" in p:
                                    self.core.memory_store.update_text(mid, p["new_text"])
                                    self.core.memory_store.update_embedding(mid, self.core.get_embedding_fn()(p["new_text"]))
                                self.core.memory_store.update_type(mid, "REFUTED_BELIEF")
                                refuted += 1
                                self.core.log(f"🛡️ [Verifier] Refuted Memory {mid}")
                            elif action == "INCREMENT_ATTEMPTS":
                                self.core.memory_store.increment_verification_attempts(mid)
                        
                        if removed > 0 or verified > 0 or refuted > 0:
                            self.core.log(f"⚙️ Verification Batch Result: {verified} Verified, {removed} Pruned, {refuted} Refuted.")
                            self.core.ui_refresh_callback('db')
                    else:
                        self.core.log("⚠️ Hod not initialized. Skipping verification.")
                finally:
                    self.core.telegram_status_callback("✅ Processing finished.")

            def run_hod_wrapper():
                if self.core.hod:
                    analysis = self.core.hod.reflect("Decider Cycle")
                    if self.core.decider:
                        self.core.decider.ingest_hod_analysis(analysis)

            def run_observer_wrapper():
                if self.core.observer:
                    signal = self.core.observer.observe()
                    if self.core.decider and signal:
                        self.core.decider.ingest_netzach_signal(signal)
                    return signal

            def remove_goal_wrapper(target):
                try:
                    target_id = None
                    try:
                        target_id = int(str(target).strip())
                    except ValueError:
                        pass
                    
                    found_items = []
                    if target_id is not None:
                        # Targeted fetch by ID
                        item = self.core.memory_store.get(target_id)
                        if item:
                            # Convert dict to tuple format expected below: (id, type, subject, text)
                            found_items = [(item['id'], item['type'], item['subject'], item['text'])]
                    else:
                        # Targeted fetch by text match
                        target_text = str(target).lower()
                        # We need to search. Since list_recent is heavy, let's use a direct query if possible or search
                        found_items = self.core.memory_store.get_active_by_type("GOAL")
                        found_items = [g for g in found_items if target_text in g[2].lower()] # g[2] is text in get_active_by_type
                    
                    removed_count = 0
                    response_msgs = []
                    
                    for item in found_items:
                        mem_id = item[0]
                        mem_type = item[1]
                        # Handle different tuple structures
                        if len(item) > 3:
                            mem_text = item[3] # list_recent format
                        else:
                            mem_text = item[2] # get_active_by_type format
                        
                        if mem_type == "GOAL":
                            self.core.memory_store.update_type(mem_id, "COMPLETED_GOAL")
                            self.core.log(f"✅ [Decider] Marked GOAL as COMPLETED: {mem_text}")
                            response_msgs.append(f"Completed '{mem_text}'")
                            removed_count += 1
                            self.core.event_bus.publish("GOAL_COMPLETED", {"goal_text": mem_text})
                        else:
                            response_msgs.append(f"Skipped ID {mem_id} (Type: {mem_type})")
                            
                    if removed_count > 0:
                        return f"✅ Success. {', '.join(response_msgs)}"
                    elif response_msgs:
                        return f"⚠️ Failed. {', '.join(response_msgs)}"
                    else:
                        return "❌ No matching goals found."
                except Exception as e:
                    return f"❌ Error removing goal: {e}"

            def list_documents_wrapper():
                try:
                    docs = self.core.document_store.list_documents(limit=50)
                    if not docs: return "No documents available."
                    lines = ["Available Documents:"]
                    for doc in docs:
                        lines.append(f"- ID {doc[0]}: {doc[1]} ({doc[4]} chunks)")
                    return "\n".join(lines)
                except Exception as e: return f"Error listing documents: {e}"

            def read_document_wrapper(target):
                try:
                    # Parse target and chunk index
                    target_str = str(target).strip()
                    start_chunk = 0

                    # Handle "filename, chunk_index" format
                    if "," in target_str:
                        parts = target_str.rsplit(",", 1)
                        if parts[1].strip().isdigit():
                            target_str = parts[0].strip()
                            start_chunk = int(parts[1].strip())

                    # Find Document
                    doc_id = None
                    if target_str.isdigit():
                        doc_id = int(target_str)

                    docs = self.core.document_store.list_documents(limit=1000)
                    selected_doc = None

                    if doc_id:
                        selected_doc = next((d for d in docs if d[0] == doc_id), None)

                    if not selected_doc:
                        target_lower = target_str.lower()
                        selected_doc = next((d for d in docs if target_lower in d[1].lower()), None)

                    if not selected_doc: return f"❌ Document '{target_str}' not found."

                    # Get Chunks
                    chunks = self.core.document_store.get_document_chunks(selected_doc[0])
                    if not chunks: return f"⚠️ Document '{selected_doc[1]}' is empty."

                    # Validate Index
                    total_chunks = len(chunks)
                    if start_chunk >= total_chunks:
                        return f"⚠️ Chunk index {start_chunk} out of range (Total: {total_chunks})."

                    # Slice Chunks (Read 5 at a time)
                    BATCH_SIZE = 5
                    end_chunk = min(start_chunk + BATCH_SIZE, total_chunks)
                    preview_chunks = chunks[start_chunk:end_chunk]

                    preview = "\n\n".join([f"[Chunk {c['chunk_index']}] {c['text']}" for c in preview_chunks])

                    footer = ""
                    if end_chunk < total_chunks:
                        footer = f"\n\n[To read more: [READ_DOC: {selected_doc[1]}, {end_chunk}]]"

                    return f"📄 Content of '{selected_doc[1]}' (Chunks {start_chunk}-{end_chunk-1} of {total_chunks}):\n{preview}{footer}"
                except Exception as e: return f"❌ Error reading document: {e}"

            def search_memory_wrapper(query):
                try:
                    emb = self.core.get_embedding_fn()(query)
                    results = self.core.memory_store.search(emb, limit=10)
                    if not results: return "No matching memories found."
                    lines = [f"Search results for '{query}':"]
                    for r in results: lines.append(f"- [{r[1]}] {r[3]} (Sim: {r[4]:.2f})")
                    return "\n".join(lines)
                except Exception as e: return f"❌ Error searching memory: {e}"

            def prune_memory_wrapper(target_id):
                try:
                    mid = int(str(target_id).strip())
                    mem = self.core.memory_store.get(mid)
                    if not mem: return f"⚠️ Memory ID {mid} not found."
                    if mem.get('deleted', 0) == 1: return f"ℹ️ Memory ID {mid} is already pruned."
                    if self.core.memory_store.soft_delete_entry(mid): return f"🗑️ Pruned memory ID {mid}."
                    return f"⚠️ Failed to prune memory ID {mid}."
                except Exception as e: return f"❌ Error pruning memory: {e}"

            def refute_memory_wrapper(target_id, reason=None):
                try:
                    mid = int(str(target_id).strip())
                    mem = self.core.memory_store.get(mid)
                    if not mem: return f"⚠️ Memory ID {mid} not found."
                    if reason:
                        new_text = f"{mem.get('text', '')} [REFUTED: {reason.replace(f'[REFUTE_MEM: {mid}]', '').strip()[:500]}]"
                        self.core.memory_store.update_text(mid, new_text)
                        self.core.memory_store.update_embedding(mid, self.core.get_embedding_fn()(new_text))
                    if self.core.memory_store.update_type(mid, "REFUTED_BELIEF"): return f"🛡️ Refuted memory ID {mid}."
                    return f"⚠️ Failed to refute memory ID {mid}."
                except Exception as e: return f"❌ Error refuting memory: {e}"

            # Initialize Continuous Observer (Netzach)
            self.core.observer = ContinuousObserver(
                memory_store=self.core.memory_store,
                reasoning_store=self.core.reasoning_store,
                meta_memory_store=self.core.meta_memory_store,
                get_settings_fn=self.core.get_settings,
                get_chat_history_fn=self.core.get_chat_history,
                get_meta_memories_fn=lambda: self.core.meta_memory_store.list_recent(limit=10),
                get_main_logs_fn=self.core.get_logs,
                get_doc_logs_fn=self.core.get_doc_logs,
                get_status_fn=self.core.get_status_text,
                event_bus=self.core.event_bus,
                get_recent_docs_fn=lambda: self.core.document_store.list_documents(limit=5),
                log_fn=self.core.log,
                stop_check_fn=self.core.stop_check,
                epigenetics=self.core.epigenetics
            )
            self.core.netzach_force = self.core.observer
            
            # Initialize Hod (Reflective Analyst)
            self.core.hod = HodAgent(
                memory_store=self.core.memory_store,
                meta_memory_store=self.core.meta_memory_store,
                reasoning_store=self.core.reasoning_store,
                document_store=self.core.document_store,
                get_settings_fn=self.core.get_settings,
                get_main_logs_fn=self.core.get_logs,
                get_doc_logs_fn=self.core.get_doc_logs,
                log_fn=self.core.log,
                meta_learner=self.core.meta_learner,
                embed_fn=self.core.get_embedding_fn(),
                event_bus=self.core.event_bus
            )
            self.core.hod_force = self.core.hod

            # Initialize Decider
            self.core.decider = Decider(
                get_settings_fn=self.core.get_settings,
                update_settings_fn=self.core.update_settings,
                memory_store=self.core.memory_store,
                document_store=self.core.document_store,
                reasoning_store=self.core.reasoning_store,
                arbiter=self.core.arbiter,
                meta_memory_store=self.core.meta_memory_store,
                actions={
                    **self.core.action_manager.get_tools(),
                    "start_daydream": start_daydream_wrapper,
                    "start_daydream_batch": start_daydream_batch_wrapper,
                    "verify_batch": verify_batch_wrapper,
                    "philosophize": philosophize_wrapper,
                    # "verify_all": verify_all_wrapper, # Add other wrappers
                    "stop_daydream": self.core.stop_daydream_fn,
                    "run_observer": run_observer_wrapper,
                    "run_hod": run_hod_wrapper,
                    "remove_goal": remove_goal_wrapper,
                    "list_documents": list_documents_wrapper,
                    "read_document": read_document_wrapper,
                    "search_memory": search_memory_wrapper,
                    "summarize": lambda: self.core.daat.run_summarization() if self.core.daat else "Daat missing",
                    "compress_reasoning": lambda: self.core.daat.run_reasoning_compression() if self.core.daat else False,
                    "consolidate_summaries": lambda: self.core.daat.consolidate_summaries() if self.core.daat else "Daat missing",
                    "sync_journal": lambda: (self.core.sync_journal_fn(), "Journal synced")[1],
                    "prune_memory": prune_memory_wrapper,
                    "refute_memory": refute_memory_wrapper,
                    "search_internet": lambda q, s: self.core.action_manager.safe_search(q, s),
                    "simulate_counterfactual": lambda p: self.core.daat.run_counterfactual_simulation(p) if self.core.daat else "Daat missing"
                },
                log_fn=self.core.log,
                chat_fn=self.core.chat_fn,
                get_chat_history_fn=self.core.get_chat_history,
                stop_check_fn=self.core.stop_check,
                event_bus=self.core.event_bus,
                hesed=self.core.hesed,
                gevurah=self.core.gevurah,
                hod=self.core.hod_force,
                netzach=self.core.netzach_force,
                binah=self.core.binah,
                dialectics=self.core.dialectics,
                keter=self.core.keter,
                daat=self.core.daat,
                malkuth=self.core.malkuth,
                yesod=self.core.yesod,
                temp_step=self.core.epigenetics.get("decider_temp_step", 0.20),
                token_step=self.core.epigenetics.get("decider_token_step", 0.20),
                crs=self.core.crs,
                epigenetics=self.core.epigenetics,
                value_core=self.core.value_core,
                stability_controller=self.core.stability_controller,
                heartbeat=self.core.heartbeat,
                global_workspace=self.core.global_workspace,
                executor=self.core.thread_pool
            )

            # ... (Event Bus Wiring) ...
            self.core.event_bus.subscribe("GOAL_COMPLETED", self.core._on_goal_completed)
            self.core.event_bus.subscribe("MEMORY_CONSOLIDATED", lambda e: self.core.log(f"🧠 Consolidated: {e.data.get('count')} memories"))
            self.core.event_bus.subscribe("DAAT:SYNTHESIZE", lambda e: self.core.daat.run_cross_domain_synthesis() if self.core.daat else None)
            self.core.event_bus.subscribe("SYSTEM:PANIC", lambda e: self.core.decider.on_panic(e) if self.core.decider else None)
            self.core.event_bus.subscribe("SYSTEM:SPONTANEOUS_SPEECH", lambda e: self.core.log(f"🗣️ Event Bus: Spontaneous Speech Triggered -> {e.data.get('context')}"))

            # Trigger initial analysis
            if self.core.hod:
                self.core.thread_pool.submit(self.core.hod.reflect, "System Startup")
            
            # Restore Identity
            self.core.identity_manager.restore_subjective_continuity()

            logging.info("🧠 Brain initialized successfully (AICore).")
            
            # Post-init validation
            if not self.core.memory_store: logging.error("❌ MemoryStore failed to initialize.")
            if not self.core.decider: logging.error("❌ Decider failed to initialize.")
            if not self.core.self_model: logging.error("❌ SelfModel failed to initialize.")
            
            # Critical Failure Check
            if not self.core.memory_store or not self.core.self_model:
                raise RuntimeError("Critical components (MemoryStore/SelfModel) failed to initialize. Aborting startup.")

        except Exception as e:
            logging.error(f"⚠️ Failed to initialize AI components fully: {e}")
            logging.warning("⚠️ Starting in SAFE MODE (Lobotomized). Some features may be unavailable.")
            # Do not raise, allow partial startup