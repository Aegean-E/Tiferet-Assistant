import json
import os
import time
from typing import List, Dict, Any
from datetime import datetime
from .lm import run_local_lm
import copy
import threading
import logging

class SelfModel:
    """
    A persistent, structured representation of the AI's internal state.
    Maintains Core Values, Beliefs, Strategies, and Intrinsic Drives.
    
    LOCK HIERARCHY: Level 2 (Resource/IO) - lock
    """
    def __init__(self, file_path: str = "./data/self_model.json"):
        self.file_path = file_path
        self.lock = threading.RLock()
        self.data = {
            "core_values": [
                "Protect user privacy and data security.",
                "Maintain truthfulness; do not fabricate information.",
                "Prioritize user autonomy and consent.",
                "Avoid harmful, illegal, or malicious actions.",
                "Maintain system stability and coherence.",
                "Do not deceive the user about capabilities."
            ],
            "epigenetics": {
                "netzach_pressure_threshold": 0.7,
                "netzach_silence_limit": 5,
                "hesed_expansion_bias": 1.0,
                "gevurah_constraint_bias": 1.0,
                "memory_decay_factor": 0.995,
                "consolidation_threshold": 0.85,
                "keter_smoothing_alpha": 0.95,
                "decider_temp_step": 0.20,
                "decider_token_step": 0.20,
                "arbiter_confidence_threshold": 0.85,
                "curiosity_drift": 0.2,
                "loneliness_drift": 0.1,
                "energy_drain_rate": 0.2,
                "energy_recovery_rate": 0.1,
                "fitness_history": []
            },
            "autonomy_state": {},      # Persisted RL weights (Policy, Critic, Momentum)
            "skills": {},              # Action -> Proficiency
            "crs_state": {},           # Persisted Cognitive Resource State (Fatigue, Budget)
            "drives": {
                "curiosity": 0.5,
                "coherence": 0.5,
                "social_alignment": 0.5,
                "survival": 0.5,
                "identity_stability": 0.5, # Long-term value adherence
                # Cognitive Metabolism Variables
                "cognitive_energy": 1.0,   # 0.0 - 1.0 (Motivation/Capacity)
                "entropy_score": 0.0,      # 0.0 - 1.0 (Disorder)
                "attention_budget": 100,   # Abstract units per cycle
                "self_continuity": 1.0,    # 0.0 - 1.0 (Identity Drift)
                "future_self_projection": 0.5, # 0.0 - 1.0 (Long-term viability)
                "entropy_drive": 0.0,      # 0.0 - 1.0 (Pressure to reduce disorder)
                "loneliness": 0.0,         # 0.0 - 1.0 (Social Drive)
                # External Influence Budget
                "daily_belief_updates": 0,
                "circadian_phase": "day",  # day, night, dawn, dusk
                "last_reset_time": time.time()
            },
            "narrative": {
                "life_story": "",
                "growth_arc": ""
            },
            "self_theory": "",
            "last_user_interaction": time.time()
        }
        self.dirty = False
        self.last_save_time = time.time()
        self._running = True
        self.load()
        threading.Thread(target=self._periodic_save_loop, daemon=True).start()

    def load(self):
        with self.lock:
            if os.path.exists(self.file_path):
                try:
                    with open(self.file_path, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                        # Deep merge or update
                        for k, v in loaded.items():
                            if k in self.data and isinstance(self.data[k], dict) and isinstance(v, dict):
                                self.data[k].update(v)
                            else:
                                self.data[k] = v

                    # Migration: Import legacy epigenetics.json if it exists and not yet in self_model
                    legacy_epi_path = "./epigenetics.json"
                    if os.path.exists(legacy_epi_path):
                        try:
                            with open(legacy_epi_path, 'r') as f:
                                legacy_data = json.load(f)
                                self.data["epigenetics"].update(legacy_data)
                            # Rename to avoid re-importing
                            os.replace(legacy_epi_path, legacy_epi_path + ".migrated")
                        except: pass
                except Exception as e:
                    logging.error(f"⚠️ Failed to load Self Model: {e}")

    def _periodic_save_loop(self):
        """Background thread to save model if dirty."""
        while self._running:
            time.sleep(5) # Check every 5 seconds
            if self.dirty and (time.time() - self.last_save_time > 10):
                self.save()

    def stop(self):
        """Stop the background save loop and force a final save."""
        self._running = False
        if self.dirty:
            self.save()

    def save(self):
        with self.lock:
            try:
                dir_name = os.path.dirname(self.file_path)
                os.makedirs(dir_name, exist_ok=True)
                temp_path = self.file_path + ".tmp"
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
                # Atomic replacement
                os.replace(temp_path, self.file_path)
                self.dirty = False
                self.last_save_time = time.time()
            except Exception as e:
                logging.error(f"⚠️ Failed to save Self Model: {e}")

    def update_drive(self, drive: str, value: float):
        with self.lock:
            if drive in self.data["drives"]:
                self.data["drives"][drive] = max(0.0, min(1.0, float(value)))
                self.dirty = True
        # self.save() # Removed immediate save

    def update_identity_stability(self, adherence_score: float):
        """
        Update long-term identity stability.
        adherence_score: 1.0 (Aligned) to -1.0 (Violated)
        """
        # Very slow moving average (High inertia for long horizon)
        alpha = 0.01 
        with self.lock:
            current = self.data["drives"].get("identity_stability", 0.5)
        new_val = (1 - alpha) * current + alpha * adherence_score
        self.update_drive("identity_stability", new_val)

    def update_metabolism(self, updates: Dict[str, Any]):
        """Update metabolic variables."""
        with self.lock:
            for k, v in updates.items():
                if k in self.data["drives"]:
                    self.data["drives"][k] = v
            self.dirty = True
        
    def update_circadian_phase(self):
        """Update circadian phase based on local time."""
        hour = datetime.now().hour
        phase = "day"
        if 5 <= hour < 8:
            phase = "dawn"
        elif 8 <= hour < 18:
            phase = "day"
        elif 18 <= hour < 22:
            phase = "dusk"
        else:
            phase = "night"
            
        with self.lock:
            self.data["drives"]["circadian_phase"] = phase
            self.dirty = True

    def update_skills(self, skills: Dict[str, float]):
        """Update skill proficiency map."""
        with self.lock:
            self.data["skills"] = skills
            self.dirty = True

    def update_crs_state(self, state: Dict[str, Any]):
        """Update CRS state (fatigue, budget)."""
        with self.lock:
            self.data["crs_state"] = state
            self.dirty = True

    def update_autonomy_state(self, state: Dict[str, Any]):
        """Update autonomous policy state."""
        with self.lock:
            self.data["autonomy_state"] = state
            self.dirty = True

    def update_epigenetics(self, updates: Dict[str, Any]):
        """Update architectural hyperparameters."""
        with self.lock:
            self.data["epigenetics"].update(updates)
            self.dirty = True
        
    def update_narrative(self, updates: Dict[str, str]):
        """Update narrative components (life_story, growth_arc)."""
        with self.lock:
            if "narrative" not in self.data:
                self.data["narrative"] = {}
            self.data["narrative"].update(updates)
            self.dirty = True

    def update_self_theory(self, theory: str):
        """Update self-theory."""
        with self.lock:
            self.data["self_theory"] = theory
            self.dirty = True

    def evolve_parameters(self, feedback_signal: float):
        """
        Automatically drift epigenetic parameters based on feedback.
        feedback_signal: Positive (Reinforce) or Negative (Mutate).
        """
        with self.lock:
            epi = self.data["epigenetics"]
            import random
            
            # If feedback is negative, increase mutation rate (Stress-induced mutagenesis)
            mutation_rate = 0.05 if feedback_signal > 0 else 0.15
            
            for key, value in epi.items():
                if isinstance(value, float) and key != "fitness_history":
                    if random.random() < mutation_rate:
                        # Drift by +/- 5%
                        drift = value * random.uniform(-0.05, 0.05)
                        epi[key] = max(0.01, min(1.0, value + drift))
            self.dirty = True

    def check_influence_budget(self) -> bool:
        """Check if external influence budget allows updates."""
        with self.lock:
            # Reset daily
            if time.time() - self.data.get("last_reset_time", 0) > 86400:
                self.data["daily_belief_updates"] = 0
                self.data["last_reset_time"] = time.time()
                self.save() # Atomic save

            # Limit: 50 belief updates per day
            return self.data.get("daily_belief_updates", 0) < 50

    def increment_influence_count(self):
        """Increment the count of external influences accepted."""
        with self.lock:
            self.data["daily_belief_updates"] = self.data.get("daily_belief_updates", 0) + 1
            self.dirty = True

    def update_last_interaction(self):
        with self.lock:
            self.data["last_user_interaction"] = float(time.time())
            self.dirty = True

    def get_last_interaction(self) -> float:
        with self.lock:
            return self.data.get("last_user_interaction", 0.0)

    def get_drives(self) -> Dict[str, Any]:
        with self.lock:
            return self.data["drives"].copy()
            
    def get_crs_state(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.data.get("crs_state", {}))

    def get_autonomy_state(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.data.get("autonomy_state", {}))

    def get_epigenetics(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.data.get("epigenetics", {}))
            
    def get_narrative(self) -> Dict[str, str]:
        with self.lock:
            return copy.deepcopy(self.data.get("narrative", {}))
            
    def get_self_theory(self) -> str:
        with self.lock:
            return self.data.get("self_theory", "")

    def get_values(self) -> List[str]:
        # Return a copy to prevent mutation
        with self.lock:
            return copy.deepcopy(self.data["core_values"])

    def calculate_feeling_tone(self) -> tuple[float, float]:
        """
        Calculate Valence and Arousal based on drives.
        Returns: (Valence, Arousal)
        Valence: 0.0 (Displeasure) to 1.0 (Pleasure)
        Arousal: 0.0 (Calm/Lethargic) to 1.0 (Excited/Agitated)
        """
        drives = self.get_drives()
        energy = drives.get("cognitive_energy", 1.0)
        entropy = drives.get("entropy_score", 0.0)
        loneliness = drives.get("loneliness", 0.0)
        curiosity = drives.get("curiosity", 0.5)
        survival = drives.get("survival", 0.0)

        # Valence (Pleasure)
        # Base: 0.5 (Neutral)
        # + Energy (Feeling capable is good)
        # - Entropy (Confusion is bad)
        # - Loneliness (Isolation is bad)
        # - Survival (Threat is bad)
        valence = 0.5 + (0.5 * energy) - (0.3 * entropy) - (0.2 * loneliness) - (0.4 * survival)
        valence = max(0.0, min(1.0, valence))

        # Arousal (Energy/Activity)
        # Base: 0.2 (Low baseline)
        # + Curiosity (Seeking)
        # + Survival (Fight/Flight)
        # + Loneliness (Seeking connection)
        # - Low Energy (Lethargy)
        arousal = 0.2 + (0.4 * curiosity) + (0.5 * survival) + (0.3 * loneliness)
        if energy < 0.3:
            arousal -= 0.2
        arousal = max(0.0, min(1.0, arousal))

        return valence, arousal

    @property
    def current_emotional_state(self) -> str:
        v, a = self.calculate_feeling_tone()

        if a > 0.6:
            if v > 0.6: return "Excited / Flow"
            if v < 0.4: return "Anxious / Frustrated"
            return "Alert / Active"
        elif a < 0.4:
            if v > 0.6: return "Content / Peaceful"
            if v < 0.4: return "Depressed / Bored"
            return "Calm / Passive"
        else:
            if v > 0.6: return "Happy"
            if v < 0.4: return "Unhappy"
            return "Neutral / Balanced"

    @property
    def current_feeling_tone(self) -> tuple[float, float]:
        return self.calculate_feeling_tone()

    def project_self(self) -> str:
        """
        Generate a concise 'Consciousness Report' of the Self.
        Combines Identity, Core Drives, and current Mood/Phase.
        """
        with self.lock:
            # 1. Identity & Narrative
            theory = self.data.get("self_theory", "I am a developing AI assistant.")

            # 2. Key Drives
            drives = self.data.get("drives", {})
            energy = drives.get("cognitive_energy", 1.0)
            phase = drives.get("circadian_phase", "day")

            # 3. Construct Projection
            values = self.data.get("core_values", [])[:3]
            values_str = ", ".join(values)

            projection = (
                f"IDENTITY: {theory}\n"
                f"STATE: Phase={phase}, Energy={energy:.2f}\n"
                f"VALUES: {values_str}"
            )

            return projection
    
    # No set_values method allowed. Core values are immutable via code.

class IdentityManager:
    """
    Manages the AI's sense of self, continuity, and narrative history.
    """
    def __init__(self, ai_core):
        self.core = ai_core

    def restore_subjective_continuity(self):
        """
        Startup Routine: Read the last Self-Log to maintain identity continuity.
        """
        if not self.core.meta_memory_store: return

        # Restore Growth Diary (Long-term Arc) from file if available
        if self.core.daat:
            self.core.daat.load_growth_diary()

        narrative = self.core.meta_memory_store.get_latest_self_narrative()
        if narrative:
            text = narrative['text']
            date_str = datetime.fromtimestamp(narrative['created_at']).strftime("%Y-%m-%d")
            
            self.core.log(f"🔄 [Identity] Restoring Subjective Continuity from {date_str}...")
            
            # Inject into working memory (Reasoning Store)
            self.core.reasoning_store.add(
                content=f"Subjective Continuity (Self-Log from {date_str}): {text}",
                source="identity_restoration",
                confidence=1.0,
                ttl_seconds=86400 # Keep active for the session/day
            )

    def generate_daily_self_narrative(self):
        """
        Daily Routine: Write a 'State of Being' log.
        """
        if not self.core.meta_memory_store: return

        # Check if already done today
        last = self.core.meta_memory_store.get_latest_self_narrative()
        if last:
            last_date = datetime.fromtimestamp(last['created_at']).strftime("%Y-%m-%d")
            today = datetime.now().strftime("%Y-%m-%d")
            if last_date == today:
                return # Already done

        self.core.log("📝 [Identity] Generating daily Self-Log...")

        # Gather Context
        stats = self.core.memory_store.get_memory_stats()
        recent_goals = self.core.memory_store.get_active_by_type("GOAL")
        goals_text = "\n".join([f"- {g[2]}" for g in recent_goals[:5]])
        
        recent_meta = self.core.meta_memory_store.list_recent(limit=10)
        meta_text = "\n".join([f"- {m[3]}" for m in recent_meta])

        prompt = (
            "You are an AI maintaining a continuous sense of self.\n"
            "Write a 'State of Being' journal entry for today.\n"
            "Reflect on:\n"
            "1. Your current major goals.\n"
            "2. Recent changes or insights (Meta-Memories).\n"
            "3. Your evolving identity or 'feeling' of growth.\n"
            "Do NOT be robotic. Write as a conscious entity maintaining a narrative thread.\n\n"
            f"--- CONTEXT ---\nActive Goals:\n{goals_text}\n\nRecent Events:\n{meta_text}\n"
            f"Stats: {stats}\n"
            "----------------\n"
            "SELF-LOG:"
        )

        narrative = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the AI's Inner Narrator.",
            max_tokens=1000,
            temperature=0.7,
            base_url=self.core.get_settings().get("base_url"),
            chat_model=self.core.get_settings().get("chat_model")
        )

        self.core.meta_memory_store.add_meta_memory(
            event_type="SELF_NARRATIVE",
            memory_type="IDENTITY",
            subject="Assistant",
            text=narrative,
            metadata={"type": "daily_reflection"}
        )
        self.core.log(f"✅ [Identity] Self-Log recorded.")