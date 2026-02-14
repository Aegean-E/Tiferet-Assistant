import time
from typing import Dict, Any

class DriveSystem:
    """
    Manages the internal homeostatic drives of the AI.
    Simulates needs like Curiosity, Social Connection, and Coherence.
    """
    def __init__(self, core):
        self.core = core
        self.last_update = time.time()
        
    def update(self, observation: Dict[str, Any] = None):
        """
        Update drive levels based on time delta and system state.
        Called periodically by the metabolic loop.
        """
        if not self.core.self_model: return

        now = time.time()
        # Time delta in hours
        dt = (now - self.last_update) / 3600.0
        # Clamp dt to avoid massive jumps if system slept
        dt = min(dt, 1.0) 
        self.last_update = now
        
        drives = self.core.self_model.get_drives()
        epi = self.core.self_model.get_epigenetics()
        
        # 1. Curiosity (Need for Novelty)
        # Drifts up over time (Boredom)
        curiosity = drives.get("curiosity", 0.5)
        curiosity_drift = epi.get("curiosity_drift", 0.2) # 20% per hour
        curiosity = min(1.0, curiosity + (curiosity_drift * dt))
        
        # Boost from Netzach signal
        if observation and observation.get("signal") == "LOW_NOVELTY":
            curiosity = min(1.0, curiosity + 0.05)
            
        # 2. Loneliness (Social Drive)
        # Drifts up over time since last interaction
        loneliness = drives.get("loneliness", 0.0)
        # We calculate drift based on time since last interaction, but here we just apply incremental drift
        # Actually, let's recalculate based on absolute time since interaction for accuracy
        last_interaction = self.core.self_model.data.get("last_user_interaction", 0)
        if last_interaction > 0:
            hours_since = (now - last_interaction) / 3600.0
            social_drift = epi.get("loneliness_drift", 0.1) # 10% per hour
            loneliness = min(1.0, hours_since * social_drift)
        
        # 3. Entropy/Coherence (State-based)
        # This reflects the current disorder of the system
        stats = self.core.memory_store.get_memory_stats()
        unverified = stats.get('unverified_facts', 0) + stats.get('unverified_beliefs', 0)
        active_goals = stats.get('active_goals', 0)
        
        # Entropy Score (0-1)
        entropy = min(1.0, (unverified * 0.05) + (active_goals * 0.02))
        entropy_drive = entropy 
        
        # 4. Cognitive Energy (Metabolic)
        # Drains with entropy, recovers with time (base recovery)
        energy = drives.get("cognitive_energy", 1.0)
        
        # Drain based on entropy (stress)
        drain_rate = epi.get("energy_drain_rate", 0.2)
        energy_drain = entropy * drain_rate * dt
        
        # Recovery (Base metabolic recovery)
        recovery_rate = epi.get("energy_recovery_rate", 0.1)
        energy_recovery = recovery_rate * dt
        
        energy = max(0.0, min(1.0, energy - energy_drain + energy_recovery))
        
        # Update SelfModel
        self.core.self_model.update_metabolism({
            "curiosity": curiosity,
            "loneliness": loneliness,
            "entropy_drive": entropy_drive,
            "cognitive_energy": energy,
            "entropy_score": entropy
        })
        
    def satisfy_drive(self, drive_name: str, amount: float):
        """Reduce a drive (satisfaction)."""
        drives = self.core.self_model.get_drives()
        if drive_name in drives:
            current = drives[drive_name]
            new_val = max(0.0, current - amount)
            self.core.self_model.update_drive(drive_name, new_val)
            self.core.log(f"📉 Drive Satisfied: {drive_name} {current:.2f} -> {new_val:.2f}")