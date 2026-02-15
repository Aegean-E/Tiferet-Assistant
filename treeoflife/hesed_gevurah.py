"""
Sephirot Forces: Hesed, Gevurah

These are stateless functions (forces) that provide raw signals to Tiferet (Decider).
They do not decide. They inform.
"""
import logging
from treeoflife.sephirah import Sephirah

class Hesed(Sephirah):
    """
    Hesed = Permission Budget (Expansion).
    Signal: 0.0 (Frozen) to 1.0 (Max Expansion).

    "How much unverified novelty is allowed right now?"
    """
    def __init__(self, memory_store, keter, log_fn=logging.info, bias: float = 1.0):
        super().__init__("Hesed", "Mercy: Permission & Expansion", log_fn)
        self.memory_store = memory_store
        self.keter = keter
        # self.log handled by super
        self.bias = bias
        self.smoothed_value = 1.0 # Initialize to max permission

    def calculate(self) -> float:
        stats = self.memory_store.get_memory_stats()
        total_mem = stats.get('total_memories', 100) # Default to 100 to avoid div by zero issues early on
        unverified = stats.get('unverified_beliefs', 0) + stats.get('unverified_facts', 0)
        
        # Base Allowance
        budget = 1.0
        
        # Penalize accumulation (Percentile-Based)
        # If unverified > 5% of total, start restricting
        ratio = unverified / max(1, total_mem)
        if ratio > 0.05: budget = 0.8
        if ratio > 0.10: budget = 0.5
        if ratio > 0.15: budget = 0.2
        if ratio > 0.20: budget = 0.0
        
        # Keter Influence
        if self.keter and hasattr(self.keter, 'last_delta'):
            delta = self.keter.last_delta
            # Delta is usually small (-0.05 to +0.05)
            if delta > 0: budget += 0.1
            if delta < 0: budget -= 0.2
            
        # Apply Epigenetic Bias
        budget *= self.bias
        
        # Smoothing (Decay/Rise regulation)
        # If budget drops, drop fast (safety). If it rises, rise slow (trust building).
        if budget < self.smoothed_value:
            alpha = 0.5 # Fast drop
        else:
            alpha = 0.1 # Slow rise
            
        self.smoothed_value = (alpha * budget) + ((1 - alpha) * self.smoothed_value)
        
        final = max(0.0, min(self.smoothed_value, 1.0))
        
        # Log only if constraining
        if final < 1.0:
            self.log(f"🌊 HESED: Unverified={unverified} | Permission={final:.2f}")
            
        return final


class Gevurah(Sephirah):
    """
    Gevurah = Constraint Budget (Restriction).
    Signal: 0.0 (Relaxed) to 1.0 (Emergency Cut).
    """
    def __init__(self, memory_store, keter=None, log_fn=logging.info, bias: float = 1.0, meta_memory_store=None):
        super().__init__("Gevurah", "Severity: Constraint & Pruning", log_fn)
        self.memory_store = memory_store
        self.keter = keter
        # self.log handled by super
        self.bias = bias
        self.meta_memory_store = meta_memory_store
        self.recommendation = None # "DROP_GOAL", "SLEEP", etc.
        self.smoothed_value = 0.0 # Initialize to no pressure

    def calculate(self) -> float:
        # 1. Fetch Deep Stats
        stats = self.memory_store.get_memory_stats()
        total_mem = stats.get('total_memories', 100)
        active_goals = stats.get('active_goals', 0)
        unverified = stats.get('unverified_facts', 0)
        self.recommendation = None
        
        pressure = 0.0
        causes = []

        # --- DEBUG LOGGING START ---
        
        # 2. Logic (RELAXED SETTINGS)
        # Percentile-Based Thresholds
        # If active goals > 2% of total memory (heuristic for "too many threads")
        # Fix: Use a fixed active window size (e.g., 500) to prevent Gevurah from weakening as DB grows
        active_window_size = 500
        goal_ratio = active_goals / active_window_size
        
        if goal_ratio > 0.04: # > 20 goals (4% of 500)
            pressure += 0.1
            causes.append(f"Goals({active_goals})>4%")
        elif goal_ratio > 0.10: # > 50 goals (10% of 500)
            pressure += 0.2
            causes.append(f"Goals({active_goals})>10%")
        
        # Memory pressure is less about count and more about coherence, handled by Keter.
        # But we can keep a sanity check for massive bloat if needed.
        # Removed hardcoded 5000/10000 limits to allow scaling.
            
        # Factor 3: Repetition Loop (Psychological Balance)
        # Check last 5 memories for identical content to detect "Stuck" state
        try:
            recent = self.memory_store.list_recent(limit=5)
            if recent:
                texts = [r[3] for r in recent] # text is index 3
                if texts:
                    latest = texts[0]
                    if texts.count(latest) > 2:
                        pressure += 0.5
                        causes.append(f"Repetition Loop detected (+0.5)")
        except Exception:
            pass
            
        # Factor 3.5: Action Loop Detection (The "Stuck" Fix)
        # If we keep verifying or failing, we must stop.
        if self.meta_memory_store:
            try:
                # Check last 10 actions
                recent_actions = self.meta_memory_store.list_recent(limit=10)
                # item: (id, event_type, subject, text, created_at, affect)
                # Filter for DECIDER_ACTION
                actions = [r[3] for r in recent_actions if r[1] == "DECIDER_ACTION"]
                
                # Check for VERIFY loop
                verify_count = sum(1 for a in actions if "verify" in a.lower())
                if verify_count >= 3:
                    pressure = 1.0 # Max pressure
                    causes.append("Action Loop (Verify)")
                    self.recommendation = "DROP_GOAL" # Signal Tiferet to drop the current goal
            except Exception: pass

        # Factor 4: Keter Coupling (Dynamic Stability)
        # If Coherence is low, Gevurah MUST rise to constrain entropy.
        if self.keter:
            try:
                keter_stats = self.keter.evaluate()
                coherence = keter_stats.get("keter", 1.0)
                if coherence < 0.3:
                    pressure += 0.5
                    causes.append(f"Critical Coherence ({coherence:.2f})")
                elif coherence < 0.5:
                    pressure += 0.2
                    causes.append(f"Low Coherence ({coherence:.2f})")
            except Exception as e:
                self.log(f"⚠️ Gevurah: Failed to evaluate Keter: {e}")

        # Apply Epigenetic Bias
        pressure *= self.bias
        
        # Smoothing
        # If pressure rises, rise fast (safety). If it drops, drop slow (caution).
        if pressure > self.smoothed_value:
            alpha = 0.5 # Fast rise
        else:
            alpha = 0.1 # Slow drop
            
        self.smoothed_value = (alpha * pressure) + ((1 - alpha) * self.smoothed_value)
        
        final_pressure = min(self.smoothed_value, 1.0)

        # 3. Log to UI
        # We log the raw stats so you can verify the 'fix' worked
        status_msg = f"🧠 STATS: Goals={active_goals} | Unverified={unverified}"
        if final_pressure > 0:
            self.log(f"🔥 GEVURAH: {status_msg} | Pressure={final_pressure:.2f} ({', '.join(causes)})")
        else:
            # Optional: Comment this out if it's too noisy, but good for confirmation
            self.log(f"✅ GEVURAH: {status_msg} | System Stable")
        
        return final_pressure