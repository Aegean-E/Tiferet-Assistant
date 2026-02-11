"""
Sephirot Forces: Hesed, Gevurah

These are stateless functions (forces) that provide raw signals to Tiferet (Decider).
They do not decide. They inform.
"""

class Hesed:
    """
    Hesed = Permission Budget (Expansion).
    Signal: 0.0 (Frozen) to 1.0 (Max Expansion).

    "How much unverified novelty is allowed right now?"
    """
    def __init__(self, memory_store, keter, log_fn=print):
        self.memory_store = memory_store
        self.keter = keter
        self.log = log_fn  # Connect to UI Logger

    def calculate(self) -> float:
        stats = self.memory_store.get_memory_stats()
        unverified = stats.get('unverified_beliefs', 0) + stats.get('unverified_facts', 0)
        
        # Base Allowance
        budget = 1.0
        
        # Penalize accumulation
        if unverified > 15: budget = 0.8
        if unverified > 30: budget = 0.5
        if unverified > 50: budget = 0.2
        if unverified > 80: budget = 0.0
        
        # Keter Influence
        if self.keter and hasattr(self.keter, 'last_delta'):
            delta = self.keter.last_delta
            # Delta is usually small (-0.05 to +0.05)
            if delta > 0: budget += 0.1
            if delta < 0: budget -= 0.2
            
        final = max(0.0, min(budget, 1.0))
        
        # Log only if constraining
        if final < 1.0:
            self.log(f"🌊 HESED: Unverified={unverified} | Permission={final:.2f}")
            
        return final


class Gevurah:
    """
    Gevurah = Constraint Budget (Restriction).
    Signal: 0.0 (Relaxed) to 1.0 (Emergency Cut).
    """
    def __init__(self, memory_store, log_fn=print):
        self.memory_store = memory_store
        self.log = log_fn  # Connect to UI Logger

    def calculate(self) -> float:
        # 1. Fetch Deep Stats
        stats = self.memory_store.get_memory_stats()
        total_mem = self.memory_store.count_all()
        
        # 2. Active Goals (Parallelism Load)
        stats = self.memory_store.get_memory_stats()
        active_goals = stats.get('active_goals', 0)
        facts = stats.get('facts', 0)
        unverified = stats.get('unverified_facts', 0)
        
        pressure = 0.0
        causes = []

        # --- DEBUG LOGGING START ---
        
        # 2. Logic (RELAXED SETTINGS)
        if active_goals > 50: 
            pressure += 0.1
            causes.append(f"Goals({active_goals})>50")
        elif active_goals > 100: 
            pressure += 0.2
            causes.append(f"Goals({active_goals})>100")
        
        if total_mem > 5000: 
            pressure += 0.1
            causes.append(f"Mem({total_mem})>5k")
        if total_mem > 10000: 
            pressure += 0.2
            
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
            
        final_pressure = min(pressure, 1.0)

        # FORCE PRINT TO CONSOLE
        print(f"📊 GEVURAH X-RAY: Pressure={final_pressure:.2f} | Causes: {', '.join(causes)}")

        # 3. Log to UI
        # We log the raw stats so you can verify the 'fix' worked
        status_msg = f"🧠 STATS: Goals={active_goals} | Facts={facts} | Unverified={unverified}"
        if final_pressure > 0:
            self.log(f"🔥 GEVURAH: {status_msg} | Pressure={final_pressure:.2f} ({', '.join(causes)})")
        else:
            # Optional: Comment this out if it's too noisy, but good for confirmation
            self.log(f"✅ GEVURAH: {status_msg} | System Stable")
        
        return final_pressure