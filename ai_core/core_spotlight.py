import time
from typing import Dict, Any, Optional

class GlobalWorkspace:
    """
    The Global Workspace (Spotlight of Attention).
    Selects the most salient information (Drive, Goal, or Event) and broadcasts it
    as CONSCIOUS_CONTENT to all modules.
    """
    def __init__(self, core):
        self.core = core
        self.current_focus = None
        self.last_update = time.time()
        self.update_interval = 10 # Seconds
        self.last_broadcast = time.time()

    def update(self):
        """
        Calculate salience of competing signals and update the Spotlight.
        Competitive Salience: Urgency, Novelty, and Emotional Weight.
        """
        if time.time() - self.last_update < self.update_interval:
            return

        self.last_update = time.time()
        candidates = []

        # 1. Drives (Internal Pressure)
        if self.core.self_model:
            drives = self.core.self_model.get_drives()
            mood = getattr(self.core.decider, 'mood', 0.5) if self.core.decider else 0.5
            
            for drive, value in drives.items():
                if drive in ["daily_belief_updates", "last_reset_time", "attention_budget", "circadian_phase"]: continue
                
                # Dynamic weight based on mood (Emotional Bias)
                # If manic (high mood), expansion/entropy is more salient.
                # If depressed (low mood), survival/caution is more salient.
                weight = 1.0
                if drive == "entropy_score": weight = 1.0 + (mood * 0.5)
                if drive == "survival": weight = 1.0 + ((1.0 - mood) * 0.5)
                if drive == "loneliness": weight = 1.2
                
                if isinstance(value, (int, float)):
                    s_val = value
                    content_prefix = "High"
                    if drive == "cognitive_energy":
                        s_val = 1.0 - value
                        content_prefix = "Low"

                    candidates.append({
                        "type": "DRIVE",
                        "content": f"{content_prefix} {drive} ({value:.2f})",
                        "salience": s_val * weight,
                        "urgency": s_val,
                        "novelty": 0.5,  # Static for drives
                        "data": {drive: value}
                    })

        # 2. Active Goals (Volition)
        if self.core.memory_store:
            goals = self.core.memory_store.get_active_by_type("GOAL")
            for g in goals:
                priority = g[4] if len(g) > 4 else 0.5
                # Urgency: Assume recent goals (higher ID) are slightly more urgent
                urgency = priority * (1.0 + (g[0] % 100) / 1000.0)
                
                candidates.append({
                    "type": "GOAL",
                    "content": f"Goal: {g[2]}",
                    "salience": priority * 1.1,
                    "urgency": urgency,
                    "novelty": 0.3,
                    "data": {"id": g[0], "text": g[2]}
                })

        # 3. Sensory Input (External Stimulus)
        last_interaction = 0
        if self.core.self_model:
             last_interaction = self.core.self_model.get_last_interaction()
             
        time_since_chat = time.time() - last_interaction
        if time_since_chat < 60:
            salience = 2.0 * (1.0 - (time_since_chat / 60.0))
            candidates.append({
                "type": "USER_INPUT",
                "content": "User is interacting",
                "salience": salience,
                "urgency": salience * 0.8,
                "novelty": 0.9, # External input is usually novel
                "data": {"time_since": time_since_chat}
            })

        # 4. Internal Events (Anomalies/Surprises)
        if self.core.meta_memory_store:
            # Check for recent SURPRISE_EVENT or PANIC
            surprises = self.core.meta_memory_store.get_by_event_type("SURPRISE_EVENT", limit=1)
            if surprises:
                t = surprises[0].get('created_at', 0)
                if time.time() - t < 300: # Last 5 mins
                    candidates.append({
                        "type": "ANOMALY",
                        "content": f"Surprise: {surprises[0]['text']}",
                        "salience": 1.5,
                        "urgency": 1.0,
                        "novelty": 1.0,
                        "data": surprises[0]
                    })

        # 5. Competition (Winner-Take-All)
        if not candidates:
            self.current_focus = None
            return

        # Weighted Salience
        # Final Salience = salience * 0.5 + urgency * 0.3 + novelty * 0.2
        for c in candidates:
            c["final_salience"] = (c["salience"] * 0.5) + (c.get("urgency", 0.5) * 0.3) + (c.get("novelty", 0.5) * 0.2)

        candidates.sort(key=lambda x: x["final_salience"], reverse=True)
        winner = candidates[0]

        # Broadcast to all modules
        is_new_focus = self.current_focus != winner["content"]
        is_urgent = winner["final_salience"] > 0.95
        time_since_last = time.time() - self.last_broadcast

        if is_new_focus or (is_urgent and time_since_last > 60):
            self.current_focus = winner["content"]
            self.last_broadcast = time.time()
            self.broadcast(winner)

    def broadcast(self, content: Dict[str, Any]):
        """
        Broadcast conscious content to all modules via EventBus.
        Mimics global availability in biological minds.
        """
        if self.core.event_bus:
            self.core.log(f"🔦 Workspace: Broadcasting '{content['content']}' (Salience: {content['final_salience']:.2f})")
            self.core.event_bus.publish(
                "CONSCIOUS_CONTENT",
                data=content,
                source="GlobalWorkspace",
                priority=10
            )
            
            # Specific Module Notifications (Broadcast to all)
            # Decider might interrupt, Hod might analyze, CRS might shift resources
            # This is handled by each module's subscription to CONSCIOUS_CONTENT.