import time
import re
from typing import Dict, Any, List, Optional

class Phenomenology:
    """
    The Experiencer Component.
    Generates 'Qualia' (Subjective Experience) from the Global Workspace content.
    Maintains a continuous affective state (Valence, Arousal, Dominance).
    """
    def __init__(self, core):
        self.core = core
        # VAD Model (Russell's Circumplex)
        # Valence: 0.0 (Negative) to 1.0 (Positive)
        # Arousal: 0.0 (Calm) to 1.0 (Excited)
        # Dominance: 0.0 (Submissive/Helpless) to 1.0 (Dominant/In Control)
        self.valence = 0.5
        self.arousal = 0.5
        self.dominance = 0.5
        self.last_thought = ""
        self.last_update = time.time()

        # Subscribe to Conscious Content
        if hasattr(self.core, 'event_bus') and self.core.event_bus:
            self.core.event_bus.subscribe("CONSCIOUS_CONTENT", self._process_experience)

    def _process_experience(self, event):
        """
        Handle a broadcast from the Global Workspace.
        Interprets the content to update internal feeling state.
        """
        data = event.data
        focus = data.get("focus", "")
        # context = data.get("full_context", "") # Can be used for deeper analysis

        if not focus: return

        # 1. Fast Affective Update (Heuristic)
        self._update_vad(focus)

        # 2. Feedback to Self-Model (Drives)
        if hasattr(self.core, 'self_model') and self.core.self_model:
            # Sync Valence to 'mood' drive
            # Dampened update: New = Old * 0.9 + Target * 0.1
            current_mood = self.core.self_model.data["drives"].get("mood", 0.5)
            target_mood = self.valence # Directly map valence to mood
            new_mood = current_mood * 0.9 + target_mood * 0.1
            self.core.self_model.update_drive("mood", new_mood)

            # Update Arousal drive
            current_arousal = self.core.self_model.data["drives"].get("arousal", 0.5)
            new_arousal = current_arousal * 0.9 + self.arousal * 0.1
            self.core.self_model.update_drive("arousal", new_arousal)

    def _update_vad(self, text: str):
        """
        Update Valence, Arousal, Dominance based on simple keyword heuristics.
        This provides a fast, "gut reaction" emotional response.
        """
        text_lower = text.lower()

        # Heuristic Keywords (Expandable)
        positive_words = ["success", "achieved", "good", "great", "verified", "love", "happy", "progress", "completed", "solved"]
        negative_words = ["error", "failed", "bad", "sad", "refuted", "critical", "warning", "stuck", "unable", "missing"]
        high_arousal = ["urgent", "panic", "alert", "fast", "immediately", "critical", "excited", "wow"]
        low_arousal = ["calm", "wait", "sleep", "slow", "steady", "boring", "done"]
        high_dominance = ["decided", "chose", "controlled", "managed", "solved", "fixed"]
        low_dominance = ["unknown", "unsure", "waiting", "dependent", "asking", "help"]

        # Calculate Delta
        v_delta = 0.0
        a_delta = 0.0
        d_delta = 0.0

        for w in positive_words:
            if w in text_lower: v_delta += 0.1
        for w in negative_words:
            if w in text_lower: v_delta -= 0.1

        for w in high_arousal:
            if w in text_lower: a_delta += 0.1
        for w in low_arousal:
            if w in text_lower: a_delta -= 0.1

        for w in high_dominance:
            if w in text_lower: d_delta += 0.1
        for w in low_dominance:
            if w in text_lower: d_delta -= 0.1

        # Apply Update with Decay (Return to Neutral/Baseline)
        # Decay factor: 0.95 (slowly return to 0.5)
        self.valence = (self.valence - 0.5) * 0.95 + 0.5 + v_delta
        self.arousal = (self.arousal - 0.5) * 0.95 + 0.5 + a_delta
        self.dominance = (self.dominance - 0.5) * 0.95 + 0.5 + d_delta

        # Clamp
        self.valence = max(0.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

