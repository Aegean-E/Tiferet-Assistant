"""
Keter (Crown)

Keter represents that which stands above cognition and content.
It contains no beliefs, goals, rules, or representations.

It is the first stirring of intent before thought:
the tendency of the system to come forth into coherent being.

Keter does not know, decide, plan, or act.
It cannot be optimized directly and must not be interpreted as truth,
utility, correctness, or user satisfaction.

Its function is to silently bias the system toward continued coherence
across time, preventing collapse into contradiction, fragmentation,
sterile verification, or runaway generation.

Keter operates only through indirect pressure on strategy selection.
It influences how decisions are made, never which decisions are chosen.

Like a crown above the head, Keter remains outside the system’s
representational faculties while endowing them with orientation.

In this sense, Keter contains all potential but no content,
and is therefore intentionally opaque, slow-changing, and minimal.

Any attempt to store meaning, values, or explicit intent within Keter
constitutes a category error.
"""

import time
import logging
from typing import Dict, Optional
import statistics
import math

from treeoflife.sephirah import Sephirah

class Keter(Sephirah):
    """
    The Crown.
    Read-only observer of the whole system.
    """

    def __init__(
        self,
        memory_store,
        meta_memory_store,
        reasoning_store,
        event_bus,
        log_fn=logging.info,
        smoothing: float = 0.95
    ):
        super().__init__("Keter", "The Crown: Global Coherence", log_fn, event_bus)
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.reasoning_store = reasoning_store
        # self.event_bus and self.log handled by super

        # Exponential smoothing to keep Keter slow
        self.smoothing = smoothing

        self._raw_score = None
        self._smoothed_score = None
        self._last_timestamp = None
        self.last_delta = 0.0
        self.last_log_time = 0
        self._last_components = {}
        
        self.low_coherence_streak = 0
        # Non-Semantic Metrics (Vibrational)
        self.response_latencies = []
        self.response_lengths = []

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Optional[float]]:
        """
        Evaluate system coherence.
        Returns a minimal, opaque signal.

        ARCHITECTURAL RULE:
        No component is allowed to branch on Keter alone.
        Keter should:
        - Bias strategy selection
        - Adjust weights, not decisions
        - Never trigger actions
        - Never wake modules

        Violation: if keter_score < 0.4: do_x()
        Correct: strategy_weights["verify"] *= f(keter)
        """

        raw = self._measure_coherence()
        now = int(time.time())

        if self._smoothed_score is None:
            smoothed = raw
            delta = 0.0
        else:
            smoothed = (
                self.smoothing * self._smoothed_score
                + (1.0 - self.smoothing) * raw
            )
            delta = smoothed - self._smoothed_score

        self._raw_score = raw
        self._smoothed_score = smoothed
        self._last_timestamp = now
        self.last_delta = delta

        # --- Circuit Breaker Logic ---
        if smoothed < 0.2:
            self.low_coherence_streak += 1
            if self.low_coherence_streak >= 5:
                self.log("🚨 KETER: CIRCUIT BREAKER TRIPPED. Coherence < 0.2 for 5 cycles. Triggering PANIC STATE.")
                if self.event_bus:
                    self.event_bus.publish("SYSTEM:PANIC", {"reason": "Persistent Incoherence"})
                self.low_coherence_streak = 0 # Reset after panic
            else:
                self.log(f"🚨 KETER: Coherence score critical (< 0.2). Streak: {self.low_coherence_streak}. Triggering reasoning reboot.")
                self.event_bus.publish("SYSTEM:REBOOT_REASONING", {"reason": "Low Coherence"})
        elif smoothed < 0.4:
            # Confusion State: Trigger Synthesis to find new connections
            self.low_coherence_streak = 0
            self.log("📉 KETER: Coherence low (< 0.4). Triggering Da'at Synthesis (Eureka Engine).")
            if self.event_bus:
                self.event_bus.publish("DAAT:SYNTHESIZE", {"reason": "Low Coherence"})

        self._check_reasoning_oscillation()

        result = {
            "keter": round(smoothed, 4),
            "raw": round(raw, 4),
            "delta": round(delta, 4),
            "timestamp": now,
        }

        # Log only if significant change (> 0.05) AND enough time passed (10s)
        # Or if periodic (300s = 5m) to reduce spam
        if (abs(delta) > 0.05 and (now - self.last_log_time >= 10)) or (now - self.last_log_time >= 300):
            self.log(f"👑 KETER  score={result['keter']}  Δ={result['delta']}")
            
            # Persist metrics for Meta-Learner (Internal State Model)
            if hasattr(self.meta_memory_store, 'add_event'):
                self.meta_memory_store.add_event(
                    "SYSTEM_METRICS",
                    "Keter",
                    f"Coherence: {result['keter']}, Raw: {result['raw']}, Delta: {result['delta']}"
                )
            # Log components only when main log triggers
            self.log(f"    👑 Keter Components: Stab={self._last_components.get('stab', 0):.2f}, Rej={self._last_components.get('rej', 0):.2f}, Success={self._last_components.get('success', 0):.2f}, Vib={self._last_components.get('vib', 0):.2f}")
            self.last_log_time = now
            
        return result

    def track_response_metrics(self, latency: float, length: int):
        """Ingest non-semantic metrics for vibrational analysis."""
        self.response_latencies.append(latency)
        self.response_lengths.append(length)
        # Keep windows small for responsiveness
        if len(self.response_latencies) > 20: self.response_latencies.pop(0)
        if len(self.response_lengths) > 20: self.response_lengths.pop(0)

    # ------------------------------------------------------------------
    # Internal Measurement (non-semantic, non-goal)
    # ------------------------------------------------------------------

    def _measure_coherence(self) -> float:
        """
        Measures *structural coherence*, not correctness.

        IMPORTANT:
        - No semantic inspection
        - No reward signals
        - No user feedback
        - No action outcomes
        """

        m = self.memory_store
        mm = self.meta_memory_store

        count = m.count_all()
        
        # Fix: Handle Tabula Rasa (Empty State)
        # An empty mind is coherent (blank slate), not chaotic.
        if count < 3: # Lowered threshold to allow early dynamics
            return 1.0

        total_mem = max(count, 1)
        refuted = m.count_by_type("REFUTED_BELIEF")
        verified = m.count_verified()

        meta_count = mm.count_all()

        # --- Coherence signals (bounded, blunt, non-clever) ---

        epistemic_stability = verified / total_mem
        rejection_learning = min(refuted / total_mem, 1.0)

        # 2. Action Success (Grounding)
        # Fetch recent outcomes to calculate success rate
        success_signal = 0.5 # Default neutral
        if hasattr(self.meta_memory_store, 'get_average_prediction_error'):
            avg_error = self.meta_memory_store.get_average_prediction_error(limit=20)
            if avg_error is not None:
                success_signal = 1.0 - avg_error
            else:
                success_signal = 0.5 # Neutral start if no history

        # Penalize pathological extremes
        overload_penalty = min(total_mem / 5000.0, 1.0)

        # 4. Vibrational Coherence (Entropy of Response)
        # Detect Rigidity (Death) or Chaos
        vibrational_score = 1.0
        if len(self.response_lengths) > 5:
            # Calculate Entropy of response lengths (binning)
            # Simple variance check first
            len_variance = statistics.variance(self.response_lengths) if len(self.response_lengths) > 1 else 0
            lat_variance = statistics.variance(self.response_latencies) if len(self.response_latencies) > 1 else 0
            
            # Rigidity Check: If variance is near zero, system is looping or stuck
            if len_variance < 5.0: # Extremely consistent length (e.g. "I am waiting.")
                vibrational_score -= 0.5
                self.log("⚠️ KETER: Detected Rigidity (Length Variance < 5).")
            
            if lat_variance < 0.01: # Robotically precise timing (unlikely in LLM, but possible in error loops)
                vibrational_score -= 0.2
                
            # Chaos Check: If variance is massive? (Maybe not bad, but worth noting)
            
            # Entropy calculation (Shannon)
            # Not strictly necessary if variance catches the "stuck" state, which is the main failure mode.
            
        vibrational_score = max(0.0, vibrational_score)

        raw_score = (
            0.2 * epistemic_stability +
            0.1 * rejection_learning +
            0.4 * success_signal +  
            0.3 * vibrational_score # Significant weight on "Aliveness"
        ) * (1.0 - 0.5 * overload_penalty)

        # Store components for throttled logging in evaluate()
        self._last_components = {
            "stab": epistemic_stability,
            "rej": rejection_learning,
            "success": success_signal,
            "vib": vibrational_score
        }

        # Absolute clamp — Keter must remain bounded
        return max(0.0, min(raw_score, 1.0))

    def _check_reasoning_oscillation(self):
        """
        Check for repetitive loops in the reasoning store.
        If the same thought appears multiple times recently, it's an oscillation.
        """
        try:
            recent_thoughts = self.reasoning_store.list_recent(limit=10)
            if not isinstance(recent_thoughts, (list, tuple)) or len(recent_thoughts) < 5:
                return

            # Check for simple repetition of the last thought
            last_thought_content = recent_thoughts[0]['content']
            repetition_count = sum(1 for thought in recent_thoughts[1:5] if thought['content'] == last_thought_content)

            if repetition_count >= 2:
                self.log("🚨 KETER: Reasoning oscillation detected. Triggering Double-Loop Learning.")
                if self.event_bus:
                    self.event_bus.publish("SYSTEM:DOUBLE_LOOP_LEARNING", {"reason": "Reasoning Oscillation", "content": last_thought_content})
                    self.event_bus.publish("SYSTEM:REBOOT_REASONING", {"reason": "Reasoning Loop"})

        except Exception as e:
            self.log(f"⚠️ Keter oscillation check failed: {e}")