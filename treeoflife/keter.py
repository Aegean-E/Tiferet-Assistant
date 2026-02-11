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
from typing import Dict, Optional


class Keter:
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
        log_fn=print,
        smoothing: float = 0.95
    ):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.reasoning_store = reasoning_store
        self.event_bus = event_bus
        self.log = log_fn

        # Exponential smoothing to keep Keter slow
        self.smoothing = smoothing

        self._raw_score = None
        self._smoothed_score = None
        self._last_timestamp = None
        self.last_delta = 0.0

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
        if smoothed < 0.2: # Critical coherence drop
            self.log("🚨 KETER: Coherence score critical (< 0.2). Triggering reasoning reboot.")
            self.event_bus.publish("SYSTEM:REBOOT_REASONING", {"reason": "Low Coherence"})
        self._check_reasoning_oscillation()

        result = {
            "keter": round(smoothed, 4),
            "raw": round(raw, 4),
            "delta": round(delta, 4),
            "timestamp": now,
        }

        self.log(f"👑 KETER  score={result['keter']}  Δ={result['delta']}")
        return result

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

        total_mem = max(m.count_all(), 1)
        refuted = m.count_by_type("REFUTED_BELIEF")
        verified = m.count_verified()

        meta_count = mm.count_all()

        # --- Coherence signals (bounded, blunt, non-clever) ---

        epistemic_stability = verified / total_mem
        rejection_learning = min(refuted / total_mem, 1.0)
        compression_signal = min(meta_count / total_mem, 1.0)

        # Penalize pathological extremes
        overload_penalty = min(total_mem / 5000.0, 1.0)

        raw_score = (
            0.4 * epistemic_stability +
            0.3 * rejection_learning +
            0.3 * compression_signal
        ) * (1.0 - 0.5 * overload_penalty)

        # Absolute clamp — Keter must remain bounded
        return max(0.0, min(raw_score, 1.0))

    def _check_reasoning_oscillation(self):
        """
        Check for repetitive loops in the reasoning store.
        If the same thought appears multiple times recently, it's an oscillation.
        """
        try:
            recent_thoughts = self.reasoning_store.list_recent(limit=10)
            if len(recent_thoughts) < 5:
                return

            # Check for simple repetition of the last thought
            last_thought_content = recent_thoughts[0]['content']
            repetition_count = sum(1 for thought in recent_thoughts[1:5] if thought['content'] == last_thought_content)

            if repetition_count >= 2:
                self.log("🚨 KETER: Reasoning oscillation detected. Triggering reboot.")
                if self.event_bus:
                    self.event_bus.publish("SYSTEM:REBOOT_REASONING", {"reason": "Reasoning Loop"})

        except Exception as e:
            self.log(f"⚠️ Keter oscillation check failed: {e}")