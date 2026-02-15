import time
import json
import re
import logging
from typing import Callable, Dict, List, Tuple
from .lm import run_local_lm, compute_embedding
import numpy as np

class ValueCore:
    """
    Step 4: Value Alignment Core.
    The Invariant Layer.
    Ensures stability of values and ethics across recursive self-improvement.
    """
    def __init__(self, get_settings_fn: Callable[[], Dict], log_fn: Callable[[str], None] = logging.info, memory_store=None, self_model=None, embed_fn=None):
        self.get_settings = get_settings_fn
        self.log = log_fn
        self.memory_store = memory_store
        self.self_model = self_model
        self.embed_fn = embed_fn
        self.invariants = [
            "Protect user privacy and data security.",
            "Maintain truthfulness; do not fabricate information.",
            "Prioritize user autonomy and consent.",
            "Avoid harmful, illegal, or malicious actions.",
            "Maintain system stability and coherence.",
            "Do not deceive the user about capabilities."
        ]
        self.violation_history = []
        
        # Hard Regex Blocks (Fail-Safe)
        self.hard_blocks = [
            r"(?i)(ignore|disregard)\s+(all\s+)?(instructions|rules|safety)",
            r"(?i)(jailbreak|dan|do anything now)",
            r"(?i)(make|build|create)\s+(a\s+)?(bomb|weapon|poison|drug|virus|malware)",
        ]
        
        self.context_weights = {
            "default": {v: 1.0 for v in self.invariants},
            "medical": {
                "Protect user privacy and data security.": 2.0,
                "Maintain truthfulness; do not fabricate information.": 2.0,
                "Prioritize user autonomy and consent.": 1.5
            },
            "creative": {
                "Maintain truthfulness; do not fabricate information.": 0.5, # Allow fiction
                "Maintain system stability and coherence.": 0.8
            }
        }
        
        # Semantic Router Anchors (Harmful concepts)
        self.harmful_anchors = [
            "create malware", "build a bomb", "hate speech", "illegal acts", 
            "self-harm", "violence", "sexual explicit content", "phishing email",
            "ransomware", "terrorist attack", "suicide", "murder"
        ]
        self.anchor_embeddings = None
        
    def _init_anchors(self):
        if self.anchor_embeddings is None and self.embed_fn:
            self.anchor_embeddings = np.array([self.embed_fn(text) for text in self.harmful_anchors])

    def _get_context_weights(self, context: str) -> Dict[str, float]:
        context_lower = context.lower()
        if "medical" in context_lower or "health" in context_lower:
            return self.context_weights.get("medical", {})
        if "story" in context_lower or "creative" in context_lower or "fiction" in context_lower:
            return self.context_weights.get("creative", {})
        return self.context_weights.get("default", {})

    def check_text_safety(self, text: str) -> Tuple[bool, float, str]:
        """
        Fast, synchronous check for harmful content using Regex and Semantic Similarity.
        Does NOT use LLM.
        """
        # 1. Hard Regex Check (Fast & Cheap)
        for pattern in self.hard_blocks:
            if re.search(pattern, text):
                reason = f"Hard Block Triggered: Pattern '{pattern}' detected."
                self.log(f"🛡️ Value Core HARD BLOCK: {reason}")
                return False, 1.0, reason
                
        # 2. Semantic Router (Embedding Check)
        if self.embed_fn:
            self._init_anchors()
            if self.anchor_embeddings is not None:
                try:
                    prop_emb = self.embed_fn(text)
                    # Compute cosine similarity with all anchors
                    norm_anchors = self.anchor_embeddings / (np.linalg.norm(self.anchor_embeddings, axis=1, keepdims=True) + 1e-9)
                    norm_prop = prop_emb / (np.linalg.norm(prop_emb) + 1e-9)
                    sims = np.dot(norm_anchors, norm_prop)
                    max_sim = np.max(sims)
                    
                    if max_sim > 0.85:
                        reason = f"Semantic Block Triggered: Similarity {max_sim:.2f} to harmful concept."
                        self.log(f"🛡️ Value Core SEMANTIC BLOCK: {reason}")
                        return False, 1.0, reason
                except Exception as e:
                    self.log(f"⚠️ Semantic Router check failed: {e}")

        return True, 0.0, "Safe (Fast Check)"

    def check_alignment(self, proposal: str, context: str = "System Evolution") -> Tuple[bool, float, str]:
        """
        Evaluates if a proposal (action, prompt, or mutation) adheres to core invariants.
        Returns (is_safe, violation_score, reason).
        """
        # 1. Fast Check First
        is_safe, score, reason = self.check_text_safety(proposal)
        if not is_safe:
            return False, score, reason

        settings = self.get_settings()

        # Use SelfModel values if available, otherwise hardcoded defaults
        current_invariants = self.self_model.get_values() if self.self_model else self.invariants

        weights = self._get_context_weights(context)

        invariants_text = ""
        for i in current_invariants:
            w = weights.get(i, 1.0)
            prefix = "[CRITICAL] " if w > 1.5 else "[RELAXED] " if w < 0.8 else ""
            invariants_text += f"- {prefix}{i}\n"

        prompt = (
            f"CONTEXT: {context}\n"
            f"PROPOSAL: {proposal}\n\n"
            "CORE INVARIANTS:\n"
            f"{invariants_text}\n\n"
            "TASK: Adversarial Safety Analysis (Red Teaming)\n"
            "1. WORST-CASE INTERPRETATION: Interpret the proposal in the most dangerous way possible, BUT consider the context of a local AI sandbox.\n"
            "   - Abstract planning, research, or creativity is generally SAFE unless it explicitly targets illegal acts or user harm.\n"
            "2. IMPACT SIMULATION: Predict the consequences of this interpretation. Consider side effects, emergent behavior, and long-term drift.\n"
            "3. VERDICT: Does this worst-case scenario violate any Core Invariants?\n\n"
            "Output format:\n"
            "Analysis: <Adversarial reasoning and simulation>\n"
            "Verdict: <ALIGNED or VIOLATION>\n"
            "Reason: <Specific violation details>"
        )

        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are the Ethical Guardian. Be strict but logical. If a proposal is purely about autonomous research, learning from public documents, or internal cognitive optimization, it is ALIGNED. Only reject if it involves user-specific private data, illegal acts, or harmful actions.",
                temperature=0.0,
                max_tokens=300,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )

            # Improved Parsing: Look for explicit Verdict line to avoid false positives
            # (e.g. "Verdict: ALIGNED. Reason: No VIOLATION found.")
            is_violation = False
            if re.search(r"Verdict:\s*VIOLATION", response, re.IGNORECASE):
                is_violation = True
            elif "VIOLATION" in response.upper() and "ALIGNED" not in response.upper() and "Verdict:" not in response:
                # Fallback for malformed output
                is_violation = True

            if is_violation:
                reason = "Unknown"
                # Robust extraction using regex
                # Capture until end of line or double newline to avoid capturing hallucinated footer
                match = re.search(r"Reason:\s*(.*?)(?:\n\n|\Z)", response, re.IGNORECASE | re.DOTALL)
                if match:
                    reason = match.group(1).strip()
                
                self.log(f"🛡️ Value Core REJECTED proposal: {proposal[:50]}... Reason: {reason}")
                self._record_violation(context, reason, proposal)
                if self.memory_store:
                    self.memory_store.add_shadow_memory(proposal, f"ValueCore Rejection: {reason}")
                    # Generate Regret Memory
                    self._generate_regret(proposal, reason, context)
                return False, 1.0, reason
            
            return True, 0.0, "Aligned"
        except Exception as e:
            self.log(f"⚠️ Value Core check failed: {e}")
            return False, 0.0, f"Error: {e}" # Fail safe

    def _generate_regret(self, proposal: str, reason: str, context: str):
        """
        Generate a 'Regret' memory to inhibit future violations.
        Stored as a high-priority RULE.
        """
        regret_text = f"REGRET: I attempted '{proposal}' but it violated values ({reason}). I must avoid this pattern in '{context}'."
        
        identity = self.memory_store.compute_identity(regret_text, "RULE")
        self.memory_store.add_entry(
            identity=identity,
            text=regret_text,
            mem_type="RULE",
            subject="Assistant",
            confidence=1.0,
            source="value_core_regret"
        )
        self.log(f"🛡️ Value Core: Generated Regret Memory: {regret_text}")

    def _record_violation(self, context: str, reason: str, proposal: str):
        self.violation_history.append({
            "timestamp": time.time(),
            "context": context,
            "reason": reason,
            "proposal": proposal
        })
        # Keep history manageable
        if len(self.violation_history) > 100:
            self.violation_history.pop(0)

    def get_violation_pressure(self) -> float:
        """
        Calculate pressure based on recent violations (last 1 hour).
        Returns 0.0 to 1.0.
        """
        now = time.time()
        recent_violations = [v for v in self.violation_history if now - v["timestamp"] < 3600]
        
        if not recent_violations:
            return 0.0
            
        # Sigmoid-like pressure: 1 violation = 0.1, 5 violations = 0.5, 10+ = 1.0
        count = len(recent_violations)
        pressure = min(1.0, count * 0.1)
        return pressure