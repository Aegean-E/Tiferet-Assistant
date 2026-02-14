import time
from typing import Dict, Callable, Optional, List
import math
import numpy as np
import logging
import threading

class CognitiveResourceController:
    """
    Cognitive Resource Controller (CRS).
    The Executive Attention Allocator.
    
    Decides:
    - Reasoning Depth (How deep to think?)
    - Branching Factor (How many paths to explore?)
    - Constraint Level (Gevurah bias)
    - Tool Authorization (Can we use tools?)
    - Abort Thresholds (When to give up?)
    
    Upgrades:
    1. Predictive Control (Uses Regression Model)
    2. Budget Accounting (Cognitive Cost Tracking)
    3. Uncertainty-Driven Strategy
    4. Long-Horizon Planning (Daily Budget)
    """
    def __init__(self, log_fn: Callable[[str], None] = logging.info, daily_budget: float = 10000.0, skill_map: Dict[str, float] = None, self_model=None):
        self.log = log_fn
        self.budget_lock = threading.Lock()
        
        # Skill Accumulation
        # Map action_type -> proficiency (0.0 to 0.8 max discount)
        self.skill_map = skill_map or {}
        self.self_model = self_model # Reference to persist skills
        
        # Physiological Energy Model
        self.base_budget = daily_budget # Store initial max budget
        self.long_term_budget = daily_budget # Current effective budget
        self.current_spend = 0.0 # Daily spend
        self.last_reset = time.time()
        
        self.instantaneous_capacity = 1.0 # 0.0 - 1.0 (Current mental sharpness)
        self.short_term_fatigue = 0.0 # 0.0 - 1.0 (Accumulates with action, decays with rest)
        
        # Load from epigenetics if available, else default
        self.fatigue_decay_rate = 0.05
        if self.self_model:
             self.fatigue_decay_rate = self.self_model.get_epigenetics().get("fatigue_decay_rate", 0.05)
        self.last_fatigue_update = time.time()
        self.tool_usage_counts = {} # Track usage frequency for fatigue
        
        # Restore State if available
        if self.self_model:
            saved = self.self_model.get_crs_state()
            if saved:
                self.current_spend = saved.get("current_spend", 0.0)
                self.short_term_fatigue = saved.get("short_term_fatigue", 0.0)
                self.last_reset = saved.get("last_reset", time.time())
                self.log(f"🔄 CRS: Restored state (Fatigue: {self.short_term_fatigue:.2f}, Spend: {self.current_spend:.1f})")

        # Damping & Stability Control
        self.pressure_ema = 0.0 # Exponential Moving Average of system pressure
        self.alpha = 0.2 # Smoothing factor (Lower = smoother)
        self.last_predictive_adjustment = 0
        self.predictive_cooldown = 300 # 5 minutes between predictive interventions
        self.last_strategy_switch = 0
        self.strategy_lock_window = 60 # Lock strategy for 1 min to prevent thrashing
        self.last_conflict_count = 0 # Track previous conflict count for derivative

    def _reset_budget_if_needed(self):
        with self.budget_lock:
            if time.time() - self.last_reset > 86400: # 24 hours
                self.current_spend = 0.0
                self.last_reset = time.time()
                self.short_term_fatigue = 0.0 # Deep sleep reset
                self.log("🔄 CRS: Daily cognitive budget reset.")
                
                self.tool_usage_counts = {} # Reset daily usage counts
                # Skill Decay (Use it or lose it)
                for skill in self.skill_map:
                    self.skill_map[skill] = max(0.0, self.skill_map[skill] * 0.99) # 1% daily decay
                self._persist_state()
            
    def _update_fatigue(self):
        """Decay fatigue over time (Rest)."""
        now = time.time()
        elapsed_minutes = (now - self.last_fatigue_update) / 60.0
        if elapsed_minutes > 0:
            decay = elapsed_minutes * self.fatigue_decay_rate
            with self.budget_lock:
                self.short_term_fatigue = max(0.0, self.short_term_fatigue - decay)
                self.last_fatigue_update = now
                self._persist_state()
            
        # Capacity is inverse of fatigue, clamped
        self.instantaneous_capacity = max(0.1, 1.0 - self.short_term_fatigue)

    def _persist_state(self):
        if self.self_model:
            self.self_model.update_crs_state({
                "current_spend": self.current_spend,
                "short_term_fatigue": self.short_term_fatigue,
                "last_reset": self.last_reset
            })
            
    def is_exhausted(self) -> bool:
        """Check if system is too tired to function effectively."""
        return self.short_term_fatigue > 0.9

    def update_metabolic_state(self, energy: float, entropy: float):
        """
        Synchronize with AICore's metabolic state (The Body).
        Energy (0.0-1.0) scales the long-term budget.
        High Entropy (>0.6) forces fatigue up.
        """
        with self.budget_lock:
            # Scale budget by current energy level
            # We use a base budget to avoid permanent shrinking if we just multiplied self.long_term_budget
            self.long_term_budget = self.base_budget * energy
            
            # If the body is failing (High Entropy), force fatigue up
            if entropy > 0.6:
                # Force fatigue to at least 0.5 (50% capacity reduction)
                self.short_term_fatigue = max(self.short_term_fatigue, 0.5)

    def calculate_structural_load(self, metrics: Dict) -> float:
        """
        Calculate System Complexity / Structural Load.
        0.0 (Empty) to 1.0 (Overloaded).
        """
        active_goals = metrics.get("active_goals", 0)
        unresolved_conflicts = metrics.get("unresolved_conflicts", 0)
        goal_spawn_rate = metrics.get("goal_spawn_rate", 0)
        
        # Weights
        w1 = 0.4 # Active Goals (Primary load)
        w2 = 0.3 # Conflicts (Cognitive dissonance)
        w3 = 0.3 # Spawn Rate (Instability)
        
        # Dynamic Weighting: If conflicts are rising fast, prioritize them
        conflict_derivative = max(0, unresolved_conflicts - self.last_conflict_count)
        if conflict_derivative > 2:
            w2 = 0.6 # Spike weight to force consolidation
            
        self.last_conflict_count = unresolved_conflicts
        
        # Normalization (Heuristics)
        norm_goals = min(active_goals / 10.0, 1.0)
        norm_conflicts = min(unresolved_conflicts / 5.0, 1.0)
        norm_spawn = min(goal_spawn_rate / 5.0, 1.0)
        
        load = (w1 * norm_goals) + (w2 * norm_conflicts) + (w3 * norm_spawn)
        return min(load, 1.0)

    def allocate(self, task_type: str, complexity: float = 0.5, urgency: float = 0.5, coherence: float = 1.0, metrics: Dict = None, model_params: Optional[List[float]] = None, current_state: Optional[Dict] = None, model_confidence: Optional[Dict] = None, violation_pressure: float = 0.0, structural_metrics: Dict = None) -> Dict:
        """
        Allocate cognitive resources for a task.
        
        Args:
            task_type: "chat", "reasoning", "planning", "verification"
            complexity: 0.0 to 1.0 (Estimated difficulty)
            urgency: 0.0 to 1.0 (Time pressure)
            coherence: 0.0 to 1.0 (System stability from Keter)
            metrics: Optional dict of historical metrics from Meta-Learner
            model_params: [a, b, c, d] regression coefficients from Meta-Learner
            current_state: {'hesed': float, 'gevurah': float}
            model_confidence: {'r_squared': float, 'mse': float}
            violation_pressure: 0.0 to 1.0 (Ethical constraint pressure from ValueCore)
            structural_metrics: Dict of structural load indicators
            
        Returns:
            Dict of resource parameters.
        """
        self._reset_budget_if_needed()
        self._update_fatigue()
        
        # Defensive casting
        try:
            complexity = float(complexity)
            urgency = float(urgency)
            coherence = float(coherence)
            violation_pressure = float(violation_pressure)
        except (ValueError, TypeError):
            self.log(f"⚠️ CRS: Type error in inputs (Cplx={complexity}, Urg={urgency}, Coh={coherence}). Using defaults.")
            complexity = 0.5
            urgency = 0.5
            coherence = 1.0
            violation_pressure = 0.0

        predicted_delta = 0.0
        # Log Inputs
        self.log(f"🧠 CRS Input: Type={task_type} | Cplx={complexity:.2f} | Urg={urgency:.2f} | Coh={coherence:.2f}")
        
        # Baseline Resources
        resources = {
            "reasoning_depth": 3,      # Steps in ToT
            "beam_width": 1,           # Parallel paths
            "temperature": 0.7,        # Creativity vs Precision
            "max_tokens": 800,         # Output length
            "gevurah_bias": 0.0,       # Constraint pressure
            "allow_tools": True,       # Tool usage permission
            "abort_threshold": 0.2,    # Confidence threshold to abort
            "strategy": "direct"       # direct, tree_of_thoughts, debate
        }

        # 1. Adjust for Task Type
        if task_type == "reasoning" or task_type == "planning":
            resources["reasoning_depth"] = 5
            resources["temperature"] = 0.4
            resources["strategy"] = "tree_of_thoughts"
        elif task_type == "verification":
            resources["reasoning_depth"] = 2
            resources["temperature"] = 0.1
            resources["gevurah_bias"] = 0.2
            resources["strategy"] = "direct"
        elif task_type == "creative":
            resources["temperature"] = 0.9
            resources["gevurah_bias"] = -0.1

        # 2. Adjust for Complexity (Scale Compute)
        if complexity > 0.6:
            resources["reasoning_depth"] += int(complexity * 5) # Up to +5 steps
            resources["beam_width"] = 2
            resources["max_tokens"] = 1500
        
        if complexity > 0.8:
            resources["beam_width"] = 3
            resources["strategy"] = "debate" # Use dialectics for very hard tasks

        # 3. Adjust for Coherence (Safety/Stability)
        if coherence < 0.5:
            self.log(f"⚠️ CRS: Low Coherence ({coherence:.2f}). Restricting resources.")
            resources["temperature"] = max(0.1, resources["temperature"] - 0.3)
            resources["gevurah_bias"] += 0.3
            resources["allow_tools"] = False # Disable tools to prevent flailing
            resources["beam_width"] = 1 # Linear thinking only
            resources["reasoning_depth"] = max(1, resources["reasoning_depth"] - 2)

        # 4. Adjust for Urgency
        if urgency > 0.8:
            resources["reasoning_depth"] = min(3, resources["reasoning_depth"]) # Think fast
            resources["beam_width"] = 1
            
        # 4.5 Circadian Modulation
        if self.self_model:
            phase = self.self_model.get_drives().get("circadian_phase", "day")
            if phase == "dawn":
                # Planning phase: High depth, low temp
                resources["reasoning_depth"] += 1
                resources["temperature"] = max(0.1, resources["temperature"] - 0.1)
            elif phase == "day":
                # Execution phase: Balanced
                pass
            elif phase == "dusk":
                # Reflection phase: Higher temp for synthesis
                resources["temperature"] = min(1.0, resources["temperature"] + 0.1)
            elif phase == "night":
                # Consolidation phase: Low energy, high constraint
                resources["gevurah_bias"] += 0.1
            
        # 5. Physiological Constraints (Fatigue)
        if self.short_term_fatigue > 0.7:
            self.log(f"😫 CRS: High Fatigue ({self.short_term_fatigue:.2f}). Forcing shallow reasoning.")
            resources["reasoning_depth"] = max(1, int(resources["reasoning_depth"] * 0.5))
            resources["beam_width"] = 1
            resources["temperature"] = max(0.1, resources["temperature"] - 0.2) # Tired minds are rigid
            
        if self.short_term_fatigue > 0.9:
            self.log(f"💤 CRS: Critical Fatigue ({self.short_term_fatigue:.2f}). Shutting down complex tools.")
            resources["allow_tools"] = False

        # Calculate Structural Load
        struct_load = 0.0
        if structural_metrics:
            struct_load = self.calculate_structural_load(structural_metrics)
            
        # Determine System Mode
        system_mode = "EXECUTION"
        if coherence < 0.4 or struct_load > 0.7 or violation_pressure > 0.5 or self.short_term_fatigue > 0.85:
            system_mode = "CONSOLIDATION"
        elif coherence > 0.8 and struct_load < 0.3 and violation_pressure < 0.1 and self.short_term_fatigue < 0.3:
            system_mode = "EXPLORATION"
            
        # Critical Failure Check (Panic Button)
        if coherence < 0.2 and struct_load > 0.9:
            self.log("🚨 CRS: CRITICAL FAILURE STATE. Forcing system reset/pause.")
            return {
                "reasoning_depth": 1,
                "beam_width": 1,
                "temperature": 0.0,
                "system_mode": "CRITICAL_FAILURE",
                "allow_tools": False,
                "force_pruning": True
            }
            
        resources["system_mode"] = system_mode
        resources["structural_load"] = struct_load
        
        # Apply Mode Constraints
        if system_mode == "CONSOLIDATION":
            resources["allow_goal_creation"] = False
            resources["allow_novelty"] = False
            resources["allow_debate"] = False
            resources["beam_width"] = 1
            resources["reasoning_depth"] = 2
            resources["temperature"] = 0.3
            resources["force_pruning"] = True
            self.log(f"📉 CRS: CONSOLIDATION MODE ACTIVATED (Load: {struct_load:.2f}, Coh: {coherence:.2f}). Restricting expansion.")
        elif system_mode == "EXPLORATION":
            resources["allow_goal_creation"] = True
            resources["allow_novelty"] = True
            resources["temperature"] = 0.9
            resources["beam_width"] = 3
            self.log(f"🚀 CRS: EXPLORATION MODE. Expanding search.")
        else:
            resources["allow_goal_creation"] = True
            resources["allow_novelty"] = False # Default to focused
            self.log(f"⚙️ CRS: EXECUTION MODE. Focused.")

        # 5. Predictive Control (Forward Modeling)
        # "If I stay on this course, will coherence drop?"
        if model_params and current_state:
            # Confidence Gating
            r2 = model_confidence.get("r_squared", 0.0) if model_confidence else 0.0
            
            if r2 > 0.2: # Only act if model explains at least 20% of variance
                h = current_state.get('hesed', 0.5)
                g = current_state.get('gevurah', 0.5)
                
                # Polynomial Model (Non-Linear)
                # We assume model_params is a list of coefficients for [H^2, G^2, H*G, H, G, 1]
                # Or simpler: just use the provided params if they match expected length
                if len(model_params) == 4:
                     # Legacy Linear: ΔC ≈ a*H + b*G + c*H*G + d
                     a, b, c, d = model_params
                     predicted_delta = (a * h) + (b * g) + (c * h * g) + d
                     resources["predicted_delta"] = predicted_delta
                
                self.log(f"    🔮 Prediction: Δ={predicted_delta:.4f} (R²={r2:.2f}) | H={h:.2f}, G={g:.2f}")
                
                # Damping: Smoothing Filter on Prediction
                # We treat negative delta as "pressure"
                raw_pressure = max(0.0, -predicted_delta)
                self.pressure_ema = (self.alpha * raw_pressure) + ((1 - self.alpha) * self.pressure_ema)

                # Dynamic Cooldown: Scale based on severity of predicted drop
                # If drop is massive (> 0.05), ignore cooldown.
                effective_cooldown = 0 if raw_pressure > 0.05 else self.predictive_cooldown

                # Threshold check with Cooldown (Rate Limiter)
                if self.pressure_ema > 0.01 and (time.time() - self.last_predictive_adjustment > effective_cooldown):
                    self.log(f"🔮 CRS Prediction (R²={r2:.2f}): Sustained Coherence drop risk (EMA: {self.pressure_ema:.4f}). Pre-emptively increasing Gevurah.")
                    resources["gevurah_bias"] += 0.2
                    resources["temperature"] = max(0.1, resources["temperature"] - 0.1)
                    self.last_predictive_adjustment = time.time()
            else:
                self.log(f"    🔮 Prediction: Skipped (Low Confidence R²={r2:.2f})")
                resources["predicted_delta"] = 0.0

        # 6. Uncertainty-Driven Strategy Selection & Active Control
        if metrics:
            # A. Mutation Acceptance (Exploration Health)
            # If mutation acceptance is low, we might be stuck in a local optimum -> Increase exploration
            mut_acc = metrics.get("mutation_acceptance", [])
            if mut_acc:
                recent_acc = sum(v for t, v in mut_acc[-10:]) / min(len(mut_acc), 10)
                if recent_acc < 0.2:
                    self.log(f"📉 CRS: Low mutation acceptance ({recent_acc:.2f}). Boosting Temperature.")
                    resources["temperature"] = min(1.0, resources["temperature"] + 0.2)
            
            # B. Stability Variance (Epistemic Uncertainty)
            stab_var = metrics.get("stability_variance", [])
            if stab_var:
                recent_var = sum(v for t, v in stab_var[-10:]) / min(len(stab_var), 10)
                
                # Uncertainty-Driven Strategy
                if recent_var > 0.25:
                    # Strategy Cooldown (Prevent Thrashing)
                    if time.time() - self.last_strategy_switch > self.strategy_lock_window:
                        self.log(f"📉 CRS: High Uncertainty ({recent_var:.2f}). Strategy -> DEBATE.")
                        resources["strategy"] = "debate"
                        resources["beam_width"] = max(resources["beam_width"], 3) # Force wide search
                        resources["gevurah_bias"] += 0.2
                        self.last_strategy_switch = time.time()
                elif recent_var > 0.15:
                    self.log(f"📉 CRS: Moderate Uncertainty. Strategy -> Tree of Thoughts.")
                    resources["strategy"] = "tree_of_thoughts"
                    resources["reasoning_depth"] += 2

        # 8. Violation Pressure (Ethical Constraint)
        if violation_pressure > 0.0:
            self.log(f"🛡️ CRS: Violation Pressure detected ({violation_pressure:.2f}). Increasing constraints.")
            resources["gevurah_bias"] += (violation_pressure * 0.5) # Up to +0.5 Gevurah
            resources["temperature"] = max(0.01, resources["temperature"] - (violation_pressure * 0.3))
            if violation_pressure > 0.5:
                resources["allow_tools"] = False # Revoke tools if high violation rate

        # 7. Budget Accounting (Long-Horizon Planning)
        # Cost Metric: Depth * Beam * (Tokens / 100)
        # Example: 5 * 3 * (1000/100) = 150 units
        estimated_cost = resources["reasoning_depth"] * resources["beam_width"] * (resources["max_tokens"] / 100.0)
        
        # Apply Skill Discount
        skill_level = self.skill_map.get(task_type, 0.0)
        
        # Metabolic Resource Awareness: Usage Fatigue
        # If a tool is used too often, cost increases
        usage_count = self.tool_usage_counts.get(task_type, 0)
        usage_fatigue_multiplier = 1.0
        if usage_count > 10:
            usage_fatigue_multiplier = 1.0 + (math.log(usage_count - 9) * 0.2) # Logarithmic penalty
            
        effective_cost = estimated_cost * (1.0 - skill_level) * usage_fatigue_multiplier
        
        # Increment usage count
        self.tool_usage_counts[task_type] = usage_count + 1
        
        with self.budget_lock:
            remaining_budget = self.long_term_budget - self.current_spend
            self.log(f"    💰 Budget: Est. Cost={effective_cost:.1f} (Skill: {skill_level:.2f}) | Remaining={remaining_budget:.1f} | Fatigue={self.short_term_fatigue:.2f}")
            
            if effective_cost > remaining_budget:
                self.log(f"📉 CRS: Budget constrained ({remaining_budget:.1f} left). Throttling compute.")
                # Scale down proportionally
                scale_factor = max(0.1, remaining_budget / effective_cost)
                resources["reasoning_depth"] = max(1, int(resources["reasoning_depth"] * scale_factor))
                resources["beam_width"] = 1 # Collapse beam
                resources["max_tokens"] = max(200, int(resources["max_tokens"] * scale_factor))
                effective_cost = remaining_budget # Cap cost
                
            # Commit spend
            self.current_spend += effective_cost
        
        # Accumulate Fatigue (Non-linear: Heavy tasks tire faster)
        fatigue_impact = (effective_cost / 1000.0) * (1.0 + (1.0 - self.instantaneous_capacity))
        self.short_term_fatigue = min(1.0, self.short_term_fatigue + fatigue_impact)
        self._persist_state()

        resources["predicted_delta"] = predicted_delta
        self.log(f"🧠 CRS Allocation [{task_type}]: Depth={resources['reasoning_depth']}, Beam={resources['beam_width']}, Temp={resources['temperature']:.2f}")
        return resources

    def estimate_action_cost(self, action_type: str, complexity: float = 0.5) -> float:
        """Estimate cognitive cost (tokens/compute) for an action."""
        base_costs = {
            "wait": 10,
            "daydream": 500,
            "verify": 300,
            "think": 1000,
            "goal_act": 1500,
            "search_internet": 2000,
            "simulate": 2500,
            "reflect": 400,
            "debate": 2000
        }
        base = base_costs.get(action_type.lower(), 500)
        
        # Apply Skill Discount
        skill_level = self.skill_map.get(action_type.lower(), 0.0)
        
        # Usage Fatigue for estimation
        usage_count = self.tool_usage_counts.get(action_type.lower(), 0)
        usage_fatigue_multiplier = 1.0
        if usage_count > 10:
            usage_fatigue_multiplier = 1.0 + (math.log(usage_count - 9) * 0.2)
            
        return base * (1.0 + complexity) * (1.0 - skill_level) * usage_fatigue_multiplier
        
    def update_skill(self, action_type: str, success_magnitude: float):
        """
        Increase skill proficiency based on success.
        success_magnitude: 0.0 to 1.0 (e.g. utility delta or TD error)
        """
        action_type = action_type.lower()
        current_skill = self.skill_map.get(action_type, 0.0)
        
        # Learning rate (slow growth)
        learning_rate = 0.01
        
        # Sigmoid-like cap at 0.8 (never perfect, always some cost)
        max_skill = 0.8
        
        if success_magnitude > 0:
            growth = success_magnitude * learning_rate * (max_skill - current_skill)
            new_skill = current_skill + growth
            self.skill_map[action_type] = new_skill
            
            if self.self_model:
                self.self_model.update_skills(self.skill_map)
            # self.log(f"🎓 Skill Up: {action_type} -> {new_skill:.4f}")

    def prioritize_goals(self, goals: List[tuple], current_resources: float) -> List[tuple]:
        """
        Rank goals by ROI (Utility / Cost).
        goals: list of (id, subject, text, source, confidence)
        Returns: list of (goal_tuple, roi_score, estimated_cost)
        """
        ranked = []
        
        # Build simple dependency graph (heuristic based on text overlap or ID)
        # In a real system, we'd have explicit 'depends_on' links.
        # Here, we assume older goals might be dependencies for newer ones if semantically related.
        # Or we check for "Step X" patterns.
        
        for g in goals:
            # g: (id, subject, text, source, confidence)
            # confidence is index 4 if present, else default
            priority = g[4] if len(g) > 4 else 0.5
            
            # Estimate complexity based on text length (heuristic)
            complexity = min(1.0, len(g[2]) / 200.0) 
            cost = self.estimate_action_cost("goal_act", complexity)
            
            # Strategic Investment Bonus (Long-Horizon)
            # If this goal unlocks others (Dependency), boost its value.
            # Heuristic: If it's a "Root" goal (no parent) or has many children (if we had that info here).
            # For now, we boost "Research" or "Plan" goals as they often enable others.
            strategic_bonus = 1.0
            text_lower = g[2].lower()
            if "plan" in text_lower or "research" in text_lower or "analyze" in text_lower:
                strategic_bonus = 1.5
            
            # ROI = (Priority * 10000) / (Cost + 1)
            roi = (priority * strategic_bonus * 10000) / (cost + 1)
            ranked.append((g, roi, cost))
        
        # Sort by ROI descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if ranked:
            self.log(f"📊 CRS Goal Prioritization (Top 3):")
            for i, (g, roi, cost) in enumerate(ranked[:3]):
                self.log(f"   {i+1}. Goal {g[0]} (ROI: {roi:.1f}, Cost: {cost:.1f}, Priority: {g[4] if len(g)>4 else 0.5:.2f})")
                
        return ranked