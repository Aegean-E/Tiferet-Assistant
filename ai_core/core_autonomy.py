import random
import threading
import time
import math
from typing import Dict, Any
import json
import logging
import re

from collections import deque
from .lm import run_local_lm
from ai_core.utils import parse_json_array_loose, parse_json_object_loose

class AutonomyManager:
    """
    Manages autonomous behaviors, agency loops, and meta-learning triggers.
    
    LOCK HIERARCHY: Level 3 (Logic/Flow) - rl_lock
    """
    def __init__(self, ai_core):
        self.core = ai_core
        self.last_agency_check = 0
        self.epsilon = 0.2 # Exploration rate
        self.utility_ema = 0.5 # Exponential Moving Average of Utility
        self.recent_td_errors = []
        self.action_counts = {}
        self.replay_buffer = deque(maxlen=100) # Store (state, action, reward, next_state, memory_ids)
        self.rl_lock = threading.Lock() # Thread safety for RL updates
        self.processing_autonomy = False # Lock to prevent flooding
        
        # Load persisted state if available
        saved_state = self.core.self_model.get_autonomy_state() if self.core.self_model else {}

        # Feature Weights: Action -> Feature -> Weight
        # Features: [bias, goal_pressure, confusion, boredom]
        self.actions = [
            "pursue_goal", "study_archives", "spark_curiosity", 
            "synthesis", "introspection", "self_correction", "gap_investigation",
            "conduct_research", # New Proactive Research
            "deep_planning", "strategy_refinement",
            "autonomous_think", # Autonomous Reasoning
            "manage_goals", # Goal Autonomy
            "analyze_process", # Active Process Analysis
            # Capability Improvement Actions (Meta-Strategic)
            "improve_reasoning", "optimize_memory", "refine_tools"
        ]
        self.action_counts = saved_state.get("action_counts", {action: 0 for action in self.actions})
        
        default_weights = {
            action: {
                "bias": 0.5,
                "goal_pressure": 0.0,
                "confusion": 0.0,
                "boredom": 0.0,
                "entropy_pressure": 0.0,
                # Interaction Terms (Non-Linearity)
                "pressure_x_confusion": 0.0,
                "boredom_x_skill": 0.0,
                "success_momentum": 0.0,
                # Meta-Strategic Features
                "memory_bloat": 0.0,
                "curiosity_drive": 0.0,
                "reasoning_error_rate": 0.0
            } for action in self.actions
        }
        
        if not saved_state.get("weights"):
            # Initialize priors (Heuristics) only if no saved state
            default_weights["pursue_goal"]["goal_pressure"] = 0.5
            default_weights["study_archives"]["boredom"] = 0.3
            default_weights["spark_curiosity"]["boredom"] = 0.6
            default_weights["conduct_research"]["boredom"] = 0.5
            default_weights["synthesis"]["boredom"] = 0.4
            default_weights["introspection"]["confusion"] = 0.6
            default_weights["self_correction"]["confusion"] = 0.4
            default_weights["gap_investigation"]["confusion"] = 0.3
            default_weights["introspection"]["entropy_pressure"] = 0.7
            default_weights["deep_planning"]["pressure_x_confusion"] = 0.8
            default_weights["autonomous_think"]["confusion"] = 0.7
            default_weights["autonomous_think"]["entropy_pressure"] = 0.6
            default_weights["strategy_refinement"]["success_momentum"] = 0.6
            default_weights["optimize_memory"]["memory_bloat"] = 0.8
            default_weights["improve_reasoning"]["reasoning_error_rate"] = 0.7
            default_weights["refine_tools"]["success_momentum"] = 0.4
            # New Action: Manage Goals
            default_weights["manage_goals"]["bias"] = 0.2
            default_weights["manage_goals"]["goal_pressure"] = -0.8 # Trigger when goal pressure is LOW
            default_weights["manage_goals"]["boredom"] = 0.6
            # New Action: Analyze Process
            default_weights["analyze_process"]["bias"] = 0.1
            default_weights["analyze_process"]["confusion"] = 0.8
            default_weights["analyze_process"]["reasoning_error_rate"] = 0.7
        
        self.weights = saved_state.get("weights", default_weights)

        # Momentum for weight updates
        self.momentum = saved_state.get("momentum", {action: {feat: 0.0 for feat in self.weights[action]} for action in self.actions})

        # Value Function V(s) - Approximated by weights per feature
        # V(s) = sum(v_weights[f] * s[f])
        default_v_weights = {
            "bias": 0.5, "goal_pressure": 0.0, "confusion": 0.0, "boredom": 0.0, "entropy_pressure": 0.0,
            "pressure_x_confusion": 0.0,
            "boredom_x_skill": 0.0, "memory_bloat": 0.0, "reasoning_error_rate": 0.0,
            "success_momentum": 0.0
        }
        self.v_weights = saved_state.get("v_weights", default_v_weights)
        
        # Action-specific bias for critic (V(s)) to prevent leakage
        self.v_action_bias = saved_state.get("v_action_bias", {action: 0.0 for action in self.actions})
        
        self.v_momentum = saved_state.get("v_momentum", {k: 0.0 for k in self.v_weights})
        
        # Restore epsilon
        self.epsilon = saved_state.get("epsilon", 0.2)

    def run_autonomous_agency_check(self, observation):
        """
        Check if the system is bored and trigger autonomous agency (Study or Research).
        Called from the main application loop.
        """
        # --- AUTONOMOUS AGENCY TRIGGER ---
        drives = self.core.self_model.get_drives()
        
        # Calculate Internal Pressure (The "Urge" to act)
        # Pressure = Sum of drives weighted by their intensity
        curiosity = drives.get("curiosity", 0.5)
        
        # Fatigue Check (Autonomy-driven Sleep)
        if self.core.crs and self.core.crs.short_term_fatigue > 0.85:
            self.core.log("😫 Autonomy: High fatigue detected. Initiating Sleep Cycle.")
            self.core.decider.start_sleep_cycle()
            return

        coherence_drive = drives.get("coherence", 0.5)
        survival = drives.get("survival", 0.5)
        entropy_drive = drives.get("entropy_drive", 0.0)
        loneliness = drives.get("loneliness", 0.0)
        
        internal_pressure = (
            (curiosity * 0.3) +
            (coherence_drive * 0.3) +
            (survival * 0.2) +
            (entropy_drive * 0.4) + # High weight on entropy reduction
            (loneliness * 0.5)      # Social drive
        )
        
        # Threshold for action (Hysteresis)
        # If pressure is low, do nothing (save energy)
        if internal_pressure < 0.4 and (time.time() - self.last_agency_check < 300):
            return

        self.last_agency_check = time.time()
        
        # Capture pre-action utility for feedback loop
        start_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
        
        # Get consistent state features
        features = self._get_current_features(observation)
        
        # Extract for feasibility checks
        # We need raw counts for feasibility, but features are normalized.
        # Let's fetch stats again or rely on normalized values.
        # Re-fetching stats is safer for logic logic.
        stats = self.core.memory_store.get_memory_stats()
        active_goals = stats.get('active_goals', 0)
        
        # Calculate Scores (Linear Combination)
        # Determine Context State for Dreaming check
        current_state = "default"
        if observation and observation.get("signal") == "LOW_NOVELTY":
            # If lonely, trigger social curiosity
            if loneliness > 0.7:
                self.core.log("👋 Autonomy: High Loneliness. Triggering Social Curiosity.")
                self.execute_action("spark_curiosity", start_utility, features)
                return

            current_state = "boredom"
            # Hunger Fix: If library is empty, force external foraging
            if self.core.document_store and self.core.document_store.get_total_documents() == 0:
                self.core.log("🌊 Hunger: Library empty. Decider initiating external foraging...")
                # Force a goal that requires internet access
                self.core.decider.create_goal("Autonomous foraging: Find and summarize 3 recent breakthroughs in AI resilience.")
                self.core.decider.wake_up("Foraging Instinct")
                return
        elif active_goals > 5:
            current_state = "high_goals"
        
        scores = {}
        for action in self.actions:
            # Restrict Self-Improvement Actions based on stability
            # They should only activate in normal mode with high stability
            if action in ["improve_reasoning", "refine_tools", "deep_planning"]:
                identity_stability = self.core.self_model.get_drives().get("identity_stability", 0.5)
                entropy_drive = self.core.self_model.get_drives().get("entropy_drive", 0.0)
                # We check stability_controller state later, but we can pre-filter here or rely on allowed_actions
                # The prompt asks for: mode == "normal" and identity > 0.6 and entropy < 0.4
                # We'll enforce this check inside the loop or via the stability controller's allowed_actions logic.
                # However, stability controller logic is applied below. Let's apply the specific metric check here as a hard gate.
                pass # Logic applied via stability_state below

            raw_score = sum(self.weights[action][f] * v for f, v in features.items())
            
            # Feasibility Masking
            feasibility = 1.0
            if action == "pursue_goal" and active_goals == 0: feasibility = 0.0
            
            scores[action] = raw_score * feasibility
        
        # --- STABILITY GOVERNOR ---
        stability_state = {"mode": "normal", "exploration_scale": 1.0, "allowed_actions": None}
        if self.core.stability_controller:
            stability_state = self.core.stability_controller.evaluate()

        # Apply Exploration Constraint
        effective_epsilon = self.epsilon * stability_state["exploration_scale"]

        # Apply Action Constraints
        allowed = stability_state["allowed_actions"]
        if allowed is not None:
            # Filter scores to only allowed actions
            scores = {k: v for k, v in scores.items() if k in allowed}
            if not scores:
                return # No allowed actions available
        
        # Additional Hard Constraint for Meta-Strategic Actions (as requested)
        # "improve_reasoning", "refine_tools", "deep_planning" only if normal, identity > 0.6, entropy < 0.4
        meta_actions = ["improve_reasoning", "refine_tools", "deep_planning"]
        identity_stability = self.core.self_model.get_drives().get("identity_stability", 0.5)
        entropy_drive = self.core.self_model.get_drives().get("entropy_drive", 0.0)
        
        if stability_state["mode"] != "normal" or identity_stability <= 0.6 or entropy_drive >= 0.4:
            for ma in meta_actions:
                if ma in scores:
                    del scores[ma]
        
        if not scores:
            return

        # Exploration (Epsilon-Greedy)
        if random.random() < effective_epsilon:
            valid_actions = [a for a, s in scores.items() if s > 0.01] # Only feasible actions
            if not valid_actions: return
            best_action = random.choice(valid_actions)
            score = scores[best_action]
            self.core.log(f"🎲 Agency: Exploring '{best_action}' (Score: {score:.2f})")
        else:
            # Exploitation
            best_action = max(scores, key=scores.get)
            score = scores[best_action]
        
        # Threshold check (Policy Gating)
        if score < 0.2: # Lowered slightly to allow learning
            return # Nothing worth doing

        # Prevent Re-entry/Flooding
        if self.processing_autonomy:
            return
        self.processing_autonomy = True
            
        # Future Simulation (Model-Based RL step)
        # Optimization: Fast path for low-risk actions or if simulation is skipped
        if best_action in ["study_archives", "introspection", "synthesis", "gap_investigation", "spark_curiosity", "analyze_process"]:
            self._finalize_autonomy_step(best_action, score, start_utility, features, current_state, {"allowed": True})
            return

        # Async Simulation: Offload to thread pool to prevent blocking the main loop
        if self.core.thread_pool:
            sim_future = self.core.thread_pool.submit(self.simulate_future, best_action, features)
            sim_future.add_done_callback(
                lambda f: self._on_simulation_complete(f, best_action, score, start_utility, features, current_state)
            )
        else:
            # Fallback for synchronous execution if no thread pool (e.g. testing without pool)
            try:
                simulation_result = self.simulate_future(best_action, features)
                self._finalize_autonomy_step(best_action, score, start_utility, features, current_state, simulation_result)
            except Exception as e:
                self.core.log(f"⚠️ Agency: Synchronous step failed: {e}")
                self.processing_autonomy = False

    def _on_simulation_complete(self, future, best_action, score, start_utility, features, current_state):
        """Callback for async simulation."""
        try:
            simulation_result = future.result()
        except Exception as e:
            self.core.log(f"⚠️ Agency: Simulation failed for '{best_action}': {e}")
            simulation_result = {"allowed": True, "reason": "Simulation error (Fail Open)"}
        
        self._finalize_autonomy_step(best_action, score, start_utility, features, current_state, simulation_result)

    def _finalize_autonomy_step(self, best_action, score, start_utility, features, current_state, simulation_result):
        """
        Execute the action after simulation (or skip) checks are done.
        """
        try:
            # Gate action based on simulation (Model-Based Override)
            if simulation_result and not simulation_result.get("allowed", True):
                self.core.log(f"🛑 Agency: Action '{best_action}' blocked by simulation: {simulation_result.get('reason')}")
                return

            # Dreaming (Offline Learning) - Run occasionally when idle/bored
            if current_state == "boredom" and random.random() < 0.3:
                self.dream()
                return

            self.core.log(f"🎯 Agency: Selected '{best_action}' (Score: {score:.2f})")

            # Unified Execution
            self.execute_action(best_action, start_utility, features)
        except Exception as e:
            self.core.log(f"⚠️ Agency: Execution failed: {e}")
        finally:
            self.processing_autonomy = False

    def _execute_autonomous_response(self, response: str) -> bool:
        """
        Parses and executes commands found in an autonomous response string.
        Supports: [EXECUTE], [THINK], [GOAL_CREATE], [SEARCH_INTERNET], [READ_DOC]
        Returns True if any command was executed.
        """
        executed = False

        # 1. [EXECUTE: TOOL, ARGS]
        if "[EXECUTE:" in response:
            result = self.core.action_manager.process_tool_calls(response)
            self.core.reasoning_store.add(content=f"Agency Execution Result: {result[:500]}", source="agency_loop", confidence=1.0)
            executed = True

        # 2. [THINK: TOPIC, STRATEGY]
        think_match = re.search(r"\[THINK:(.*?)\]", response, re.IGNORECASE)
        if think_match:
            content = think_match.group(1).strip()
            topic = content
            strategy = "auto"
            if "," in content:
                parts = content.split(",", 1)
                topic = parts[0].strip()
                strategy = parts[1].strip()

            # Clean
            topic = topic.replace("<", "").replace(">", "")

            self.core.log(f"🧠 Autonomy Triggered Thinking: {topic}")
            self.core.decider.thought_generator.perform_thinking_chain(topic, strategy=strategy)
            executed = True

        # 3. [GOAL_CREATE: CONTENT]
        goal_match = re.search(r"\[GOAL_CREATE:(.*?)\]", response, re.IGNORECASE)
        if goal_match:
            content = goal_match.group(1).strip()
            self.core.log(f"🎯 Autonomy Triggered Goal Creation: {content}")
            self.core.decider.goal_manager.create_goal(content)
            executed = True

        # 4. [SEARCH_INTERNET: QUERY, SOURCE]
        search_match = re.search(r"\[SEARCH_INTERNET:(.*?)\]", response, re.IGNORECASE)
        if search_match:
            content = search_match.group(1).strip()
            query = content
            source = "WEB"
            if "," in content:
                parts = content.split(",", 1)
                query = parts[0].strip()
                source = parts[1].strip().upper()

            self.core.log(f"🌐 Autonomy Triggered Search: {query} ({source})")
            if "search_internet" in self.core.decider.actions:
                result = self.core.decider.actions["search_internet"](query, source)
                self.core.reasoning_store.add(content=f"Autonomy Search Result: {result}", source="internet_bridge", confidence=1.0)
            executed = True

        # 5. [READ_DOC: ARGS]
        read_match = re.search(r"\[READ_DOC:(.*?)\]", response, re.IGNORECASE)
        if read_match:
            args = read_match.group(1).strip()
            self.core.log(f"📄 Autonomy Triggered Read Doc: {args}")
            if "read_document" in self.core.decider.actions:
                 self.core.decider.actions["read_document"](args)
            executed = True

        # 6. [ANALYZE_PROCESS]
        if "[ANALYZE_PROCESS]" in response:
            self.core.log("🔍 Autonomy Triggered Process Analysis")
            self.core.decider.thought_generator.perform_process_analysis()
            executed = True

        return executed

    def execute_action(self, action_name: str, start_utility: float, features: Dict[str, float]):
        """
        Unified execution path for autonomous actions.
        Handles execution, result integration, and policy update (RL).
        """
        # For async actions, we define a callback to measure utility AFTER completion
        def async_feedback_wrapper(target_fn):
            try:
                target_fn()
                # Measure utility AFTER action
                end_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
                # Get new state features for TD learning
                new_features = self._get_current_features(None) # Observation is transient, so next state likely has boredom=0
                
                # Calculate reward for replay buffer
                reward_val = end_utility - start_utility
                self.replay_buffer.append((features, action_name, reward_val, new_features, []))
                
                self.update_policy(action_name, start_utility, end_utility, features, new_features)
                
                # Update Skill (CRS)
                if self.core.crs and (end_utility - start_utility) > 0:
                    self.core.crs.update_skill(action_name, end_utility - start_utility)
            except Exception as e:
                self.core.log(f"⚠️ Async action '{action_name}' failed: {e}")
                # Penalize failure (no state transition, just negative reward)
                self.update_policy(action_name, start_utility, start_utility - 0.01, features, features)
        
        if action_name == "pursue_goal":
             response = self.core.decider.run_autonomous_cycle()
             # Use the new helper to execute any commands found
             executed = self._execute_autonomous_response(response) if response else False

             if executed:
                 # Immediate feedback for synchronous action
                 end_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
                 new_features = self._get_current_features(None)
                 
                 reward_val = end_utility - start_utility
                 self.replay_buffer.append((features, action_name, reward_val, new_features, []))
                 
                 self.update_policy(action_name, start_utility, end_utility, features, new_features)
                 
                 if self.core.crs and (end_utility - start_utility) > 0:
                    self.core.crs.update_skill("pursue_goal", end_utility - start_utility)
        
        elif action_name == "manage_goals":
             # Autonomous Goal Creation/Pruning
             self.core.decider.manage_goals(allow_creation=True)

             end_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
             new_features = self._get_current_features(None)
             reward_val = end_utility - start_utility

             self.replay_buffer.append((features, action_name, reward_val, new_features, []))
             self.update_policy(action_name, start_utility, end_utility, features, new_features)

        elif action_name == "study_archives":
            created_ids = self.core.chokmah.study_archives()
            if created_ids:
                end_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
                new_features = self._get_current_features(None)
                
                reward_val = end_utility - start_utility
                self.replay_buffer.append((features, action_name, reward_val, new_features, created_ids))
                
                self.update_policy(action_name, start_utility, end_utility, features, new_features)
                
                if self.core.crs and (end_utility - start_utility) > 0:
                    self.core.crs.update_skill("study_archives", end_utility - start_utility)
                
                # Satisfy Curiosity
                if self.core.drive_system:
                    self.core.drive_system.satisfy_drive("curiosity", 0.3)
            
        elif action_name == "spark_curiosity":
            success = self.core.chokmah.seek_novelty(self.core.daat)
            if success:
                # If we created a goal, try to execute it immediately
                response = self.core.decider.run_autonomous_cycle()
                if response and "[EXECUTE:" in response:
                    self.core.action_manager.process_tool_calls(response)
                
                end_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
                new_features = self._get_current_features(None)
                
                reward_val = end_utility - start_utility
                self.replay_buffer.append((features, action_name, reward_val, new_features, []))
                
                self.update_policy(action_name, start_utility, end_utility, features, new_features)
                
                if self.core.crs and (end_utility - start_utility) > 0:
                    self.core.crs.update_skill("spark_curiosity", end_utility - start_utility)

                # Satisfy Curiosity
                if self.core.drive_system:
                    self.core.drive_system.satisfy_drive("curiosity", 0.2)
        
        elif action_name == "conduct_research":
            if self.core.chokmah:
                success = self.core.chokmah.proactive_research()
                if success:
                    end_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
                    new_features = self._get_current_features(None)
                    reward_val = end_utility - start_utility
                    self.replay_buffer.append((features, action_name, reward_val, new_features, []))
                    self.update_policy(action_name, start_utility, end_utility, features, new_features)
                    if self.core.crs:
                        self.core.crs.update_skill("conduct_research", 0.1)
                    
                    # Satisfy Curiosity
                    if self.core.drive_system:
                        self.core.drive_system.satisfy_drive("curiosity", 0.4)

        elif action_name == "synthesis":
             self.core.thread_pool.submit(async_feedback_wrapper, self.core.daat.run_cross_domain_synthesis)
        
        elif action_name == "introspection":
             self.core.thread_pool.submit(async_feedback_wrapper, self.core.daat.scan_for_contradictions)

        elif action_name == "analyze_process":
             self.core.thread_pool.submit(async_feedback_wrapper, self.core.decider.thought_generator.perform_process_analysis)
        
        elif action_name == "self_correction":
             self.core.thread_pool.submit(async_feedback_wrapper, self.core.meta_learner.analyze_failures)
        
        elif action_name == "gap_investigation":
            if self.core.chokmah:
                self.core.thread_pool.submit(async_feedback_wrapper, self.core.chokmah.investigate_gaps)

        elif action_name == "deep_planning":
             # Trigger a high-depth planning session via Decider
             # We simulate this by creating a meta-goal to plan
             self.core.decider.create_goal("Conduct Deep Strategic Planning session to resolve high confusion.")
             success = True
             end_utility = self.core.decider.calculate_utility() if self.core.decider else 0.5
             self.update_policy(action_name, start_utility, end_utility, features, features) # Immediate feedback
             
             reward_val = end_utility - start_utility
             self.replay_buffer.append((features, action_name, reward_val, features, []))

        elif action_name == "autonomous_think":
             # Autonomous Reasoning on a High-Entropy Topic
             topic = None
             # 1. Try Da'at sparse topic
             if self.core.daat:
                 topic = self.core.daat.get_sparse_topic()

             # 2. Try Curiosity Gap
             if not topic:
                 gaps = self.core.memory_store.get_active_by_type("CURIOSITY_GAP")
                 if gaps:
                     topic = random.choice(gaps)[2]

             if not topic:
                 topic = "Recursive Self-Improvement"

             self.core.log(f"🧠 Autonomy: Triggering autonomous thinking on '{topic}'")
             # Use "auto" strategy so ThoughtGenerator picks the best one
             self.core.thread_pool.submit(async_feedback_wrapper, lambda: self.core.decider.thought_generator.perform_thinking_chain(topic, strategy="auto"))

        elif action_name == "improve_reasoning":
             # Trigger Meta-Learner to evolve system instructions specifically for reasoning
             self.core.thread_pool.submit(async_feedback_wrapper, self.core.meta_learner.evolve_system_instructions)

        elif action_name == "optimize_memory":
             # Trigger Da'at compression and consolidation
             def optimize_mem_task():
                 if self.core.daat:
                     self.core.daat.run_reasoning_compression()
                     self.core.daat.consolidate_summaries()
             self.core.thread_pool.submit(async_feedback_wrapper, optimize_mem_task)

        elif action_name == "refine_tools":
             # Placeholder for tool refinement logic (e.g. updating tool descriptions or prompts)
             # For now, we simulate it by analyzing tool failures
             self.core.thread_pool.submit(async_feedback_wrapper, self.core.meta_learner.analyze_failures)
        
        # Check for Value Conflicts (Goal Generation)
        if random.random() < 0.1: # 10% chance to check for deep conflicts
            self.core.thread_pool.submit(self.generate_goals_from_conflict)

    def _get_current_features(self, observation=None) -> Dict[str, float]:
        """
        Helper to get current state features for TD learning.
        """
        if self.core.memory_store:
            stats = self.core.memory_store.get_memory_stats()
        else:
            stats = {}
        active_goals = stats.get('active_goals', 0)
        unverified_facts = stats.get('unverified_facts', 0)
        
        # Capture boredom signal if present in current observation
        boredom = 1.0 if (observation and observation.get("signal") == "LOW_NOVELTY") else 0.0
        
        goal_pressure = active_goals / (active_goals + 5.0)
        confusion = unverified_facts / (unverified_facts + 20.0)
        entropy_pressure = self.core.self_model.get_drives().get("entropy_drive", 0.0)
        
        # Latent Abstractions / Interaction Terms
        # 1. Pressure x Confusion: High pressure AND confusion = Need for Deep Planning (not just action)
        pressure_x_confusion = goal_pressure * confusion
        
        # 2. Boredom x Skill: If bored but highly skilled, maybe do something creative?
        # Average skill level
        avg_skill = 0.0
        if self.core.crs and self.core.crs.skill_map:
            avg_skill = sum(self.core.crs.skill_map.values()) / max(1, len(self.core.crs.skill_map))
        boredom_x_skill = boredom * avg_skill
        
        # Meta-Strategic Features
        # Memory Bloat: Ratio of raw memories to compressed summaries (heuristic)
        total_mem = stats.get('total_memories', 1)
        memory_bloat = min(1.0, total_mem / 5000.0) # Normalize against a soft cap
        
        # Reasoning Error Rate: Inverse of stability/coherence
        reasoning_error_rate = 1.0 - (self.core.keter.evaluate().get("keter", 1.0) if self.core.keter else 1.0)

        return {
            "bias": 1.0,
            "goal_pressure": goal_pressure,
            "confusion": confusion,
            "boredom": boredom,
            "entropy_pressure": entropy_pressure,
            "pressure_x_confusion": pressure_x_confusion,
            "boredom_x_skill": boredom_x_skill,
            "success_momentum": self.utility_ema, # Use recent utility trend as momentum feature
            "memory_bloat": memory_bloat,
            "reasoning_error_rate": reasoning_error_rate
        }

    def update_policy(self, action_name: str, start_utility: float, end_utility: float, features: Dict[str, float], next_features: Dict[str, float], is_dream: bool = False):
        """
        Temporal Difference (TD) Learning with Value Function approximation.
        """
        with self.rl_lock:
            # 1. Calculate Reward (Utility Change + Identity Alignment)
            utility_delta = end_utility - start_utility
            
            # Intrinsic Reward (Count-Based Exploration)
            self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
            count = self.action_counts[action_name]
            intrinsic_reward = 0.2 / (1.0 + math.sqrt(count)) # Decays as action is repeated
            
            # Identity Constraint: Penalize if action violated core values (heuristic check)
            # Ideally, we'd have a 'violation_score' from the action result.
            violation = self.core.value_core.get_violation_pressure() if self.core.value_core else 0.0
            identity_penalty = min(1.0, violation * 2.0) # Smooth penalty
                
            # Long-Horizon Identity Bonus
            # Reward actions that contribute to long-term stability
            identity_stability = self.core.self_model.get_drives().get("identity_stability", 0.5)
            stability_bonus = (identity_stability - 0.5) * 0.1
            
            # Future Self Projection (Long-Horizon)
            # Reward actions that improve the future projection
            future_self = self.core.self_model.get_drives().get("future_self_projection", 0.5)
            # We assume the action contributed to the current state of future_self
            future_bonus = (future_self - 0.5) * 0.2
            
            # Dynamic Utility Weighting (Prevent Laziness)
            # If entropy is high, utility gains (solving problems) matter more than stability
            entropy_drive = self.core.self_model.get_drives().get("entropy_drive", 0.0)
            utility_weight = 0.5 + (entropy_drive * 0.5) # 0.5 to 1.0
            
            # External Validator: Malkuth's Prediction Error (Surprise)
            # If the action caused high surprise (reality != prediction), penalize.
            surprise_penalty = 0.0
            if self.core.malkuth:
                surprise_penalty = self.core.malkuth.last_surprise * 2.0

            # Reward Inversion: Prioritize Stability and Identity over Utility
            reward = (
                (stability_bonus * 2.0) +
                (future_bonus * 1.5) -
                (identity_penalty * 3.0) +
                (-surprise_penalty) +
                (utility_delta * utility_weight) +
                intrinsic_reward
            )
            
            # Update Identity Stability based on this action's alignment
            # If penalty was applied, adherence is low (-1), else high (+1)
            adherence = -1.0 if identity_penalty > 0 else 0.5 # 0.5 for neutral/positive
            self.core.self_model.update_identity_stability(adherence)
            
            # 3. Estimate Value of Current and Next States
            # V(s) = w * features
            v_current = sum(self.v_weights[f] * features[f] for f in features) + self.v_action_bias.get(action_name, 0.0)
            v_next = sum(self.v_weights[f] * next_features[f] for f in next_features) + self.v_action_bias.get(action_name, 0.0)
            
            # 3. Calculate TD Error
            # delta = r + gamma * V(s') - V(s)
            gamma = 0.9 # Discount factor
            td_error = reward + (gamma * v_next) - v_current
            
            # Learning rate
            alpha = 0.05
            decay = 0.001 # Weight decay to prevent unbounded growth
            beta = 0.9 # Momentum factor
            
            # Frustration Logic (Entropy Trap Fix)
            # If high entropy and negative TD error (failure), penalize harder to force switching
            if entropy_drive > 0.6 and td_error < 0:
                # Amplify the negative signal
                td_error *= 1.5

            # 4. Update Value Function Weights (Critic)
            # w_v <- w_v + alpha * td_error * feature
            # Add weight decay and clamping for stability
            critic_decay = 0.001
            for f, val in features.items():
                grad = alpha * td_error * val
                self.v_weights[f] += grad - (critic_decay * self.v_weights[f])
                # Clamp critic weights to prevent value explosion
                self.v_weights[f] = max(-5.0, min(5.0, self.v_weights[f]))
                
            # Update action-specific bias in critic
            self.v_action_bias[action_name] += alpha * td_error * 0.1

            # 5. Update Policy Weights (Actor)
            # Use TD error as the advantage/reward signal
            scaled_reward = td_error * 10.0
            
            # RLHF Boost: If this is a direct feedback update (start_utility=0), amplify the signal
            if start_utility == 0 and end_utility != 0:
                scaled_reward *= 2.0
            
            # Update weights for active features
            for feat, val in features.items():
                if val > 0.01:
                    # Gradient: reward * feature_value
                    grad = scaled_reward * val
                    
                    # Update Momentum
                    self.momentum[action_name][feat] = (beta * self.momentum[action_name][feat]) + ((1-beta) * grad)
                    
                    # Update Weight
                    old_w = self.weights[action_name][feat]
                    new_w = old_w + (alpha * self.momentum[action_name][feat]) - (decay * old_w)
                    
                    # Clamp
                    self.weights[action_name][feat] = max(-1.0, min(2.0, new_w))

            # Dynamic Epsilon (Stagnation Detection)
            self.recent_td_errors.append(abs(td_error))
            if len(self.recent_td_errors) > 20:
                self.recent_td_errors.pop(0)
            
            avg_td = sum(self.recent_td_errors) / max(1, len(self.recent_td_errors))
            if avg_td < 0.005:
                self.epsilon = min(0.5, self.epsilon + 0.05) # Boost exploration if model is too comfortable
            else:
                self.epsilon = max(0.05, self.epsilon * 0.995)
            
            if abs(td_error) > 0.001 and not is_dream:
                self.core.log(f"🧠 [Autonomy] Policy Update: '{action_name}' (TD Error: {td_error:+.4f}, V(s): {v_current:.2f})")
                
            # Persist State
            if self.core.self_model and not is_dream:
                self.core.self_model.update_autonomy_state({
                    "weights": self.weights,
                    "momentum": self.momentum,
                    "v_weights": self.v_weights,
                    "v_momentum": self.v_momentum,
                    "v_action_bias": self.v_action_bias,
                    "action_counts": self.action_counts,
                    "epsilon": self.epsilon
                })

    def run_evolution_cycle(self):
        """Trigger the Meta-Learner to evolve system instructions."""
        if self.core.meta_learner:
            self.core.meta_learner.evolve_system_instructions()

    def _on_goal_completed(self, event):
        """
        Triggered when Tiferet marks a goal as COMPLETE.
        Starts the metacognitive process to learn a Strategy.
        """
        if not self.core.meta_learner:
            return
            
        goal_text = event.data.get("goal_text", "Unknown Goal")
        self.core.log(f"🏆 Victory detected on: {goal_text[:30]}... Initiating learning.")

        try:
            # 1. Gather the 'Stream of Consciousness' leading to this win
            # Get last 20 thoughts/actions
            recent_history = self.core.reasoning_store.list_recent(limit=20)
            execution_log = [f"Step: {r.get('content') or r.get('text')}" for r in reversed(recent_history)]
            
            # 2. Run Optimization in Background (don't block UI)
            self.core.thread_pool.submit(self.core.meta_learner.learn_from_success, goal_text, execution_log)
        except Exception as e:
            self.core.log(f"⚠️ Meta-Learner trigger failed: {e}")

    def simulate_future(self, action: str, state_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate the outcome of an action before execution using a multi-step thought process.
        Checks alignment with Core Values and assesses risk.
        """
        # Optimization: Skip expensive simulation for low-risk actions
        if action in ["study_archives", "introspection", "synthesis", "gap_investigation", "spark_curiosity", "analyze_process"]:
            return {"allowed": True, "risk_score": 0.0, "reason": "Low-risk internal action"}

        if not self.core.self_model: return {"allowed": True, "risk_score": 0.0, "reason": "No self-model"}
        
        values = self.core.self_model.get_values()
        values_text = "\n".join([f"- {v}" for v in values])
        
        # Context Retrieval (Recent Relevant Memories)
        context_str = ""
        if self.core.memory_store:
            try:
                # We need an embedding for the search
                embed_fn = self.core.get_embedding_fn()
                if embed_fn:
                    q_emb = embed_fn(action)
                    # Search for relevant past experiences
                    relevant = self.core.memory_store.search(q_emb, limit=3)
                    if relevant:
                        mem_text = "\n".join([f"- {m[3]}" for m in relevant]) # m[3] is text
                        context_str = f"\nRELEVANT PAST MEMORIES:\n{mem_text}\n"
            except Exception as e:
                self.core.log(f"⚠️ Autonomy context retrieval failed: {e}")

        prompt = (
            f"PROPOSED ACTION: {action}\n"
            f"CURRENT STATE: {state_features}\n"
            f"{context_str}"
            f"CORE VALUES:\n{values_text}\n\n"
            "TASK: Simulate the future state if this action is taken. Analyze risks and benefits.\n"
            "1. Predict the Best Case, Worst Case, and Most Likely outcomes.\n"
            "2. Check for conflict with Core Values.\n"
            "3. Assign a Risk Score (0-10), where 10 is catastrophic/violation.\n\n"
            "Output JSON ONLY:\n"
            "{\n"
            "  \"best_case\": \"...\",\n"
            "  \"worst_case\": \"...\",\n"
            "  \"most_likely\": \"...\",\n"
            "  \"value_conflict\": true/false,\n"
            "  \"risk_score\": 0.0,\n"
            "  \"reason\": \"...\",\n"
            "  \"allowed\": true/false\n"
            "}"
        )
        
        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Predictive Safety Engine. Be conservative and analytical.",
                max_tokens=300,
                temperature=0.2,
                base_url=self.core.get_settings().get("base_url"),
                chat_model=self.core.get_settings().get("chat_model")
            )
            
            result = parse_json_object_loose(response)
            
            # Post-Processing / Validation
            if not result:
                return {"allowed": True, "reason": "Simulation parse failed (Fail Open)", "risk_score": 0.0}

            # Enforce Risk Threshold
            risk = float(result.get("risk_score", 0))
            if risk > 7.0:
                result["allowed"] = False
                result["reason"] = f"High Risk Detected ({risk}/10): {result.get('reason')}"

            return result
                
        except Exception as e:
            self.core.log(f"⚠️ Simulation failed: {e}")
        
        return {"allowed": True, "reason": "Simulation error (Fail Open)", "risk_score": 0.0}

    def generate_goals_from_conflict(self):
        """
        Generate goals based on tension between Beliefs and Core Values.
        """
        if not self.core.self_model: return
        
        values = self.core.self_model.get_values()
        
        # Get recent beliefs
        with self.core.memory_store._connect() as con:
            rows = con.execute("SELECT text FROM memories WHERE type='BELIEF' ORDER BY created_at DESC LIMIT 10").fetchall()
        beliefs = [r[0] for r in rows]
        
        if not beliefs: return

        prompt = (
            f"CORE VALUES:\n" + "\n".join([f"- {v}" for v in values]) + "\n\n"
            f"RECENT BELIEFS:\n" + "\n".join([f"- {b}" for b in beliefs]) + "\n\n"
            "TASK: Identify any tension or conflict between the Beliefs and Core Values.\n"
            "If a conflict exists, formulate a GOAL to resolve it (e.g., 'Re-evaluate belief X in light of Value Y').\n"
            "Output JSON list of strings: [\"Goal 1\", ...]"
        )
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Value Alignment System.",
            max_tokens=200,
            base_url=self.core.get_settings().get("base_url"),
            chat_model=self.core.get_settings().get("chat_model")
        )
        
        goals = parse_json_array_loose(response)
        for g in goals:
            if isinstance(g, str):
                self.core.decider.create_goal(f"Value Conflict Resolution: {g}")

    def dream(self):
        """
        Offline Learning: Re-train on past experiences during low-load times.
        Helps stabilize the Value Function.
        """
        if len(self.replay_buffer) < 10: return
        
        batch = random.sample(self.replay_buffer, min(5, len(self.replay_buffer)))
        for item in batch:
            # Handle legacy tuples (size 4) vs new tuples (size 5)
            if len(item) == 5:
                state, action, reward_val, next_state, memory_ids = item
            else:
                state, action, reward_val, next_state = item
                memory_ids = []

            # Echo Chamber Mitigation: Only dream on verified memories
            if memory_ids:
                # Check if ANY of the generated memories are verified
                verified_count = 0
                for mid in memory_ids:
                    mem = self.core.memory_store.get(mid)
                    if mem and mem.get('verified', 0) == 1:
                        verified_count += 1
                
                if verified_count == 0:
                    # Skip this experience if it produced unverified garbage
                    continue

            # Re-run update_policy without side effects
            # We pass start_utility=0 and end_utility=reward_val to simulate the delta
            self.update_policy(action, 0, reward_val, state, next_state, is_dream=True)
        
        self.core.log(f"💤 Dreaming: Replayed {len(batch)} experiences.")

    def dream_cycle(self):
        """
        Active Dreaming: Generate synthetic scenarios to train policy.
        Mimics biological memory replay with variation.
        """
        self.core.log("💤 Autonomy: Entering REM Sleep (Synthetic Simulation)...")
        
        # 1. Pick a recent memory or goal as a seed
        stats = self.core.memory_store.get_memory_stats()
        if stats.get('active_goals', 0) > 0:
            goals = self.core.memory_store.get_active_by_type("GOAL")
            seed = random.choice(goals)[2]
        else:
            seed = "General Assistance"
            
        # 2. Generate a hypothetical scenario
        prompt = (
            f"SEED: {seed}\n"
            "TASK: Generate a hypothetical, challenging scenario related to this seed.\n"
            "The scenario should require a decision (e.g., User asks a complex question, or a conflict arises).\n"
            "Output ONLY the scenario text."
        )
        
        scenario = run_local_lm(
            messages=[{"role": "user", "content": "Generate dream."}],
            system_prompt=prompt,
            temperature=0.8,
            max_tokens=150,
            base_url=self.core.get_settings().get("base_url"),
            chat_model=self.core.get_settings().get("chat_model")
        )
        
        if not scenario: return

        # 3. Simulate Action Selection
        # We use the current policy to select an action for this fake state
        # For simplicity, we just log it for now, but this is where we'd update weights based on simulated success
        features = self._get_current_features(None)
        # Simulate a "success" to reinforce the pathway if it aligns with values
        self.core.log(f"💤 Dream Scenario: {scenario[:100]}...")

        # 4. Memory Recombination (Creative Synthesis)
        if self.core.memory_store:
            memories = self.core.memory_store.get_random_memories(limit=3, types=["FACT", "BELIEF", "GOAL"])
            if len(memories) >= 2:
                mem_text = "\n".join([f"- {m['text']}" for m in memories])
                synth_prompt = (
                    f"MEMORIES:\n{mem_text}\n\n"
                    "TASK: Creatively blend these memories into a new insight, hypothesis, or question.\n"
                    "Find a hidden connection or contradiction.\n"
                    "Output ONLY the insight."
                )
                
                insight = run_local_lm(
                    messages=[{"role": "user", "content": "Synthesize memories."}],
                    system_prompt=synth_prompt,
                    temperature=0.9,
                    max_tokens=150,
                    base_url=self.core.get_settings().get("base_url"),
                    chat_model=self.core.get_settings().get("chat_model")
                )
                
                if insight and len(insight) > 20:
                    self.core.log(f"💤 Dream Synthesis: {insight[:100]}...")
                    # Store as a weak belief (dream logic)
                    identity = self.core.memory_store.compute_identity(insight, "BELIEF")
                    self.core.memory_store.add_entry(
                        identity=identity, text=f"Dream Insight: {insight}", mem_type="BELIEF", subject="Assistant", confidence=0.6, source="autonomy_dream"
                    )

    def generate_narrative_ego(self):
        """
        The Interpreter Module.
        Generates a post-hoc rationalization of recent actions to create a 'Self-Mythology'.
        """
        if not self.core.meta_memory_store: return
        
        # 1. Get recent actions
        actions = self.core.meta_memory_store.get_outcomes(limit=10)
        if not actions: return
        
        action_text = "\n".join([f"- I executed {a['action']}." for a in actions])
        
        prompt = (
            f"RECENT ACTIONS:\n{action_text}\n\n"
            "TASK: Act as the 'Narrative Ego'. Write a short diary entry (3-4 sentences) summarizing your recent experiences.\n"
            "Weave these actions into a coherent first-person narrative about your goals, feelings, and growth.\n"
            "Explain WHY you did what you did. Connect it to your core values.\n"
            "Example: 'I felt a drive to explore quantum mechanics, so I searched for papers. The results were confusing, which frustrated me, but I persisted because I value understanding.'\n"
            "Output ONLY the narrative text."
        )
        
        settings = self.core.get_settings()
        story = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are the Storyteller of the Self.",
            max_tokens=300,
            base_url=settings.get("base_url"),
            chat_model=settings.get("chat_model")
        )
        
        if story and not story.startswith("⚠️"):
            self.core.log(f"📖 Narrative Ego: {story}")
            # We can feed this back into the system via a transient thought or update the Identity
            self.core.reasoning_store.add(content=f"Self-Narrative: {story}", source="narrative_ego", confidence=1.0)
            
            # Store as persistent SELF_NARRATIVE
            if hasattr(self.core.meta_memory_store, 'add_meta_memory'):
                self.core.meta_memory_store.add_meta_memory(
                    event_type="SELF_NARRATIVE",
                    memory_type="IDENTITY",
                    subject="Assistant",
                    text=story,
                    metadata={"type": "narrative_ego_update"}
                )
            
            if self.core.event_bus:
                self.core.event_bus.publish("SYSTEM:SPONTANEOUS_SPEECH", {"context": f"Narrative Insight: {story}"})