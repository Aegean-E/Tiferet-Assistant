"""
Malkuth (Kingdom): The Causal Engine.
Responsible for grounding AI reasoning in physical reality via simulation and causal checks.
"""

import os
import json
import ast
import time
import logging
import math
from typing import Dict, Any, List, Optional
from ai_core.lm import run_local_lm
from ai_core.utils import parse_json_object_loose
from ai_core.safety_checks import PluginSafetyValidator

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

class SensoryPacket:
    """Standardized return format for Malkuth actions."""
    def __init__(self, content: str, success: bool, metadata: dict = None):
        self.content = content
        self.success = success
        self.metadata = metadata or {}

class SimplePhysicsEngine:
    """
    Zero-dependency, lightweight 2D physics engine for causal simulation.
    Supports: Circles, Gravity, Drag, Elastic Collisions.
    """
    def __init__(self, gravity=(0, -9.81), drag=0.0):
        self.objects = []
        self.gravity = gravity
        self.drag = drag
        self.t = 0.0
        self.log = []

    def add_object(self, name, mass, pos, vel, radius=0.5, static=False):
        self.objects.append({
            "name": name,
            "m": mass,
            "inv_m": 1.0/mass if mass > 0 and not static else 0.0,
            "p": list(pos), # [x, y]
            "v": list(vel), # [vx, vy]
            "radius": radius,
            "static": static
        })

    def step(self, dt):
        self.t += dt
        # 1. Integrate
        for obj in self.objects:
            if obj["static"]: continue
            
            # Gravity
            obj["v"][0] += self.gravity[0] * dt
            obj["v"][1] += self.gravity[1] * dt
            
            # Drag
            obj["v"][0] *= (1.0 - self.drag * dt)
            obj["v"][1] *= (1.0 - self.drag * dt)
            
            # Position
            obj["p"][0] += obj["v"][0] * dt
            obj["p"][1] += obj["v"][1] * dt
            
            # Ground Plane (y=0)
            if obj["p"][1] - obj["radius"] <= 0:
                obj["p"][1] = obj["radius"]
                if obj["v"][1] < 0:
                    obj["v"][1] = -obj["v"][1] * 0.6 # Bounce
                    if abs(obj["v"][1]) < 0.5: obj["v"][1] = 0
                    self.log.append(f"[{self.t:.2f}s] {obj['name']} hit ground. Pos={obj['p']}")

        # 2. Collisions
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                self._resolve(self.objects[i], self.objects[j])

    def _resolve(self, a, b):
        dx = b["p"][0] - a["p"][0]
        dy = b["p"][1] - a["p"][1]
        dist_sq = dx*dx + dy*dy
        rad_sum = a["radius"] + b["radius"]
        
        if dist_sq < rad_sum * rad_sum:
            dist = math.sqrt(dist_sq) or 0.001
            nx, ny = dx / dist, dy / dist
            
            # Relative velocity
            rvx = b["v"][0] - a["v"][0]
            rvy = b["v"][1] - a["v"][1]
            vel_norm = rvx * nx + rvy * ny
            
            if vel_norm > 0: return
            
            j = -(1 + 0.7) * vel_norm / (a["inv_m"] + b["inv_m"])
            ix, iy = j * nx, j * ny
            
            if not a["static"]: a["v"][0] -= ix * a["inv_m"]; a["v"][1] -= iy * a["inv_m"]
            if not b["static"]: b["v"][0] += ix * b["inv_m"]; b["v"][1] += iy * b["inv_m"]
            self.log.append(f"[{self.t:.2f}s] Collision: {a['name']} <-> {b['name']}")

class Malkuth:
    def __init__(self, memory_store=None, meta_memory_store=None, log_fn=logging.info, event_bus=None, value_core=None):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.log = log_fn
        self.event_bus = event_bus
        self.value_core = value_core
        self.plugin_validator = PluginSafetyValidator(log_fn=self.log)
        self.causal_graph = nx.DiGraph() if NX_AVAILABLE else None
        
        # Step 1: World Model Layer Persistence
        self.world_model_path = "./data/world_model.json"
        self.world_state = self._load_world_model()
        self.prediction_history = [] # Track (prediction, actual, error)
        self.last_surprise = 0.0
        self.causal_rules = [] # Inferred causal rules
        self.output_locked = False

    def lock_output(self):
        """Emergency lock for output gate."""
        self.output_locked = True

    def describe_image(self, image_path: str) -> str:
        """
        Sensory Perception: Describe an image using a vision model.
        """
        self.log(f"👁️ [Malkuth] Perceiving image: {image_path}")
        
        prompt = "Describe this image in detail. Identify key objects, text, and the overall context."
        
        try:
            description = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Visual Perception Engine.",
                temperature=0.2,
                max_tokens=300,
                images=[image_path]
            )
            return description
        except Exception as e:
            self.log(f"⚠️ Vision perception failed: {e}")
            return "Image perception failed."

    def _load_world_model(self):
        if os.path.exists(self.world_model_path):
            try:
                with open(self.world_model_path, 'r') as f:
                    return json.load(f)
            except: pass
        return {"entities": {}, "relationships": [], "history": []}

    def _save_world_model(self):
        temp_path = self.world_model_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(self.world_state, f, indent=2)
        # Atomic replacement
        os.replace(temp_path, self.world_model_path)

    def verify_physical_possibility(self, scenario: str, estimation: str) -> str:
        """
        Checks if a scenario and its estimation are physically consistent
        with known causal rules (thermodynamics, conservation laws).
        Uses a Causal Graph to detect violations.
        """
        self.log(f"🌍 [Malkuth] Verifying physical consistency for: {scenario}")
        
        if not NX_AVAILABLE:
            return "Causal Engine (NetworkX) not available. Performing heuristic check."

        # 1. Extract Causal Model from Scenario via LLM
        prompt = (
            f"SCENARIO: {scenario}\n"
            f"ESTIMATION: {estimation}\n\n"
            "TASK: Extract the Causal Graph implied by this scenario.\n"
            "Identify nodes (entities/variables) and edges (causal links: increases, decreases, converts_to).\n"
            "Output JSON: {'nodes': ['A', 'B'], 'edges': [{'source': 'A', 'target': 'B', 'relation': 'increases'}]}\n"
            "Include 'Energy' or 'Mass' as nodes if relevant to check conservation laws."
        )
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Causal Graph Extractor.",
            temperature=0.0,
            max_tokens=300
        )
        
        try:
            data = parse_json_object_loose(response)
            
            # 2. Build Temporary Graph
            G = nx.DiGraph()
            for node in data.get('nodes', []):
                if isinstance(node, str):
                    G.add_node(node)
                else:
                    # Fallback if LLM returns objects
                    G.add_node(str(node))
            for edge in data.get('edges', []):
                G.add_edge(str(edge['source']), str(edge['target']), relation=edge.get('relation', 'causes'))
            
            # 3. Check Constraints (Thermodynamics / Conservation)
            # Heuristic: Check for 'Free Energy' loops (A increases B, B increases A, without input)
            violations = []
            try:
                cycles = list(nx.simple_cycles(G))
                for cycle in cycles:
                    # Check if cycle implies runaway positive feedback without external input
                    relations = []
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        relations.append(G[u][v]['relation'])
                    
                    if all(r in ['increases', 'promotes', 'generates', 'produces', 'causes'] for r in relations):
                        violations.append(f"Runaway positive feedback loop found in {cycle}. Violates conservation of energy unless external input exists.")
            except Exception:
                pass

            # Check 2: Creation Ex Nihilo (Source nodes)
            try:
                for node in G.nodes():
                    if G.in_degree(node) == 0:
                        # Check outgoing edges
                        out_edges = G.out_edges(node, data=True)
                        for u, v, data in out_edges:
                            rel = data.get('relation', '').lower()
                            if rel in ['generates', 'produces', 'creates', 'synthesizes']:
                                violations.append(f"Potential Creation Ex Nihilo: '{u}' {rel} '{v}' but has no input.")
            except Exception:
                pass

            if violations:
                return "⚠️ VIOLATIONS DETECTED:\n" + "\n".join(violations)

            return "Consistent with causal model (No obvious violations detected)."
            
        except Exception as e:
            self.log(f"⚠️ Causal verification failed: {e}")
            return "Verification inconclusive (Parsing Error)."

    def simulate_shock(self, system_state: dict, shock: dict) -> dict:
        """
        Simulate the effect of a shock on a system state.
        """
        self.log(f"🌍 [Malkuth] Simulating shock: {shock} on system.")
        
        # Simple 1-step propagation
        new_state = system_state.copy()
        impact_notes = []
        
        for key, val in shock.items():
            if key in new_state:
                old_val = new_state[key]
                # Heuristic update
                if isinstance(old_val, (int, float)) and isinstance(val, (int, float)):
                    new_state[key] = old_val + val
                    impact_notes.append(f"{key}: {old_val} -> {new_state[key]}")
        
        return {"status": "simulated", "new_state": new_state, "notes": ", ".join(impact_notes)}

    def create_note(self, content: str) -> int:
        """Manually create an Assistant Note (NOTE)."""
        if self.output_locked:
            self.log("🛑 Malkuth: Output locked. Cannot create note.")
            return None
        if not self.memory_store:
            return None
        mem_type = "NOTE"
        text = content

        # Use a deterministic identity based on content for NOTES to allow versioning if edited
        identity = self.memory_store.compute_identity(text, mem_type)
        mid = self.memory_store.add_entry(
            identity=identity,
            text=text,
            mem_type=mem_type,
            subject="Assistant", 
            confidence=1.0,
            source="malkuth_tool"
        )
        self.log(f"📝 [Malkuth] Created Assistant Note (ID: {mid}): {text}")
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event("MEMORY_CREATED", "Assistant", f"Created Note: {text}")
        return mid

    def edit_note(self, mem_id: int, content: str):
        """Edit an Assistant Note by superseding it."""
        if self.output_locked:
            self.log("🛑 Malkuth: Output locked. Cannot edit note.")
            return
        if not self.memory_store:
            return
        old_mem = self.memory_store.get(mem_id)
        if not old_mem:
            self.log(f"⚠️ [Malkuth] Tried to edit non-existent memory ID: {mem_id}")
            return

        mid = self.memory_store.add_entry(
            identity=old_mem['identity'],
            text=content,
            mem_type="NOTE",
            subject=old_mem['subject'],
            confidence=1.0,
            source="malkuth_edit",
            parent_id=mem_id
        )
        self.log(f"📝 [Malkuth] Edited Assistant Note (ID: {mem_id} -> {mid}): {content}")
        if hasattr(self.meta_memory_store, 'add_event'):
            self.meta_memory_store.add_event("MEMORY_EDITED", "Assistant", f"Edited Note {mem_id} -> {mid}")

    def write_plugin(self, filename: str, content: str) -> str:
        """
        Write a Python plugin to the 'plugins' directory with strict safety checks.
        """
        if self.output_locked:
            self.log("🛑 Malkuth: Output locked. Cannot write plugin.")
            return "Error: System Panic Active. Output Locked."

        # 1. Basic Validation
        if not filename.endswith(".py"):
            return "Error: Plugin filename must end with .py"

        # 2. Safety Analysis (Static + Semantic)
        is_safe, reason = self.plugin_validator.validate(content, self.value_core)

        if not is_safe:
            self.log(f"🛑 [Malkuth] Plugin blocked by Safety Validator: {reason}")
            return f"Error: Plugin rejected by Safety Validator. Reason: {reason}"

        # 3. Write File
        plugins_dir = os.path.abspath("./plugins")
        os.makedirs(plugins_dir, exist_ok=True)

        safe_filename = os.path.basename(filename)
        file_path = os.path.join(plugins_dir, safe_filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.log(f"🔌 [Malkuth] Created new plugin: {safe_filename}")
            return f"Success: Plugin '{safe_filename}' created. You must ENABLE it in settings to use it."
        except Exception as e:
            return f"Error writing plugin file: {e}"

    def write_file(self, filename: str, content: str) -> str:
        """Write content to a file in the 'works' directory."""
        if self.output_locked:
            self.log("🛑 Malkuth: Output locked. Cannot write file.")
            return "Error: System Panic Active. Output Locked."

        # Define the allowed directory
        works_dir = os.path.abspath("./works")
        os.makedirs(works_dir, exist_ok=True)
        
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        file_path = os.path.abspath(os.path.join(works_dir, safe_filename))
        
        # Ensure path is within works_dir
        if not file_path.startswith(works_dir):
            self.log(f"⚠️ Path traversal attempt blocked: {filename}")
            return "Error: Invalid filename."
            
        # Security: Limit file extensions
        allowed_exts = {".txt", ".md", ".json", ".csv", ".log"}
        if os.path.splitext(safe_filename)[1].lower() not in allowed_exts:
            return f"Error: File type not allowed. Allowed: {', '.join(allowed_exts)}"
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
        except Exception as e:
            return f"Failed to write file: {e}"

    def run_causal_inference(self, treatment: str, outcome: str, context: str) -> str:
        """
        Perform Bayesian Causal Inference using DoWhy.
        1. Construct Causal Graph (LLM).
        2. Generate Synthetic Data based on priors (LLM).
        3. Estimate Effect and P-Value (DoWhy).
        """
        if not DOWHY_AVAILABLE:
            return "⚠️ DoWhy/Pandas not installed. Install 'dowhy', 'pandas', 'numpy' to use Causal Inference."

        self.log(f"🧬 [Malkuth] Causal Inference: Effect of '{treatment}' on '{outcome}' in '{context}'")

        # 1. Parameterize Simulation via LLM
        prompt = (
            f"CONTEXT: {context}\n"
            f"HYPOTHESIS: Treatment '{treatment}' affects Outcome '{outcome}'.\n\n"
            "TASK: Parameterize a Causal Model for simulation.\n"
            "1. Define a simple Causal Graph (DOT format). Include at least one Confounder if plausible.\n"
            "2. Define parameters for generating synthetic data (N=200).\n"
            "   - Assume linear relationships.\n"
            "   - Provide 'treatment_effect' (expected slope).\n"
            "\n"
            "OUTPUT JSON ONLY:\n"
            "{\n"
            "  \"graph_dot\": \"digraph { T -> Y; C -> T; C -> Y; }\",\n"
            "  \"treatment_node\": \"T\",\n"
            "  \"outcome_node\": \"Y\",\n"
            "  \"confounders\": [\"C\"],\n"
            "  \"params\": {\n"
            "    \"T_mean\": 10, \"T_std\": 2,\n"
            "    \"C_mean\": 5, \"C_std\": 1,\n"
            "    \"Y_base\": 50,\n"
            "    \"effect_T_Y\": 2.5,\n"
            "    \"effect_C_Y\": 1.0,\n"
            "    \"effect_C_T\": 0.5,\n"
            "    \"noise_std\": 1.0\n"
            "  }\n"
            "}"
        )

        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Causal Inference Engineer.",
                temperature=0.1,
                max_tokens=600
            )
            
            # Parse JSON
            data = parse_json_object_loose(response)
            
            graph_dot = data["graph_dot"]
            t_node = data["treatment_node"]
            y_node = data["outcome_node"]
            confounders = data.get("confounders", [])
            p = data["params"]
            
            # 2. Generate Synthetic Data
            N = 200
            df = pd.DataFrame()
            
            # Generate Confounders
            for c in confounders:
                df[c] = np.random.normal(p.get(f"{c}_mean", 0), p.get(f"{c}_std", 1), N)
            
            # Generate Treatment (influenced by confounders)
            df[t_node] = np.random.normal(p.get("T_mean", 0), p.get("T_std", 1), N)
            for c in confounders:
                df[t_node] += df[c] * p.get(f"effect_{c}_T", 0)
            
            # Generate Outcome
            df[y_node] = p.get("Y_base", 0) + (df[t_node] * p.get("effect_T_Y", 0))
            for c in confounders:
                df[y_node] += df[c] * p.get(f"effect_{c}_Y", 0)
            df[y_node] += np.random.normal(0, p.get("noise_std", 1), N)
            
            # 3. DoWhy Analysis
            model = CausalModel(
                data=df,
                treatment=t_node,
                outcome=y_node,
                graph=graph_dot.replace("\n", " ")
            )
            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
            refute = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
            
            return (f"📊 **Causal Analysis Results**\n"
                    f"Treatment: {t_node} -> Outcome: {y_node}\n"
                    f"Estimated Effect: {estimate.value:.4f}\n"
                    f"P-Value (Refutation): {refute.refutation_result['p_value']:.4f}\n"
                    f"Interpretation: {'Statistically Significant' if refute.refutation_result['p_value'] < 0.05 else 'Not Significant'}")

        except Exception as e:
            self.log(f"⚠️ Causal Inference failed: {e}")
            return f"Causal Inference Error: {e}"

    def run_physics_simulation(self, scenario: str) -> str:
        """
        Run a lightweight 2D physics simulation for a given scenario.
        """
        self.log(f"🧬 [Malkuth] Running Physics Simulation: {scenario}")
        
        # 1. Parse Scenario
        prompt = (
            f"SCENARIO: {scenario}\n"
            "TASK: Convert this into a 2D physics simulation configuration.\n"
            "Assume standard gravity (0, -9.81) unless specified.\n"
            "Output JSON:\n"
            "{\n"
            "  \"gravity\": [0, -9.81],\n"
            "  \"drag\": 0.0,\n"
            "  \"duration\": 5.0,\n"
            "  \"objects\": [\n"
            "    {\"name\": \"Ball\", \"mass\": 1.0, \"pos\": [0, 10], \"vel\": [0, 0], \"radius\": 0.5, \"static\": false}\n"
            "  ]\n"
            "}"
        )
        
        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a Physics Engine Configurator.",
                temperature=0.1,
                max_tokens=400
            )
            config = parse_json_object_loose(response)
            
            # 2. Run Simulation
            sim = SimplePhysicsEngine(
                gravity=tuple(config.get("gravity", [0, -9.81])),
                drag=config.get("drag", 0.0)
            )
            
            for obj in config.get("objects", []):
                sim.add_object(obj["name"], obj["mass"], obj["pos"], obj.get("vel", [0,0]), obj.get("radius", 0.5), obj.get("static", False))
            
            duration = config.get("duration", 5.0)
            dt = 0.1
            steps = int(duration / dt)
            
            for _ in range(steps):
                sim.step(dt)
                
            log_str = "\n".join(sim.log) if sim.log else "No significant events (collisions/ground hits)."
            final_state = ", ".join([f"{o['name']} pos={o['p']}" for o in sim.objects])
            
            return f"Simulation Log:\n{log_str}\n\nFinal State:\n{final_state}"
            
        except Exception as e:
            return f"Simulation failed: {e}"

    def register_outcome(self, action: str, prediction: str, observation: str):
        """
        The Active Inference Loop.
        1. Compare Prediction vs Observation.
        2. Calculate Surprise (Prediction Error).
        3. Update World Model.
        """
        self.log(f"🌍 [Malkuth] Registering Outcome for: {action}")
        
        prompt = (
            f"ACTION: {action}\n"
            f"PREDICTION: {prediction}\n"
            f"OBSERVATION: {observation}\n\n"
            "TASK: Update World Model & Calculate Surprise.\n"
            "1. Did the observation match the prediction?\n"
            "2. Update entity states based on observation.\n"
            "3. Assign a 'Surprise Score' (0.0 = Expected, 1.0 = Shocking).\n"
            "Output JSON: {\"surprise\": 0.1, \"entities\": {\"User\": {\"state\": \"...\"}}}"
        )
        
        # If prediction was missing/empty, surprise is naturally higher if outcome is complex
        if not prediction:
            prompt = prompt.replace(f"PREDICTION: {prediction}", "PREDICTION: None (Unexpected Action)")
        
        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a State Tracking Engine.",
                temperature=0.1,
                max_tokens=300
            )
            
            data = parse_json_object_loose(response)
            
            self.last_surprise = data.get("surprise", 0.0)
            
            # Update internal state
            timestamp = time.time()
            for entity, info in data.get("entities", {}).items():
                self.world_state["entities"][entity] = {
                    "state": info.get("state"),
                    "last_cause": action,
                    "last_updated": timestamp
                }
            
            self.log(f"🌍 [Malkuth] World Model Updated. Surprise: {self.last_surprise:.2f}")
            
            self._save_world_model()

            if self.event_bus and self.last_surprise > 0.2:
                self.event_bus.publish("SURPRISE_EVENT", {"text": f"Prediction Error: {self.last_surprise:.2f}", "action": action})

            
        except Exception as e:
            self.log(f"⚠️ World Model update failed: {e}")

    def get_prediction(self, entity: str) -> str:
        """Get the latest prediction for an entity."""
        ent_data = self.world_state.get("entities", {}).get(entity)
        if ent_data:
            return f"Prediction for {entity}: {ent_data.get('prediction')} (State: {ent_data.get('state')})"
        return "No prediction available."

    def predict_action_outcome(self, action: str, current_context: str) -> str:
        """
        Option A: Internal Simulator.
        Predicts the outcome of an action before execution.
        """
        self.log(f"🔮 [Malkuth] Simulating outcome for: {action}")
        
        # Calibrate confidence based on historical accuracy
        base_confidence = 0.8
        if self.prediction_history:
            avg_error = sum([h['error'] for h in self.prediction_history]) / len(self.prediction_history)
            # Simple calibration: Higher error -> Lower confidence
            base_confidence = max(0.1, 1.0 - avg_error)

        prompt = (
            f"CURRENT CONTEXT:\n{current_context[:1000]}...\n\n"
            f"PROPOSED ACTION: {action}\n\n"
            "TASK: Simulate the immediate outcome.\n"
            "1. Predicted State Change: What happens?\n"
            "2. Success Probability: 0.0 - 1.0\n"
            "3. Resource Cost: Low/Med/High\n"
            "4. Side Effects: Any negative consequences?\n"
            f"Output concise JSON: {{\"outcome\": \"...\", \"probability\": {base_confidence:.2f}, \"cost\": \"Med\", \"risks\": \"...\"}}"
        )
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Predictive Engine.",
            temperature=0.1,
            max_tokens=200
        )
        
        # Ensure valid JSON string is returned
        data = parse_json_object_loose(response)
        
        if self.event_bus:
            self.event_bus.publish("MALKUTH_PREDICTION", {"text": f"Simulated outcome for {action}: {data.get('outcome', 'Unknown')}"})
            
        return json.dumps(data) if data else response
        
    def predict_system_state(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Predict future system load and drives.
        """
        # Simple linear extrapolation for now, could be replaced by LLM or regression model
        predicted = {}
        for k, v in current_metrics.items():
            # Assume slight regression to mean or trend continuation
            predicted[k] = v # Placeholder
        return predicted

    def infer_causal_rules(self):
        """
        Analyze history to infer causal rules (X -> Y).
        """
        # This would typically involve analyzing the event log for correlations
        # For this implementation, we'll use a placeholder logic or LLM call on recent history
        # to extract potential causal links.
        pass

    def generate_counterfactual_goals(self):
        """
        Generate goals to test causal models (Counterfactual Curiosity).
        """
        # Identify uncertain causal rules and create goals to test them
        # e.g. "Test if increasing temperature actually increases creativity"
        pass

    def simulate_futures(self, action: str, context: str, n_scenarios: int = 3) -> Dict[str, Any]:
        """
        Generate multiple potential futures (Optimistic, Neutral, Pessimistic).
        Returns aggregated utility score and details.
        """
        self.log(f"🔮 [Malkuth] Simulating {n_scenarios} futures for: {action}")
        
        prompt = (
            f"ACTION: {action}\n"
            f"CONTEXT: {context[:1000]}...\n\n"
            f"TASK: Simulate {n_scenarios} distinct potential outcomes (futures) for this action.\n"
            "1. Optimistic: Best case scenario.\n"
            "2. Neutral: Most likely scenario.\n"
            "3. Pessimistic: Worst case (risks/side effects).\n\n"
            "For each, estimate a 'Utility Score' (-1.0 to 1.0) where 1.0 is perfect alignment with goals/safety.\n"
            "Output JSON: {\"futures\": [{\"type\": \"Optimistic\", \"outcome\": \"...\", \"utility\": 0.9}, ...], \"expected_utility\": 0.5}"
        )
        
        response = run_local_lm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a Monte Carlo Simulator.",
            temperature=0.3,
            max_tokens=500
        )
        
        data = parse_json_object_loose(response)
        
        # Calculate expected utility (weighted average if probs provided, else simple average)
        futures = data.get("futures", [])
        if futures:
            total_util = sum(f.get("utility", 0) for f in futures)
            avg_util = total_util / len(futures)
            data["expected_utility"] = avg_util
            self.log(f"🔮 Simulation Result: Expected Utility {avg_util:.2f}")
        else:
            data["expected_utility"] = 0.0
            
        return data

    def make_prediction(self, claim: str, timeframe: str) -> str:
        """
        Fix 3: Make Verification Real.
        Stores a measurable prediction to be verified later.
        """
        if not self.memory_store: return "Memory store unavailable."
        
        text = f"PREDICTION: {claim} [Timeframe: {timeframe}]"
        identity = self.memory_store.compute_identity(text, "BELIEF")
        
        self.memory_store.add_entry(
            identity=identity,
            text=text,
            mem_type="BELIEF", # Stored as belief until verified
            subject="Assistant",
            confidence=0.5, # Tentative
            source="malkuth_prediction"
        )
        self.log(f"🔮 [Malkuth] Registered Prediction: {claim}")
        return f"Prediction registered. ID stored for verification in {timeframe}."

    def update_prediction_model(self, prediction_id: int, actual_outcome: str, error: float):
        """
        Close the loop: Learn from prediction error.
        """
        self.prediction_history.append({
            "id": prediction_id,
            "actual": actual_outcome,
            "error": error,
            "timestamp": time.time()
        })
        # Keep history manageable
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
        self.log(f"🧠 [Malkuth] Updated World Model with prediction error: {error:.4f}")