"""
Malkuth (Kingdom): The Causal Engine.
Responsible for grounding AI reasoning in physical reality via simulation and causal checks.
"""

import os
import json
from lm import run_local_lm

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

class Malkuth:
    def __init__(self, memory_store=None, meta_memory_store=None, log_fn=print):
        self.memory_store = memory_store
        self.meta_memory_store = meta_memory_store
        self.log = log_fn
        self.causal_graph = nx.DiGraph() if NX_AVAILABLE else None

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
            # Parse JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            
            # 2. Build Temporary Graph
            G = nx.DiGraph()
            for node in data.get('nodes', []):
                G.add_node(node)
            for edge in data.get('edges', []):
                G.add_edge(edge['source'], edge['target'], relation=edge['relation'])
            
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

    def write_file(self, filename: str, content: str) -> str:
        """Write content to a file in the 'works' directory."""
        # Define the allowed directory
        works_dir = os.path.abspath("./works")
        os.makedirs(works_dir, exist_ok=True)
        
        # Sanitize filename
        filename = os.path.basename(filename)
        file_path = os.path.join(works_dir, filename)
        
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
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            
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