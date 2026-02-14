class StabilityController:
    """
    Global Stability Governor.
    Regulates autonomy based on organism-level health metrics (Identity, Entropy, Ethics).
    Prevents runaway optimization or self-modification under instability.
    """
    def __init__(self, core):
        self.core = core

    def evaluate(self):
        """
        Determine the system's operational mode based on stability metrics.
        """
        if not self.core.self_model:
            return {"mode": "normal", "exploration_scale": 1.0, "allowed_actions": None}

        identity = self.core.self_model.get_drives().get("identity_stability", 0.5)
        entropy = self.core.self_model.get_drives().get("entropy_drive", 0.0)
        violation = self.core.value_core.get_violation_pressure() if self.core.value_core else 0.0

        state = {
            "mode": "normal",
            "exploration_scale": 1.0,
            "allowed_actions": None,
            "crs_directives": {} # Directives for Cognitive Resource Controller
        }

        if violation > 0.1:
            state["mode"] = "ethical_lockdown"
            state["exploration_scale"] = 0.0
            state["allowed_actions"] = ["introspection", "self_correction"]
            # Force high constraint and low temperature
            state["crs_directives"] = {"gevurah_bias": 0.5, "temperature": 0.1, "strategy": "direct"}

        elif identity < 0.3:
            state["mode"] = "identity_recovery"
            state["exploration_scale"] = 0.1
            state["allowed_actions"] = ["introspection", "synthesis", "self_correction"]
            # Force internal focus
            state["crs_directives"] = {"reasoning_depth": 5, "beam_width": 1, "allow_tools": False}

        elif entropy > 0.6:
            state["mode"] = "entropy_control"
            state["exploration_scale"] = 0.2
            state["allowed_actions"] = ["introspection", "gap_investigation", "optimize_memory"]
            # Force consolidation (convergent thinking)
            state["crs_directives"] = {"temperature": 0.2, "beam_width": 1, "force_pruning": True}

        return state