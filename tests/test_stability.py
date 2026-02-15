import unittest
from unittest.mock import MagicMock
import sys
import os

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_core.core_stability import StabilityController

class TestStabilityController(unittest.TestCase):
    def setUp(self):
        # Create a mock for AICore
        self.mock_core = MagicMock()

        # Create mocks for self_model and value_core
        self.mock_self_model = MagicMock()
        self.mock_value_core = MagicMock()

        # Attach mocks to core
        self.mock_core.self_model = self.mock_self_model
        self.mock_core.value_core = self.mock_value_core

        # Initialize StabilityController
        self.stability_controller = StabilityController(self.mock_core)

    def test_evaluate_no_self_model(self):
        """Test that evaluate returns normal mode when self_model is None."""
        self.mock_core.self_model = None
        state = self.stability_controller.evaluate()
        self.assertEqual(state["mode"], "normal")
        self.assertEqual(state["exploration_scale"], 1.0)
        self.assertIsNone(state["allowed_actions"])

    def test_evaluate_ethical_lockdown(self):
        """Test that evaluate returns ethical_lockdown when violation > 0.1."""
        # Setup self_model drives
        self.mock_self_model.get_drives.return_value = {
            "identity_stability": 0.5,
            "entropy_drive": 0.0
        }
        # Setup value_core violation
        self.mock_value_core.get_violation_pressure.return_value = 0.2

        state = self.stability_controller.evaluate()

        self.assertEqual(state["mode"], "ethical_lockdown")
        self.assertEqual(state["exploration_scale"], 0.0)
        self.assertEqual(state["allowed_actions"], ["introspection", "self_correction"])
        self.assertEqual(state["crs_directives"]["gevurah_bias"], 0.5)

    def test_evaluate_identity_recovery(self):
        """Test that evaluate returns identity_recovery when identity < 0.3."""
        # Setup self_model drives
        self.mock_self_model.get_drives.return_value = {
            "identity_stability": 0.2,
            "entropy_drive": 0.0
        }
        # Setup value_core violation (below threshold)
        self.mock_value_core.get_violation_pressure.return_value = 0.0

        state = self.stability_controller.evaluate()

        self.assertEqual(state["mode"], "identity_recovery")
        self.assertEqual(state["exploration_scale"], 0.1)
        self.assertEqual(state["allowed_actions"], ["introspection", "synthesis", "self_correction"])
        self.assertEqual(state["crs_directives"]["reasoning_depth"], 5)

    def test_evaluate_entropy_control(self):
        """Test that evaluate returns entropy_control when entropy > 0.6."""
        # Setup self_model drives
        self.mock_self_model.get_drives.return_value = {
            "identity_stability": 0.5,
            "entropy_drive": 0.7
        }
        # Setup value_core violation (below threshold)
        self.mock_value_core.get_violation_pressure.return_value = 0.0

        state = self.stability_controller.evaluate()

        self.assertEqual(state["mode"], "entropy_control")
        self.assertEqual(state["exploration_scale"], 0.2)
        self.assertEqual(state["allowed_actions"], ["introspection", "gap_investigation", "optimize_memory"])
        self.assertTrue(state["crs_directives"]["force_pruning"])

    def test_evaluate_normal_operation(self):
        """Test that evaluate returns normal mode under stable conditions."""
        # Setup self_model drives
        self.mock_self_model.get_drives.return_value = {
            "identity_stability": 0.5,
            "entropy_drive": 0.0
        }
        # Setup value_core violation
        self.mock_value_core.get_violation_pressure.return_value = 0.0

        state = self.stability_controller.evaluate()

        self.assertEqual(state["mode"], "normal")
        self.assertEqual(state["exploration_scale"], 1.0)
        self.assertIsNone(state["allowed_actions"])
        self.assertEqual(state["crs_directives"], {})

    def test_evaluate_missing_value_core(self):
        """Test that evaluate handles missing value_core gracefully."""
        # Remove value_core
        self.mock_core.value_core = None

        # Setup self_model drives
        self.mock_self_model.get_drives.return_value = {
            "identity_stability": 0.5,
            "entropy_drive": 0.0
        }

        state = self.stability_controller.evaluate()

        self.assertEqual(state["mode"], "normal")
        # Ensure violation was treated as 0.0
        self.assertEqual(state["exploration_scale"], 1.0)

if __name__ == '__main__':
    unittest.main()
