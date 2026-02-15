import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()
sys.modules['requests'] = MagicMock()

# Import after path setup
from ai_core.core_autonomy import AutonomyManager
from treeoflife.chokmah import Chokmah

class TestProactiveMessaging(unittest.TestCase):
    def setUp(self):
        # Create a mock core
        self.mock_core = MagicMock()

        # Mock Self Model Drives
        self.mock_core.self_model.get_drives.return_value = {
            "loneliness": 0.9,
            "entropy_drive": 0.1,
            "curiosity": 0.5,
            "identity_stability": 0.9
        }
        # Ensure get_autonomy_state returns something safe
        self.mock_core.self_model.get_autonomy_state.return_value = {}

        # Mock Memory Store Stats
        self.mock_core.memory_store.get_memory_stats.return_value = {
            "active_goals": 2,
            "unverified_facts": 5,
            "total_memories": 100
        }

        # Mock Keter
        self.mock_core.keter.evaluate.return_value = {"keter": 0.8}

        # Mock Decider
        self.mock_core.decider.calculate_utility.return_value = 0.6

        # Mock Value Core
        self.mock_core.value_core.get_violation_pressure.return_value = 0.0

        # Mock Malkuth
        self.mock_core.malkuth.last_surprise = 0.0

        # Mock Stability Controller
        self.mock_core.stability_controller = None

        # Initialize AutonomyManager with mock core
        self.autonomy = AutonomyManager(self.mock_core)

        # Initialize Chokmah with mocks
        self.mock_memory_store = MagicMock()
        self.mock_document_store = MagicMock()
        self.mock_settings_fn = lambda: {"base_url": "http://localhost:11434"}
        self.chokmah = Chokmah(
            self.mock_memory_store,
            self.mock_document_store,
            self.mock_settings_fn,
            log_fn=lambda x: None
        )

    def test_initiate_conversation_structure(self):
        """Test that initiate_conversation is registered with correct weights."""
        self.assertIn("initiate_conversation", self.autonomy.actions)
        weights = self.autonomy.weights["initiate_conversation"]
        self.assertEqual(weights["loneliness"], 0.8)
        self.assertEqual(weights["boredom"], 0.2)

    def test_features_include_loneliness(self):
        """Test that _get_current_features extracts loneliness correctly."""
        features = self.autonomy._get_current_features(None)
        self.assertIn("loneliness", features)
        self.assertEqual(features["loneliness"], 0.9)

    def test_execute_action_initiate_conversation(self):
        """Test the execution logic of initiate_conversation."""
        # Setup specific mocks for this test
        self.mock_core.chokmah.pick_conversation_starter.return_value = "Test Context"

        # Execute
        self.autonomy.execute_action("initiate_conversation", 0.5, {})

        # Verify Chokmah called
        self.mock_core.chokmah.pick_conversation_starter.assert_called_once()

        # Verify Event Published
        self.mock_core.event_bus.publish.assert_called_with(
            "SYSTEM:SPONTANEOUS_SPEECH",
            {"context": "Test Context"}
        )

        # Verify Drive Satisfied
        # Check if drive_system was accessed
        if self.mock_core.drive_system:
             self.mock_core.drive_system.satisfy_drive.assert_called_with("loneliness", 0.4)

    def test_chokmah_pick_starter_checkin(self):
        """Test Chokmah picks checkin."""
        with patch('random.choices', return_value=["checkin"]):
            result = self.chokmah.pick_conversation_starter()
            self.assertIn("Social Check-in", result)

    def test_chokmah_pick_starter_fact(self):
        """Test Chokmah picks fact."""
        self.mock_memory_store.get_random_fact.return_value = "The mitochondria is the powerhouse of the cell."

        with patch('random.choices', return_value=["fact"]):
            result = self.chokmah.pick_conversation_starter()
            self.assertIn("Fact Share", result)
            self.assertIn("The mitochondria", result)

    def test_chokmah_pick_starter_gap(self):
        """Test Chokmah picks gap."""
        self.mock_memory_store.get_curiosity_gap_candidate.return_value = "I like apples."

        with patch('random.choices', return_value=["gap"]):
            result = self.chokmah.pick_conversation_starter()
            self.assertIn("Curiosity Gap", result)
            self.assertIn("I like apples", result)

    def test_chokmah_pick_starter_goal(self):
        """Test Chokmah picks goal."""
        # Mock get_active_by_type
        # Assume standard structure: (id, created_at, text, ...)
        self.mock_memory_store.get_active_by_type.return_value = [(1, 123456, "Build a spaceship")]

        with patch('random.choices', return_value=["goal"]):
            result = self.chokmah.pick_conversation_starter()
            self.assertIn("Goal Update", result)
            self.assertIn("Build a spaceship", result)

if __name__ == '__main__':
    unittest.main()
