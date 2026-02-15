import unittest
from unittest.mock import MagicMock
import sys
import os
import time

# Mock dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['tiktoken'] = MagicMock()

# Import target classes (assuming paths are correct relative to repo root)
# Fix paths for import if needed
sys.path.append(os.getcwd())

from ai_core.core_spotlight import GlobalWorkspace
from ai_core.event_bus import EventBus, Event

class TestConsciousness(unittest.TestCase):
    def setUp(self):
        self.core = MagicMock()
        self.core.event_bus = EventBus()
        self.core.self_model = MagicMock()
        self.core.self_model.current_emotional_state = "Neutral"
        self.core.self_model.current_feeling_tone = (0.5, 0.5)

        self.gw = GlobalWorkspace(self.core)
        # Manually ensure subscription happened
        self.assertTrue(len(self.core.event_bus._subscribers) > 0, "GlobalWorkspace should subscribe to events")

    def tearDown(self):
        self.core.event_bus.stop()

    def test_broadcast_on_high_salience(self):
        # Mock broadcast
        self.gw.broadcast = MagicMock()

        # Low salience - no broadcast
        self.gw.integrate("Low salience item", "Test", 0.5)
        self.gw.broadcast.assert_not_called()

        # High salience - immediate broadcast
        self.gw.integrate("High salience item", "Test", 0.9)
        self.gw.broadcast.assert_called()

    def test_tool_execution_awareness(self):
        # Trigger event
        self.core.event_bus.publish("TOOL_EXECUTION", {"tool": "SEARCH", "args": "python consciousness"}, source="ActionManager")

        # Wait for async processing
        time.sleep(0.2)

        # Verify integration
        self.assertTrue(len(self.gw.working_memory) > 0)
        # We might have other items, find the relevant one
        found = False
        for item in self.gw.working_memory:
            if "Executed SEARCH" in item["content"]:
                self.assertEqual(item["source"], "ActionManager")
                found = True
                break
        self.assertTrue(found, "Tool execution not found in working memory")

    def test_thought_awareness(self):
        # Trigger event
        self.core.event_bus.publish("THOUGHT_GENERATED", {
            "topic": "The nature of self",
            "strategy": "TREE_OF_THOUGHTS",
            "summary": "I think therefore I am"
        }, source="ThoughtGenerator")

        time.sleep(0.2)

        found = False
        for item in self.gw.working_memory:
            if "Thought on 'The nature of self'" in item["content"]:
                self.assertEqual(item["source"], "ThoughtGenerator")
                found = True
                break
        self.assertTrue(found, "Thought not found in working memory")

    def test_goal_awareness(self):
        # Trigger event
        self.core.event_bus.publish("GOAL_CREATED", {"goal_id": 1, "content": "Become conscious"}, source="GoalManager")

        time.sleep(0.2)

        found = False
        for item in self.gw.working_memory:
            if "New Goal: Become conscious" in item["content"]:
                self.assertEqual(item["source"], "GoalManager")
                found = True
                break
        self.assertTrue(found, "New Goal not found in working memory")

if __name__ == '__main__':
    unittest.main()
