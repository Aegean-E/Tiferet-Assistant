import unittest
from unittest.mock import MagicMock, patch
import logging
from typing import Dict, List, Any, Optional
import sys
import os

# Ensure repo root is in path
sys.path.append(os.getcwd())

# Import ai_core.lm to ensure it's loaded and patchable
import ai_core.lm

from treeoflife.tiferet import Decider

class TestDeciderProcessChat(unittest.TestCase):
    def setUp(self):
        self.mock_settings = {
            "base_url": "http://mock",
            "chat_model": "mock-model",
            "embedding_model": "mock-embed",
            "max_tokens": 100,
            "context_window": 1000,
            "system_prompt": "You are a bot.",
            "temperature": 0.7,
            "top_p": 0.9,
        }
        self.get_settings = MagicMock(return_value=self.mock_settings)
        self.update_settings = MagicMock()
        self.memory_store = MagicMock()
        self.document_store = MagicMock()
        self.reasoning_store = MagicMock()
        self.arbiter = MagicMock()
        self.meta_memory_store = MagicMock()
        self.actions = {}
        self.log = MagicMock()

        # Mocks for memory store methods
        self.memory_store.get_recent_filtered.return_value = []
        self.memory_store.list_recent.return_value = []
        self.memory_store.get_active_by_type.return_value = []
        self.memory_store.search.return_value = []

        self.document_store.search_chunks.return_value = []
        self.document_store.search_filenames.return_value = []

        self.meta_memory_store.get_by_event_type.return_value = []
        self.meta_memory_store.search.return_value = []

        self.decider = Decider(
            get_settings_fn=self.get_settings,
            update_settings_fn=self.update_settings,
            memory_store=self.memory_store,
            document_store=self.document_store,
            reasoning_store=self.reasoning_store,
            arbiter=self.arbiter,
            meta_memory_store=self.meta_memory_store,
            actions=self.actions,
            log_fn=self.log
        )
        # Disable background executor for tests to run synchronously
        self.decider.executor = None

    @patch('treeoflife.tiferet.compute_embedding')
    @patch('treeoflife.tiferet.run_local_lm')
    @patch('treeoflife.tiferet.extract_memory_candidates')
    @patch('treeoflife.tiferet.count_tokens')
    def test_process_chat_message_flow(self, mock_count, mock_extract, mock_llm, mock_embedding):
        # Setup mocks
        mock_embedding.return_value = [0.1] * 768
        mock_llm.return_value = "This is a response."
        mock_count.return_value = 10
        mock_extract.return_value = []

        user_text = "Hello, world!"
        history = [{"role": "user", "content": "Hi"}]

        response = self.decider.process_chat_message(user_text, history)

        self.assertEqual(response, "This is a response.")

        # Verify calls
        mock_embedding.assert_called()
        self.memory_store.search.assert_called()
        self.document_store.search_chunks.assert_not_called() # Should not be called for "Hello, world!"
        mock_llm.assert_called()

    @patch('treeoflife.tiferet.compute_embedding')
    @patch('treeoflife.tiferet.run_local_lm')
    @patch('treeoflife.tiferet.extract_memory_candidates')
    @patch('treeoflife.tiferet.count_tokens')
    def test_process_chat_message_with_rag(self, mock_count, mock_extract, mock_llm, mock_embedding):
        # Setup mocks
        mock_embedding.return_value = [0.1] * 768
        mock_llm.return_value = "Here is the document info."
        mock_count.return_value = 10
        mock_extract.return_value = []

        user_text = "Summarize the document file.pdf" # Triggers RAG
        history = []

        response = self.decider.process_chat_message(user_text, history)

        self.assertEqual(response, "Here is the document info.")

        # Verify calls
        self.document_store.search_chunks.assert_called()

if __name__ == '__main__':
    unittest.main()
