import unittest
import numpy as np
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock faiss if not present
if 'faiss' not in sys.modules:
    sys.modules['faiss'] = MagicMock()

import memory.memory
from memory.memory import MemoryStore

class TestMemorySearch(unittest.TestCase):
    def setUp(self):
        self.db_path = "./data/test_memory_search.sqlite3"
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.store = MemoryStore(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.db_path + "-wal"):
            os.remove(self.db_path + "-wal")
        if os.path.exists(self.db_path + "-shm"):
            os.remove(self.db_path + "-shm")

    def test_search_refuted_fallback(self):
        """Test the fallback path (no FAISS)"""
        original_faiss_avail = memory.memory.FAISS_AVAILABLE
        memory.memory.FAISS_AVAILABLE = False
        try:
            # Insert data
            emb1 = np.array([1.0, 0.0], dtype='float32')
            emb2 = np.array([0.0, 1.0], dtype='float32')
            self.store.add_entry(identity="id1", text="t1", mem_type="REFUTED_BELIEF", confidence=1.0, source="test", embedding=emb1)
            self.store.add_entry(identity="id2", text="t2", mem_type="REFUTED_BELIEF", confidence=1.0, source="test", embedding=emb2)

            # Query close to emb1
            query = np.array([0.9, 0.1], dtype='float32')
            # Normalize query
            query = query / np.linalg.norm(query)

            res = self.store.search_refuted(query, limit=1)

            self.assertEqual(len(res), 1)
            self.assertEqual(res[0][2], "t1") # text
            self.assertTrue(res[0][4] > 0.8) # similarity high

        finally:
            memory.memory.FAISS_AVAILABLE = original_faiss_avail

    def test_search_refuted_faiss(self):
        """Test the FAISS path"""
        original_faiss_avail = memory.memory.FAISS_AVAILABLE
        memory.memory.FAISS_AVAILABLE = True

        # Inject mock index
        mock_index = MagicMock()
        mock_index.ntotal = 100
        self.store.faiss_index = mock_index

        try:
            # Insert data (needed for ID retrieval from DB)
            emb1 = np.array([1.0, 0.0], dtype='float32')
            emb2 = np.array([0.0, 1.0], dtype='float32')

            # We must set side_effect for add_with_ids to avoid errors if add_entry calls it
            # But MagicMock accepts all calls by default so it's fine.

            id1 = self.store.add_entry(identity="id1", text="t1", mem_type="REFUTED_BELIEF", confidence=1.0, source="test", embedding=emb1)
            id2 = self.store.add_entry(identity="id2", text="t2", mem_type="REFUTED_BELIEF", confidence=1.0, source="test", embedding=emb2)

            # Mock reconstruct behavior
            def reconstruct_side_effect(mid):
                if mid == id1: return emb1
                if mid == id2: return emb2
                raise Exception(f"ID {mid} not found in index")

            mock_index.reconstruct.side_effect = reconstruct_side_effect

            # Query
            query = np.array([0.9, 0.1], dtype='float32')
            query = query / np.linalg.norm(query)

            # Run search
            # This should call reconstruct internally if implemented
            # If not implemented yet, it falls back to DB JSON scan (which still works)
            # But we want to test if it uses FAISS once implemented.
            # Currently it uses fallback.
            res = self.store.search_refuted(query, limit=1)

            # Verify results
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0][2], "t1")

            # Check if reconstruct was called (after implementation)
            # This ensures we are using the optimized path
            mock_index.reconstruct.assert_called()

        finally:
             memory.memory.FAISS_AVAILABLE = original_faiss_avail
             self.store.faiss_index = None

    def test_search_fallback(self):
        """Test the search fallback path (no FAISS)"""
        original_faiss_avail = memory.memory.FAISS_AVAILABLE
        memory.memory.FAISS_AVAILABLE = False
        try:
            # Insert data
            emb1 = np.array([1.0, 0.0], dtype='float32')
            emb2 = np.array([0.0, 1.0], dtype='float32')
            self.store.add_entry(identity="id1", text="t1", mem_type="FACT", confidence=1.0, source="test", embedding=emb1)
            self.store.add_entry(identity="id2", text="t2", mem_type="FACT", confidence=1.0, source="test", embedding=emb2)

            # Query close to emb1
            query = np.array([0.9, 0.1], dtype='float32')
            query = query / np.linalg.norm(query)

            res = self.store.search(query, limit=1)

            self.assertEqual(len(res), 1)
            self.assertEqual(res[0][3], "t1") # text is at index 3 in search results
            self.assertTrue(res[0][4] > 0.8) # similarity high

        finally:
            memory.memory.FAISS_AVAILABLE = original_faiss_avail

if __name__ == '__main__':
    unittest.main()
