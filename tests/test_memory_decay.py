
import unittest
import sqlite3
import os
import shutil
import logging
from memory.memory import MemoryStore

# Disable logging for tests
logging.basicConfig(level=logging.ERROR)

class TestMemoryDecay(unittest.TestCase):
    def setUp(self):
        self.db_path = "./data/test_memory_decay.sqlite3"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.store = MemoryStore(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.db_path.replace(".sqlite3", ".faiss")):
            os.remove(self.db_path.replace(".sqlite3", ".faiss"))

    def test_decay_propagation(self):
        # Create a chain: A -> B -> C
        id_a = self.store.add_entry(identity="A", text="A", mem_type="FACT", confidence=1.0, source="test")
        id_b = self.store.add_entry(identity="B", text="B", mem_type="FACT", confidence=1.0, source="test")
        id_c = self.store.add_entry(identity="C", text="C", mem_type="FACT", confidence=1.0, source="test")

        self.store.link_memory_directional(id_a, id_b, "RELATED", strength=1.0)
        self.store.link_memory_directional(id_b, id_c, "RELATED", strength=0.5)

        # Decay from A
        # Decay factor = 0.5
        # B is neighbor of A (str 1.0). Drop = (1-0.5)*1.0 = 0.5. New = 1.0*(1-0.5)=0.5.
        # C is neighbor of B (str 0.5). Drop = (1-0.5)*0.5 = 0.25. New = 1.0*(1-0.25)=0.75.

        self.store.decay_confidence_network(id_a, decay_factor=0.5, max_depth=3)

        mem_b = self.store.get(id_b)
        mem_c = self.store.get(id_c)

        self.assertAlmostEqual(mem_b['confidence'], 0.5)
        self.assertAlmostEqual(mem_c['confidence'], 0.75)

    def test_visited_handling(self):
        # Diamond: A -> B, A -> C, B -> D, C -> D
        id_a = self.store.add_entry(identity="A", text="A", mem_type="FACT", confidence=1.0, source="test")
        id_b = self.store.add_entry(identity="B", text="B", mem_type="FACT", confidence=1.0, source="test")
        id_c = self.store.add_entry(identity="C", text="C", mem_type="FACT", confidence=1.0, source="test")
        id_d = self.store.add_entry(identity="D", text="D", mem_type="FACT", confidence=1.0, source="test")

        self.store.link_memory_directional(id_a, id_b, "RELATED", strength=1.0)
        self.store.link_memory_directional(id_a, id_c, "RELATED", strength=1.0)
        self.store.link_memory_directional(id_b, id_d, "RELATED", strength=1.0)
        self.store.link_memory_directional(id_c, id_d, "RELATED", strength=0.5) # Weaker link

        # Decay from A.
        # B and C are level 1. Both str 1.0.
        # Drop = 0.5. New conf = 0.5.

        # D is level 2. Reached from B (str 1.0) and C (str 0.5).
        # Should pick MAX (1.0).
        # Drop = 0.5. New conf = 0.5.

        self.store.decay_confidence_network(id_a, decay_factor=0.5, max_depth=3)

        mem_b = self.store.get(id_b)
        mem_c = self.store.get(id_c)
        mem_d = self.store.get(id_d)

        self.assertAlmostEqual(mem_b['confidence'], 0.5)
        self.assertAlmostEqual(mem_c['confidence'], 0.5)
        self.assertAlmostEqual(mem_d['confidence'], 0.5)

if __name__ == '__main__':
    unittest.main()
