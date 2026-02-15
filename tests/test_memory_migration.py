import unittest
import numpy as np
import json
import os
import sys
import sqlite3

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock faiss if not present
if 'faiss' not in sys.modules:
    sys.modules['faiss'] = None

from memory.memory import MemoryStore

class TestMemoryMigration(unittest.TestCase):
    def setUp(self):
        self.db_path = "./data/test_migration.sqlite3"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        # 1. Create DB with legacy schema and data MANUALLY
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = sqlite3.connect(self.db_path)
        con.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity TEXT NOT NULL,
                parent_id INTEGER,
                type TEXT NOT NULL,
                subject TEXT NOT NULL,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                conflict_with TEXT,
                created_at INTEGER NOT NULL,
                embedding TEXT, -- Old Type
                affect REAL DEFAULT 0.5,
                verified INTEGER DEFAULT 0,
                epistemic_origin TEXT DEFAULT 'INFERENCE',
                deleted INTEGER DEFAULT 0,
                flags TEXT,
                completed INTEGER DEFAULT 0,
                progress REAL DEFAULT 0.0,
                verification_attempts INTEGER DEFAULT 0
            )
        """)

        # Insert 10 items with JSON embeddings
        self.dim = 768
        for i in range(10):
            emb = np.random.rand(self.dim).astype('float32')
            emb_json = json.dumps(emb.tolist())
            con.execute("""
                INSERT INTO memories (identity, type, subject, text, confidence, source, created_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (f"id_{i}", "FACT", "User", f"Text {i}", 0.9, "test", 1234567890, emb_json))

        con.commit()
        con.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.db_path + "-wal"):
            os.remove(self.db_path + "-wal")
        if os.path.exists(self.db_path + "-shm"):
            os.remove(self.db_path + "-shm")

    def test_migration_on_init(self):
        """Test that initializing MemoryStore migrates JSON to BLOB."""

        # Initialize store - should trigger migration
        store = MemoryStore(db_path=self.db_path)

        # Check data type in DB
        with store._connect() as con:
            rows = con.execute("SELECT embedding FROM memories").fetchall()

        for r in rows:
            self.assertIsInstance(r[0], bytes, "Embedding should be bytes after migration")

            # Verify we can load it back to numpy
            emb = np.frombuffer(r[0], dtype='float32')
            self.assertEqual(emb.shape[0], self.dim)

    def test_search_after_migration(self):
        """Test that search works after migration."""
        store = MemoryStore(db_path=self.db_path)

        # Perform search
        query = np.random.rand(self.dim).astype('float32')
        results = store.search(query, limit=5)

        self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()
