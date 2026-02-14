import os
import sqlite3
import threading
import faiss
import logging
from typing import Dict, Optional

class BaseStore:
    """Base class for stores using SQLite and FAISS."""
    def __init__(self, db_path: str, log_fn=logging.info):
        self.db_path = db_path
        self.log = log_fn
        self.write_lock = threading.Lock()
        self.faiss_lock = threading.Lock()
        self.faiss_index = None
        self.unsaved_faiss_changes = 0
        self.faiss_save_threshold = 50

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _save_faiss_index(self, force: bool = False):
        if not self.faiss_index: return
        if not force:
            self.unsaved_faiss_changes += 1
            if self.unsaved_faiss_changes < self.faiss_save_threshold:
                return
        try:
            index_path = self.db_path.replace(".sqlite3", ".faiss")
            temp_path = index_path + ".tmp"
            faiss.write_index(self.faiss_index, temp_path)
            os.replace(temp_path, index_path)
            self.unsaved_faiss_changes = 0
        except Exception as e:
            self.log(f"⚠️ Failed to save FAISS index: {e}")

    def vacuum(self):
        try:
            with self._connect() as con:
                con.execute("VACUUM")
        except Exception as e:
            self.log(f"⚠️ Database vacuum failed: {e}")