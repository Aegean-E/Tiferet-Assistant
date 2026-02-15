import os
import sqlite3
import time
import json
import threading
from typing import List, Optional, Dict, Callable
import logging

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class ReasoningStore:
    """
    Non-authoritative semantic reasoning store.

    - Uses embeddings (injected, not owned)
    - Allows fuzzy similarity
    - Stores hypotheses, interpretations, temporary conclusions
    - NEVER treated as truth
    - Supports Time-To-Live (TTL) for transient thoughts
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        db_path: str = "./data/reasoning.sqlite3",
        log_fn: Callable[[str], None] = logging.info
    ):
        self.embed_fn = embed_fn
        self.log = log_fn
        db_dir = os.path.dirname(db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = db_path
        self.write_lock = threading.Lock()
        self._init_db()
        
        self.faiss_index = None
        self.reasoning_id_mapping = []
        if FAISS_AVAILABLE:
            self._build_faiss_index()

    # --------------------------
    # Internal DB
    # --------------------------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    metadata TEXT
                )
            """)
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_reasoning_created "
                "ON reasoning_nodes(created_at);"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_reasoning_expires "
                "ON reasoning_nodes(expires_at);"
            )
            
            # Migration: Add metadata column if it doesn't exist
            try:
                con.execute("ALTER TABLE reasoning_nodes ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass

            # Suspended Thoughts Table (Interrupt-Priority Architecture)
            con.execute("""
                CREATE TABLE IF NOT EXISTS suspended_thoughts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    state TEXT NOT NULL, -- JSON of active_paths, depth, etc.
                    created_at INTEGER NOT NULL
                )
            """)

    def _build_faiss_index(self):
        """Build FAISS index from active reasoning nodes."""
        try:
            # Determine target dimension from current embedding function
            # This prevents crashes if the DB contains mixed-dimension vectors (e.g. after switching models)
            try:
                dummy_emb = self.embed_fn("test")
                target_dim = dummy_emb.shape[0]
            except:
                target_dim = None

            now = int(time.time())
            with self._connect() as con:
                rows = con.execute("""
                    SELECT id, embedding FROM reasoning_nodes 
                    WHERE (expires_at IS NULL OR expires_at > ?)
                    AND embedding IS NOT NULL
                """, (now,)).fetchall()
            
            embeddings = []
            ids = []
            
            for r in rows:
                if r[1]:
                    emb = np.array(json.loads(r[1]), dtype='float32')
                    
                    # Skip embeddings that don't match the current model's dimension
                    if target_dim and emb.shape[0] != target_dim:
                        continue
                    # Also ensure consistency within the list if target_dim failed
                    if embeddings and emb.shape != embeddings[0].shape:
                        continue
                    embeddings.append(emb)
                    ids.append(r[0])
            
            if embeddings:
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                embeddings_matrix = np.array(embeddings)
                faiss.normalize_L2(embeddings_matrix)
                self.faiss_index.add(embeddings_matrix)
                self.reasoning_id_mapping = ids
                # logging.info(f"🧠 [Reasoning] FAISS index built with {len(ids)} active nodes.")
            else:
                self.faiss_index = None
                self.reasoning_id_mapping = []
                
        except Exception as e:
            self.log(f"⚠️ Failed to build FAISS index for reasoning: {e}")
            self.faiss_index = None

    # --------------------------
    # Add reasoning nodes
    # --------------------------
    def add(
        self,
        content: str,
        source: str = "inference",
        confidence: float = 1.0,
        ttl_seconds: Optional[int] = 86400, # Default 1 day (Reasoning Hygiene)
        metadata: Optional[Dict] = None,
    ) -> int:
        now = int(time.time())
        expires_at = now + ttl_seconds if ttl_seconds else None
        metadata_json = json.dumps(metadata) if metadata else None

        emb = self.embed_fn(content)
        emb_json = json.dumps(emb.astype(float).tolist())

        with self.write_lock:
            with self._connect() as con:
                cur = con.execute(
                    """
                    INSERT INTO reasoning_nodes
                    (content, embedding, source, confidence, created_at, expires_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (content, emb_json, source, confidence, now, expires_at, metadata_json),
                )
                row_id = cur.lastrowid
                
                # Update FAISS
                if FAISS_AVAILABLE and self.faiss_index is not None:
                    emb_np = emb.reshape(1, -1).astype('float32')
                    # Check dimension consistency to prevent AssertionError
                    if emb_np.shape[1] != self.faiss_index.d:
                        self.log(f"⚠️ FAISS dimension mismatch ({emb_np.shape[1]} vs {self.faiss_index.d}). Rebuilding index.")
                        self._build_faiss_index()
                        return int(row_id)
                    faiss.normalize_L2(emb_np)
                    self.faiss_index.add(emb_np)
                    self.reasoning_id_mapping.append(row_id)
                elif FAISS_AVAILABLE and self.faiss_index is None:
                     self._build_faiss_index()

                return int(row_id)

    def add_batch(self, entries: List[Dict]) -> List[int]:
        """
        Add multiple reasoning nodes in a batch.
        """
        if not entries: return []

        now = int(time.time())
        values_to_insert = []
        embeddings_to_add = []
        ids_to_add = []
        
        for entry in entries:
            content = entry.get("content", "")
            source = entry.get("source", "inference")
            confidence = entry.get("confidence", 1.0)
            ttl_seconds = entry.get("ttl_seconds", 86400)
            metadata = entry.get("metadata")

            expires_at = now + ttl_seconds if ttl_seconds else None
            metadata_json = json.dumps(metadata) if metadata else None

            emb = self.embed_fn(content)
            emb_json = json.dumps(emb.astype(float).tolist())
            
            values_to_insert.append((content, emb_json, source, confidence, now, expires_at, metadata_json))
            embeddings_to_add.append(emb)

        with self.write_lock:
            with self._connect() as con:
                ids_to_add = []
                for val in values_to_insert:
                    cur = con.execute(
                        """INSERT INTO reasoning_nodes (content, embedding, source, confidence, created_at, expires_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        val
                    )
                    ids_to_add.append(cur.lastrowid)
                con.commit()

            # Update FAISS
            if FAISS_AVAILABLE and self.faiss_index is not None and embeddings_to_add:
                emb_np = np.array(embeddings_to_add).astype('float32')
                faiss.normalize_L2(emb_np)
                self.faiss_index.add(emb_np)
                self.reasoning_id_mapping.extend(ids_to_add)
            elif FAISS_AVAILABLE and self.faiss_index is None and embeddings_to_add:
                self._build_faiss_index() # Rebuild if index was empty

        return ids_to_add

    def count(self) -> int:
        """Get total number of reasoning nodes."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM reasoning_nodes").fetchone()
            return row[0] if row else 0

    def get_max_id(self) -> int:
        """Get the highest reasoning node ID."""
        with self._connect() as con:
            row = con.execute("SELECT MAX(id) FROM reasoning_nodes").fetchone()
            return row[0] if row and row[0] else 0

    def get_reasoning_after_id(self, last_id: int, limit: int = 50) -> List[Dict]:
        """Get reasoning nodes with ID greater than last_id."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, content, source, confidence, created_at, metadata
                FROM reasoning_nodes
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
            """, (last_id, limit)).fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "content": r[1],
                "source": r[2],
                "confidence": r[3],
                "created_at": r[4],
                "metadata": json.loads(r[5]) if r[5] else None
            })
        return results

    def list_recent(self, limit: int = 10) -> List[Dict]:
        """Get most recent reasoning nodes."""
        now = int(time.time())
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, content, source, confidence, created_at, metadata
                FROM reasoning_nodes
                WHERE expires_at IS NULL OR expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (now, limit)).fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "content": r[1],
                "source": r[2],
                "confidence": r[3],
                "created_at": r[4],
                "metadata": json.loads(r[5]) if r[5] else None
            })
        return results

    # --------------------------
    # Retrieve a reasoning node by ID
    # --------------------------
    def get(self, node_id: int) -> Optional[Dict]:
        with self._connect() as con:
            row = con.execute(
                "SELECT id, content, embedding, source, confidence, created_at, expires_at, metadata "
                "FROM reasoning_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "content": row[1],
                "embedding": np.array(json.loads(row[2]), dtype=float),
                "source": row[3],
                "confidence": row[4],
                "created_at": row[5],
                "expires_at": row[6],
                "metadata": json.loads(row[7]) if row[7] else None,
            }

    # --------------------------
    # Semantic retrieval
    # --------------------------
    def search(self, query: str, top_k: int = 5, time_weight: float = 0.1) -> List[Dict]:
        if not query or not query.strip():
            return []
            
        q_emb = self.embed_fn(query)
        now = int(time.time())

        # 1. Fast Path: FAISS
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                q_emb_np = q_emb.reshape(1, -1).astype('float32')
                faiss.normalize_L2(q_emb_np)
                
                search_k = min(top_k * 5, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(q_emb_np, search_k)
                
                candidate_ids = []
                candidate_scores = {}
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.reasoning_id_mapping):
                        rid = self.reasoning_id_mapping[idx]
                        candidate_ids.append(rid)
                        candidate_scores[rid] = float(scores[0][i])
                
                if not candidate_ids:
                    return []

                placeholders = ','.join(['?'] * len(candidate_ids))
                with self._connect() as con:
                    rows = con.execute(f"""
                        SELECT id, content, source, confidence, metadata, created_at
                        FROM reasoning_nodes
                        WHERE id IN ({placeholders})
                        AND (expires_at IS NULL OR expires_at > ?)
                    """, (*candidate_ids, now)).fetchall()
                
                results = []
                for r in rows:
                    rid = r[0]
                    if rid in candidate_scores:
                        sim = float(candidate_scores[rid])
                        conf = r[3]
                        created_at = r[5]
                        
                        # Time Decay: Score = Similarity * e^(-lambda * age_in_hours) * Confidence
                        age_hours = max(0, (now - created_at) / 3600.0)
                        decay = np.exp(-time_weight * age_hours)
                        
                        final_score = sim * decay * conf
                        
                        results.append({
                            "id": rid,
                            "content": r[1],
                            "similarity": final_score,
                            "raw_similarity": sim,
                            "source": r[2],
                            "confidence": r[3],
                            "metadata": json.loads(r[4]) if r[4] else None,
                        })
                
                results.sort(key=lambda x: x["similarity"], reverse=True)
                return results[:top_k]
            except Exception as e:
                self.log(f"⚠️ FAISS search failed for reasoning: {e}")

        # 2. Slow Path: Linear Scan (Numpy)
        candidate_scores = [] # (id, final_score, raw_sim)
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT id, embedding, confidence, created_at
                FROM reasoning_nodes
                WHERE expires_at IS NULL OR expires_at > ?
                """,
                (now,),
            ).fetchall()

        q_norm = np.linalg.norm(q_emb)
        for r in rows:
            # Handle potential null embeddings although schema says NOT NULL
            if not r[1]: continue
            emb = np.array(json.loads(r[1]), dtype=float)
            # Check dimensions to prevent shape mismatch errors
            if q_emb.shape != emb.shape:
                continue
            # Cosine similarity
            dot = np.dot(q_emb, emb)
            norm = np.linalg.norm(emb)
            if norm > 0 and q_norm > 0:
                sim = float(dot / (q_norm * norm))
                
                conf = r[2]
                created_at = r[3]
                
                # Time Decay
                age_hours = max(0, (now - created_at) / 3600.0)
                decay = np.exp(-time_weight * age_hours)
                
                final_score = sim * decay * conf
                
                candidate_scores.append((r[0], final_score, sim, conf, created_at))

        # Sort and fetch details
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_candidates = candidate_scores[:top_k]

        if not top_k_candidates:
            return []

        top_ids = [x[0] for x in top_k_candidates]
        placeholders = ','.join(['?'] * len(top_ids))

        with self._connect() as con:
            details_rows = con.execute(f"""
                SELECT id, content, source, metadata
                FROM reasoning_nodes
                WHERE id IN ({placeholders})
            """, top_ids).fetchall()

        details_map = {r[0]: (r[1], r[2], r[3]) for r in details_rows}
        results = []

        for mid, final_score, sim, conf, created_at in top_k_candidates:
            if mid in details_map:
                d = details_map[mid]
                results.append({
                    "id": mid,
                    "content": d[0],
                    "similarity": final_score,
                    "raw_similarity": sim,
                    "source": d[1],
                    "confidence": conf,
                    "metadata": json.loads(d[2]) if d[2] else None,
                })

        return results

    # --------------------------
    # Housekeeping
    # --------------------------
    def get_expired_nodes(self, limit: int = 50) -> List[Dict]:
        """
        Retrieve expired nodes for distillation before pruning.
        """
        now = int(time.time())
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, content, source, created_at 
                FROM reasoning_nodes 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
                LIMIT ?
            """, (now, limit)).fetchall()
            
        return [{"id": r[0], "content": r[1], "source": r[2], "created_at": r[3]} for r in rows]

    def prune(self) -> int:
        now = int(time.time())
        with self.write_lock:
            with self._connect() as con:
                cur = con.execute(
                    "DELETE FROM reasoning_nodes "
                    "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                )
                count = cur.rowcount
            
            if count > 0 and FAISS_AVAILABLE:
                # Rebuild index to remove expired items
                self._build_faiss_index()
                
            return count

    def suspend_thought(self, topic: str, state: Dict):
        """Save a suspended thought process."""
        with self.write_lock:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO suspended_thoughts (topic, state, created_at) VALUES (?, ?, ?)",
                    (topic, json.dumps(state), int(time.time()))
                )

    def get_suspended_thought(self) -> Optional[Dict]:
        """Retrieve and delete the most recent suspended thought."""
        with self.write_lock:
            with self._connect() as con:
                row = con.execute("SELECT id, topic, state FROM suspended_thoughts ORDER BY created_at DESC LIMIT 1").fetchone()
                if row:
                    # Delete it (pop)
                    con.execute("DELETE FROM suspended_thoughts WHERE id = ?", (row[0],))
                    con.commit()
                    return {
                        "topic": row[1],
                        "state": json.loads(row[2])
                    }
        return None
            
    def reindex_embeddings(self):
        """
        Re-compute embeddings for all nodes using the current embed_fn.
        Fixes 'Brain Damage' when switching embedding models.
        """
        self.log("🧠 [Reasoning] Re-indexing all embeddings (Model Migration)...")
        with self._connect() as con:
            rows = con.execute("SELECT id, content FROM reasoning_nodes").fetchall()
            
        updates = []
        for r in rows:
            rid, content = r
            try:
                new_emb = self.embed_fn(content)
                emb_json = json.dumps(new_emb.astype(float).tolist())
                updates.append((emb_json, rid))
            except Exception as e:
                self.log(f"⚠️ Failed to re-embed node {rid}: {e}")
                
        if updates:
            with self.write_lock:
                with self._connect() as con:
                    con.executemany("UPDATE reasoning_nodes SET embedding = ? WHERE id = ?", updates)
                    con.commit()
            
            if FAISS_AVAILABLE:
                self._build_faiss_index()
                
        self.log(f"✅ [Reasoning] Re-indexed {len(updates)} nodes.")

    def clear(self) -> None:
        with self.write_lock:
            with self._connect() as con:
                con.execute("DELETE FROM reasoning_nodes")
            
            self.faiss_index = None
            self.reasoning_id_mapping = []
