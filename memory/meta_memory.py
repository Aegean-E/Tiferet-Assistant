import os
import sqlite3
import time
import json
import threading
from typing import List, Tuple, Optional, Dict
import numpy as np
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MetaMemoryStore:
    """
    Meta-Memory Store: Tracks changes and reflections about memories.

    This is separate from the main memory store to keep meta-cognition
    distinct from actual memories.

    Meta-memories enable:
    - Self-reflection ("I used to be called Ada")
    - Temporal reasoning ("My name changed 3 times this week")
    - Change tracking ("User renamed me on Feb 4th")
    """

    def __init__(self, db_path: str = "./data/meta_memory.sqlite3", embed_fn=None):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.write_lock = threading.Lock()
        self._init_db()
        
        self.faiss_index = None
        self.meta_id_mapping = []

        self._migrate_json_to_blob()

        if FAISS_AVAILABLE:
            self._build_faiss_index()

    def _migrate_json_to_blob(self):
        """Migrate legacy JSON embeddings to BLOB format for performance."""
        try:
            with self._connect() as con:
                # Check if we have unmigrated embeddings
                count = con.execute("SELECT COUNT(*) FROM meta_memories WHERE embedding IS NOT NULL AND embedding_blob IS NULL").fetchone()[0]

            if count > 0:
                logging.info(f"🔄 [Meta-Memory] Migrating {count} embeddings to binary format...")
                batch_size = 1000

                with self._connect() as con:
                    cursor = con.execute("SELECT id, embedding FROM meta_memories WHERE embedding IS NOT NULL AND embedding_blob IS NULL")
                    while True:
                        rows = cursor.fetchmany(batch_size)
                        if not rows: break

                        current_batch = []
                        for mid, emb_json in rows:
                            try:
                                emb = np.array(json.loads(emb_json), dtype='float32')
                                current_batch.append((emb.tobytes(), mid))
                            except Exception:
                                continue

                        if current_batch:
                            try:
                                with self.write_lock:
                                    with self._connect() as con_write:
                                        con_write.executemany("UPDATE meta_memories SET embedding_blob = ? WHERE id = ?", current_batch)
                                        con_write.commit()
                            except Exception as e:
                                logging.error(f"❌ [Meta-Memory] Batch update failed: {e}")

                logging.info("✅ [Meta-Memory] Migration complete.")
        except Exception as e:
            logging.error(f"❌ [Meta-Memory] Migration failed: {e}")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS meta_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,      -- VERSION_UPDATE, CONFLICT_DETECTED, etc.
                    memory_type TEXT NOT NULL,     -- IDENTITY, FACT, PREFERENCE, etc.
                    subject TEXT NOT NULL,         -- User | Assistant
                    text TEXT NOT NULL,            -- Human-readable description
                    old_id INTEGER,                -- Reference to old memory
                    new_id INTEGER,                -- Reference to new memory
                    old_value TEXT,                -- Old value (extracted)
                    new_value TEXT,                -- New value (extracted)
                    metadata TEXT,                 -- Additional JSON metadata
                    created_at INTEGER NOT NULL,
                    embedding TEXT,
                    affect REAL                    -- Emotional valence (0.0 to 1.0)
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_event_type ON meta_memories(event_type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_subject ON meta_memories(subject);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_meta_created ON meta_memories(created_at);")

            # Migration: Add embedding column if it doesn't exist
            try:
                con.execute("ALTER TABLE meta_memories ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass

            # Migration: Add affect column if it doesn't exist
            try:
                con.execute("ALTER TABLE meta_memories ADD COLUMN affect REAL")
            except sqlite3.OperationalError:
                pass

            # Migration: Add embedding_blob column for faster retrieval
            try:
                con.execute("ALTER TABLE meta_memories ADD COLUMN embedding_blob BLOB")
            except sqlite3.OperationalError:
                pass

            # Migration: Unify 'Decider' subject to 'Assistant'
            con.execute("UPDATE meta_memories SET subject = 'Assistant' WHERE subject = 'Decider'")
            con.execute("UPDATE meta_memories SET subject = 'Assistant' WHERE subject = 'Daat'")

            # Outcomes Table (Phase II: Credit Assignment)
            con.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    trigger_state TEXT,  -- JSON snapshot of system state
                    result TEXT,         -- JSON result (deltas)
                    timestamp INTEGER NOT NULL,
                    context_metadata TEXT -- JSON snapshot of system DNA (epigenetics, prompt version)
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_outcome_action ON outcomes(action);")

    def _build_faiss_index(self):
        """Build FAISS index from active meta-memories."""
        try:
            # Reset index state to prevent stale data if DB is empty
            self.faiss_index = None
            self.meta_id_mapping = []

            with self._connect() as con:
                rows = con.execute("SELECT id, embedding, embedding_blob FROM meta_memories WHERE embedding IS NOT NULL OR embedding_blob IS NOT NULL").fetchall()
            
            embeddings = []
            ids = []
            
            for r in rows:
                # r[0]=id, r[1]=json, r[2]=blob
                try:
                    if r[2]:
                        emb = np.frombuffer(r[2], dtype='float32')
                    elif r[1]:
                        emb = np.array(json.loads(r[1]), dtype='float32')
                    else:
                        continue
                    embeddings.append(emb)
                    ids.append(r[0])
                except:
                    continue
            
            if embeddings:
                dimension = len(embeddings[0])
                self.faiss_index = faiss.IndexFlatIP(dimension)
                embeddings_matrix = np.array(embeddings)
                faiss.normalize_L2(embeddings_matrix)
                self.faiss_index.add(embeddings_matrix)
                self.meta_id_mapping = ids
                logging.info(f"🧠 [Meta-Memory] FAISS index built with {len(ids)} records.")
        except Exception as e:
            logging.error(f"⚠️ Failed to build FAISS index for meta-memory: {e}")
            self.faiss_index = None

    def reindex_embeddings(self, embed_fn):
        """Re-compute all embeddings using new model."""
        logging.info("🔄 [Meta-Memory] Re-indexing all meta-memories...")
        with self._connect() as con:
            rows = con.execute("SELECT id, text FROM meta_memories").fetchall()
        
        updates = []
        for mid, text in rows:
            try:
                emb = embed_fn(text)
                emb_json = json.dumps(emb.tolist())
                emb_blob = emb.tobytes()
                updates.append((emb_json, emb_blob, mid))
            except Exception as e:
                logging.error(f"⚠️ Failed to re-embed meta-memory {mid}: {e}")
        
        if updates:
            with self.write_lock:
                with self._connect() as con:
                    con.executemany("UPDATE meta_memories SET embedding = ?, embedding_blob = ? WHERE id = ?", updates)
                    con.commit()
            
            if FAISS_AVAILABLE:
                self._build_faiss_index()
        logging.info(f"✅ [Meta-Memory] Re-indexed {len(updates)} items.")

    def add_meta_memory(
        self,
        event_type: str,
        memory_type: str,
        subject: str,
        text: str,
        old_id: Optional[int] = None,
        new_id: Optional[int] = None,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        metadata: Optional[dict] = None,
        affect: Optional[float] = None
    ) -> int:
        """
        Add a meta-memory about a memory change or event.

        Args:
            event_type: Type of event (VERSION_UPDATE, CONFLICT_DETECTED, etc.)
            memory_type: Type of memory affected (IDENTITY, FACT, etc.)
            subject: Who the memory is about (User, Assistant)
            text: Human-readable description
            old_id: ID of old memory (if applicable)
            new_id: ID of new memory (if applicable)
            old_value: Old value (extracted from memory text)
            new_value: New value (extracted from memory text)
            metadata: Additional structured data
            affect: Emotional valence (0.0 to 1.0)

        Returns:
            ID of created meta-memory
        """
        # Enforce subject unification
        if subject in ["Decider", "Daat"]:
            subject = "Assistant"

        metadata_json = json.dumps(metadata) if metadata else None
        
        # Generate embedding
        embedding = None
        embedding_json = None
        embedding_blob = None
        if self.embed_fn:
            try:
                embedding = self.embed_fn(text)
                embedding_json = json.dumps(embedding.tolist())
                embedding_blob = embedding.tobytes()
            except Exception as e:
                logging.error(f"⚠️ Failed to generate embedding for meta-memory: {e}")

        with self.write_lock:
            with self._connect() as con:
                cur = con.execute("""
                    INSERT INTO meta_memories (
                        event_type,
                        memory_type,
                        subject,
                        text,
                        old_id,
                        new_id,
                        old_value,
                        new_value,
                        metadata,
                        created_at,
                        embedding,
                        embedding_blob,
                        affect
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_type.upper(),
                    memory_type.upper(),
                    subject,
                    text.strip(),
                    old_id,
                    new_id,
                    old_value,
                    new_value,
                    metadata_json,
                    int(time.time()),
                    embedding_json,
                    embedding_blob,
                    affect
                ))
                row_id = cur.lastrowid
                
                # Update FAISS
                if FAISS_AVAILABLE and embedding is not None:
                    if self.faiss_index is None:
                        dimension = len(embedding)
                        self.faiss_index = faiss.IndexFlatIP(dimension)
                        self.meta_id_mapping = []
                    
                    emb_np = embedding.reshape(1, -1).astype('float32')
                    faiss.normalize_L2(emb_np)
                    self.faiss_index.add(emb_np)
                    self.meta_id_mapping.append(row_id)

                return row_id

    def add_outcome(self, action: str, trigger_state: Dict, result: Dict, context_metadata: Optional[Dict] = None):
        """
        Record an action outcome for Reinforcement Learning.
        """
        with self.write_lock:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO outcomes (action, trigger_state, result, timestamp, context_metadata) VALUES (?, ?, ?, ?, ?)",
                    (action, json.dumps(trigger_state), json.dumps(result), int(time.time()), json.dumps(context_metadata) if context_metadata else None)
                )

    def get_outcomes(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve recent outcomes for policy analysis.
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT action, trigger_state, result, timestamp, context_metadata
                FROM outcomes
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)).fetchall()
        
        outcomes = []
        for r in rows:
            try:
                trigger_state = json.loads(r[1]) if r[1] else {}
                if not isinstance(trigger_state, dict): trigger_state = {}
            except: trigger_state = {}
            
            try:
                result = json.loads(r[2]) if r[2] else {}
                if not isinstance(result, dict): result = {}
            except: result = {}

            try:
                context_metadata = json.loads(r[4]) if r[4] else {}
                if not isinstance(context_metadata, dict): context_metadata = {}
            except: context_metadata = {}

            outcomes.append({
                "action": r[0],
                "trigger_state": trigger_state,
                "result": result,
                "timestamp": r[3],
                "context_metadata": context_metadata
            })
        return outcomes

    def get_average_prediction_error(self, limit: int = 20) -> Optional[float]:
        """
        Calculate the rolling average of prediction error from recent outcomes.
        Used for Utility calculation.
        """
        outcomes = self.get_outcomes(limit=limit * 5) # Fetch more to filter
        errors = []
        for o in outcomes:
            res = o.get("result", {})
            if "prediction_error" in res:
                errors.append(float(res["prediction_error"]))
                if len(errors) >= limit:
                    break
        
        if not errors:
            return None # Signal no data
            
        return sum(errors) / len(errors)

    def add_event(self, event_type: str, subject: str, text: str, affect: Optional[float] = None) -> int:
        """
        Helper to log system events (Netzach, Hod, Decider actions) to meta-memory.
        Wraps add_meta_memory with a default memory_type.
        """
        return self.add_meta_memory(
            event_type=event_type,
            memory_type="SYSTEM_EVENT",
            subject=subject,
            text=text,
            affect=affect
        )

    def get_max_id(self) -> int:
        """Get the highest meta-memory ID."""
        with self._connect() as con:
            row = con.execute("SELECT MAX(id) FROM meta_memories").fetchone()
            return row[0] if row and row[0] else 0

    def get(self, event_id: int) -> Optional[Dict]:
        """Retrieve a specific meta-memory by ID."""
        with self._connect() as con:
            row = con.execute("""
                              SELECT id, event_type, subject, text, created_at
                              FROM meta_memories
                              WHERE id = ?
                              """, (event_id,)).fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "type": row[1],
            "subject": row[2],
            "text": row[3],
            "created_at": row[4]
        }
            
    def count_all(self) -> int:
        """Count all meta-memories."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM meta_memories").fetchone()
            return row[0] if row else 0

    def get_meta_memories_after_id(self, last_id: int, limit: int = 50) -> List[Tuple]:
        """Get meta-memories with ID greater than last_id."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, event_type, subject, text, created_at, affect
                FROM meta_memories
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
            """, (last_id, limit)).fetchall()
        return rows

    def list_recent(self, limit: int = 30) -> List[Tuple[int, str, str, str, str, Optional[float]]]:
        """
        Get recent meta-memories.

        Returns: List of (id, event_type, subject, text, created_at, affect)
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, event_type, subject, text, created_at, affect
                FROM meta_memories
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> List[Dict]:
        """Semantic search for meta-memories."""
        if self.faiss_index and self.faiss_index.ntotal > 0:
            try:
                q_emb = query_embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(q_emb)
                
                search_k = min(limit * 5, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(q_emb, search_k)
                
                candidate_ids = []
                candidate_scores = {}
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.meta_id_mapping):
                        mid = self.meta_id_mapping[idx]
                        candidate_ids.append(mid)
                        candidate_scores[mid] = float(scores[0][i])
                
                if not candidate_ids:
                    return []

                placeholders = ','.join(['?'] * len(candidate_ids))
                with self._connect() as con:
                    rows = con.execute(f"""
                        SELECT id, event_type, subject, text, created_at, affect
                        FROM meta_memories
                        WHERE id IN ({placeholders})
                    """, candidate_ids).fetchall()
                
                results = []
                for r in rows:
                    mid = r[0]
                    if mid in candidate_scores:
                        results.append({
                            'id': mid,
                            'event_type': r[1],
                            'subject': r[2],
                            'text': r[3],
                            'created_at': r[4],
                            'affect': r[5],
                            'similarity': candidate_scores[mid]
                        })
                
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:limit]
            except Exception as e:
                logging.error(f"⚠️ FAISS search failed for meta-memory: {e}")
        
        # 2. Slow Path: Linear Scan (Fallback)
        try:
            with self._connect() as con:
                rows = con.execute("SELECT id, event_type, subject, text, created_at, embedding, affect, embedding_blob FROM meta_memories WHERE embedding IS NOT NULL OR embedding_blob IS NOT NULL").fetchall()
            
            results = []
            q_norm = np.linalg.norm(query_embedding)
            
            for r in rows:
                # r[5]=json, r[7]=blob
                try:
                    if r[7]:
                        emb = np.frombuffer(r[7], dtype='float32')
                    elif r[5]:
                        emb = np.array(json.loads(r[5]), dtype='float32')
                    else:
                        continue

                    emb_norm = np.linalg.norm(emb)
                    if q_norm > 0 and emb_norm > 0:
                        sim = np.dot(query_embedding, emb) / (q_norm * emb_norm)
                        results.append({
                            'id': r[0],
                            'event_type': r[1],
                            'subject': r[2],
                            'text': r[3],
                            'created_at': r[4],
                            'affect': r[6],
                            'similarity': float(sim)
                        })
                except:
                    continue
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
        except Exception as e:
            logging.error(f"⚠️ Linear search failed for meta-memory: {e}")
        
        return []

    def get_by_subject(self, subject: str, limit: int = 30) -> List[dict]:
        """
        Get meta-memories for a specific subject (User or Assistant).
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, event_type, memory_type, subject, text,
                       old_id, new_id, old_value, new_value, metadata, created_at, affect
                FROM meta_memories
                WHERE subject = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (subject, limit)).fetchall()

        result = []
        for r in rows:
            result.append({
                'id': r[0],
                'event_type': r[1],
                'memory_type': r[2],
                'subject': r[3],
                'text': r[4],
                'old_id': r[5],
                'new_id': r[6],
                'old_value': r[7],
                'new_value': r[8],
                'metadata': json.loads(r[9]) if r[9] else None,
                'created_at': r[10],
                'affect': r[11],
            })
        return result

    def get_by_event_type(self, event_type: str, limit: int = 30) -> List[dict]:
        """
        Get meta-memories by event type (e.g., VERSION_UPDATE).
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, event_type, memory_type, subject, text,
                       old_id, new_id, old_value, new_value, metadata, created_at, affect
                FROM meta_memories
                WHERE event_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (event_type.upper(), limit)).fetchall()

        result = []
        for r in rows:
            result.append({
                'id': r[0],
                'event_type': r[1],
                'memory_type': r[2],
                'subject': r[3],
                'text': r[4],
                'old_id': r[5],
                'new_id': r[6],
                'old_value': r[7],
                'new_value': r[8],
                'metadata': json.loads(r[9]) if r[9] else None,
                'created_at': r[10],
                'affect': r[11],
            })
        return result

    def delete_by_event_type(self, event_type: str) -> int:
        """Delete meta-memories by event type."""
        with self.write_lock:
            with self._connect() as con:
                cur = con.execute("DELETE FROM meta_memories WHERE event_type = ?", (event_type.upper(),))
                con.commit()
                count = cur.rowcount
            
            if count > 0 and FAISS_AVAILABLE:
                # Rebuild index to remove deleted items
                self._build_faiss_index()
                
            return count

    def delete_entries(self, ids: List[int]) -> int:
        """Delete specific meta-memories by ID."""
        if not ids:
            return 0
        
        with self.write_lock:
            with self._connect() as con:
                placeholders = ','.join(['?'] * len(ids))
                cur = con.execute(f"DELETE FROM meta_memories WHERE id IN ({placeholders})", ids)
                con.commit()
                count = cur.rowcount
            
            if count > 0 and FAISS_AVAILABLE:
                self._build_faiss_index()
                
            return count

    def get_latest_self_narrative(self) -> Optional[Dict]:
        """Retrieve the most recent SELF_NARRATIVE entry."""
        with self._connect() as con:
            row = con.execute("""
                SELECT id, text, created_at, metadata
                FROM meta_memories
                WHERE event_type = 'SELF_NARRATIVE'
                ORDER BY created_at DESC
                LIMIT 1
            """).fetchone()
        
        if row:
            return {
                "id": row[0],
                "text": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {}
            }
        return None

    def clear(self):
        """DANGEROUS: Clears all meta-memories."""
        with self.write_lock:
            with self._connect() as con:
                con.execute("DELETE FROM meta_memories")
            
            self.faiss_index = None
            self.meta_id_mapping = []

    def prune_events(self, max_age_seconds: int = 259200, event_types: List[str] = None, prune_all: bool = False) -> int:
        """
        Prune old meta-memories of specific types.
        
        If prune_all is True, event_types is ignored and ALL events older than threshold are deleted.
        """
        cutoff = int(time.time()) - max_age_seconds
        
        with self.write_lock:
            with self._connect() as con:
                if prune_all:
                    cur = con.execute("DELETE FROM meta_memories WHERE created_at < ?", (cutoff,))
                else:
                    if event_types is None:
                        # Default noisy types to clean up
                        event_types = [
                            "DECIDER_ACTION", 
                            "NETZACH_ACTION", 
                            "HOD_INSTRUCTION", 
                            "DECIDER_OBSERVATION_RECEIVED", 
                            "TOOL_EXECUTION",
                            "NETZACH_INFO",
                            "NETZACH_INSTRUCTION",
                            "STRATEGIC_THOUGHT",
                            "CHAIN_OF_THOUGHT"
                        ]
                    
                    if not event_types:
                        return 0
                        
                    placeholders = ','.join(['?'] * len(event_types))
                    params = [cutoff] + event_types
                    
                    cur = con.execute(f"""
                        DELETE FROM meta_memories 
                        WHERE created_at < ? AND event_type IN ({placeholders})
                    """, params)

                con.commit()
                count = cur.rowcount
            
            if count > 0 and FAISS_AVAILABLE:
                # Rebuild index since we removed items to keep IDs in sync
                self._build_faiss_index()
                
            return count
