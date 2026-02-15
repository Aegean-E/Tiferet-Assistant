import os
import sqlite3
import time
import re
import json
import threading
import random
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class MemoryStore:
    """
    Immutable, append-only memory ledger.

    Responsibilities:
    - Store memory items with versioning
    - Maintain identities for duplicate/version tracking
    - Support basic conflict detection
    """

    def __init__(self, db_path: str = "./data/memory.sqlite3", config : Dict = None):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.write_lock = threading.Lock()
        self.faiss_lock = threading.Lock()
        self._init_db()
        self._migrate_embeddings()
        self.self_model = None # Injected later
        self.unsaved_faiss_changes = 0
        self.faiss_save_threshold = int(config.get("faiss_save_threshold", 50)) if config else 50
        
        self.faiss_index_type = config.get("faiss_index_type", "IndexFlatIP") if config else "IndexFlatIP"
        self.faiss_nlist = int(config.get("faiss_nlist", 100)) if config else 100
        self.faiss_nprobe = int(config.get("faiss_nprobe", 10)) if config else 10
        self.new_mems_since_train = 0
        
        self.faiss_index = None
        if FAISS_AVAILABLE:
            if not self._load_faiss_index():
                self._build_faiss_index()
            else:
                self._sync_faiss_index()

    # --------------------------
    # Internal DB
    # --------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _migrate_embeddings(self):
        """Migrate legacy JSON embeddings to binary blobs."""
        try:
            with self._connect() as con:
                # Check if we have any JSON embeddings (starting with '[')
                # Use LIMIT 1 to check existence quickly
                check = con.execute("SELECT 1 FROM memories WHERE embedding LIKE '[%' LIMIT 1").fetchone()
                if not check:
                    return

                logging.info("📦 [Memory] Migrating legacy JSON embeddings to binary blobs...")

                # Fetch all JSON embeddings
                rows = con.execute("SELECT id, embedding FROM memories WHERE embedding LIKE '[%'").fetchall()

                updates = []
                for mid, emb_json in rows:
                    try:
                        if isinstance(emb_json, str):
                            emb_array = np.array(json.loads(emb_json), dtype='float32')
                            updates.append((emb_array.tobytes(), mid))
                    except:
                        continue

                if updates:
                    with self.write_lock:
                        con.executemany("UPDATE memories SET embedding = ? WHERE id = ?", updates)
                        con.commit()
                    logging.info(f"✅ [Memory] Migrated {len(updates)} embeddings to binary.")
        except Exception as e:
            logging.error(f"⚠️ Migration failed: {e}")

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL,
                    parent_id INTEGER,
                    type TEXT NOT NULL,        -- FACT | PREFERENCE | GOAL | RULE | PERMISSION | IDENTITY | REFUTED_BELIEF
                    subject TEXT NOT NULL,     -- User | Assistant
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    conflict_with TEXT,
                    created_at INTEGER NOT NULL,
                    embedding TEXT,
                    affect REAL DEFAULT 0.5,   -- Emotional valence (0.0 to 1.0)
                    verified INTEGER DEFAULT 0,
                    epistemic_origin TEXT DEFAULT 'INFERENCE' -- DIRECT_EXPERIENCE, INFERENCE, SECOND_HAND_SOURCE, AXIOM
                )
            """)
            
            # Migration: Add deleted column
            try:
                con.execute("ALTER TABLE memories ADD COLUMN deleted INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            con.execute("CREATE INDEX IF NOT EXISTS idx_identity ON memories(identity);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_subject ON memories(subject);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_parent_id ON memories(parent_id);")

            # Performance Optimizations
            con.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_parent_created ON memories(parent_id, created_at);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_identity_created ON memories(identity, created_at);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_deleted ON memories(deleted);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_verified ON memories(verified);")

            # Migration: Add embedding column if it doesn't exist (for existing DBs)
            try:
                con.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass
            
            # Migration: Add verified column if it doesn't exist
            try:
                con.execute("ALTER TABLE memories ADD COLUMN verified INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            # Migration: Add verification_attempts column if it doesn't exist
            try:
                con.execute("ALTER TABLE memories ADD COLUMN verification_attempts INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            # Migration: Add completed column if it doesn't exist (for GOAL tracking)
            try:
                con.execute("ALTER TABLE memories ADD COLUMN completed INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass

            # Migration: Add flags column
            try:
                con.execute("ALTER TABLE memories ADD COLUMN flags TEXT")
            except sqlite3.OperationalError:
                pass

            # Migration: Add progress column if it doesn't exist (for GOAL tracking)
            try:
                con.execute("ALTER TABLE memories ADD COLUMN progress REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass

            # Migration: Add affect column
            try:
                con.execute("ALTER TABLE memories ADD COLUMN affect REAL DEFAULT 0.5")
            except sqlite3.OperationalError:
                pass

            # Migration: Add epistemic_origin column
            try:
                con.execute("ALTER TABLE memories ADD COLUMN epistemic_origin TEXT DEFAULT 'INFERENCE'")
            except sqlite3.OperationalError:
                pass

            # New table for tracking consolidation history to avoid re-checking pairs
            con.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_history (
                    id_a INTEGER NOT NULL,
                    id_b INTEGER NOT NULL,
                    similarity REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (id_a, id_b)
                )
            """)
            
            # Migration: Unify 'Decider' subject to 'Assistant'
            con.execute("UPDATE memories SET subject = 'Assistant' WHERE subject = 'Decider'")
            con.execute("UPDATE memories SET subject = 'Assistant' WHERE subject = 'Daat'")

            # Reminders table for Netzach
            con.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    due_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    completed INTEGER DEFAULT 0
                )
            """)
            
            # Memory Links Table (Graph RAG)
            con.execute("""
                CREATE TABLE IF NOT EXISTS memory_links (
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL, -- 'SUPPORTS', 'CONTRADICTS', 'RELATED_TO'
                    strength REAL DEFAULT 1.0,
                    created_at INTEGER,
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY(source_id) REFERENCES memories(id),
                    FOREIGN KEY(target_id) REFERENCES memories(id)
                )
            """)
            
            # Archive Table (Cold Storage)
            con.execute("""
                CREATE TABLE IF NOT EXISTS memory_archive (
                    id INTEGER PRIMARY KEY,
                    original_id INTEGER,
                    identity TEXT,
                    type TEXT,
                    subject TEXT,
                    text TEXT,
                    archived_at INTEGER
                )
            """)

            # Shadow Memory Table (Adversarial Error Log)
            con.execute("""
                CREATE TABLE IF NOT EXISTS shadow_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)
            
    def vacuum(self):
        """Reclaim unused disk space."""
        try:
            with self._connect() as con:
                con.execute("VACUUM")
        except Exception as e:
            logging.error(f"⚠️ Memory vacuum failed: {e}")

    def optimize(self):
        """Rebuild FAISS index and vacuum DB to reclaim space."""
        self._build_faiss_index()
        self.vacuum()

    def _save_faiss_index(self, force: bool = False):
        """Save FAISS index to disk (Atomic)."""
        if not FAISS_AVAILABLE or not self.faiss_index: return
        
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
            logging.error(f"⚠️ Failed to save FAISS index: {e}")

    def save_index(self):
        """Force save FAISS index to disk."""
        self._save_faiss_index(force=True)

    def _load_faiss_index(self) -> bool:
        """Load FAISS index from disk."""
        if not FAISS_AVAILABLE: return False
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        if os.path.exists(index_path):
            try:
                self.faiss_index = faiss.read_index(index_path)
                logging.info(f"🧠 [Memory] Loaded FAISS index ({self.faiss_index.ntotal} vectors).")
                return True
            except Exception as e:
                logging.error(f"⚠️ Failed to load FAISS index: {e}")
                return False
        return False

    def _sync_faiss_index(self):
        """Sync any missing embeddings to FAISS (if DB grew while FAISS was offline)."""
        if not FAISS_AVAILABLE or not self.faiss_index: return
        
        # Get count from DB
        with self._connect() as con:
            db_count = con.execute("SELECT COUNT(*) FROM memories WHERE parent_id IS NULL AND deleted = 0 AND embedding IS NOT NULL").fetchone()[0]
            
        index_count = self.faiss_index.ntotal
        
        if index_count == db_count:
            return # In sync
            
        # If mismatch, rebuild (simplest safe strategy for IndexIDMap without complex diffing)
        logging.info(f"🧠 [Memory] Syncing FAISS index: {index_count} vs {db_count} in DB. Rebuilding...")
        self._build_faiss_index()

    def _build_faiss_index(self):
        """
        Build FAISS index from active memories in DB.
        This runs on startup to cache embeddings.
        """
        if not FAISS_AVAILABLE: return
        with self.faiss_lock:
            try:
            # Fetch all active memories with embeddings
            # Use cursor to fetch in batches to avoid OOM
                with self._connect() as con:
                    cur = con.execute("""
                    SELECT id, embedding FROM memories 
                    WHERE parent_id IS NULL 
                    AND deleted = 0
                    AND embedding IS NOT NULL
                """)
            
                batch_size = 5000
                total_loaded = 0
                
                # Reset index
                self.faiss_index = None
                
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                        
                    embeddings = []
                    ids = []
                    
                    for r in rows:
                        if r[1]:
                            try:
                                # Handle binary blob or JSON
                                if isinstance(r[1], bytes):
                                    emb = np.frombuffer(r[1], dtype='float32')
                                else:
                                    emb = np.array(json.loads(r[1]), dtype='float32')
                                embeddings.append(emb)
                                ids.append(r[0])
                            except:
                                continue
                    
                    if not embeddings:
                        self.faiss_index = None
                        return

                    dimension = len(embeddings[0])
                    embs_np = np.array(embeddings).astype('float32')
                    faiss.normalize_L2(embs_np)
                    ids_np = np.array(ids).astype('int64')

                    if self.faiss_index_type == "IndexIVFFlat" and total_loaded >= self.faiss_nlist:
                        quantizer = faiss.IndexFlatIP(dimension)
                        ivf_index = faiss.IndexIVFFlat(quantizer, dimension, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
                        self.faiss_index = faiss.IndexIDMap(ivf_index)
                        self.faiss_index.train(embs_np)
                    else:
                        quantizer = faiss.IndexFlatIP(dimension)
                        self.faiss_index = faiss.IndexIDMap(quantizer)
                    
                    self.faiss_index.add_with_ids(embs_np, ids_np)
                    self.new_mems_since_train = 0

                if total_loaded > 0:
                    logging.info(f"🧠 [Memory] FAISS index built with {total_loaded} active memories.")
                    self._save_faiss_index(force=True)
                    
            except Exception as e:
                logging.error(f"⚠️ Failed to build FAISS index for memory: {e}")
                self.faiss_index = None

    def reindex_embeddings(self, embed_fn):
        """Re-compute all embeddings using new model."""
        logging.info("🔄 [Memory] Re-indexing all memories...")
        with self._connect() as con:
            rows = con.execute("SELECT id, text FROM memories WHERE deleted=0").fetchall()
        
        updates = []
        for mid, text in rows:
            try:
                emb = embed_fn(text)
                # Store as binary blob
                emb_blob = emb.tobytes()
                updates.append((emb_blob, mid))
            except Exception as e:
                logging.error(f"⚠️ Failed to re-embed memory {mid}: {e}")
        
        if updates:
            with self.write_lock:
                with self._connect() as con:
                    con.executemany("UPDATE memories SET embedding = ? WHERE id = ?", updates)
                    con.commit()
            
            if FAISS_AVAILABLE:
                self._build_faiss_index()
        logging.info(f"✅ [Memory] Re-indexed {len(updates)} items.")

    # --------------------------
    # Identity
    # --------------------------

    def compute_identity(self, text: str, mem_type: str = None) -> str:
        """
        Deterministic identity for a memory item.
        
        - IDENTITY type: Uses broad patterns (e.g., "User name is") to force versioning.
        - Other types: Uses full normalized text to allow multiple distinct items.
        """
        text_lower = " ".join(text.lower().strip().split())
        # Remove punctuation for identity hashing to prevent near-identical collisions
        text_lower = re.sub(r'[^\w\s]', '', text_lower)

        # Normalize pronouns for identity consistency
        # This ensures "Your name is X" and "Assistant name is X" map to the same identity slot
        text_lower = text_lower.replace("your name", "assistant name")
        text_lower = text_lower.replace("my name", "user name")

        # Patterns for identifying unique "slots" in identity
        # We check these REGARDLESS of mem_type to catch "FACTS" that are actually identities
        patterns = [
            ("name is", "name is"),
            ("lives in", "lives in"),
            ("works at", "works at"),
            ("occupation is", "occupation is"),
            ("is currently a", "is currently a"),
            ("is now called", "name is"),
            ("is known as", "name is"),
            ("identity is", "identity is"),
        ]

        for trigger, norm in patterns:
            if trigger in text_lower:
                parts = text_lower.split(trigger)
                identity_base = parts[0] + norm
                return hashlib.sha256(identity_base.encode("utf-8")).hexdigest()

        # Default: use full normalized text
        return hashlib.sha256(text_lower.encode("utf-8")).hexdigest()

    def exists_identity(self, identity: str) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM memories WHERE identity = ? LIMIT 1",
                (identity,),
            ).fetchone()
        return row is not None

    def get_by_identity(self, identity: str) -> List[Dict]:
        """
        Returns all versions of a claim, ordered chronologically.
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, parent_id, type, subject, text, confidence, source, conflict_with, created_at
                FROM memories
                WHERE identity = ?
                ORDER BY created_at ASC
            """, (identity,)).fetchall()

        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "parent_id": r[1],
                "type": r[2],
                "subject": r[3],
                "text": r[4],
                "confidence": r[5],
                "source": r[6],
                "conflict_with": json.loads(r[7] or "[]"),
                "created_at": r[8],
            })
        return result

    def get(self, memory_id: int) -> Optional[Dict]:
        """Retrieve a specific memory by ID."""
        with self._connect() as con:
            row = con.execute("""
                SELECT id, identity, parent_id, type, subject, text, confidence, source, created_at, verified, flags, verification_attempts
                FROM memories
                WHERE id = ?
            """, (memory_id,)).fetchone()
        
        if not row:
            return None
            
        return {
            "id": row[0], "identity": row[1], "parent_id": row[2],
            "type": row[3], "subject": row[4], "text": row[5],
            "confidence": row[6], "source": row[7], "created_at": row[8],
            "verified": row[9], "flags": row[10], "verification_attempts": row[11]
        }

    def get_shadow_memories(self, limit: int = 3) -> List[Dict]:
        """Retrieve recent shadow memories (mistakes/rejections)."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT text, reason, created_at FROM shadow_memories
                ORDER BY created_at DESC LIMIT ?
            """, (limit,)).fetchall()
        
        return [{"text": r[0], "reason": r[1], "created_at": r[2]} for r in rows]

    def get_embedding(self, memory_id: int) -> Optional[np.ndarray]:
        """Retrieve the embedding vector for a memory ID."""
        with self._connect() as con:
            row = con.execute("SELECT embedding FROM memories WHERE id = ?", (memory_id,)).fetchone()
        
        if row and row[0]:
            try:
                # Handle binary blob (new format)
                if isinstance(row[0], bytes):
                    return np.frombuffer(row[0], dtype='float32')
                # Handle JSON string (legacy format)
                return np.array(json.loads(row[0]), dtype='float32')
            except:
                pass
        return None

    # --------------------------
    # Add memory (append-only)
    # --------------------------

    def add_entry(
        self,
        *,
        identity: str,
        text: str,
        mem_type: str,
        subject: str = "User",
        confidence: float,
        source: str,
        parent_id: Optional[int] = None,
        conflicts: Optional[List[int]] = None,
        created_at: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        progress: float = 0.0,
        affect: float = 0.5,
        epistemic_origin: str = "INFERENCE",
    ) -> int:
        """
        Append a new memory event.

        identity MUST be provided
        subject should be 'User' or 'Assistant' (default: 'User')
        parent_id enables version chaining
        """
        if not identity:
            raise ValueError("identity must be explicitly provided")

        # Enforce subject unification
        if subject in ["Decider", "Daat"]:
            subject = "Assistant"

        # External Influence Budget Check
        if mem_type.upper() == "BELIEF" and self.self_model:
            if not self.self_model.check_influence_budget():
                logging.warning(f"🛡️ Memory: Belief update rejected (Daily Influence Budget Exceeded).")
                return -1 # Indicate failure/rejection
            self.self_model.increment_influence_count()

        conflicts_json = json.dumps(conflicts or [])
        timestamp = created_at if created_at is not None else int(time.time())

        # Binary embedding storage
        embedding_blob = embedding.tobytes() if embedding is not None else None

        with self.write_lock:
            with self._connect() as con:
                cur = con.execute("""
                    INSERT INTO memories (
                        identity,
                        parent_id,
                        type,
                        subject,
                        text,
                        confidence,
                        source,
                        conflict_with,
                        created_at,
                        embedding,
                        progress,
                        affect,
                        epistemic_origin
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    identity,
                    parent_id,
                    mem_type.upper(),
                    subject,
                    text.strip()[:10000],
                    float(confidence),
                    source,
                    conflicts_json,
                    timestamp,
                    embedding_blob,
                    progress,
                    affect,
                    epistemic_origin
                ))
                row_id = cur.lastrowid

            # Update FAISS if available (Outside DB context but inside write_lock)
            if FAISS_AVAILABLE and embedding is not None:
                try:
                    with self.faiss_lock:
                        if self.faiss_index is None:
                            dimension = len(embedding)
                            quantizer = faiss.IndexFlatIP(dimension)
                            self.faiss_index = faiss.IndexIDMap(quantizer)

                        emb_np = embedding.reshape(1, -1).astype('float32')
                        faiss.normalize_L2(emb_np)
                    
                    self.new_mems_since_train += 1
                    if self.faiss_index_type == "IndexIVFFlat" and self.new_mems_since_train >= 500:
                        # Trigger full rebuild to re-train centroids
                        self._build_faiss_index()
                    else:
                        self.faiss_index.add_with_ids(emb_np, np.array([row_id]).astype('int64'))
                        self._save_faiss_index()
                except Exception as e:
                    logging.error(f"❌ MemoryStore: FAISS update failed for ID {row_id}. Index may be out of sync: {e}")
                    # Note: The DB record exists. _sync_faiss_index will repair this on next reboot.

            return row_id

    def add_shadow_memory(self, text: str, reason: str):
        """Add a rejected thought or hallucination to the Shadow."""
        with self.write_lock:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO shadow_memories (text, reason, created_at) VALUES (?, ?, ?)",
                    (text, reason, int(time.time()))
                )

    # --------------------------
    # Query helpers
    # --------------------------

    def get_max_id(self) -> int:
        """Get the highest memory ID."""
        with self._connect() as con:
            row = con.execute("SELECT MAX(id) FROM memories").fetchone()
            return row[0] if row and row[0] else 0
            
    def count_all(self) -> int:
        """Count all active memories (excluding archived)."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM memories WHERE deleted = 0 AND type != 'ARCHIVED_GOAL'").fetchone()
            return row[0] if row else 0

    def count_by_type(self, mem_type: str) -> int:
        """Count active memories of a specific type."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM memories WHERE type = ? AND deleted = 0", (mem_type,)).fetchone()
            return row[0] if row else 0

    def count_verified(self) -> int:
        """Count verified memories."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM memories WHERE verified = 1 AND deleted = 0").fetchone()
            return row[0] if row else 0

    def get_memories_after_id(self, last_id: int, limit: int = 50) -> List[Tuple]:
        """Get active memories with ID greater than last_id."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT m.id, m.type, m.subject, m.text, m.source, m.verified, m.flags
                FROM memories m
                WHERE m.id > ?
                AND m.parent_id IS NULL
                AND m.deleted = 0
                ORDER BY m.id ASC
                LIMIT ?
            """, (last_id, limit)).fetchall()
        return rows

    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about active memories."""
        stats = {}
        with self._connect() as con:
            # Count active goals
            row = con.execute("SELECT COUNT(*) FROM memories WHERE type = 'GOAL' AND parent_id IS NULL AND deleted = 0").fetchone()
            stats['active_goals'] = row[0] if row else 0
            
            # Count unverified beliefs
            row = con.execute("SELECT COUNT(*) FROM memories WHERE type = 'BELIEF' AND verified = 0 AND parent_id IS NULL AND deleted = 0 AND source = 'daydream'").fetchone()
            stats['unverified_beliefs'] = row[0] if row else 0
            
            # Count unverified facts
            row = con.execute("SELECT COUNT(*) FROM memories WHERE type = 'FACT' AND verified = 0 AND parent_id IS NULL AND deleted = 0 AND source = 'daydream'").fetchone()
            stats['unverified_facts'] = row[0] if row else 0
            stats['total_memories'] = self.count_all()
            
        return stats

    def list_recent(self, limit: Optional[int] = 30, offset: int = 0) -> List[Tuple[int, str, str, str, str, int]]:
        """
        Get recent memories, excluding old superseded versions.
        Returns: (id, type, subject, text, source, verified, flags, confidence)
        A memory is hidden if:
        1. It has a parent_id set (meaning it was superseded/consolidated).
        2. There's a newer memory with the EXACT same identity.
        3. There's a newer memory that explicitly points to it as a parent (supersedes it).
        """
        with self._connect() as con:
            query = """
                SELECT m.id, m.type, m.subject, m.text, m.source, m.verified, m.flags, m.confidence, m.progress
                FROM memories m
                WHERE m.parent_id IS NULL
                AND m.deleted = 0
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE newer.identity = m.identity
                    AND newer.created_at > m.created_at
                )
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE newer.parent_id = m.id
                    AND newer.created_at > m.created_at
                )
                ORDER BY m.created_at DESC
            """
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                rows = con.execute(query, (limit, offset)).fetchall()
            else:
                rows = con.execute(query).fetchall()
        return rows

    def get_recent_filtered(self, limit: int = 20, exclude_sources: List[str] = None) -> List[Tuple]:
        """
        Get recent memories with specific source filtering.
        Used to separate chat context from autonomous daydreaming.
        """
        with self._connect() as con:
            query = """
                SELECT m.id, m.type, m.subject, m.text, m.source, m.verified, m.flags, m.confidence, m.progress
                FROM memories m
                WHERE m.parent_id IS NULL
                AND m.deleted = 0
            """
            params = []
            if exclude_sources:
                placeholders = ','.join(['?'] * len(exclude_sources))
                query += f" AND m.source NOT IN ({placeholders})"
                params.extend(exclude_sources)
            
            query += """
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE newer.identity = m.identity
                    AND newer.created_at > m.created_at
                )
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE newer.parent_id = m.id
                    AND newer.created_at > m.created_at
                )
                ORDER BY m.created_at DESC
                LIMIT ?
            """
            params.append(limit)
            rows = con.execute(query, params).fetchall()
        return rows

    def get_active_by_type(self, mem_type: str) -> List[Tuple[int, str, str, str, float, float]]:
        """Get all active memories of a specific type (id, subject, text, source, confidence, progress)."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT m.id, m.subject, m.text, m.source, m.confidence, m.progress
                FROM memories m
                WHERE m.type = ?
                AND m.parent_id IS NULL
                AND m.deleted = 0
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE newer.identity = m.identity
                    AND newer.created_at > m.created_at
                )
                AND NOT EXISTS (
                    SELECT 1 FROM memories newer
                    WHERE newer.parent_id = m.id
                    AND newer.created_at > m.created_at
                )
            """, (mem_type.upper(),)).fetchall()
        return rows

    def get_refuted_memories(self, limit: int = 20) -> List[Tuple]:
        """Get recent refuted beliefs."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT m.id, m.type, m.subject, m.text, m.source, m.verified
                FROM memories m
                WHERE m.type = 'REFUTED_BELIEF'
                AND m.parent_id IS NULL
                AND m.deleted = 0
                ORDER BY m.created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows

    def search(self, query_embedding: np.ndarray, limit: int = 5, target_affect: Optional[float] = None) -> List[Tuple[int, str, str, str, float]]:
        """
        Semantic search for memories using cosine similarity.
        Returns: List of (id, type, subject, text, similarity)
        """
        candidate_ids = []
        candidate_scores = {}

        # 1. Fast Path: Use FAISS if available
        with self.faiss_lock:
            if self.faiss_index and self.faiss_index.ntotal > 0:
                try:
                    q_emb = query_embedding.reshape(1, -1).astype('float32')
                    faiss.normalize_L2(q_emb)
                    
                    # Set nprobe for IVF indexes
                    if self.faiss_index_type == "IndexIVFFlat":
                        try:
                            inner_index = faiss.downcast_index(self.faiss_index.index)
                            inner_index.nprobe = self.faiss_nprobe
                        except:
                            pass

                    # Search more candidates than needed to account for filtered/inactive ones
                    search_k = min(limit * 10, self.faiss_index.ntotal)
                    t_faiss = time.time()
                    scores, indices = self.faiss_index.search(q_emb, search_k)
                    logging.debug(f"⏱️ [Memory] FAISS search took {time.time()-t_faiss:.3f}s")
                    
                    for i, idx in enumerate(indices[0]):
                        if idx != -1:
                            candidate_ids.append(int(idx))
                            candidate_scores[int(idx)] = float(scores[0][i])
                except Exception as e:
                    logging.error(f"⚠️ FAISS search failed: {e}")

        if candidate_ids:
            try:
                # Verify candidates are still active in DB
                placeholders = ','.join(['?'] * len(candidate_ids))
                with self._connect() as con:
                    rows = con.execute(f"""
                        SELECT m.id, m.type, m.subject, m.text, m.confidence, m.affect
                        FROM memories m
                        WHERE m.id IN ({placeholders})
                        AND m.parent_id IS NULL
                        AND m.deleted = 0
                    """, candidate_ids).fetchall()
                
                results = []
                for r in rows:
                    mid = r[0]
                    if mid in candidate_scores:
                        sim = candidate_scores[mid]
                        conf = r[4] if r[4] is not None else 0.5
                        # Epistemic Weighting
                        weighted_score = sim * (0.5 + (0.5 * conf))

                        # Mood Congruence
                        if target_affect is not None:
                            mem_affect = r[5] if len(r) > 5 and r[5] is not None else 0.5
                            affect_sim = 1.0 - abs(target_affect - mem_affect)
                            weighted_score = (weighted_score * 0.8) + (affect_sim * 0.2)
                        
                        results.append((mid, r[1], r[2], r[3], round(float(weighted_score), 4)))
                
                if results:
                    results.sort(key=lambda x: x[4], reverse=True)
                    return results[:limit]
            except Exception as e:
                logging.error(f"⚠️ Search candidate verification failed: {e}")

        # 2. Slow Path: Linear Scan (Fallback)
        try:
            with self._connect() as con:
                cursor = con.execute("""
                    SELECT m.id, m.embedding, m.confidence, m.affect
                    FROM memories m
                    WHERE m.parent_id IS NULL
                    AND m.deleted = 0
                    AND m.embedding IS NOT NULL
                """)

                candidate_scores = []
                q_norm = np.linalg.norm(query_embedding)
                if q_norm == 0: return []

                BATCH_SIZE = 2000 # Increased batch size since we fetch less data

                while True:
                    rows = cursor.fetchmany(BATCH_SIZE)
                    if not rows:
                        break
                    
                    embeddings = []
                    batch_meta = [] # (id, confidence, affect)

                    for r in rows:
                        if not r[1]: continue
                        try:
                            # Handle binary blob or JSON
                            if isinstance(r[1], bytes):
                                emb = np.frombuffer(r[1], dtype='float32')
                            else:
                                emb = np.array(json.loads(r[1]), dtype='float32')

                            if emb.shape[0] != query_embedding.shape[0]:
                                continue
                            embeddings.append(emb)
                            batch_meta.append((r[0], r[2], r[3]))
                        except:
                            continue

                    if not embeddings: continue

                    E = np.array(embeddings, dtype='float32')

                    # Vectorized cosine similarity
                    dots = np.dot(E, query_embedding)
                    norms = np.linalg.norm(E, axis=1)
                    norms[norms == 0] = 1e-10

                    sims = dots / (norms * q_norm)

                    for i, sim in enumerate(sims):
                        mid, conf, mem_affect = batch_meta[i]
                        conf = conf if conf is not None else 0.5
                        mem_affect = mem_affect if mem_affect is not None else 0.5

                        # Calculate weighted score
                        weighted_score = float(sim) * (0.5 + (0.5 * conf))

                        if target_affect is not None:
                            affect_sim = 1.0 - abs(target_affect - mem_affect)
                            weighted_score = (weighted_score * 0.8) + (affect_sim * 0.2)
                            
                        candidate_scores.append((mid, round(float(weighted_score), 4)))

                # Sort and keep top K
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                top_k = candidate_scores[:limit]

                if not top_k:
                    return []

                # Fetch full details for top K
                top_ids = [x[0] for x in top_k]
                placeholders = ','.join(['?'] * len(top_ids))

                with self._connect() as con:
                    details_rows = con.execute(f"""
                        SELECT m.id, m.type, m.subject, m.text
                        FROM memories m
                        WHERE m.id IN ({placeholders})
                    """, top_ids).fetchall()

                # Map back to results preserving order
                details_map = {r[0]: (r[1], r[2], r[3]) for r in details_rows}
                results = []
                
                for mid, score in top_k:
                    if mid in details_map:
                        d = details_map[mid]
                        results.append((mid, d[0], d[1], d[2], score))

                return results

        except Exception as e:
            logging.error(f"⚠️ Linear search fallback failed: {e}")
            return []

    def range_search(self, query_embedding: np.ndarray, threshold: float) -> Tuple[List[int], List[float]]:
        """
        Perform range search on FAISS index (find all vectors within radius).
        Returns (indices, distances).
        Used for clustering and density analysis.
        """
        if FAISS_AVAILABLE:
            with self.faiss_lock:
                if self.faiss_index:
                    try:
                        q_emb = query_embedding.reshape(1, -1).astype('float32')
                        faiss.normalize_L2(q_emb)
                        
                        lims, D, I = self.faiss_index.range_search(q_emb, threshold)
                        
                        mapped_ids = []
                        distances = []
                        
                        for i in range(lims[0], lims[1]):
                            faiss_idx = int(I[i])
                            mapped_ids.append(faiss_idx)
                            distances.append(float(D[i]))
                                
                        return mapped_ids, distances
                    except Exception as e:
                        logging.error(f"⚠️ FAISS range search failed: {e}")

        # 2. Fallback: Linear Scan (Numpy)
        try:
            with self._connect() as con:
                cursor = con.execute("""
                    SELECT id, embedding FROM memories 
                    WHERE parent_id IS NULL AND deleted = 0 AND embedding IS NOT NULL
                """)
                
                mapped_ids = []
                distances = []
                q_norm = np.linalg.norm(query_embedding)
                
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows: break
                    for r in rows:
                        try:
                            if isinstance(r[1], bytes):
                                mem_emb = np.frombuffer(r[1], dtype='float32')
                            else:
                                mem_emb = np.array(json.loads(r[1]), dtype='float32')
                        except: continue

                        if query_embedding.shape != mem_emb.shape: continue
                        m_norm = np.linalg.norm(mem_emb)
                        if m_norm > 0 and q_norm > 0:
                            sim = np.dot(query_embedding, mem_emb) / (q_norm * m_norm)
                            if sim >= threshold:
                                mapped_ids.append(r[0])
                                distances.append(float(sim))
                return mapped_ids, distances
        except Exception as e:
            logging.error(f"⚠️ Linear range search fallback failed: {e}")
            return [], []

    def get_memory_history(self, identity: str) -> List[Dict]:
        """
        Get full version history for a memory via parent_id chain.

        Use this to retrieve old consolidated/linked versions.
        LLM can call this to access previous versions of a memory
        (e.g., old names, previous preferences, etc.)

        Args:
            identity: The identity hash (e.g., "assistant name is")

        Returns: List of all versions in order (oldest → newest)
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, parent_id, type, subject, text, confidence, created_at
                FROM memories
                WHERE identity = ?
                ORDER BY created_at ASC
            """, (identity,)).fetchall()

        versions = []
        for r in rows:
            versions.append({
                'id': r[0],
                'parent_id': r[1],
                'type': r[2],
                'subject': r[3],
                'text': r[4],
                'confidence': r[5],
                'created_at': r[6],
            })
        return versions

    # --------------------------
    # Associative Memory (Graph)
    # --------------------------

    def link_memories(self, id_a: int, id_b: int, relation: str, strength: float = 1.0):
        """Create a bidirectional semantic link between two memories."""
        with self.write_lock:
            with self._connect() as con:
                ts = int(time.time())
                con.execute("INSERT OR REPLACE INTO memory_links VALUES (?, ?, ?, ?, ?)", 
                           (id_a, id_b, relation, strength, ts))
                con.execute("INSERT OR REPLACE INTO memory_links VALUES (?, ?, ?, ?, ?)", 
                           (id_b, id_a, relation, strength, ts))

    def link_memory_directional(self, source_id: int, target_id: int, relation: str, strength: float = 1.0):
        """Create a directional semantic link (Source -> Target)."""
        with self.write_lock:
            with self._connect() as con:
                ts = int(time.time())
                con.execute("INSERT OR REPLACE INTO memory_links VALUES (?, ?, ?, ?, ?)", 
                           (source_id, target_id, relation, strength, ts))

    def get_associated_memories(self, memory_id: int, min_strength: float = 0.5) -> List[Dict]:
        """Retrieve memories linked to the given ID (Graph Traversal)."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT m.id, m.text, m.type, l.relation_type, l.strength
                FROM memory_links l
                JOIN memories m ON l.target_id = m.id
                WHERE l.source_id = ? AND l.strength >= ?
            """, (memory_id, min_strength)).fetchall()
        
        return [{"id": r[0], "text": r[1], "type": r[2], "relation": r[3], "strength": r[4]} for r in rows]

    # --------------------------
    # Dangerous operations
    # --------------------------

    def clear(self):
        """DANGEROUS: Clears the entire memory ledger."""
        with self._connect() as con:
            con.execute("DELETE FROM memories")

    def clear_by_type(self, mem_type: str) -> int:
        """
        DANGEROUS: Clears all memories of a specific type.
        
        Args:
            mem_type: Memory type to clear (FACT, PREFERENCE, GOAL, etc.)
        
        Returns:
            Number of memories deleted
        """
        with self._connect() as con:
            # Get IDs first for FAISS removal
            rows = con.execute("SELECT id FROM memories WHERE type = ? AND deleted = 0", (mem_type.upper(),)).fetchall()
            ids = [r[0] for r in rows]
            
            cur = con.execute(
                "UPDATE memories SET deleted = 1 WHERE type = ?",
                (mem_type.upper(),)
            )
            con.commit()
            
            if ids and FAISS_AVAILABLE and self.faiss_index:
                self.faiss_index.remove_ids(np.array(ids).astype('int64'))
                self._save_faiss_index()
                
            return cur.rowcount

    def delete_entry(self, memory_id: int) -> bool:
        """
        DANGEROUS: Delete a specific memory entry by ID.
        Used for removing hallucinations or corrupted data.
        """
        with self._connect() as con:
            cur = con.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            con.commit()
            success = cur.rowcount > 0
            
        if success and FAISS_AVAILABLE and self.faiss_index:
            self.faiss_index.remove_ids(np.array([memory_id]).astype('int64'))
            self._save_faiss_index()
            
        return success

    def soft_delete_entry(self, memory_id: int) -> bool:
        """
        Soft delete a memory entry by marking it as deleted.
        Preserves the record but hides it from active queries.
        """
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET deleted = 1 WHERE id = ?", (memory_id,))
            con.commit()
            success = cur.rowcount > 0
            
        if success and FAISS_AVAILABLE and self.faiss_index:
            self.faiss_index.remove_ids(np.array([memory_id]).astype('int64'))
            self._save_faiss_index()
            
        return success

    def set_flag(self, memory_id: int, flag: Optional[str]) -> bool:
        """Set or clear a flag on a memory entry."""
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET flags = ? WHERE id = ?", (flag, memory_id))
            con.commit()
            return cur.rowcount > 0

    def update_type(self, memory_id: int, new_type: str) -> bool:
        """Update the type of a memory entry (e.g. to REFUTED_BELIEF)."""
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET type = ? WHERE id = ?", (new_type.upper(), memory_id))
            con.commit()
            return cur.rowcount > 0

    def update_text(self, memory_id: int, new_text: str) -> bool:
        """Update the text of a memory entry."""
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET text = ? WHERE id = ?", (new_text, memory_id))
            con.commit()
            return cur.rowcount > 0

    def update_embedding(self, memory_id: int, embedding: np.ndarray) -> bool:
        """Update the embedding of a memory entry."""
        embedding_blob = embedding.tobytes()
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET embedding = ? WHERE id = ?", (embedding_blob, memory_id))
            con.commit()
            
        if FAISS_AVAILABLE and self.faiss_index:
            # FAISS doesn't support update easily, so remove and add
            self.faiss_index.remove_ids(np.array([memory_id]).astype('int64'))
            emb_np = embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(emb_np)
            self.faiss_index.add_with_ids(emb_np, np.array([memory_id]).astype('int64'))
            self._save_faiss_index()
            
            return cur.rowcount > 0

    def update_confidence(self, memory_id: int, new_confidence: float) -> bool:
        """Update the confidence of a memory entry."""
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET confidence = ? WHERE id = ?", (new_confidence, memory_id))
            con.commit()
            return cur.rowcount > 0

    def update_progress(self, memory_id: int, progress: float) -> bool:
        """Update the progress of a goal memory."""
        with self._connect() as con:
            cur = con.execute("UPDATE memories SET progress = ? WHERE id = ?", (max(0.0, min(1.0, progress)), memory_id))
            con.commit()
            return cur.rowcount > 0

    def decay_confidence_network(self, seed_id: int, decay_factor: float = 0.85, max_depth: int = 2):
        """
        Bayesian-like update: Lower confidence of memories linked to the seed.
        Used when a memory is refuted, weakening the credibility of its neighbors.
        Propagates recursively up to max_depth.
        """
        # BFS for propagation (Optimized with Batched Level-by-Level)
        affected_count = 0
        current_layer_ids = [seed_id]
        visited = {seed_id}

        with self._connect() as con:
            for depth in range(max_depth):
                if not current_layer_ids:
                    break
                
                next_layer_candidates = {} # target_id -> strength

                # 1. Fetch outgoing links for current layer (Chunked to avoid SQL limits)
                chunk_size = 900
                for i in range(0, len(current_layer_ids), chunk_size):
                    chunk = current_layer_ids[i:i + chunk_size]
                    placeholders = ','.join(['?'] * len(chunk))

                    rows = con.execute(f"""
                        SELECT l.target_id, l.strength
                        FROM memory_links l
                        JOIN memories m ON l.target_id = m.id
                        WHERE l.source_id IN ({placeholders}) AND l.strength >= 0.1
                    """, chunk).fetchall()

                    for target_id, strength in rows:
                        if target_id not in visited:
                            # Use max strength if multiple paths lead to same node in this layer
                            if target_id not in next_layer_candidates:
                                next_layer_candidates[target_id] = strength
                            else:
                                next_layer_candidates[target_id] = max(next_layer_candidates[target_id], strength)

                if not next_layer_candidates:
                    break

                targets_to_update = list(next_layer_candidates.keys())
                visited.update(targets_to_update)

                # 2. Fetch current confidence and prepare updates
                updates = []

                for i in range(0, len(targets_to_update), chunk_size):
                    chunk = targets_to_update[i:i + chunk_size]
                    placeholders = ','.join(['?'] * len(chunk))
                    
                    conf_rows = con.execute(
                        f"SELECT id, confidence FROM memories WHERE id IN ({placeholders})",
                        chunk
                    ).fetchall()

                    for mid, current_conf in conf_rows:
                        strength = next_layer_candidates[mid]
                        drop = (1.0 - decay_factor) * strength
                        new_conf = max(0.1, current_conf * (1.0 - drop))
                        updates.append((new_conf, mid))

                # 3. Batch Update
                if updates:
                    con.executemany("UPDATE memories SET confidence = ? WHERE id = ?", updates)
                    affected_count += len(updates)

                current_layer_ids = targets_to_update

            con.commit()

            if affected_count > 0:
                logging.info(f"📉 [Memory] Decayed confidence for {affected_count} nodes linked to ID {seed_id} (Depth {max_depth})")

    def mark_verified(self, memory_id: int) -> None:
        """Mark a memory as verified against source."""
        with self._connect() as con:
            con.execute("UPDATE memories SET verified = 1 WHERE id = ?", (memory_id,))
            con.commit()

    def decay_memories(self, decay_rate: float = 0.01):
        """
        Apply passive decay to memory confidence.
        Excludes: IDENTITY, RULE, PERMISSION, and Verified Facts.
        """
        with self._connect() as con:
            # Decay unverified facts and beliefs
            con.execute(f"""
                UPDATE memories 
                SET confidence = MAX(0.0, confidence - {decay_rate})
                WHERE type IN ('FACT', 'BELIEF') 
                AND verified = 0 
                AND deleted = 0
            """)
            con.commit()

    def archive_weak_memories(self, threshold: float = 0.1) -> int:
        """
        Move low-confidence memories to cold storage (Archive).
        Returns number of archived items.
        """
        with self.write_lock:
            with self._connect() as con:
                # Get IDs to remove
                rows = con.execute("SELECT id FROM memories WHERE confidence < ? AND deleted = 0 AND type NOT IN ('IDENTITY', 'RULE', 'PERMISSION')", (threshold,)).fetchall()
                ids_to_remove = [r[0] for r in rows]
                
                if not ids_to_remove: return 0

                # 1. Copy to archive
                con.execute(f"""
                    INSERT INTO memory_archive (original_id, identity, type, subject, text, archived_at)
                    SELECT id, identity, type, subject, text, {int(time.time())}
                    FROM memories
                    WHERE confidence < ? AND deleted = 0 AND type NOT IN ('IDENTITY', 'RULE', 'PERMISSION')
                """, (threshold,))
                
                # 2. Soft delete from main table
                cur = con.execute("UPDATE memories SET deleted = 1 WHERE confidence < ? AND deleted = 0 AND type NOT IN ('IDENTITY', 'RULE', 'PERMISSION')", (threshold,))
                count = cur.rowcount
                con.commit()
            
            # Remove from FAISS
            if FAISS_AVAILABLE and self.faiss_index:
                 self.faiss_index.remove_ids(np.array(ids_to_remove).astype('int64'))
                 self._save_faiss_index()
                 
            return count

    def increment_verification_attempts(self, memory_id: int) -> int:
        """
        Increment the verification attempts counter for a memory.
        Returns the new count.
        """
        with self._connect() as con:
            con.execute("UPDATE memories SET verification_attempts = COALESCE(verification_attempts, 0) + 1 WHERE id = ?", (memory_id,))
            con.commit()
            row = con.execute("SELECT verification_attempts FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return row[0] if row else 0

    def get_comparison_similarity(self, id1: int, id2: int) -> Optional[float]:
        """Check if two memories have been compared before."""
        if id1 > id2:
            id1, id2 = id2, id1
        with self._connect() as con:
            row = con.execute(
                "SELECT similarity FROM consolidation_history WHERE id_a = ? AND id_b = ?",
                (id1, id2)
            ).fetchone()
        return row[0] if row else None

    def record_comparison(self, id1: int, id2: int, similarity: float) -> None:
        """Record that two memories have been compared."""
        if id1 > id2:
            id1, id2 = id2, id1
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO consolidation_history (id_a, id_b, similarity, created_at) VALUES (?, ?, ?, ?)",
                (id1, id2, float(similarity), int(time.time()))
            )
            con.commit()

    def sanitize_sources(self) -> int:
        """
        Auto-heal corrupted source tags (e.g., remove quotes).
        Returns number of rows fixed.
        """
        import re
        count = 0
        updates = []
        try:
            with self._connect() as con:
                # Find candidates
                cursor = con.execute("SELECT id, text FROM memories WHERE text LIKE '%[Source: \"%' OR text LIKE '%[Source: ''%'")
                rows = cursor.fetchall()
                
                for mid, text in rows:
                    # Fix quotes
                    new_text = re.sub(r'\[Source: "(.*?)"\]', r'[Source: \1]', text)
                    new_text = re.sub(r"\[Source: '(.*?)'\]", r'[Source: \1]', new_text)

                    if new_text != text:
                        updates.append((new_text, mid))

                if updates:
                    con.executemany("UPDATE memories SET text = ? WHERE id = ?", updates)
                    con.commit()
                    count = len(updates)
                    logging.info(f"🧹 [MemoryStore] Auto-sanitized {count} source tags.")
        except Exception as e:
            logging.error(f"❌ Error in sanitize_sources: {e}")
        return count

    def get_curiosity_spark(self) -> Optional[Tuple[int, str, str]]:
        """
        Get a random memory, biasing towards unverified facts for curiosity.
        Returns: (id, text, type) or None
        """
        with self._connect() as con:
            row = None
            
            # 1. Priority: Dissonance (Conflicting memories)
            # Look for memories with conflicts recorded
            row = con.execute("""
                SELECT id, text, type FROM memories 
                WHERE (conflict_with IS NOT NULL AND conflict_with != '[]')
                AND deleted=0
                ORDER BY RANDOM() LIMIT 1
            """).fetchone()
            
            if not row:
                # 2. Priority: Uncertainty (Medium confidence 0.3 - 0.7)
                row = con.execute("""
                    SELECT id, text, type FROM memories 
                    WHERE confidence BETWEEN 0.3 AND 0.7
                    AND deleted=0
                    ORDER BY RANDOM() LIMIT 1
                """).fetchone()
            
            if not row:
                # 3. Fallback: Unverified Facts
                row = con.execute("""
                    SELECT id, text, type FROM memories 
                    WHERE type='FACT' AND verified=0 AND deleted=0
                    ORDER BY RANDOM() LIMIT 1
                """).fetchone()
            
            # Fallback to random if no unverified facts found, or if dice rolled > 0.5
            if not row:
                row = con.execute("SELECT id, text, type FROM memories WHERE deleted=0 ORDER BY RANDOM() LIMIT 1").fetchone()
                
            return row

    def get_random_memories(self, limit: int = 3, types: List[str] = None) -> List[Dict]:
        """Get random memories for dreaming/recombination."""
        query = "SELECT id, type, subject, text, confidence FROM memories WHERE deleted=0 AND parent_id IS NULL"
        params = []
        if types:
            placeholders = ','.join(['?'] * len(types))
            query += f" AND type IN ({placeholders})"
            params.extend(types)
        
        query += " ORDER BY RANDOM() LIMIT ?"
        params.append(limit)
        
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
            
        return [
            {"id": r[0], "type": r[1], "subject": r[2], "text": r[3], "confidence": r[4]} 
            for r in rows
        ]

    def search_refuted(self, query_embedding: np.ndarray, limit: int = 3) -> List[Tuple]:
        """Specific search for REFUTED_BELIEF memories."""
        # 1. Fast Path: Use FAISS if available
        if FAISS_AVAILABLE and self.faiss_index:
            try:
                result = self._search_refuted_faiss(query_embedding, limit)
                if result is not None:
                    return result
            except Exception as e:
                logging.error(f"⚠️ FAISS refuted search failed: {e}. Falling back.")
                # Fallback to linear scan

        # 2. Optimized Fallback
        return self._search_refuted_fallback(query_embedding, limit)

    def _search_refuted_faiss(self, query_embedding: np.ndarray, limit: int = 3) -> Optional[List[Tuple]]:
        """Use FAISS reconstruction for fast search within a subset."""
        # Fetch IDs first (DB is source of truth for 'REFUTED_BELIEF')
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, subject, text, source
                FROM memories
                WHERE type='REFUTED_BELIEF' AND deleted=0
            """).fetchall()

        if not rows: return []

        target_ids = [r[0] for r in rows]
        vectors = []
        valid_indices = []

        # Retrieve vectors from FAISS index directly (No JSON parsing)
        # Note: reconstruct might fail if ID not in index.
        # We process one by one to handle potential missing IDs gracefully.
        # In future: could use reconstruct_batch if IDs are guaranteed.
        for i, mid in enumerate(target_ids):
            try:
                vec = self.faiss_index.reconstruct(mid)
                vectors.append(vec)
                valid_indices.append(i)
            except:
                pass

        # If we found candidate rows but no vectors in FAISS, index is out of sync.
        # Fallback to DB scan to ensure we find the data.
        if not vectors:
            return None

        # Vectorized similarity check
        E = np.array(vectors, dtype='float32')
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0: return []

        # Assuming FAISS vectors are normalized (L2), dot product is cosine similarity
        # If not, we should normalize E.
        # But _build_faiss_index does `faiss.normalize_L2`.
        # So we trust they are normalized.

        # Normalize query
        query_normed = query_embedding / q_norm

        sims = np.dot(E, query_normed)

        results = []
        for j, sim in enumerate(sims):
            if sim > 0.1:
                idx = valid_indices[j]
                r = rows[idx]
                results.append((r[0], r[1], r[2], r[3], float(sim)))

        results.sort(key=lambda x: x[4], reverse=True)
        return results[:limit]

    def _search_refuted_fallback(self, query_embedding: np.ndarray, limit: int = 3) -> List[Tuple]:
        """Optimized linear scan using vectorized Numpy operations."""
        candidate_scores = []
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0: return []
        
        with self._connect() as con:
            cursor = con.execute("""
                SELECT id, embedding
                FROM memories 
                WHERE type='REFUTED_BELIEF' AND deleted=0 AND embedding IS NOT NULL
            """)
            
            BATCH_SIZE = 2000

            while True:
                rows = cursor.fetchmany(BATCH_SIZE)
                if not rows: break
                
                embeddings = []
                batch_ids = []

                for r in rows:
                    if not r[1]: continue
                    try:
                        if isinstance(r[1], bytes):
                            emb = np.frombuffer(r[1], dtype='float32')
                        else:
                            emb = np.array(json.loads(r[1]), dtype='float32')

                        # Check dimension
                        if emb.shape[0] != query_embedding.shape[0]:
                            continue
                        embeddings.append(emb)
                        batch_ids.append(r[0])
                    except: continue

                if not embeddings: continue

                E = np.array(embeddings, dtype='float32')

                if E.shape[1] != query_embedding.shape[0]:
                    continue

                # Vectorized cosine similarity
                dots = np.dot(E, query_embedding)
                norms = np.linalg.norm(E, axis=1)

                # Avoid divide by zero
                norms[norms == 0] = 1e-10

                sims = dots / (norms * q_norm)

                # Filter > 0.1
                mask = sims > 0.1
                indices = np.where(mask)[0]

                for idx in indices:
                    mid = batch_ids[idx]
                    candidate_scores.append((mid, float(sims[idx])))
        
        # Sort and fetch details
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = candidate_scores[:limit]

        if not top_k:
            return []

        top_ids = [x[0] for x in top_k]
        placeholders = ','.join(['?'] * len(top_ids))

        with self._connect() as con:
            details_rows = con.execute(f"""
                SELECT id, subject, text, source
                FROM memories
                WHERE id IN ({placeholders})
            """, top_ids).fetchall()

        details_map = {r[0]: (r[1], r[2], r[3]) for r in details_rows}
        results = []

        for mid, score in top_k:
            if mid in details_map:
                d = details_map[mid]
                results.append((mid, d[0], d[1], d[2], score))

        return results

    # --------------------------
    # Reminders (Temporal Awareness)
    # --------------------------

    def add_reminder(self, text: str, due_at: float) -> int:
        with self.write_lock:
            with self._connect() as con:
                cur = con.execute("INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)", (text, int(due_at), int(time.time())))
                return cur.lastrowid

    def get_due_reminders(self) -> List[Dict]:
        now = int(time.time())
        with self._connect() as con:
            rows = con.execute("SELECT id, text, due_at FROM reminders WHERE due_at <= ? AND completed = 0", (now,)).fetchall()
        return [{"id": r[0], "text": r[1], "due_at": r[2]} for r in rows]

    def complete_reminder(self, reminder_id: int):
        with self.write_lock:
            with self._connect() as con:
                con.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))
