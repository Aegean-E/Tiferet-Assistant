"""
FAISS-Enhanced Document Store

This version uses FAISS for fast vector similarity search
while maintaining SQLite for metadata storage.

Performance improvement: ~10-50x faster for 10,000+ chunks
"""

import os
import logging
import sqlite3
import time
import hashlib
import json
import threading
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ FAISS not installed. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False
    logging.warning("⚠️ FAISS not installed. Install with: pip install faiss-cpu")


class FaissDocumentStore:
    """
    FAISS-enhanced document store with fast vector search.
    
    Uses:
    - SQLite for metadata (filenames, page numbers, etc.)
    - FAISS for vector embeddings (fast similarity search)
    """

    def __init__(self, db_path: str = "./data/documents_faiss.sqlite3", embed_fn=None):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.embed_fn = embed_fn
        self.faiss_index = None
        self.index_lock = threading.Lock()
        self._init_db()
        
        # Load FAISS configuration from settings (assuming self.core.get_settings is available or passed)
        settings = self._load_settings() # Temporary load for init
        self.faiss_index_type = settings.get("faiss_index_type", "IndexFlatIP")
        self.faiss_nlist = settings.get("faiss_nlist", 100) # For IndexIVFFlat
        self.faiss_nprobe = settings.get("faiss_nprobe", 10)

        self._load_faiss_index()
        self._sync_faiss_index()

    # --------------------------
    # Database Initialization
    # --------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            # Documents table (metadata)
            con.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER NOT NULL,
                    upload_source TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)

            # Chunks table (text segments with metadata)
            con.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER,
                    created_at INTEGER NOT NULL,
                    embedding TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            con.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)")
            
            # Migration: Add embedding column if it doesn't exist
            try:
                con.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")
            except sqlite3.OperationalError:
                pass

            # 1. Enable FTS5 Virtual Table for Keyword Search
            con.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts 
                USING fts5(text, content='chunks', content_rowid='id');
            """)
            
            # 2. Trigger to keep FTS synced with Chunks
            con.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
            """)
            con.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                END;
            """)

    # --------------------------
    # FAISS Integration
    # --------------------------

    def _load_faiss_index(self):
        """Load or create FAISS index"""
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        
        if os.path.exists(index_path):
            try:
                with self.index_lock:
                    # Load existing index
                    self.faiss_index = faiss.read_index(index_path)
            except Exception as e:
                logging.error(f"⚠️ FAISS index file is corrupted or incompatible: {e}")
                logging.info("🔧 Deleting corrupted index and creating a fresh one...")
                try:
                    os.remove(index_path)
                except:
                    pass
                self._create_empty_index()
        else:
            self._create_empty_index()
            
    def _create_empty_index(self):
        """Helper to create a new empty index"""
        dimension = self._detect_embedding_dimension()
        with self.index_lock:
            logging.info(f"🔧 Creating FAISS index with dimension: {dimension}")
            if self.faiss_index_type == "IndexIVFFlat":
                # For IndexIVFFlat, we need to train the index
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIDMap(faiss.IndexIVFFlat(quantizer, dimension, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT))
                # Training will happen when the first batch of embeddings is added
            else:
                # Default to IndexFlatIP
                self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

    def _load_settings(self):
        # This is a temporary local load for init, as AICore might not be fully ready
        # In a real scenario, settings would be passed or accessed via a central config manager
        try:
            with open("./settings.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"⚠️ Could not load settings.json for FAISS config: {e}. Using defaults.")
            return {}
            
    def _detect_embedding_dimension(self):
        """Detect the embedding dimension from the model"""
        try:
            # Try to get dimension from existing chunks
            with self._connect() as con:
                sample_chunk = con.execute("""
                    SELECT text FROM chunks LIMIT 1
                """).fetchone()
                
                if sample_chunk:
                    # Regenerate embedding for sample text
                    if self.embed_fn:
                        sample_embedding = self.embed_fn(sample_chunk[0])
                    else:
                        from ai_core.lm import compute_embedding
                        sample_embedding = compute_embedding(sample_chunk[0])
                    return len(sample_embedding)
        except:
            pass
        
        # Fallback: create test embedding to detect dimension
        try:
            test_embedding = self.embed_fn("test") if self.embed_fn else np.random.rand(1536)
            return len(test_embedding)
        except Exception as e:
            logging.warning(f"⚠️ Could not detect embedding dimension: {e}")
            logging.info("🔧 Using default dimension 1536")
            return 1536  # Default fallback
            
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        index_path = self.db_path.replace(".sqlite3", ".faiss")
        temp_path = index_path + ".tmp"
        try:
            with self.index_lock:
                if self.faiss_index:
                    faiss.write_index(self.faiss_index, temp_path)
                    if os.path.exists(index_path):
                        os.remove(index_path)
                    os.rename(temp_path, index_path)
        except Exception as e:
            logging.error(f"⚠️ Failed to save FAISS index: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _add_embeddings_to_faiss(self, embeddings: List[np.ndarray], chunk_ids: List[int], save_index: bool = True):
        """Add embeddings to FAISS index"""
        if not embeddings:
            return
            
        # Defensive check for inhomogeneous shapes
        valid_embeddings = []
        valid_ids = []
        expected_dim = self.faiss_index.d if self.faiss_index else None
        
        for emb, cid in zip(embeddings, chunk_ids):
            if expected_dim is None or len(emb) == expected_dim:
                valid_embeddings.append(emb)
                valid_ids.append(cid)
        
        if not valid_embeddings: return
        embeddings_array = np.array(valid_embeddings).astype('float32')
        
        # Normalize for inner product search
        faiss.normalize_L2(embeddings_array)
        
        with self.index_lock:
            # Train IndexIVFFlat if not already trained
            if self.faiss_index_type == "IndexIVFFlat" and not self.faiss_index.is_trained:
                logging.info(f"🔧 Training FAISS IndexIVFFlat with {len(embeddings)} vectors (nlist={self.faiss_nlist})...")
                # Ensure enough data for training
                if len(embeddings_array) >= self.faiss_nlist:
                    self.faiss_index.train(embeddings_array)
            if self.faiss_index:
                # Add to index
                ids_array = np.array(valid_ids).astype('int64')
                self.faiss_index.add_with_ids(embeddings_array, ids_array)
        
        # Save index
        if save_index:
            self._save_faiss_index()

    def _sync_faiss_index(self):
        """Ensure FAISS index is in sync with SQLite chunks."""
        try:
            with self._connect() as con:
                count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            
            # If DB has data but FAISS is empty/None or mismatched, rebuild
            with self.index_lock:
                index_count = self.faiss_index.ntotal if self.faiss_index else 0
            
            if count > 0 and (index_count == 0 or index_count != count):
                logging.warning(f"⚠️ FAISS index out of sync (Index: {index_count}, DB: {count}). Rebuilding...")
                self._rebuild_index()
        except Exception as e:
            logging.error(f"⚠️ Error syncing FAISS index: {e}")

    def _rebuild_index(self):
        """Rebuild FAISS index from SQLite chunks (re-embedding if necessary)."""
        # Reset index first
        dimension = 1536 # Default, will be updated by first batch if possible
        with self.index_lock:
            self.faiss_index = None 

        with self._connect() as con:
            cur = con.execute("SELECT id, text, embedding FROM chunks ORDER BY id")
        
            batch_size = 1000
            total_processed = 0

            logging.info(f"🔄 Rebuilding index (batch size: {batch_size})...")
            
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                
                batch_embeddings = []
                batch_ids = []
                batch_updates = []

                for r in rows:
                    chunk_id = r[0]
                    text = r[1]
                    emb_json = r[2]
                    
                    emb = None
                    if emb_json:
                        try:
                            emb = np.array(json.loads(emb_json), dtype='float32')
                        except:
                            pass
                    
                    if emb is None:
                        # Re-compute if missing (Legacy data recovery)
                        if self.embed_fn:
                            try:
                                emb = self.embed_fn(text).astype('float32')
                                batch_updates.append((json.dumps(emb.tolist()), chunk_id))
                            except Exception as e:
                                logging.error(f"❌ Failed to compute embedding for chunk {chunk_id}: {e}")
                                continue

                    if emb is not None:
                        batch_embeddings.append(emb)
                        batch_ids.append(chunk_id)
                
                # Initialize index on first batch if needed
                with self.index_lock:
                    if self.faiss_index is None and batch_embeddings:
                        dimension = len(batch_embeddings[0])
                        self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

                # Add batch to FAISS immediately to free memory
                if batch_embeddings and self.faiss_index:
                    self._add_embeddings_to_faiss(batch_embeddings, batch_ids, save_index=False)

                # Save recovered embeddings to DB immediately
                if batch_updates:
                    with self._connect() as update_con:
                        update_con.executemany("UPDATE chunks SET embedding = ? WHERE id = ?", batch_updates)
                        update_con.commit()

                total_processed += len(rows)
                logging.info(f"   Processed {total_processed} chunks...")
            
            # Ensure index exists even if DB was empty
            with self.index_lock:
                if self.faiss_index is None:
                    # Can't call _create_empty_index here because it takes lock
                    # Just create it directly or release lock
                    pass
            if not self.faiss_index:
                self._create_empty_index()
            
            # Save once at the end
            self._save_faiss_index()
            
            logging.info(f"✅ FAISS index rebuilt successfully ({total_processed} chunks).")

    def reindex_embeddings(self, embed_fn):
        """Re-compute all embeddings using new model."""
        logging.info("🔄 [Documents] Re-indexing all chunks...")
        with self._connect() as con:
            rows = con.execute("SELECT id, text FROM chunks").fetchall()
        
        updates = []
        for mid, text in rows:
            try:
                emb = embed_fn(text)
                emb_json = json.dumps(emb.tolist())
                updates.append((emb_json, mid))
            except Exception as e:
                logging.error(f"⚠️ Failed to re-embed chunk {mid}: {e}")
        
        if updates:
            with self._connect() as con:
                con.executemany("UPDATE chunks SET embedding = ? WHERE id = ?", updates)
                con.commit()
            
            self._rebuild_index()
        logging.info(f"✅ [Documents] Re-indexed {len(updates)} chunks.")

    # --------------------------
    # Document Management
    # --------------------------

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def document_exists(self, file_hash: str) -> bool:
        """Check if document already exists in database."""
        with self._connect() as con:
            row = con.execute(
                "SELECT 1 FROM documents WHERE file_hash = ? LIMIT 1",
                (file_hash,)
            ).fetchone()
        return row is not None

    def add_document(
        self,
        file_hash: str,
        filename: str,
        file_type: str,
        file_size: int,
        page_count: Optional[int],
        chunks: List[Dict],  # [{'text': str, 'embedding': np.ndarray, 'page_number': int}, ...]
        upload_source: str = "telegram"
    ) -> int:
        """
        Add document and its chunks to database with FAISS indexing.
        """
        timestamp = int(time.time())
        chunk_embeddings = []
        chunk_ids = []

        with self._connect() as con:
            # Insert document metadata
            cur = con.execute("""
                INSERT INTO documents (
                    file_hash, filename, file_type, file_size, 
                    page_count, chunk_count, upload_source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_hash,
                filename,
                file_type,
                file_size,
                page_count,
                len(chunks),
                upload_source,
                timestamp
            ))
            document_id = cur.lastrowid

            # Insert chunks and collect embeddings
            for idx, chunk in enumerate(chunks):
                con.execute("""
                    INSERT INTO chunks (
                        document_id, chunk_index, text,
                        page_number, created_at, embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    document_id,
                    idx,
                    chunk['text'],
                    chunk.get('page_number'),
                    timestamp,
                    json.dumps(chunk['embedding'].tolist())
                ))
                
                # Store embedding for FAISS
                chunk_embeddings.append(chunk['embedding'])
                # Get the chunk ID (we'll need to query it after commit)
                chunk_ids.append(None)  # Will be filled after commit

            con.commit()

            # Get actual chunk IDs
            chunk_rows = con.execute("""
                SELECT id FROM chunks 
                WHERE document_id = ? 
                ORDER BY chunk_index
            """, (document_id,)).fetchall()
            
            chunk_ids = [row[0] for row in chunk_rows]

        # Add embeddings to FAISS
        self._add_embeddings_to_faiss(chunk_embeddings, chunk_ids, save_index=True)
        
        return document_id

    # --------------------------
    # Fast Semantic Search
    # --------------------------

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Fast semantic search using FAISS.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            document_id: Optionally filter by specific document
        
        Returns:
            List of chunks with similarity scores
        """
        # OPTIMIZATION: Local Search (Document-Specific)
        # If we know the document, search ONLY its chunks via SQLite + Numpy
        # This avoids the issue where relevant chunks are pushed out of the top-k by other documents
        if document_id is not None:
            with self._connect() as con:
                # Fetch all chunks for this document that have embeddings
                rows = con.execute("""
                    SELECT id, chunk_index, text, page_number, embedding 
                    FROM chunks 
                    WHERE document_id = ? AND embedding IS NOT NULL
                """, (document_id,)).fetchall()
            
            if not rows:
                return []

            results = []
            q_norm = np.linalg.norm(query_embedding)
            
            for r in rows:
                emb = np.array(json.loads(r[4]), dtype='float32')
                emb_norm = np.linalg.norm(emb)
                if q_norm > 0 and emb_norm > 0:
                    sim = np.dot(query_embedding, emb) / (q_norm * emb_norm)
                    results.append({
                        'chunk_id': r[0],
                        'document_id': document_id,
                        'chunk_index': r[1],
                        'text': r[2],
                        'page_number': r[3],
                        'filename': '', # Caller knows filename
                        'similarity': float(sim)
                    })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]

        # STANDARD: Global Search via FAISS
        with self.index_lock:
            if not self.faiss_index or self.faiss_index.ntotal == 0:
                return []

            # Normalize query embedding
            query_array = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_array)

            # Set nprobe for IVF indexes
            if self.faiss_index_type == "IndexIVFFlat":
                try:
                    inner_index = faiss.downcast_index(self.faiss_index.index)
                    inner_index.nprobe = self.faiss_nprobe
                except:
                    pass

            # Search using FAISS
            t_faiss = time.time()
            scores, indices = self.faiss_index.search(query_array, min(top_k * 2, self.faiss_index.ntotal))
            logging.debug(f"⏱️ [Docs] FAISS search took {time.time()-t_faiss:.3f}s")
        
        # Get chunk IDs from mapping
        results = []
        valid_results = 0
        
        with self._connect() as con:
            for i in range(len(indices[0])):
                if valid_results >= top_k:
                    break
                    
                chunk_id = indices[0][i]
                if chunk_id == -1:
                    continue
                    
                # chunk_id is now the actual DB ID
                score = float(scores[0][i])
                
                # Get chunk details
                row = con.execute("""
                    SELECT c.chunk_index, c.text, c.page_number, d.filename, d.id
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id = ?
                """, (chunk_id,)).fetchone()
                
                if not row:
                    continue
                    
                # Apply document filter if specified
                if document_id and row[4] != document_id:
                    continue
                
                results.append({
                    'chunk_id': chunk_id,
                    'document_id': row[4],
                    'chunk_index': row[0],
                    'text': row[1],
                    'page_number': row[2],
                    'filename': row[3],
                    'similarity': score
                })
                valid_results += 1

        return results

    # --------------------------
    # Document Queries
    # --------------------------

    def search_filenames(self, query: str, limit: int = 5) -> List[str]:
        """Search for filenames containing query terms."""
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return []
            
        with self._connect() as con:
            rows = con.execute("SELECT filename FROM documents").fetchall()
            
        matches = []
        for (filename,) in rows:
            fn_lower = filename.lower()
            score = sum(1 for t in terms if t in fn_lower)
            if score > 0:
                matches.append((score, filename))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]

    def get_document_by_filename(self, filename: str) -> Optional[int]:
        """Get document ID by exact filename."""
        with self._connect() as con:
            row = con.execute("SELECT id FROM documents WHERE filename = ?", (filename,)).fetchone()
        return row[0] if row else None

    def list_documents(self, limit: int = 1000) -> List[Tuple]:
        """List all documents."""
        with self._connect() as con:
            rows = con.execute("""
                SELECT id, filename, file_type, page_count, chunk_count, created_at
                FROM documents
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows

    def get_document_chunks(self, document_id: int, include_embeddings: bool = False) -> List[Dict]:
        """Get all chunks for a specific document."""
        query = "SELECT chunk_index, text, page_number"
        if include_embeddings:
            query += ", embedding"

        query += " FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC"

        with self._connect() as con:
            rows = con.execute(query, (document_id,)).fetchall()

        results = []
        for r in rows:
            item = {
                'chunk_index': r[0],
                'text': r[1],
                'page_number': r[2]
            }
            if include_embeddings and r[3]:
                try:
                    item['embedding'] = np.array(json.loads(r[3]), dtype='float32')
                except:
                    pass
            results.append(item)

        return results

    def get_specific_chunks(self, document_id: int, indices: List[int], include_embeddings: bool = False) -> List[Dict]:
        """Retrieve specific chunks by their indices within a document."""
        if not indices: return []
        
        placeholders = ','.join(['?'] * len(indices))
        query = f"SELECT chunk_index, text, page_number"
        if include_embeddings:
            query += ", embedding"
        query += f" FROM chunks WHERE document_id = ? AND chunk_index IN ({placeholders}) ORDER BY chunk_index ASC"
        
        with self._connect() as con:
            rows = con.execute(query, [document_id] + indices).fetchall()
            
        results = []
        for r in rows:
            item = {
                'chunk_index': r[0],
                'text': r[1],
                'page_number': r[2]
            }
            # ... (embedding handling if needed)
            results.append(item)
        return results

    def get_chunk_by_index(self, document_id: int, chunk_index: int) -> Optional[Dict]:
        """Retrieve a specific chunk by its index within a document."""
        with self._connect() as con:
            row = con.execute("""
                SELECT id, text, page_number 
                FROM chunks 
                WHERE document_id = ? AND chunk_index = ?
            """, (document_id, chunk_index)).fetchone()
            
        if row:
            return {
                'chunk_id': row[0],
                'text': row[1],
                'page_number': row[2]
            }
        return None

    def delete_document(self, document_id: int) -> bool:
        """Delete document and all its chunks (and remove from FAISS)."""
        with self._connect() as con:
            # Check if document exists
            exists = con.execute(
                "SELECT 1 FROM documents WHERE id = ?",
                (document_id,)
            ).fetchone()

            if not exists:
                return False

            # Get chunk IDs to remove from FAISS
            chunk_rows = con.execute("""
                SELECT id FROM chunks WHERE document_id = ?
            """, (document_id,)).fetchall()
            
            chunk_ids_to_remove = [row[0] for row in chunk_rows]

            # Delete from SQLite
            con.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            con.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            con.commit()

        # Remove from FAISS immediately
        if self.faiss_index and chunk_ids_to_remove:
            with self.index_lock:
                self.faiss_index.remove_ids(np.array(chunk_ids_to_remove).astype('int64'))
                self._save_faiss_index()

        return True

    def find_broken_documents(self) -> List[Dict]:
        """Find documents with integrity issues (0 chunks, count mismatch, missing embeddings)."""
        broken = []
        with self._connect() as con:
            # Check 1: Metadata mismatch or empty chunks
            rows = con.execute("""
                SELECT d.id, d.filename, d.chunk_count, COUNT(c.id) as actual_chunks
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id
                HAVING d.chunk_count != actual_chunks OR d.chunk_count = 0
            """).fetchall()
            
            for r in rows:
                issue = "No chunks found" if r[2] == 0 else f"Chunk count mismatch (Meta: {r[2]}, Actual: {r[3]})"
                broken.append({'id': r[0], 'filename': r[1], 'issue': issue})

            # Check 2: Missing embeddings
            rows_emb = con.execute("""
                SELECT d.id, d.filename, COUNT(c.id)
                FROM documents d
                JOIN chunks c ON d.id = c.document_id
                WHERE c.embedding IS NULL
                GROUP BY d.id
            """).fetchall()

            for r in rows_emb:
                if not any(b['id'] == r[0] for b in broken):
                    broken.append({'id': r[0], 'filename': r[1], 'issue': f"Missing embeddings for {r[2]} chunks"})
        
        return broken

    def get_orphaned_chunk_count(self) -> int:
        """Count chunks that have no parent document."""
        with self._connect() as con:
            row = con.execute("""
                SELECT COUNT(c.id) 
                FROM chunks c 
                LEFT JOIN documents d ON c.document_id = d.id 
                WHERE d.id IS NULL
            """).fetchone()
        return row[0] if row else 0

    def delete_orphaned_chunks(self) -> int:
        """Delete chunks that have no parent document."""
        chunk_ids_to_remove = []
        with self._connect() as con:
            # Get IDs first
            rows = con.execute("""
                SELECT id FROM chunks 
                WHERE document_id NOT IN (SELECT id FROM documents)
            """).fetchall()
            chunk_ids_to_remove = [r[0] for r in rows]
            
            cur = con.execute("""
                DELETE FROM chunks 
                WHERE document_id NOT IN (SELECT id FROM documents)
            """)
            con.commit()
            count = cur.rowcount
        
        if count > 0 and self.faiss_index and chunk_ids_to_remove:
            with self.index_lock:
                self.faiss_index.remove_ids(np.array(chunk_ids_to_remove).astype('int64'))
                self._save_faiss_index()
            
        return count

    def optimize(self):
        """Rebuild FAISS index to remove ghost vectors (deleted documents)."""
        self._rebuild_index()
        self.vacuum() # Also vacuum SQLite

    # --------------------------
    # Utilities
    # --------------------------

    def get_total_documents(self) -> int:
        """Get total number of documents."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def get_total_chunks(self) -> int:
        """Get total number of chunks across all documents."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def get_search_stats(self) -> Dict:
        """Get FAISS search statistics."""
        return {
            'total_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
            'dimension': self.faiss_index.d if self.faiss_index else 0,
            'index_type': str(type(self.faiss_index).__name__) if self.faiss_index else 'None'
        }

    def vacuum(self):
        """Reclaim unused disk space in SQLite."""
        try:
            with self._connect() as con:
                con.execute("VACUUM")
        except Exception as e:
            logging.warning(f"⚠️ Document vacuum failed: {e}")

    def clear(self):
        """DANGEROUS: Clear all documents and chunks."""
        with self._connect() as con:
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM documents")
            con.commit()
        
        # Clear FAISS index
        dimension = 1536  # Adjust as needed
        self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self._save_faiss_index()
