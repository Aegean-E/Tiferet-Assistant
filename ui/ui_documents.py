import tkinter as tk
import logging
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
from datetime import datetime
import threading
import os
import config

class DocumentsUI:
    """Mixin for Documents UI tab"""

    def setup_documents_tab(self):
        """Setup documents interface"""
        # Upload and search frame
        controls_frame = ttk.Frame(self.docs_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Upload button
        upload_button = ttk.Button(controls_frame, text="Upload Documents", command=self.upload_documents,
                                   bootstyle="secondary")
        upload_button.pack(side=tk.LEFT, padx=(0, 5))

        # Stop Processing button
        stop_docs_button = ttk.Button(controls_frame, text="Stop Processing", command=self.stop_processing,
                                      bootstyle="danger")
        stop_docs_button.pack(side=tk.LEFT, padx=(0, 5))

        # Refresh button
        refresh_button = ttk.Button(controls_frame, text="Refresh", command=self.refresh_documents,
                                    bootstyle="secondary")
        refresh_button.pack(side=tk.LEFT, padx=(0, 5))

        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(controls_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.RIGHT, padx=(0, 5))
        self.search_entry.bind('<KeyRelease>', self.filter_documents)
        self.search_entry.bind('<FocusIn>', self.on_search_focus_in)
        self.search_entry.bind('<FocusOut>', self.on_search_focus_out)

        # Set placeholder text
        self.set_placeholder_text()

        # Clear search button
        clear_search_button = ttk.Button(controls_frame, text="Clear", command=self.clear_search, bootstyle="secondary")
        clear_search_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Documents treeview
        docs_tree_frame = ttk.Frame(self.docs_frame)
        docs_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        columns = ("ID", "Filename", "Type", "Pages", "Chunks", "Date Added")
        self.docs_tree = ttk.Treeview(docs_tree_frame, columns=columns, show="headings", height=15)

        # Define headings with click events for sorting and arrows
        for col in columns:
            self.docs_tree.heading(col, text=f"{col} ↕", command=lambda c=col: self.sort_column(c))
            self.docs_tree.column(col, width=100)

        docs_scrollbar = ttk.Scrollbar(docs_tree_frame, orient=tk.VERTICAL, command=self.docs_tree.yview)
        self.docs_tree.configure(yscrollcommand=docs_scrollbar.set)

        self.docs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        docs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Results label
        self.results_var = tk.StringVar(value="Showing all documents")
        self.results_label = ttk.Label(self.docs_frame, textvariable=self.results_var, bootstyle="secondary")
        self.results_label.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Buttons frame
        docs_buttons_frame = ttk.Frame(self.docs_frame)
        docs_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        delete_button = ttk.Button(docs_buttons_frame, text="Delete Selected", command=self.delete_selected_document,
                                   bootstyle="secondary")
        delete_button.pack(side=tk.LEFT)

        delete_all_button = ttk.Button(docs_buttons_frame, text="Delete All", command=self.delete_all_documents,
                                       bootstyle="secondary")
        delete_all_button.pack(side=tk.LEFT, padx=(5, 0))

        check_integrity_button = ttk.Button(docs_buttons_frame, text="Check Integrity",
                                            command=self.check_document_integrity, bootstyle="warning")
        check_integrity_button.pack(side=tk.LEFT, padx=(5, 0))

        report_button = ttk.Button(docs_buttons_frame, text="Gen. Report", 
                                   command=self.generate_document_report, bootstyle="info")
        report_button.pack(side=tk.LEFT, padx=(5, 0))
        
        rebuild_button = ttk.Button(docs_buttons_frame, text="Rebuild Index", 
                                   command=self.rebuild_index, bootstyle="warning")
        rebuild_button.pack(side=tk.LEFT, padx=(5, 0))

        # Clear log button on the right side
        clear_log_button = ttk.Button(docs_buttons_frame, text="Clear Log", command=self.clear_debug_log,
                                      bootstyle="secondary")
        clear_log_button.pack(side=tk.RIGHT)

        # Store original documents list
        self.original_docs = []
        # Track sort direction for each column
        self.sort_directions = {col: False for col in columns}  # False = ascending, True = descending

        # Initialize debug log first
        self.setup_debug_log()

    def setup_debug_log(self):
        """Setup debug log frame in the documents tab"""
        # Create debug log frame using regular Frame instead of LabelFrame to avoid ttkbootstrap issues
        debug_frame = ttk.Frame(self.docs_frame)
        debug_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add a label for the log section
        log_label = ttk.Label(debug_frame, text="Document Processing Log", font=("Arial", 10, "bold"))
        log_label.pack(anchor=tk.W, padx=5, pady=(5, 0))

        # Create text widget for logs - smaller height
        self.debug_log = tk.Text(debug_frame, height=6, state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(debug_frame, orient=tk.VERTICAL, command=self.debug_log.yview)
        self.debug_log.configure(yscrollcommand=scrollbar.set)

        self.debug_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=(0, 5))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 5))

    def log_debug_message(self, message):
        """Log a debug message to the debug log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        
        # Buffer for thread-safe access
        if not hasattr(self, 'debug_log_buffer'): self.debug_log_buffer = []
        self.debug_log_buffer.append(formatted)
        if len(self.debug_log_buffer) > 1000:
            self.debug_log_buffer = self.debug_log_buffer[-1000:]

        if hasattr(self, 'debug_log'):
            self.root.after(0, lambda: self._log_debug_safe(formatted))

    def _log_debug_safe(self, formatted_message):
        try:
            self.debug_log.config(state=tk.NORMAL)
            self.debug_log.insert(tk.END, formatted_message)

            # Limit log size to prevent lag (keep last 500 lines)
            num_lines = int(self.debug_log.index('end-1c').split('.')[0])
            if num_lines > 500:
                self.debug_log.delete("1.0", f"{num_lines - 500 + 1}.0")

            self.debug_log.see(tk.END)  # Auto-scroll to the end
            self.debug_log.config(state=tk.DISABLED)
        except Exception:
            pass

    def clear_debug_log(self):
        """Clear the debug log"""
        self.debug_log.config(state=tk.NORMAL)
        self.debug_log.delete(1.0, tk.END)
        self.debug_log.config(state=tk.DISABLED)

    def clear_search(self):
        """Clear the search field"""
        self.search_var.set("")
        self.search_entry.delete(0, tk.END)  # Actually clear the entry field
        self.set_placeholder_text()  # Set placeholder text
        self.filter_documents()
        self.search_entry.focus_set()  # Set focus back to search box

    def set_placeholder_text(self):
        """Set placeholder text in search box"""
        self.search_entry.delete(0, tk.END)  # Clear any existing text
        self.search_entry.insert(0, "Search documents...")
        self.is_showing_placeholder = True  # Mark that we're showing placeholder

    def on_search_focus_in(self, event):
        """Handle focus in for search entry"""
        if self.is_showing_placeholder:
            self.search_entry.delete(0, tk.END)
            self.is_showing_placeholder = False  # Mark that we're no longer showing placeholder

    def on_search_focus_out(self, event):
        """Handle focus out for search entry"""
        if not self.search_var.get():  # If search box is empty
            self.set_placeholder_text()

    def sort_column(self, col):
        """Sort the treeview by the given column with toggle functionality"""
        # Toggle sort direction for this column
        self.sort_directions[col] = not self.sort_directions[col]
        is_descending = self.sort_directions[col]

        # Update heading text to show sort direction
        arrow = " ↓" if is_descending else " ↑"
        for c in self.sort_directions.keys():
            current_text = self.docs_tree.heading(c)['text']
            # Remove any existing arrows
            clean_text = current_text.split()[0]  # Get just the column name
            if c == col:
                # This is the column being sorted, show the current sort direction
                self.docs_tree.heading(c, text=f"{clean_text}{arrow}")
            else:
                # Other columns show the default arrow
                self.docs_tree.heading(c, text=f"{clean_text} ↕")

        # Get all items
        items = [(self.docs_tree.set(k, col), k) for k in self.docs_tree.get_children('')]

        # Determine if we're sorting numbers or text
        is_numeric = col in ['ID', 'Pages', 'Chunks']

        # Sort based on column type
        if is_numeric:
            # Convert to integer for comparison, handle 'N/A' values
            def sort_key(item):
                val = item[0]
                if val == 'N/A' or val == '':
                    return float('-inf') if is_descending else float('inf')  # Put N/A at the end
                try:
                    return int(val)
                except ValueError:
                    try:
                        return float(val)
                    except ValueError:
                        return val  # Fallback to string comparison if conversion fails

            items.sort(key=sort_key, reverse=is_descending)
        else:
            # For text columns, sort alphabetically (case-insensitive)
            items.sort(key=lambda x: x[0].lower(), reverse=is_descending)

        # Rearrange items in sorted order
        for index, (val, k) in enumerate(items):
            self.docs_tree.move(k, '', index)

        # Log sort action
        if hasattr(self, 'debug_log'):
            sort_order = "descending" if is_descending else "ascending"
            self.log_debug_message(f"Sorted by {col} ({sort_order})")

    def refresh_documents(self):
        """Refresh documents list"""
        if not getattr(self, 'document_store', None):
            return
        
        def fetch_data():
            try:
                # Get total count and list
                total_docs = self.document_store.get_total_documents()
                all_docs = self.document_store.list_documents(limit=5000)
                
                # Get stats for logging
                total_chunks = self.document_store.get_total_chunks()
                orphans = self.document_store.get_orphaned_chunk_count()
                
                # Update UI on main thread
                self.root.after(0, lambda: self._update_docs_ui(all_docs, total_docs, total_chunks, orphans))
            except Exception as e:
                logging.error(f"Error refreshing documents: {e}")

        threading.Thread(target=fetch_data, daemon=True).start()

    def _update_docs_ui(self, all_docs, total_docs, total_chunks, orphans):
        """Update the documents UI with fetched data"""
        try:
            # Store original documents
            self.original_docs = all_docs

            # Clear current tree
            for item in self.docs_tree.get_children():
                self.docs_tree.delete(item)

            # Add documents to tree
            for doc in all_docs:
                doc_id, filename, file_type, page_count, chunk_count, created_at = doc
                date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                page_info = str(page_count) if page_count else "N/A"

                self.docs_tree.insert("", "end", values=(
                    doc_id,
                    filename,
                    file_type.upper(),
                    page_info,
                    chunk_count,
                    date_str
                ))

            # Reset sort directions and heading arrows
            columns = ("ID", "Filename", "Type", "Pages", "Chunks", "Date Added")
            for col in columns:
                self.sort_directions[col] = False
                self.docs_tree.heading(col, text=f"{col} ↕")

            # Calculate Max ID to show user (explains gaps)
            max_id = max(doc[0] for doc in all_docs) if all_docs else 0

            # Update results label
            if len(all_docs) < total_docs:
                self.results_var.set(f"Showing {len(all_docs)} of {total_docs} docs (Limit: 5000) | Max ID: {max_id}")
            else:
                self.results_var.set(f"Showing all {len(all_docs)} docs | Max ID: {max_id}")

            # Only log if debug_log has been initialized
            if hasattr(self, 'debug_log'):
                msg = f"Refreshed document list: {len(all_docs)} documents loaded (Total Docs: {total_docs}, Total Chunks: {total_chunks})"
                if orphans > 0:
                    msg += f" [⚠️ {orphans} Orphaned Chunks]"
                self.log_debug_message(msg)
        except Exception as e:
            logging.error(f"Error updating docs UI: {e}")

    def filter_documents(self, event=None):
        """Filter documents based on search term"""
        # Check if we're currently showing the placeholder
        if self.is_showing_placeholder:
            search_term = ""
        else:
            search_term = self.search_var.get().strip().lower()

        # Clear current tree
        for item in self.docs_tree.get_children():
            self.docs_tree.delete(item)

        if not search_term:
            # No search term, show all documents
            filtered_docs = self.original_docs
            self.results_var.set(f"Showing all {len(filtered_docs)} documents")
        else:
            # Filter documents based on search term
            filtered_docs = []
            for doc in self.original_docs:
                doc_id, filename, file_type, page_count, chunk_count, created_at = doc
                # Check if search term is in filename, type, or other fields
                if (search_term in filename.lower() or
                        search_term in file_type.lower() or
                        search_term in str(page_count) or
                        search_term in str(chunk_count)):
                    filtered_docs.append(doc)

            self.results_var.set(f"Showing {len(filtered_docs)} of {len(self.original_docs)} documents")

        # Add filtered documents to tree
        for doc in filtered_docs:
            doc_id, filename, file_type, page_count, chunk_count, created_at = doc
            date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            page_info = str(page_count) if page_count else "N/A"

            self.docs_tree.insert("", "end", values=(
                doc_id,
                filename,
                file_type.upper(),
                page_info,
                chunk_count,
                date_str
            ))

        # After filtering, if there was a sort applied, reapply it
        # Find which column is currently sorted (has an arrow other than ↕)
        for col in self.sort_directions.keys():
            current_text = self.docs_tree.heading(col)['text']
            if "↓" in current_text or "↑" in current_text:
                # Reapply the sort to the filtered results
                self.sort_column(col)
                break

        # Log filter action
        if hasattr(self, 'debug_log'):
            if search_term:
                self.log_debug_message(f"Filtered documents with search term: '{search_term}'")
            else:
                self.log_debug_message("Cleared document filter, showing all documents")

    def delete_selected_document(self):
        """Delete selected document"""
        selected = self.docs_tree.selection()
        if not selected:
            messagebox.showwarning("Delete", "Please select a document to delete")
            return

        values = self.docs_tree.item(selected[0], "values")
        doc_id = int(values[0])

        if messagebox.askyesno("Confirm Delete", f"Delete document {values[1]}?"):
            success = self.document_store.delete_document(doc_id)
            if success:
                if hasattr(self, 'debug_log'):
                    self.log_debug_message(f"Deleted document: {values[1]} (ID: {doc_id})")
                self.refresh_documents()  # This will update original_docs
                self.status_var.set("Document deleted")

    def delete_all_documents(self):
        """Delete all documents"""
        doc_count = len(self.docs_tree.get_children())
        if doc_count == 0:
            messagebox.showinfo("Delete All", "No documents to delete")
            return

        if messagebox.askyesno("Confirm Delete All", f"Delete all {doc_count} documents?"):
            # Get all doc IDs first
            docs = self.document_store.list_documents(limit=1000)
            deleted_count = 0
            for doc in docs:
                if self.document_store.delete_document(doc[0]):
                    deleted_count += 1

            if hasattr(self, 'debug_log'):
                self.log_debug_message(f"Deleted all {deleted_count} documents")
            self.refresh_documents()  # This will update original_docs
            self.status_var.set(f"Deleted {deleted_count} documents")

    def rebuild_index(self):
        """Rebuild FAISS index from DB"""
        if not getattr(self, 'document_store', None): return
        
        if messagebox.askyesno("Rebuild Index", "This will wipe the FAISS index and rebuild it from the SQLite database. This may take a while for large datasets. Continue?"):
            def run_rebuild():
                self.root.after(0, lambda: self.status_var.set("Rebuilding index..."))
                try:
                    self.document_store.optimize() # optimize calls _rebuild_index
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Index rebuilt successfully."))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to rebuild index: {e}"))
                finally:
                    self.root.after(0, lambda: self.status_var.set("Ready"))
            
            threading.Thread(target=run_rebuild, daemon=True).start()

    def check_document_integrity(self):
        """Check for broken or incomplete documents"""
        if not hasattr(self.document_store, 'find_broken_documents'):
            messagebox.showinfo("Info", "Integrity check not supported by current document store.")
            return

        # Ensure we have the latest list for comparison
        self.refresh_documents()

        broken_docs = self.document_store.find_broken_documents()
        orphans = self.document_store.get_orphaned_chunk_count()

        # Check for ghost files (files on disk not in DB)
        ghost_files = []
        # Only check the application's specific upload directory
        upload_dir = config.UPLOADED_DOCS_DIR
        if os.path.exists(upload_dir):
            # Get all filenames in DB (case insensitive for Windows safety)
            db_filenames = {doc[1].lower() for doc in self.original_docs}
            
            for f in os.listdir(upload_dir):
                if os.path.isfile(os.path.join(upload_dir, f)):
                    if f.lower() not in db_filenames:
                        ghost_files.append(f)

        if not broken_docs and orphans == 0 and not ghost_files:
            messagebox.showinfo("Integrity Check", "✅ No issues found.\nDatabase and File System are in sync.")
            return

        msg = f"⚠️ Found issues:\n- {len(broken_docs)} broken document(s)\n- {orphans} orphaned chunks\n\n"
        for doc in broken_docs[:10]:
            msg += f"• {doc['filename']} (ID: {doc['id']})\n  Issue: {doc['issue']}\n"

        if len(broken_docs) > 10:
            msg += f"\n...and {len(broken_docs) - 10} more."

        if ghost_files:
            msg += f"\n\n⚠️ Found {len(ghost_files)} files in '{upload_dir}' NOT in Database (Ghost Files):\n"
            for f in ghost_files[:5]:
                msg += f"• {f}\n"
            if len(ghost_files) > 5:
                msg += f"...and {len(ghost_files)-5} more.\n"

        msg += "\nDo you want to clean up these issues?"

        if messagebox.askyesno("Integrity Check", msg):
            deleted_count = 0
            for doc in broken_docs:
                if self.document_store.delete_document(doc['id']):
                    deleted_count += 1
            
            orphans_deleted = 0
            if orphans > 0:
                orphans_deleted = self.document_store.delete_orphaned_chunks()

            ghosts_deleted = 0
            if ghost_files:
                if messagebox.askyesno("Delete Ghost Files?", f"Delete {len(ghost_files)} files from disk that are not in the database?"):
                    for f in ghost_files:
                        try:
                            os.remove(os.path.join(upload_dir, f))
                            ghosts_deleted += 1
                        except Exception as e:
                            logging.error(f"Error deleting ghost file {f}: {e}")

            self.refresh_documents()
            messagebox.showinfo("Cleanup", f"🗑️ Cleanup Report:\n- {deleted_count} broken documents removed from DB\n- {orphans_deleted} orphaned chunks removed from DB\n- {ghosts_deleted} ghost files deleted from disk")

    def generate_document_report(self):
        """Generate a detailed report of database vs file system state"""
        try:
            upload_dir = config.UPLOADED_DOCS_DIR
            report_lines = []
            report_lines.append(f"Document System Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("=" * 60)
            
            # 1. Database State
            # Ensure we have fresh data
            total_docs_count = self.document_store.get_total_documents()
            # Fetch all documents (add buffer to limit just in case)
            all_docs = self.document_store.list_documents(limit=total_docs_count + 500) 
            
            db_filenames = {doc[1]: doc[0] for doc in all_docs} # filename -> id
            max_id = max([d[0] for d in all_docs]) if all_docs else 0
            
            report_lines.append(f"\n[DATABASE]")
            report_lines.append(f"Total Documents in DB: {len(all_docs)}")
            report_lines.append(f"Max ID: {max_id}")
            
            # 2. File System State
            disk_files = []
            if os.path.exists(upload_dir):
                disk_files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
            
            report_lines.append(f"\n[FILE SYSTEM] ({upload_dir})")
            report_lines.append(f"Total Files on Disk: {len(disk_files)}")
            
            # 3. Comparison
            db_set = set(db_filenames.keys())
            disk_set = set(disk_files)
            
            # Case insensitive comparison maps
            db_map_lower = {f.lower(): f for f in db_set}
            disk_map_lower = {f.lower(): f for f in disk_set}
            
            # In DB but not on Disk (Missing Files)
            missing_on_disk = []
            for f in db_set:
                if f.lower() not in disk_map_lower:
                    missing_on_disk.append(f)
            
            # On Disk but not in DB (Ghost Files)
            ghost_files = []
            for f in disk_set:
                if f.lower() not in db_map_lower:
                    ghost_files.append(f)
            
            report_lines.append(f"\n[DISCREPANCIES]")
            if missing_on_disk:
                report_lines.append(f"⚠️ In DB but MISSING on Disk ({len(missing_on_disk)}):")
                for f in missing_on_disk:
                    report_lines.append(f"  - ID {db_filenames[f]}: {f}")
            else:
                report_lines.append("✅ No missing files (All DB entries exist on disk).")
                
            if ghost_files:
                report_lines.append(f"⚠️ On Disk but NOT in DB ({len(ghost_files)}):")
                for f in ghost_files:
                    report_lines.append(f"  - {f}")
            else:
                report_lines.append("✅ No ghost files (All disk files are indexed).")

            # 4. Full List
            report_lines.append(f"\n[FULL DATABASE LIST]")
            report_lines.append(f"{'ID':<6} | {'Chunks':<6} | {'Filename'}")
            report_lines.append("-" * 60)
            
            # Sort by ID
            sorted_docs = sorted(all_docs, key=lambda x: x[0])
            for doc in sorted_docs:
                # doc: (id, filename, file_type, page_count, chunk_count, created_at)
                report_lines.append(f"{doc[0]:<6} | {doc[4]:<6} | {doc[1]}")

            # Write to file
            report_path = os.path.abspath("document_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            
            # Open file
            if os.name == 'nt':
                os.startfile(report_path)
            else:
                messagebox.showinfo("Report Generated", f"Report saved to:\n{report_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
            logging.error(f"Report generation error: {e}")
