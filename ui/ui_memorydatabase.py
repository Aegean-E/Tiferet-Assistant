import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import os
from datetime import datetime

class MemoryDatabaseUI:
    """Mixin for Memory Database UI tab"""

    def setup_database_tab(self):
        """Setup database viewer interface"""
        # Database Notebook (Memories vs Meta-Memories)
        db_notebook = ttk.Notebook(self.database_frame)
        db_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Refresh button (Placed at top right, inline with tabs)
        refresh_button = ttk.Button(self.database_frame, text="🔄 Refresh", command=self.refresh_database_view,
                                    bootstyle="secondary")
        refresh_button.place(relx=1.0, x=-5, y=2, anchor="ne")

        # Compress Summaries button (Placed to the left of Refresh)
        self.compress_button = ttk.Button(self.database_frame, text="🗜️ Compress", command=self.compress_summaries,
                                   bootstyle="info-outline")
        self.compress_button.place(relx=1.0, x=-95, y=2, anchor="ne")

        # Export Summaries button (Placed to the left of Refresh)
        export_button = ttk.Button(self.database_frame, text="💾 Export Summaries", command=self.export_summaries,
                                   bootstyle="info")
        export_button.place(relx=1.0, x=-185, y=2, anchor="ne")

        # Verify All button (Placed to the left of Export)
        verify_all_button = ttk.Button(self.database_frame, text="🧹 Verify All", command=self.verify_all_memory_sources,
                                       bootstyle="warning")
        verify_all_button.place(relx=1.0, x=-325, y=2, anchor="ne")

        # Verify Batch button (Placed to the left of Verify All)
        verify_button = ttk.Button(self.database_frame, text="🧹 Verify Sources", command=self.verify_memory_sources,
                                   bootstyle="warning")
        verify_button.place(relx=1.0, x=-425, y=2, anchor="ne")

        # Stop Verification button (Placed to the left of Verify Sources)
        stop_verify_button = ttk.Button(self.database_frame, text="🛑 Stop", command=self.stop_processing,
                                        bootstyle="danger")
        stop_verify_button.place(relx=1.0, x=-535, y=2, anchor="ne")

        # Sync Journal button (Placed to the left of Stop)
        sync_journal_button = ttk.Button(self.database_frame, text="📓 Sync Journal", command=self.sync_journal,
                                        bootstyle="success-outline")
        sync_journal_button.place(relx=1.0, x=-615, y=2, anchor="ne")

        # Stats Label (Verified / Total)
        # Moved to notebook header (tab bar) via place() and event binding
        self.memory_stats_var = tk.StringVar(value="Verified: 0 / 0")
        self.stats_label = ttk.Label(self.notebook, textvariable=self.memory_stats_var, bootstyle="info")
        
        # Bind tab change to show/hide stats
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        # Trigger once to set initial state
        self.root.after(100, lambda: self._on_tab_changed(None))

        # Summaries Tab
        summaries_frame = ttk.Frame(db_notebook)
        db_notebook.add(summaries_frame, text="Summaries")
        self.summaries_scrollable_frame = self._setup_scrollable_frame(summaries_frame)

        # Chat Memories Tab
        chat_memories_frame = ttk.Frame(db_notebook)
        db_notebook.add(chat_memories_frame, text="Chat Memories")
        self.chat_memories_scrollable_frame = self._setup_scrollable_frame(chat_memories_frame)

        # Daydream Memories Tab
        daydream_memories_frame = ttk.Frame(db_notebook)
        db_notebook.add(daydream_memories_frame, text="Daydream Memories")
        self.daydream_memories_scrollable_frame = self._setup_scrollable_frame(daydream_memories_frame)

        # Journal Tab (formerly Assistant Notes)
        notes_frame = ttk.Frame(db_notebook)
        db_notebook.add(notes_frame, text="Journal")
        self.notes_scrollable_frame = self._setup_scrollable_frame(notes_frame)

        # Meta-Memories Tab
        meta_frame = ttk.Frame(db_notebook)
        db_notebook.add(meta_frame, text="Meta-Memories")

        self.meta_memories_text = scrolledtext.ScrolledText(
            meta_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 10),
            bg="#1e1e1e",
            fg="white"
        )
        self.meta_memories_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _on_tab_changed(self, event):
        """Handle tab change to show/hide stats label in the header"""
        try:
            if not hasattr(self, 'stats_label'): return
            
            # Get index of the currently selected tab
            current_idx = self.notebook.index("current")
            db_idx = self.notebook.index(self.database_frame)
            
            if current_idx == db_idx:
                self.stats_label.place(relx=1.0, x=-5, y=2, anchor="ne")
            else:
                self.stats_label.place_forget()
        except Exception:
            pass

    def _setup_scrollable_frame(self, parent_frame):
        """Helper to setup a scrollable frame structure"""
        canvas = tk.Canvas(parent_frame, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel support
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Attach handler to frame for child widgets to use
        scrollable_frame.on_mousewheel = _on_mousewheel

        return scrollable_frame

    def refresh_database_view(self):
        """Refresh the database views"""
        if not hasattr(self, 'memory_store') or not hasattr(self, 'meta_memory_store'):
            return

        # Update status immediately
        if hasattr(self, 'status_var'):
            self.status_var.set("Refreshing database...")

        def fetch_data():
            try:
                # Fetch data in background thread
                mem_items = self.memory_store.list_recent(limit=None)
                
                # Calculate stats
                total_count = len(mem_items)
                verified_count = sum(1 for item in mem_items if len(item) > 5 and item[5] == 1)
                unverified_beliefs = sum(1 for item in mem_items if item[1] == 'BELIEF' and (len(item) <= 5 or item[5] == 0) and len(item) > 4 and item[4] == 'daydream')
                unverified_facts = sum(1 for item in mem_items if item[1] == 'FACT' and (len(item) <= 5 or item[5] == 0) and len(item) > 4 and item[4] == 'daydream')
                stats_text = f"Total: {total_count} | Verified: {verified_count} | Unverified Beliefs: {unverified_beliefs} | Unverified Facts: {unverified_facts}"

                # Fetch Summaries
                summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=30)
                analyses = self.meta_memory_store.get_by_event_type("HOD_ANALYSIS", limit=30)
                summary_items = summaries + analyses
                summary_items.sort(key=lambda x: x['created_at'], reverse=True)

                # Fetch Meta-Memories
                meta_items = self.meta_memory_store.list_recent(limit=75)

                # Schedule UI update on main thread
                self.root.after(0, lambda: self._update_database_ui(mem_items, stats_text, summary_items, meta_items))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Refresh Error", f"Failed to refresh data: {e}"))
                self.root.after(0, lambda: self.status_var.set("Refresh failed"))

        # Start background thread
        threading.Thread(target=fetch_data, daemon=True).start()

    def _update_database_ui(self, mem_items, stats_text, summary_items, meta_items):
        """Update UI elements with fetched data (Main Thread)"""
        try:
            # 1. Clear existing widgets
            for widget in self.summaries_scrollable_frame.winfo_children(): widget.destroy()
            for widget in self.chat_memories_scrollable_frame.winfo_children(): widget.destroy()
            for widget in self.daydream_memories_scrollable_frame.winfo_children(): widget.destroy()
            for widget in self.notes_scrollable_frame.winfo_children(): widget.destroy()

            # 2. Update Stats
            self.memory_stats_var.set(stats_text)

            # 3. Populate Memories
            chat_items = []
            daydream_items = []
            note_items = []

            # Internal sources that should go to Daydream/System tab
            system_sources = {
                'daydream', 'chokmah_gap_investigation', 'autonomous_curiosity', 
                'daat_lattice', 'daat_cluster', 'daat_synthesis', 
                'architect_decomposition', 'decider_autonomous', 'hod_verification',
                'model_stress', 'model_revision', 'agency_loop', 'meta_learner'
            }

            for item in mem_items:
                source = item[4] if len(item) > 4 else ''
                if item[1] == "NOTE":
                    note_items.append(item)
                elif source in system_sources or source.startswith('autonomous_reading') or source.startswith('synthesis') or source.startswith('daat'):
                    daydream_items.append(item)
                else:
                    chat_items.append(item)

            self._populate_memory_tab(self.chat_memories_scrollable_frame, chat_items)
            self._populate_memory_tab(self.daydream_memories_scrollable_frame, daydream_items)
            self._populate_memory_tab(self.notes_scrollable_frame, note_items)

            # 4. Populate Summaries
            if not summary_items:
                lbl = ttk.Label(self.summaries_scrollable_frame, text="📜 No session summaries found.", padding=20)
                lbl.pack(anchor="center")
            else:
                for item in summary_items:
                    self._create_summary_card(self.summaries_scrollable_frame, item)

            # 5. Populate Meta-Memories
            self.meta_memories_text.config(state=tk.NORMAL)
            self.meta_memories_text.delete(1.0, tk.END)
            
            if not meta_items:
                self.meta_memories_text.insert(tk.END, "🧠 No meta-memories.")
            else:
                for (_id, event_type, subject, text, created_at) in meta_items:
                    event_emoji = {
                        "MEMORY_CREATED": "✨", "VERSION_UPDATE": "🔄", "DECIDER_CHAT": "💬",
                        "CONFLICT_DETECTED": "⚠️", "CONSOLIDATION": "🔗", "HOD_ANALYSIS": "🔮",
                        "SESSION_SUMMARY": "📅"
                    }.get(event_type, "🧠")

                    try:
                        date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = str(created_at)

                    self.meta_memories_text.insert(tk.END, f"[{date_str}] {event_emoji} [{subject}] {text}\n")
            
            self.meta_memories_text.config(state=tk.DISABLED)
            self.status_var.set("Ready")
        except Exception as e:
            print(f"UI Update Error: {e}")

    def export_summaries(self):
        """Export session summaries to a text file"""
        if not hasattr(self, 'meta_memory_store'):
            messagebox.showerror("Error", "Meta-memory store not initialized.")
            return

        try:
            # Fetch all summaries
            summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1000)
            analyses = self.meta_memory_store.get_by_event_type("HOD_ANALYSIS", limit=1000)
            
            all_items = summaries + analyses
            all_items.sort(key=lambda x: x['created_at'], reverse=True)

            if not all_items:
                messagebox.showinfo("Export", "No summaries to export.")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Summaries"
            )

            if not file_path:
                return

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("SESSION SUMMARIES & ANALYSES\n")
                f.write("============================\n\n")
                
                for item in all_items:
                    date_str = datetime.fromtimestamp(item['created_at']).strftime("%Y-%m-%d %H:%M:%S")
                    type_str = "SUMMARY" if item.get('event_type') == "SESSION_SUMMARY" else "ANALYSIS"
                    subject = item.get('subject', 'Unknown')
                    
                    f.write(f"[{date_str}] [{type_str}] [{subject}]\n")
                    f.write(f"{item['text']}\n")
                    f.write("-" * 50 + "\n\n")

            messagebox.showinfo("Export", f"Successfully exported {len(all_items)} items to {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export summaries: {e}")

    def compress_summaries(self):
        """Trigger summary consolidation"""
        if hasattr(self, 'daat') and self.daat:
            # Disable button to prevent multiple clicks
            if hasattr(self, 'compress_button'):
                self.compress_button.config(state=tk.DISABLED)
            
            # Update status
            if hasattr(self, 'status_var'):
                self.status_var.set("Compressing summaries... (This may take a while)")

            # Run in background to avoid freezing UI
            def run_compression():
                try:
                    result = self.daat.consolidate_summaries()
                    self.root.after(0, lambda: messagebox.showinfo("Compression", result))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Compression failed: {e}"))
                finally:
                    self.root.after(0, self.refresh_database_view)
                    if hasattr(self, 'compress_button'):
                        self.root.after(0, lambda: self.compress_button.config(state=tk.NORMAL))
                    if hasattr(self, 'status_var'):
                        self.root.after(0, lambda: self.status_var.set("Ready"))

            threading.Thread(target=run_compression, daemon=True).start()
        else:
            messagebox.showerror("Error", "Da'at not initialized.")

    def _create_summary_card(self, parent, item):
        """Create a card for a session summary"""
        date_str = datetime.fromtimestamp(item['created_at']).strftime("%Y-%m-%d %H:%M")
        
        # Different style for Hod Analysis
        style_color = "info" if item.get('event_type') == "SESSION_SUMMARY" else "secondary"
        prefix = "📅" if item.get('event_type') == "SESSION_SUMMARY" else "🔮"
        
        text = f"{prefix} {date_str} - {item['subject']}"
        
        # Robust LabelFrame creation handling different ttk/ttkbootstrap versions
        try:
            # Try bootstyle first (standard for ttkbootstrap)
            card = ttk.LabelFrame(parent, text=text, bootstyle=style_color)
        except Exception:
            try:
                # Try style (standard for ttk)
                style_name = f"{style_color.title()}.TLabelframe"
                card = ttk.LabelFrame(parent, text=text, style=style_name)
            except Exception:
                # Fallback to no style (standard tk or broken ttk)
                card = ttk.LabelFrame(parent, text=text)

        card.pack(fill=tk.X, pady=5, padx=5)
        
        lbl = ttk.Label(card, text=item['text'], wraplength=780, justify=tk.LEFT)
        lbl.pack(fill=tk.X, padx=5, pady=5)

    def _populate_memory_tab(self, parent_frame, items):
        """Populate a memory tab with sections"""
        if not items:
            lbl = ttk.Label(parent_frame, text="🧠 No saved memories.", padding=20)
            lbl.pack(anchor="center")
            return

        type_emoji = {
            "IDENTITY": "👤", "FACT": "📌", "PREFERENCE": "❤️",
            "GOAL": "🎯", "RULE": "⚖️", "PERMISSION": "✅", "BELIEF": "💭",
            "NOTE": "📓",
            "REFUTED_BELIEF": "🛡️",
            "COMPLETED_GOAL": "🏁"
        }

        grouped = {}
        for item in items:
            # item: (id, type, subject, text, source)
            _id, mem_type, subject, text = item[:4]
            grouped.setdefault(mem_type, []).append((_id, subject, text))

        hierarchy = ["NOTE", "PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "COMPLETED_GOAL", "FACT", "BELIEF", "REFUTED_BELIEF"]

        for mem_type in hierarchy:
            if mem_type in grouped:
                all_items = grouped[mem_type]
                total_count = len(all_items)
                # Limit to latest 50 items per type to prevent lag
                self._create_memory_section(parent_frame, mem_type, all_items[:50], type_emoji, total_count)
                del grouped[mem_type]

        for mem_type, remaining in grouped.items():
            total_count = len(remaining)
            # Limit to latest 50 items per type
            self._create_memory_section(parent_frame, mem_type, remaining[:50], type_emoji, total_count)

    def on_memory_right_click(self, event, widget):
        """Handle right-click on memory text widget"""
        try:
            index = widget.index(f"@{event.x},{event.y}")
            tags = widget.tag_names(index)
            
            mem_id = None
            for tag in tags:
                if tag.startswith("mem_"):
                    try:
                        mem_id = int(tag.split("_")[1])
                        break
                    except:
                        pass
            
            if mem_id:
                menu = tk.Menu(self.root, tearoff=0)
                menu.add_command(label=f"❌ Delete Memory ID: {mem_id}", command=lambda: self.delete_memory_action(mem_id))
                menu.tk_popup(event.x_root, event.y_root)
        except Exception as e:
            print(f"Right-click error: {e}")

    def delete_memory_action(self, mem_id):
        """Delete a memory by ID and refresh view"""
        if not hasattr(self, 'memory_store'):
            return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete memory ID {mem_id}?"):
            try:
                if self.memory_store.delete_entry(mem_id):
                    self.status_var.set(f"Deleted memory {mem_id}")
                    self.refresh_database_view()
                else:
                    messagebox.showerror("Error", f"Failed to delete memory ID {mem_id}")
            except Exception as e:
                messagebox.showerror("Error", f"Error deleting memory: {e}")

    def _create_memory_section(self, parent, mem_type, items, type_emoji, total_count=None):
        """Helper to create a collapsible section for a memory type"""
        emoji = type_emoji.get(mem_type, "💡")
        count = len(items)
        if total_count is None:
            total_count = count

        if count < total_count:
            title = f"{emoji} {mem_type} ({count}/{total_count})"
        else:
            title = f"{emoji} {mem_type} ({total_count})"

        # Container for the whole section
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, expand=False, padx=5, pady=2)

        # Content container (holds canvas + scrollbar)
        content_container = ttk.Frame(container)

        # Toggle state
        is_open = tk.BooleanVar(value=True)

        def toggle():
            if is_open.get():
                content_container.pack_forget()
                is_open.set(False)
                toggle_btn.configure(text=f"▶ {title}")
            else:
                content_container.pack(fill=tk.X, expand=False, padx=10, pady=5)
                is_open.set(True)
                toggle_btn.configure(text=f"▼ {title}")

        # Header Button
        toggle_btn = ttk.Button(
            container,
            text=f"▼ {title}",
            command=toggle,
            bootstyle="secondary-outline",
            cursor="hand2"
        )
        toggle_btn.pack(fill=tk.X)

        # Pack content initially
        content_container.pack(fill=tk.X, expand=True, padx=10, pady=5)

        # Use a Text widget for performance (instead of hundreds of Frames)
        # Calculate height: approx 3 lines per item, max 20 lines
        widget_height = min(len(items) * 3, 20)
        if widget_height < 3: widget_height = 3

        text_widget = scrolledtext.ScrolledText(
            content_container,
            wrap=tk.WORD,
            height=widget_height,
            font=("Segoe UI", 9) if os.name == 'nt' else ("Arial", 9),
            bg="#2b2b2b",
            fg="white",
            borderwidth=0,
            highlightthickness=0
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags
        text_widget.tag_config("header", foreground="#61afef", font=("Segoe UI", 9, "bold"))
        text_widget.tag_config("content", foreground="#dcdcdc")
        text_widget.tag_config("separator", foreground="#4e4e4e")

        for _id, subject, text in items:
            header = f"[ID:{_id}] [{subject}]\n"
            content = f"{text}\n"
            sep = "-" * 80 + "\n"
            
            # Add unique tag for ID
            mem_tag = f"mem_{_id}"
            
            text_widget.insert(tk.END, header, ("header", mem_tag))
            text_widget.insert(tk.END, content, ("content", mem_tag))
            text_widget.insert(tk.END, sep, ("separator", mem_tag))

        text_widget.config(state=tk.DISABLED)
        
        # Bind right-click
        text_widget.bind("<Button-3>", lambda e: self.on_memory_right_click(e, text_widget))