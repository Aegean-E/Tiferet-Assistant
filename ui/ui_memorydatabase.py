import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import os
from datetime import datetime
import logging

class MemoryDatabaseUI:
    """Mixin for Memory Database UI tab"""

    def setup_database_tab(self):
        """Setup database viewer interface"""
        # Pagination State
        self.db_page = 0
        self.db_page_size = 50

        # Database Notebook (Memories vs Meta-Memories)
        db_notebook = ttk.Notebook(self.database_frame)
        db_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Refresh button (Placed at top right, inline with tabs)
        refresh_button = ttk.Button(self.database_frame, text="🔄 Refresh", command=self.refresh_database_view,
                                    bootstyle="secondary")
        refresh_button.place(relx=1.0, x=-5, y=2, anchor="ne")
        
        # Export Memories button (Placed to the left of Refresh)
        export_mem_button = ttk.Button(self.database_frame, text="💾 Export Memories", command=self.export_memories,
                                   bootstyle="info")
        export_mem_button.place(relx=1.0, x=-105, y=2, anchor="ne")

        # Export Summaries button (Placed to the left of Export Memories)
        export_button = ttk.Button(self.database_frame, text="💾 Export Summaries", command=self.export_summaries,
                                   bootstyle="info")
        export_button.place(relx=1.0, x=-245, y=2, anchor="ne")

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

        # Chronicles Tab (formerly Journal/Assistant Notes)
        notes_frame = ttk.Frame(db_notebook)
        db_notebook.add(notes_frame, text="Chronicles")
        self.notes_scrollable_frame = self._setup_scrollable_frame(notes_frame)

        # Meta-Memories Tab
        meta_frame = ttk.Frame(db_notebook)
        db_notebook.add(meta_frame, text="Meta-Memories")

        self.meta_memories_text = scrolledtext.ScrolledText(
            meta_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 10)
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

    def next_db_page(self):
        self.db_page += 1
        self.page_label.config(text=f"Page {self.db_page + 1}")
        self.refresh_database_view()

    def prev_db_page(self):
        if self.db_page > 0:
            self.db_page -= 1
            self.page_label.config(text=f"Page {self.db_page + 1}")
            self.refresh_database_view()

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
        if not getattr(self, 'memory_store', None) or not getattr(self, 'meta_memory_store', None):
            return

        # Update status immediately
        if hasattr(self, 'status_var'):
            self.status_var.set("Refreshing database...")

        def fetch_data():
            try:
                # Fetch data in background thread
                mem_items = self.memory_store.list_recent(limit=self.db_page_size, offset=self.db_page * self.db_page_size)
                stats = self.memory_store.get_memory_stats()
                
                # Calculate stats
                stats_text = f"Total: {stats.get('total_memories', 0)} | Unverified Beliefs: {stats.get('unverified_beliefs', 0)} | Unverified Facts: {stats.get('unverified_facts', 0)} | Goals: {stats.get('active_goals', 0)}"

                # Fetch Summaries
                summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=30)
                analyses = self.meta_memory_store.get_by_event_type("HOD_ANALYSIS", limit=30)
                narratives = self.meta_memory_store.get_by_event_type("SELF_NARRATIVE", limit=30)
                
                summary_items = summaries + analyses
                summary_items.sort(key=lambda x: x['created_at'], reverse=True)

                # Fetch Meta-Memories
                meta_items = self.meta_memory_store.list_recent(limit=75)

                # Schedule UI update on main thread
                self.root.after(0, lambda: self._update_database_ui(mem_items, stats_text, summary_items, meta_items, narratives))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Refresh Error", f"Failed to refresh data: {e}"))
                self.root.after(0, lambda: self.status_var.set("Refresh failed"))

        # Start background thread
        threading.Thread(target=fetch_data, daemon=True).start()

    def _update_database_ui(self, mem_items, stats_text, summary_items, meta_items, narratives=None):
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
                'architect_decomposition', 'decider_autonomous', 'hod_verification', 'hod',
                'model_stress', 'model_revision', 'agency_loop', 'meta_learner', 
                'malkuth_tool', 'malkuth_edit', 'daat_growth_arc', 'daat_growth_restore',
                'meta_learner_self_model', 'meta_learner_rl'
            }

            for item in mem_items:
                source = item[4] if len(item) > 4 else ''
                text = item[3] if len(item) > 3 else ''
                
                # Skip empty memories
                if not text or not text.strip():
                    continue

                if item[1] == "NOTE":
                    note_items.append(item)
                elif source == "meta_learner_self_model":
                    # Override type for display
                    new_item = list(item)
                    new_item[1] = "SELF-KNOWLEDGE"
                    note_items.append(tuple(new_item))
                elif source in ["daat_life_story_update", "daat_growth_arc", "daat_growth_restore"]:
                    new_item = list(item)
                    new_item[1] = "NARRATIVE"
                    note_items.append(tuple(new_item))
                elif source in system_sources or source.startswith('autonomous_reading') or source.startswith('synthesis') or source.startswith('daat'):
                    daydream_items.append(item)
                else:
                    chat_items.append(item)

            # Add narratives to note_items (Chronicles)
            if narratives:
                for n in narratives:
                    # Convert dict to tuple: (id, type, subject, text, source)
                    note_items.append((n['id'], "NARRATIVE", n['subject'], n['text'], "meta_memory"))

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
                for item in meta_items:
                    # Handle variable length tuples (legacy vs new)
                    _id = item[0]
                    event_type = item[1]
                    subject = item[2]
                    text = item[3]
                    created_at = item[4]
                    affect = item[5] if len(item) > 5 else None
                    
                    event_emoji = {
                        "MEMORY_CREATED": "✨", "VERSION_UPDATE": "🔄", "DECIDER_CHAT": "💬",
                        "CONFLICT_DETECTED": "⚠️", "CONSOLIDATION": "🔗", "HOD_ANALYSIS": "🔮", 
                        "SESSION_SUMMARY": "📅", "SELF_NARRATIVE": "📖"
                    }.get(event_type, "🧠")

                    try:
                        date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = str(created_at)

                    affect_str = ""
                    if affect is not None:
                        affect_str = f" (Mood: {affect:.2f})"

                    self.meta_memories_text.insert(tk.END, f"[{date_str}] {event_emoji} [{subject}] {text}{affect_str}\n")
            
            self.meta_memories_text.config(state=tk.DISABLED)
            self.status_var.set("Ready")
        except Exception as e:
            logging.error(f"UI Update Error: {e}")

    def export_summaries(self):
        """Export session summaries to a text file"""
        if not getattr(self, 'meta_memory_store', None):
            messagebox.showerror("Error", "Meta-memory store not initialized.")
            return

        try:
            # Fetch all summaries
            summaries = self.meta_memory_store.get_by_event_type("SESSION_SUMMARY", limit=1000)
            analyses = self.meta_memory_store.get_by_event_type("HOD_ANALYSIS", limit=1000)
            narratives = self.meta_memory_store.get_by_event_type("SELF_NARRATIVE", limit=1000)
            
            all_items = summaries + analyses + narratives
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
                    if item.get('event_type') == "SESSION_SUMMARY":
                        type_str = "SUMMARY"
                    elif item.get('event_type') == "HOD_ANALYSIS":
                        type_str = "ANALYSIS"
                    else:
                        type_str = "NARRATIVE"
                    subject = item.get('subject', 'Unknown')
                    
                    f.write(f"[{date_str}] [{type_str}] [{subject}]\n")
                    f.write(f"{item['text']}\n")
                    f.write("-" * 50 + "\n\n")

            messagebox.showinfo("Export", f"Successfully exported {len(all_items)} items to {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export summaries: {e}")

    def export_memories(self):
        """Export all memories to a text file"""
        if not getattr(self, 'memory_store', None):
            messagebox.showerror("Error", "Memory store not initialized.")
            return

        try:
            # Fetch all memories
            # list_recent returns (id, type, subject, text, source, verified, flags, confidence)
            all_items = self.memory_store.list_recent(limit=None)
            
            if not all_items:
                messagebox.showinfo("Export", "No memories to export.")
                return

            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Memories"
            )

            if not file_path:
                return

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("ACTIVE MEMORIES EXPORT\n")
                f.write("======================\n\n")
                
                for item in all_items:
                    # item: (id, type, subject, text, source, verified, flags, confidence)
                    _id, mtype, subject, text = item[:4]
                    source = item[4] if len(item) > 4 else "unknown"
                    conf = item[7] if len(item) > 7 else 0.0
                    
                    f.write(f"[{mtype}] [{subject}] (ID:{_id}, Conf:{conf:.2f})\n")
                    f.write(f"{text}\n")
                    f.write(f"Source: {source}\n")
                    f.write("-" * 50 + "\n\n")

            messagebox.showinfo("Export", f"Successfully exported {len(all_items)} memories to {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export memories: {e}")

    def _create_summary_card(self, parent, item):
        """Create a card for a session summary"""
        date_str = datetime.fromtimestamp(item['created_at']).strftime("%Y-%m-%d %H:%M")
        
        # Different style for Hod Analysis
        event_type = item.get('event_type')
        
        if event_type == "SESSION_SUMMARY":
            style_color = "info"
            prefix = "📅"
        elif event_type == "HOD_ANALYSIS":
            style_color = "secondary"
            prefix = "🔮"
        elif event_type == "SELF_NARRATIVE":
            style_color = "success"
            prefix = "📖"
        else:
            style_color = "secondary"
            prefix = "📝"
        
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
            "NARRATIVE": "📖",
            "REFUTED_BELIEF": "🛡️",
            "COMPLETED_GOAL": "🏁",
            "SELF-KNOWLEDGE": "🧠"
        }

        grouped = {}
        for item in items:
            # item: (id, type, subject, text, source)
            _id, mem_type, subject, text = item[:4]
            grouped.setdefault(mem_type, []).append((_id, subject, text))

        hierarchy = ["NARRATIVE", "NOTE", "SELF-KNOWLEDGE", "PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "COMPLETED_GOAL", "FACT", "BELIEF", "REFUTED_BELIEF"]

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
            logging.error(f"Right-click error: {e}")

    def delete_memory_action(self, mem_id):
        """Delete a memory by ID and refresh view"""
        if not getattr(self, 'memory_store', None):
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
        
        # Store items for lazy loading
        container.items = items

        # Content container (holds canvas + scrollbar)
        content_container = ttk.Frame(container)

        # Toggle state
        is_open = tk.BooleanVar(value=True)

        # Lazy load flag
        is_loaded = [False] # Mutable closure hack

        def toggle():
            if is_open.get():
                content_container.pack_forget()
                is_open.set(False)
                toggle_btn.configure(text=f"▶ {title}")
            else:
                content_container.pack(fill=tk.X, expand=False, padx=10, pady=5)
                is_open.set(True)
                toggle_btn.configure(text=f"▼ {title}")
                if not is_loaded[0]:
                    populate_text()

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
        # Calculate height based on type
        if mem_type in ["NARRATIVE", "NOTE", "IDENTITY", "SELF-KNOWLEDGE"]:
            # These tend to be longer
            lines_per_item = 15
            max_lines = 200 # Increased for longer narratives
        else:
            lines_per_item = 3
            max_lines = 25

        widget_height = min(len(items) * lines_per_item, max_lines)
        if widget_height < 5: widget_height = 5

        def populate_text():
            text_widget = scrolledtext.ScrolledText(
                content_container,
                wrap=tk.WORD,
                height=widget_height,
                font=("Segoe UI", 10) if os.name == 'nt' else ("Arial", 10),
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

            for _id, subject, text in container.items:
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
            is_loaded[0] = True

        # Initial population
        populate_text()