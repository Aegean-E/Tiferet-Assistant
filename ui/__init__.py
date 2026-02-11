"""
UI Module for AI Telegram Desktop Assistant
Contains the DesktopAssistantUI mixin and UI-related helpers.
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import sys
import os
import json
import re
import threading
from datetime import datetime

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow not installed. Images will not be displayed. Run: pip install Pillow")
    Image = None

# Import prompts
from lm import DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT
from treeoflife.chokmah import DAYDREAM_EXTRACTOR_PROMPT as DEFAULT_DAYDREAM_EXTRACTOR_PROMPT

# Import UI Mixins
from .ui_documents import DocumentsUI
from .ui_settings import SettingsUI
from .ui_memorydatabase import MemoryDatabaseUI


class StdoutRedirector:
    def __init__(self, app, original_stream):
        self.app = app
        self.original_stream = original_stream

    def write(self, string):
        if self.original_stream:
            try:
                self.original_stream.write(string)
            except:
                pass
        self.app.log_to_main(string)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except:
                pass


class DesktopAssistantUI(DocumentsUI, SettingsUI, MemoryDatabaseUI):
    """UI Mixin for DesktopAssistantApp"""

    def setup_ui(self):
        """Setup the main UI"""
        # Initialize buffers for thread-safe logging
        self.log_buffer = []
        self.debug_log_buffer = []

        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Chat tab
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="💬 Chat")

        # Logs tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📝 Logs")

        # Events tab
        self.events_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.events_frame, text="⚡ Events")

        # Database tab
        self.database_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.database_frame, text="🗄️ Memory Database")

        # Documents tab
        self.docs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.docs_frame, text="📚 Documents")

        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="⚙️ Settings")

        # Help tab
        self.help_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.help_frame, text="❓ Help")

        # About tab
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="ℹ️ About")

        self.setup_chat_tab()
        self.setup_logs_tab()
        self.setup_events_tab()
        self.setup_database_tab()
        self.setup_documents_tab()
        self.setup_settings_tab()
        self.setup_help_tab()
        self.setup_about_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, bootstyle="secondary")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_dark_theme(self):
        """Setup dark theme"""
        self.style = ttk.Style()
        if self.settings.get("theme", "dark") == "dark":
            # Configure dark theme
            self.style.theme_use("clam")
            self.style.configure(".", background="#2b2b2b", foreground="white")
            self.style.configure("TFrame", background="#2b2b2b")
            self.style.configure("TLabel", background="#2b2b2b", foreground="white")
            self.style.configure("TButton", background="#3a3a3a", foreground="white")
            self.style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b")
            self.style.map("TButton", background=[("active", "#4a4a4a")])

    def setup_chat_tab(self):
        """Setup chat interface"""
        # Determine font for emoji support
        chat_font = ("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 10)
        entry_font = ("Segoe UI Emoji", 11) if os.name == 'nt' else ("Arial", 11)

        # Create PanedWindow for split view
        self.chat_paned = ttk.Panedwindow(self.chat_frame, orient=tk.VERTICAL)
        self.chat_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Top: Assistant Chat ---
        assistant_frame = ttk.Frame(self.chat_paned)
        self.chat_paned.add(assistant_frame, weight=3)

        self.chat_history = scrolledtext.ScrolledText(
            assistant_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="white",
            font=chat_font
        )
        self.chat_history.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Bottom: Netzach Observations ---
        self.netzach_container = ttk.Frame(self.chat_paned)
        # Note: We do NOT add it here immediately; toggle_thoughts_view handles it.

        ttk.Label(self.netzach_container, text="👁️ Internal Thought Process", font=("Arial", 9, "bold"),
                  bootstyle="info").pack(anchor=tk.W)

        self.netzach_history = scrolledtext.ScrolledText(
            self.netzach_container,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#1a1a2e",  # Slightly bluer/darker for distinction
            fg="#a0a0ff",  # Soft blue text
            font=("Consolas", 10),
            height=5
        )
        self.netzach_history.pack(fill=tk.BOTH, expand=True)

        # Input frame
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Configure large emoji style
        self.style.configure("Big.Link.TButton", font=("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 15))

        # Image button
        image_btn = ttk.Button(input_frame, text="📷", command=self.send_image, style="Big.Link.TButton")
        image_btn.pack(side=tk.LEFT, padx=(0, 2))

        # Message entry
        self.message_entry = ttk.Entry(input_frame, font=entry_font)
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind("<Return>", self.send_message)

        # Buttons ordered Right-to-Left to appear Left-to-Right:
        # STOP ALL <- STOP DAYDREAM <- DAYDREAM <- CHAT MODE <- CONNECT <- SEND

        # Stop button (STOP ALL)
        stop_button = ttk.Button(input_frame, text="Stop", command=self.stop_processing, bootstyle="danger")
        stop_button.pack(side=tk.RIGHT)

        # Daydream button (DAYDREAM)
        daydream_button = ttk.Button(input_frame, text="Daydream", command=self.start_daydream, bootstyle="info")
        daydream_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Show Thoughts Toggle
        self.show_thoughts_var = tk.BooleanVar(value=True)
        self.thoughts_btn = ttk.Checkbutton(
            input_frame,
            text="Show Thoughts",
            variable=self.show_thoughts_var,
            bootstyle="info-toolbutton",
            command=self.toggle_thoughts_view
        )
        self.thoughts_btn.pack(side=tk.RIGHT, padx=(0, 5))

        # Chat Mode button (CHAT MODE)
        self.chat_mode_btn = ttk.Checkbutton(
            input_frame,
            text="Chat Mode",
            variable=self.chat_mode_var,
            bootstyle="warning-toolbutton",
            command=self.on_chat_mode_toggle
        )
        self.chat_mode_btn.pack(side=tk.RIGHT, padx=(0, 5))

        # Connect/Disconnect button (CONNECT)
        self.connect_button = ttk.Button(input_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Send button (SEND)
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Initialize view state
        self.toggle_thoughts_view()

    def toggle_thoughts_view(self):
        """Toggle the visibility of the internal thought process pane"""
        if self.show_thoughts_var.get():
            self.chat_paned.add(self.netzach_container, weight=1)
        else:
            try:
                self.chat_paned.forget(self.netzach_container)
            except tk.TclError:
                pass

    def add_netzach_message(self, message):
        """Add message to Netzach's observation window"""
        self.netzach_history.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] ✨ {message}\n\n"

        self.netzach_history.insert(tk.END, formatted_msg)
        # Limit size
        if int(self.netzach_history.index('end-1c').split('.')[0]) > 100:
            self.netzach_history.delete("1.0", "2.0")

        self.netzach_history.see(tk.END)
        self.netzach_history.config(state=tk.DISABLED)

    def toggle_chat_input(self, enabled: bool):
        """Enable or disable chat input widgets"""
        state = tk.NORMAL if enabled else tk.DISABLED
        if hasattr(self, 'message_entry'):
            self.message_entry.config(state=state)
        if hasattr(self, 'send_button'):
            self.send_button.config(state=state)

    def setup_logs_tab(self):
        """Setup logs interface"""
        # Controls frame
        controls_frame = ttk.Frame(self.logs_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Clear button
        clear_button = ttk.Button(controls_frame, text="Clear Logs", command=self.clear_main_log, bootstyle="secondary")
        clear_button.pack(side=tk.RIGHT)

        # Log text area
        self.main_log_text = scrolledtext.ScrolledText(
            self.logs_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4"
        )
        self.main_log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def redirect_logging(self):
        """Redirect stdout and stderr to the logs tab"""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        sys.stdout = StdoutRedirector(self, self.original_stdout)
        sys.stderr = StdoutRedirector(self, self.original_stderr)

    def log_to_main(self, message):
        """Thread-safe logging to main log widget"""
        # Buffer for thread-safe access
        if not hasattr(self, 'log_buffer'): self.log_buffer = []
        
        if message is None:
            return
        if not isinstance(message, str):
            message = str(message)
            
        self.log_buffer.append(message)
        if len(self.log_buffer) > 5000:
            self.log_buffer = self.log_buffer[-4000:]

        if hasattr(self, 'main_log_text'):
            self.root.after(0, lambda: self._log_to_main_safe(message))

    def _log_to_main_safe(self, message):
        """Internal method to update log widget"""
        try:
            self.main_log_text.config(state=tk.NORMAL)
            self.main_log_text.insert(tk.END, message)

            # Limit log size to prevent lag (keep last 2000 lines)
            num_lines = int(self.main_log_text.index('end-1c').split('.')[0])
            if num_lines > 2000:
                self.main_log_text.delete("1.0", f"{num_lines - 2000 + 1}.0")

            self.main_log_text.see(tk.END)
            self.main_log_text.config(state=tk.DISABLED)
        except Exception:
            pass

    def clear_main_log(self):
        """Clear the main log"""
        self.main_log_text.config(state=tk.NORMAL)
        self.main_log_text.delete(1.0, tk.END)
        self.main_log_text.config(state=tk.DISABLED)

    def setup_events_tab(self):
        """Setup events visualization interface"""
        # Controls
        controls_frame = ttk.Frame(self.events_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        refresh_btn = ttk.Button(controls_frame, text="🔄 Refresh Events", command=self.refresh_events_log, bootstyle="info")
        refresh_btn.pack(side=tk.LEFT)

        # Treeview for events
        columns = ("Time", "Source", "Type", "Priority", "Data")
        self.events_tree = ttk.Treeview(self.events_frame, columns=columns, show="headings", height=20)
        
        self.events_tree.heading("Time", text="Time")
        self.events_tree.column("Time", width=80)
        
        self.events_tree.heading("Source", text="Source")
        self.events_tree.column("Source", width=100)
        
        self.events_tree.heading("Type", text="Event Type")
        self.events_tree.column("Type", width=150)
        
        self.events_tree.heading("Priority", text="Pri")
        self.events_tree.column("Priority", width=40, anchor="center")
        
        self.events_tree.heading("Data", text="Data Payload")
        self.events_tree.column("Data", width=400)

        scrollbar = ttk.Scrollbar(self.events_frame, orient=tk.VERTICAL, command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=scrollbar.set)
        
        self.events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def refresh_events_log(self):
        """Fetch and display recent events from the EventBus"""
        if not hasattr(self, 'ai_core') or not self.ai_core:
            return

        events = self.ai_core.get_event_logs(limit=100)
        
        # Clear current
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
            
        # Populate (Newest first)
        for e in reversed(events):
            time_str = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
            self.events_tree.insert("", "end", values=(time_str, e.source, e.type, e.priority, str(e.data)[:200]))

    def add_chat_message(self, sender, message, message_type="incoming", image_path=None, trigger_processing=True):
        """Add message to chat history"""
        self.chat_history.config(state=tk.NORMAL)

        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Insert Header
        self.chat_history.insert(tk.END, f"[{timestamp}] {sender}:\n", message_type)

        # Insert Image if present
        if image_path and os.path.exists(image_path) and Image:
            try:
                # Load and resize to fixed thumbnail
                img = Image.open(image_path)
                thumbnail_size = (250, 250) # Uniform size box
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Keep reference to prevent GC
                if not hasattr(self, 'image_references'):
                    self.image_references = []
                self.image_references.append(photo)
                
                # Insert image
                self.chat_history.image_create(tk.END, image=photo, padx=5, pady=5)
                
                # Bind click event to open original
                img_tag = f"img_{len(self.image_references)}"
                self.chat_history.tag_add(img_tag, "end-1c")
                self.chat_history.tag_bind(img_tag, "<Button-1>", lambda e, p=image_path: self.open_image(p))
                self.chat_history.tag_bind(img_tag, "<Enter>", lambda e: self.chat_history.config(cursor="hand2"))
                self.chat_history.tag_bind(img_tag, "<Leave>", lambda e: self.chat_history.config(cursor=""))
                
                self.chat_history.insert(tk.END, "\n")
            except Exception as e:
                print(f"Error displaying image: {e}")
                self.chat_history.insert(tk.END, f"[Image Error: {e}]\n", message_type)

        # Insert Message Text
        self.chat_history.insert(tk.END, f"{message}\n\n", message_type)

        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

        # If this was an outgoing message (from local user), process it
        if message_type == "outgoing" and trigger_processing:
            # Run processing in background thread to avoid freezing GUI
            import threading
            threading.Thread(target=self.process_message_thread, args=(message, True)).start()

    def open_image(self, path):
        """Open image in default viewer"""
        try:
            if os.name == 'nt':
                os.startfile(path)
            else:
                # Cross platform fallback using PIL
                if Image:
                    img = Image.open(path)
                    img.show()
        except Exception as e:
            print(f"Error opening image: {e}")

    def setup_help_tab(self):
        """Setup help tab"""
        help_text = scrolledtext.ScrolledText(self.help_frame, wrap=tk.WORD, font=("Arial", 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            help_path = os.path.join(base_dir, "docs", "help.md")
            with open(help_path, "r", encoding="utf-8") as f:
                content = f.read()
            help_text.insert(tk.END, content)
        except Exception as e:
            help_text.insert(tk.END, f"Error loading help.md: {e}\n\nAI Desktop Assistant Help\n\nCommands:\n/status - Check system status\n/memories - View all memories\n/chatmemories - View chat memories\n/documents - List documents\n/resetmemory - Clear all memories\n/resetchat - Clear chat history")
            
        help_text.config(state=tk.DISABLED)

    def setup_about_tab(self):
        """Setup about tab"""
        about_text = scrolledtext.ScrolledText(self.about_frame, wrap=tk.WORD, font=("Arial", 10))
        about_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            about_path = os.path.join(base_dir, "docs", "about.md")
            with open(about_path, "r", encoding="utf-8") as f:
                content = f.read()
            about_text.insert(tk.END, content)
        except Exception as e:
            about_text.insert(tk.END, f"Error loading about.md: {e}\n\nAI Desktop Assistant v1.0\nA local AI assistant with memory and document processing capabilities.")
            
        about_text.config(state=tk.DISABLED)