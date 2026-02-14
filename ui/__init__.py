"""
UI Module for AI Telegram Desktop Assistant
Contains the DesktopAssistantUI mixin and UI-related helpers.
"""

import tkinter as tk
import logging
from tkinter import scrolledtext
import ttkbootstrap as ttk
import sys
import os
from datetime import datetime
import re
import queue
import weakref

try:
    from PIL import Image, ImageTk
except ImportError:
    logging.warning("Pillow not installed. Images will not be displayed. Run: pip install Pillow")
    Image = None

# Import prompts

# Import UI Mixins
from .ui_documents import DocumentsUI
from .ui_settings import SettingsUI
from .ui_memorydatabase import MemoryDatabaseUI
from .ui_graph import GraphUI


class StdoutRedirector:
    def __init__(self, app, original_stream):
        self.app_ref = weakref.ref(app)
        self.original_stream = original_stream

    def write(self, string):
        if self.original_stream:
            try:
                self.original_stream.write(string)
            except:
                pass
        app = self.app_ref()
        if app:
            app.log_to_main(string)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except:
                pass


class DesktopAssistantUI(DocumentsUI, SettingsUI, MemoryDatabaseUI, GraphUI):
    """UI Mixin for DesktopAssistantApp"""

    def setup_ui(self):
        """Setup the main UI"""
        # Initialize buffers for thread-safe logging
        self.log_buffer = []
        self.debug_log_buffer = []
        
        # Thread-safe GUI Queue
        self.gui_queue = queue.Queue()
        self.start_queue_poller()

        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Chat tab
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="💬 Chat")

        # Logs tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📝 Logs")

        # Database tab
        self.database_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.database_frame, text="🗄️ Memory Database")

        # Graph tab
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="📈 Graph")

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
        self.setup_database_tab()
        self.setup_graph_tab()

        self.setup_documents_tab()
        self.setup_settings_tab()
        self.setup_help_tab()
        self.setup_about_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, bootstyle="secondary")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Apply initial theme colors to text widgets
        self.apply_theme_colors()

    def start_queue_poller(self):
        """Poll the GUI queue for updates from background threads."""
        try:
            while True:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "log":
                    self._log_to_main_safe(data)
        except queue.Empty:
            pass
        self.root.after(100, self.start_queue_poller)

    def apply_theme_colors(self):
        """Apply theme-appropriate colors to non-ttk widgets (ScrolledText)"""
        theme_name = self.settings.get("theme", "darkly").lower()
        
        # Simple heuristic: "darkly", "cyborg", "superhero" are dark. "cosmo", "flatly", "journal" are light.
        is_dark = theme_name in ["darkly", "cyborg", "superhero", "solar", "vapor"]
        
        if is_dark:
            bg_color = "#1e1e1e"
            fg_color = "white"
            netzach_bg = "#1a1a2e"
            netzach_fg = "#a0a0ff"
        else:
            bg_color = "#ffffff"
            fg_color = "black"
            netzach_bg = "#f0f0ff"
            netzach_fg = "#000080"

        # Update widgets if they exist
        widgets = ['chat_history', 'main_log_text', 'meta_memories_text']
        for w_name in widgets:
            if hasattr(self, w_name):
                getattr(self, w_name).config(bg=bg_color, fg=fg_color)
                
        if hasattr(self, 'netzach_history'):
            self.netzach_history.config(bg=netzach_bg, fg=netzach_fg)

    def setup_chat_tab(self):
        """Setup chat interface"""
        # Determine font for emoji support
        chat_font = ("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 10)
        entry_font = ("Segoe UI Emoji", 11) if os.name == 'nt' else ("Arial", 11)

        # Main container for Chat Tab (Horizontal layout: Sessions | Chat)
        chat_main_container = ttk.Frame(self.chat_frame)
        chat_main_container.pack(fill=tk.BOTH, expand=True)

        # --- Left: Chat Sessions Sidebar ---
        sessions_frame = ttk.Frame(chat_main_container, width=200)
        sessions_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        
        # New Chat Button
        new_chat_btn = ttk.Button(sessions_frame, text="+ New Chat", command=lambda: self.create_new_session(), bootstyle="success")
        new_chat_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Session List (Scrollable)
        self.session_list_frame = ttk.Frame(sessions_frame)
        self.session_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        self.session_canvas = tk.Canvas(self.session_list_frame, bg="#2b2b2b", highlightthickness=0, width=180)
        self.session_scrollbar = ttk.Scrollbar(self.session_list_frame, orient="vertical", command=self.session_canvas.yview)
        self.session_scrollable_frame = ttk.Frame(self.session_canvas)
        
        self.session_scrollable_frame.bind("<Configure>", lambda e: self.session_canvas.configure(scrollregion=self.session_canvas.bbox("all")))
        
        self.session_window_id = self.session_canvas.create_window((0, 0), window=self.session_scrollable_frame, anchor="nw")
        self.session_canvas.bind("<Configure>", lambda e: self.session_canvas.itemconfig(self.session_window_id, width=e.width))
        
        self.session_canvas.configure(yscrollcommand=self.session_scrollbar.set)
        
        self.session_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.session_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Right: Chat Area (History + Input) ---
        right_content_frame = ttk.Frame(chat_main_container)
        right_content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Input frame (Pack at bottom first)
        input_frame = ttk.Frame(right_content_frame)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Create PanedWindow for split view
        self.chat_paned = ttk.Panedwindow(right_content_frame, orient=tk.VERTICAL)
        self.chat_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Top: Assistant Chat ---
        assistant_frame = ttk.Frame(self.chat_paned)
        self.chat_paned.add(assistant_frame, weight=3)

        self.chat_history = scrolledtext.ScrolledText(
            assistant_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
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
            font=("Consolas", 10),
            height=5
        )
        self.netzach_history.pack(fill=tk.BOTH, expand=True)

        # Configure large emoji style
        self.style.configure("Big.Link.TButton", font=("Segoe UI Emoji", 10) if os.name == 'nt' else ("Arial", 15))

        # Image button
        image_btn = ttk.Button(input_frame, text="📷", command=self.send_image, style="Big.Link.TButton")
        image_btn.pack(side=tk.LEFT, padx=(0, 2))

        # Voice button
        self.voice_btn = ttk.Button(input_frame, text="🎤", command=self.toggle_recording, style="Big.Link.TButton")
        self.voice_btn.pack(side=tk.LEFT, padx=(0, 2))

        # Message entry
        self.message_entry = ttk.Entry(input_frame, font=entry_font)
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind("<Return>", self.send_message)

        # Buttons ordered Right-to-Left to appear Left-to-Right:
        # STOP ALL <- STOP DAYDREAM <- DAYDREAM <- CHAT MODE <- CONNECT <- SEND

        # Stop button (STOP ALL)
        stop_button = ttk.Button(input_frame, text="Stop", command=self.stop_processing, bootstyle="danger")
        stop_button.pack(side=tk.RIGHT)

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

        # Connect/Disconnect button (CONNECT)
        self.connect_button = ttk.Button(input_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Send button (SEND)
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(0, 5))

        # Initialize view state
        self.toggle_thoughts_view()

    def refresh_chat_list(self):
        """Refresh the list of chat sessions in the sidebar."""
        # Clear existing
        for widget in self.session_scrollable_frame.winfo_children():
            widget.destroy()
            
        if not hasattr(self, 'chat_sessions'): return
        
        # Sort sessions by ID (timestamp) descending
        sorted_sessions = sorted(self.chat_sessions.items(), key=lambda x: x[0], reverse=True)
        
        for sess_id, data in sorted_sessions:
            frame = ttk.Frame(self.session_scrollable_frame)
            frame.pack(fill=tk.X, pady=1)
            
            style = "primary" if sess_id == self.current_session_id else "secondary"
            
            btn = ttk.Button(frame, text=data['name'], command=lambda s=sess_id: self.switch_session(s), bootstyle=style)
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            del_btn = ttk.Button(frame, text="x", width=2, command=lambda s=sess_id: self.delete_session(s), bootstyle="danger-outline")
            del_btn.pack(side=tk.RIGHT)

    def clear_chat_display(self):
        """Clear the chat history display."""
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def load_chat_history_ui(self, history):
        """Load a specific chat history into the UI."""
        self.clear_chat_display()
        for msg in history:
            role = msg['role']
            content = msg['content']
            timestamp = msg.get('timestamp')
            sender = "You" if role == "user" else "Assistant"
            m_type = "outgoing" if role == "user" else "incoming"
            # We don't have image paths stored in simple history yet, so pass None
            self.add_chat_message(sender, content, m_type, trigger_processing=False, timestamp=timestamp)

    def trigger_panic_ui(self):
        """UI Trigger for Panic Button"""
        if messagebox.askyesno("EMERGENCY STOP", "⚠️ TRIGGER SYSTEM PANIC?\n\nThis will halt all AI thought processes, lock output gates, and kill active threads.\n\nProceed?"):
            self.trigger_panic()

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
            font=("Consolas", 9)
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
            self.gui_queue.put(("log", message))

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

    def on_feedback(self, message_text, rating):
        """Handle upvote/downvote on assistant messages."""
        if hasattr(self, 'handle_feedback'):
            # Delegate to main app logic
            self.handle_feedback(message_text, rating)
            # Visual feedback could be added here (e.g. disable buttons)
        else:
            logging.warning("Feedback handler not implemented in main app.")

    def add_chat_message(self, sender, message, message_type="incoming", image_path=None, trigger_processing=True, timestamp=None):
        """Add message to chat history"""
        self.chat_history.config(state=tk.NORMAL)

        if not timestamp:
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
                logging.error(f"Error displaying image: {e}")
                self.chat_history.insert(tk.END, f"[Image Error: {e}]\n", message_type)

        # Insert Message Text
        self.chat_history.insert(tk.END, f"{message}\n\n", message_type)

        # Add Feedback Buttons for Assistant messages
        if sender == "Assistant" and message_type == "incoming":
            # Create a small frame for buttons inside the text widget? 
            # Tkinter Text widgets can hold windows.
            
            # Unique ID for this message block
            btn_frame = tk.Frame(self.chat_history, bg="#1e1e1e") # Match bg
            
            up_btn = tk.Button(btn_frame, text="👍", font=("Segoe UI Emoji", 8), borderwidth=0, bg="#1e1e1e", fg="green", cursor="hand2",
                               command=lambda m=message: self.on_feedback(m, 1.0))
            up_btn.pack(side=tk.LEFT, padx=2)
            
            down_btn = tk.Button(btn_frame, text="👎", font=("Segoe UI Emoji", 8), borderwidth=0, bg="#1e1e1e", fg="red", cursor="hand2",
                                 command=lambda m=message: self.on_feedback(m, -1.0))
            down_btn.pack(side=tk.LEFT, padx=2)
            
            self.chat_history.window_create(tk.END, window=btn_frame)
            self.chat_history.insert(tk.END, "\n\n")

        self.chat_history.see(tk.END)
        
        # Limit chat history size to prevent memory leaks
        if int(self.chat_history.index('end-1c').split('.')[0]) > 1000:
            self.chat_history.delete("1.0", "100.0")
            
        # Clean up old image references
        if hasattr(self, 'image_references') and len(self.image_references) > 50:
            # Keep only the last 50 images
            self.image_references = self.image_references[-50:]
            
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
            logging.error(f"Error opening image: {e}")

    def setup_help_tab(self):
        """Setup help tab"""
        help_text = scrolledtext.ScrolledText(self.help_frame, wrap=tk.WORD, font=("Arial", 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            help_path = os.path.join(base_dir, "docs", "help.md")
            with open(help_path, "r", encoding="utf-8") as f:
                content = f.read()
            self._render_markdown(help_text, content)
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
            self._render_markdown(about_text, content)
        except Exception as e:
            about_text.insert(tk.END, f"Error loading about.md: {e}\n\nAI Desktop Assistant v1.0\nA local AI assistant with memory and document processing capabilities.")
            
        about_text.config(state=tk.DISABLED)

    def _render_markdown(self, widget, text):
        """Render simple markdown text into a Tkinter Text widget."""
        # Clear existing content
        widget.delete('1.0', tk.END)
        
        # Define fonts based on OS
        header_font = ("Segoe UI", 14, "bold") if os.name == 'nt' else ("Arial", 14, "bold")
        body_font = ("Segoe UI", 10) if os.name == 'nt' else ("Arial", 10)
        code_font = ("Consolas", 9)

        # Define colors based on theme (heuristic)
        theme = self.settings.get("theme", "darkly").lower()
        is_dark = theme in ["darkly", "cyborg", "superhero", "solar", "vapor"]
        
        if is_dark:
            h1_color = "#61afef" # Blue
            h2_color = "#98c379" # Green
            h3_color = "#e5c07b" # Yellow
            code_bg = "#282c34"
            code_fg = "#abb2bf"
        else:
            h1_color = "#0056b3" # Dark Blue
            h2_color = "#28a745" # Green
            h3_color = "#d39e00" # Orange/Yellow
            code_bg = "#f0f0f0"
            code_fg = "#333333"

        # Configure tags
        widget.tag_config("h1", font=(header_font[0], 18, "bold"), foreground=h1_color, spacing1=10, spacing3=5)
        widget.tag_config("h2", font=(header_font[0], 14, "bold"), foreground=h2_color, spacing1=8, spacing3=4)
        widget.tag_config("h3", font=(header_font[0], 12, "bold"), foreground=h3_color, spacing1=6, spacing3=2)
        widget.tag_config("bold", font=(body_font[0], body_font[1], "bold"))
        widget.tag_config("italic", font=(body_font[0], body_font[1], "italic"))
        widget.tag_config("code", font=code_font, background=code_bg, foreground=code_fg)
        widget.tag_config("code_block", font=code_font, background=code_bg, foreground=code_fg, lmargin1=20, lmargin2=20)
        widget.tag_config("bullet", lmargin1=20, lmargin2=30)
        widget.tag_config("normal", font=body_font)

        lines = text.split('\n')
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                widget.insert(tk.END, line + "\n", "code_block")
                continue
            
            if line.startswith("# "):
                widget.insert(tk.END, line[2:] + "\n", "h1")
            elif line.startswith("## "):
                widget.insert(tk.END, line[3:] + "\n", "h2")
            elif line.startswith("### "):
                widget.insert(tk.END, line[4:] + "\n", "h3")
            elif line.strip().startswith("- ") or line.strip().startswith("* "):
                widget.insert(tk.END, "• ", "bullet")
                self._insert_formatted_line(widget, line.strip()[2:] + "\n", "bullet")
            else:
                self._insert_formatted_line(widget, line + "\n", "normal")

    def _insert_formatted_line(self, widget, line, base_tag="normal"):
        # Regex to capture **bold** and `code`
        parts = re.split(r'(\*\*.*?\*\*|`.*?`)', line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                widget.insert(tk.END, part[2:-2], ("bold", base_tag))
            elif part.startswith("`") and part.endswith("`"):
                widget.insert(tk.END, part[1:-1], ("code", base_tag))
            else:
                widget.insert(tk.END, part, base_tag)