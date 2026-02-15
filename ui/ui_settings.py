import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import ttkbootstrap as ttk
import json
import os
import logging
from docs.default_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT, DAYDREAM_EXTRACTOR_PROMPT as DEFAULT_DAYDREAM_EXTRACTOR_PROMPT

try:
    from ttkbootstrap.widgets import ToolTip
except ImportError:
    try:
        from ttkbootstrap.tooltip import ToolTip
    except ImportError:
        ToolTip = None

class SettingsUI:
    """Mixin for Settings UI tab"""

    def create_tooltip(self, widget, text):
        """Helper to create a tooltip if available"""
        if ToolTip:
            ToolTip(widget, text=text, bootstyle="info")

    def setup_settings_tab(self):
        """Setup settings interface"""
        # Buttons frame - Pack at bottom first to ensure visibility
        buttons_frame = ttk.Frame(self.settings_frame)
        buttons_frame.pack(side=tk.BOTTOM, pady=10)

        # Save button
        save_settings_button = ttk.Button(buttons_frame, text="Save Settings", command=self.save_settings_from_ui,
                                          bootstyle="primary")
        save_settings_button.pack(side=tk.LEFT, padx=5)

        # Load button - opens file dialog
        load_settings_button = ttk.Button(buttons_frame, text="Load Settings",
                                          command=self.load_settings_from_file_dialog, bootstyle="secondary")
        load_settings_button.pack(side=tk.LEFT, padx=5)

        settings_notebook = ttk.Notebook(self.settings_frame)
        settings_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 1. Model Settings
        model_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(model_frame, text=" Model")
        self.setup_model_settings(model_frame)

        # 2. Generation Settings
        gen_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(gen_frame, text=" Generation")
        self.setup_generation_settings(gen_frame)

        # 3. Prompts Settings
        prompts_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(prompts_frame, text=" Prompts")
        self.setup_prompt_settings(prompts_frame)

        # 4. Memory Settings
        memory_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(memory_frame, text=" Memory")
        self.setup_memory_settings(memory_frame)

        # 5. General Settings
        general_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(general_frame, text=" General")
        self.setup_general_settings(general_frame)

        # 6. Bridges Settings
        bridges_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(bridges_frame, text=" Bridges")
        self.setup_bridge_settings(bridges_frame)

        # 7. Appearance Settings
        appearance_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(appearance_frame, text=" Appearance")
        self.setup_appearance_settings(appearance_frame)

        # 8. Plugins Settings
        plugins_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(plugins_frame, text=" Plugins")
        self.setup_plugins_tab(plugins_frame)

        # 9. Permissions Settings
        perms_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(perms_frame, text=" Permissions")
        self.setup_permissions_tab(perms_frame)

    def setup_model_settings(self, parent):
        # URL settings box
        url_box = ttk.LabelFrame(parent, text="API URLs")
        url_box.pack(fill=tk.X, padx=5, pady=5)

        lbl = ttk.Label(url_box, text="Base URL:")
        lbl.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl, "The base URL for the LLM API (e.g., LM Studio, Ollama).")

        self.base_url_var = tk.StringVar(value="http://127.0.0.1:1234/v1")
        base_url_entry = ttk.Entry(url_box, textvariable=self.base_url_var, width=50)
        base_url_entry.grid(row=0, column=1, padx=5, pady=5)

        # Model settings box
        model_box = ttk.LabelFrame(parent, text="Model Names")
        model_box.pack(fill=tk.X, padx=5, pady=5)

        lbl_chat = ttk.Label(model_box, text="Chat Model:")
        lbl_chat.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_chat, "The name of the LLM model to use for chat and reasoning.")

        self.chat_model_var = tk.StringVar(value="qwen2.5-vl-7b-instruct-abliterated")
        chat_model_entry = ttk.Entry(model_box, textvariable=self.chat_model_var, width=50)
        chat_model_entry.grid(row=0, column=1, padx=5, pady=5)

        lbl_embed = ttk.Label(model_box, text="Embedding Model:")
        lbl_embed.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_embed, "The name of the model used for generating vector embeddings (Memory/RAG).")

        self.embedding_model_var = tk.StringVar(value="text-embedding-nomic-embed-text-v1.5")
        embedding_model_entry = ttk.Entry(model_box, textvariable=self.embedding_model_var, width=50)
        embedding_model_entry.grid(row=1, column=1, padx=5, pady=5)

    def setup_generation_settings(self, parent):
        # Creativity Group (Temperature & Top P)
        creativity_box = ttk.LabelFrame(parent, text="Creativity & Randomness")
        creativity_box.pack(fill=tk.X, padx=5, pady=5)
        creativity_box.columnconfigure(1, weight=1)

        # Temperature
        lbl_temp = ttk.Label(creativity_box, text="Temperature:")
        lbl_temp.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_temp, "Controls randomness. Lower (0.1) is deterministic, Higher (1.0+) is creative/chaotic.")

        self.temperature_var = tk.DoubleVar(value=0.7)
        temp_scale = ttk.Scale(creativity_box, from_=0.0, to=2.0, variable=self.temperature_var,
                               command=lambda v: self.temperature_var.set(round(float(v), 2)))
        temp_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        temp_entry = ttk.Entry(creativity_box, textvariable=self.temperature_var, width=10)
        temp_entry.grid(row=0, column=2, padx=5, pady=5)

        # Top P
        lbl_top_p = ttk.Label(creativity_box, text="Top P:")
        lbl_top_p.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_top_p, "Nucleus sampling. Limits choices to top P% probability mass. Lower = more focused.")

        self.top_p_var = tk.DoubleVar(value=0.94)
        top_p_scale = ttk.Scale(creativity_box, from_=0.0, to=1.0, variable=self.top_p_var,
                                command=lambda v: self.top_p_var.set(round(float(v), 2)))
        top_p_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        top_p_entry = ttk.Entry(creativity_box, textvariable=self.top_p_var, width=10)
        top_p_entry.grid(row=1, column=2, padx=5, pady=5)

        # Auto-Adjust Step
        lbl_step = ttk.Label(creativity_box, text="Auto-Adjust Step:")
        lbl_step.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_step, "How much to increase Temperature when retrying after a failure.")

        self.temperature_step_var = tk.DoubleVar(value=0.20)
        step_scale = ttk.Scale(creativity_box, from_=0.01, to=0.50, variable=self.temperature_step_var,
                               command=lambda v: self.temperature_step_var.set(round(float(v), 2)))
        step_scale.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        step_entry = ttk.Entry(creativity_box, textvariable=self.temperature_step_var, width=10)
        step_entry.grid(row=2, column=2, padx=5, pady=5)

        # Constraints Group
        constraints_box = ttk.LabelFrame(parent, text="Constraints")
        constraints_box.pack(fill=tk.X, padx=5, pady=5)

        # Max Tokens
        lbl_tokens = ttk.Label(constraints_box, text="Max Tokens:")
        lbl_tokens.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_tokens, "Maximum number of tokens the model can generate in a response.")

        self.max_tokens_var = tk.IntVar(value=800)
        max_tokens_entry = ttk.Entry(constraints_box, textvariable=self.max_tokens_var, width=10)
        max_tokens_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

    def setup_prompt_settings(self, parent):
        prompts_box = ttk.LabelFrame(parent, text="System & Instruction Prompts")
        prompts_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        prompts_box.columnconfigure(0, weight=1)
        prompts_box.rowconfigure(1, weight=1)
        prompts_box.rowconfigure(4, weight=1)
        prompts_box.rowconfigure(7, weight=1)

        # Helper to reset prompt
        def reset_prompt(target_widget, default_text):
            if messagebox.askyesno("Reset Prompt", "Are you sure you want to reset this prompt to default?"):
                target_widget.delete(1.0, tk.END)
                target_widget.insert(tk.END, default_text)

        # System Prompt
        header_frame_1 = ttk.Frame(prompts_box)
        header_frame_1.grid(row=0, column=0, sticky=tk.EW)

        lbl_sys = ttk.Label(header_frame_1, text="System Prompt:")
        lbl_sys.pack(side=tk.LEFT, padx=5, pady=5)
        self.create_tooltip(lbl_sys, "The core personality and instructions for the AI.")

        btn_reset_sys = ttk.Button(header_frame_1, text="Reset", style="link",
                                   command=lambda: reset_prompt(self.system_prompt_text, DEFAULT_SYSTEM_PROMPT))
        btn_reset_sys.pack(side=tk.RIGHT, padx=5)

        self.system_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=8, width=60)
        self.system_prompt_text.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Memory Extractor Prompt
        header_frame_2 = ttk.Frame(prompts_box)
        header_frame_2.grid(row=3, column=0, sticky=tk.EW)

        lbl_mem = ttk.Label(header_frame_2, text="Memory Extractor Prompt:")
        lbl_mem.pack(side=tk.LEFT, padx=5, pady=5)
        self.create_tooltip(lbl_mem, "Instructions for extracting memories (Facts, Goals, etc.) from conversation.")

        btn_reset_mem = ttk.Button(header_frame_2, text="Reset", style="link",
                                   command=lambda: reset_prompt(self.memory_extractor_prompt_text, DEFAULT_MEMORY_EXTRACTOR_PROMPT))
        btn_reset_mem.pack(side=tk.RIGHT, padx=5)

        self.memory_extractor_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=8, width=60)
        self.memory_extractor_prompt_text.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Daydream Extractor Prompt
        header_frame_3 = ttk.Frame(prompts_box)
        header_frame_3.grid(row=6, column=0, sticky=tk.EW)

        lbl_day = ttk.Label(header_frame_3, text="Daydream Extractor Prompt:")
        lbl_day.pack(side=tk.LEFT, padx=5, pady=5)
        self.create_tooltip(lbl_day, "Instructions for extracting insights from internal monologues (Daydreaming).")

        btn_reset_day = ttk.Button(header_frame_3, text="Reset", style="link",
                                   command=lambda: reset_prompt(self.daydream_extractor_prompt_text, DEFAULT_DAYDREAM_EXTRACTOR_PROMPT))
        btn_reset_day.pack(side=tk.RIGHT, padx=5)

        self.daydream_extractor_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=8, width=60)
        self.daydream_extractor_prompt_text.grid(row=7, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

    def setup_memory_settings(self, parent):
        # Cycles & Limits
        limits_box = ttk.LabelFrame(parent, text="Cycles & Limits")
        limits_box.pack(fill=tk.NONE, anchor=tk.W, padx=5, pady=5)

        lbl_cycle = ttk.Label(limits_box, text="Daydream Cycles (Before Verification):")
        lbl_cycle.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_cycle, "How many thought cycles to run before verifying facts against memory.")

        self.daydream_cycle_limit_var = tk.IntVar(value=15)
        ttk.Entry(limits_box, textvariable=self.daydream_cycle_limit_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        lbl_incon = ttk.Label(limits_box, text="Inconclusive Deletion Limit:")
        lbl_incon.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_incon, "Delete a memory if it produces this many 'Inconclusive' verification results.")

        self.max_inconclusive_attempts_var = tk.IntVar(value=3)
        ttk.Entry(limits_box, textvariable=self.max_inconclusive_attempts_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        lbl_ret = ttk.Label(limits_box, text="Retrieval Failure Deletion Limit:")
        lbl_ret.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_ret, "Delete a memory if it fails to be retrieved/verified this many times.")

        self.max_retrieval_failures_var = tk.IntVar(value=3)
        ttk.Entry(limits_box, textvariable=self.max_retrieval_failures_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        lbl_conc = ttk.Label(limits_box, text="Verification Concurrency:")
        lbl_conc.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_conc, "Number of parallel threads used for memory verification.")

        self.concurrency_var = tk.IntVar(value=4)
        ttk.Entry(limits_box, textvariable=self.concurrency_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        # Consolidation Thresholds
        thresholds_box = ttk.LabelFrame(parent, text="Consolidation Thresholds (0.0 - 1.0)")
        thresholds_box.pack(fill=tk.NONE, anchor=tk.W, padx=5, pady=5)
        self.create_tooltip(thresholds_box, "Minimum confidence score required to save a new memory of this type.")

        self.threshold_vars = {}
        # Hierarchy order: PERMISSION -> RULE -> IDENTITY -> PREFERENCE -> GOAL -> FACT -> BELIEF
        types = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]

        # Create single column of entries
        for i, t in enumerate(types):
            ttk.Label(thresholds_box, text=f"{t}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.DoubleVar(value=0.9)
            self.threshold_vars[t] = var
            ttk.Entry(thresholds_box, textvariable=var, width=8).grid(row=i, column=1, padx=5, pady=5)

        # FAISS Settings
        faiss_frame = ttk.LabelFrame(parent, text="FAISS Index Settings (Advanced)")
        faiss_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(faiss_frame, text="Index Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.faiss_index_type_var = tk.StringVar(value="IndexFlatIP")
        faiss_type_combo = ttk.Combobox(faiss_frame, textvariable=self.faiss_index_type_var, values=["IndexFlatIP", "IndexIVFFlat"])
        faiss_type_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(faiss_frame, text="nlist (for IndexIVFFlat):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.faiss_nlist_var = tk.IntVar(value=100)
        faiss_nlist_entry = ttk.Entry(faiss_frame, textvariable=self.faiss_nlist_var, width=10)
        faiss_nlist_entry.grid(row=1, column=1, padx=5, pady=5)

    def setup_general_settings(self, parent):
        # Startup settings
        startup_box = ttk.LabelFrame(parent, text="Startup Settings")
        startup_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(startup_box, text="Initial AI Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.ai_mode_var = tk.StringVar(value="Daydream")
        ai_mode_combo = ttk.Combobox(startup_box, textvariable=self.ai_mode_var, values=["Chat", "Daydream"],
                                     state="readonly", width=15)
        ai_mode_combo.grid(row=0, column=1, padx=5, pady=5)

        # Storage Settings
        storage_box = ttk.LabelFrame(parent, text="Storage & Backup")
        storage_box.pack(fill=tk.X, padx=5, pady=5)

        lbl_backup = ttk.Label(storage_box, text="Backup Directory:")
        lbl_backup.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        self.backup_dir_var = tk.StringVar(value="./data/backups")

        backup_frame = ttk.Frame(storage_box)
        backup_frame.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)

        ttk.Entry(backup_frame, textvariable=self.backup_dir_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)

        def select_backup_dir():
            path = filedialog.askdirectory(initialdir=self.backup_dir_var.get())
            if path:
                self.backup_dir_var.set(path)

        ttk.Button(backup_frame, text="Browse...", command=select_backup_dir, bootstyle="secondary-outline").pack(side=tk.LEFT, padx=5)

    def setup_bridge_settings(self, parent):
        # Telegram bridge settings box
        telegram_box = ttk.LabelFrame(parent, text="Telegram Bridge Settings")
        telegram_box.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)

        # Telegram settings inside the box
        lbl_token = ttk.Label(telegram_box, text="Bot Token:")
        lbl_token.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_token, "The API token provided by BotFather.")

        self.bot_token_var = tk.StringVar()
        bot_token_entry = ttk.Entry(telegram_box, textvariable=self.bot_token_var, width=50)
        bot_token_entry.grid(row=0, column=1, padx=5, pady=5)

        lbl_chat = ttk.Label(telegram_box, text="Chat ID:")
        lbl_chat.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.create_tooltip(lbl_chat, "Your numerical User ID on Telegram.")

        self.chat_id_var = tk.StringVar()
        chat_id_entry = ttk.Entry(telegram_box, textvariable=self.chat_id_var, width=50)
        chat_id_entry.grid(row=1, column=1, padx=5, pady=5)

        # Telegram bridge toggle
        self.telegram_bridge_enabled = tk.BooleanVar()
        telegram_toggle = ttk.Checkbutton(
            telegram_box,
            text="Enable Telegram Bridge",
            variable=self.telegram_bridge_enabled,
            bootstyle="round-toggle"
        )
        telegram_toggle.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)

    def setup_appearance_settings(self, parent):
        appearance_frame = ttk.Frame(parent)
        appearance_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(appearance_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.theme_var = tk.StringVar(value="Darkly")
        theme_combo = ttk.Combobox(appearance_frame, textvariable=self.theme_var, values=[
            "Cosmo", "Cyborg", "Darkly"
        ])
        theme_combo.grid(row=0, column=1, padx=5, pady=5)

    def load_settings_into_ui(self):
        """Load settings into UI fields"""
        self.bot_token_var.set(self.settings.get("bot_token", ""))
        self.chat_id_var.set(str(self.settings.get("chat_id", "")))
        self.theme_var.set(self.settings.get("theme", "Darkly"))
        self.telegram_bridge_enabled.set(self.settings.get("telegram_bridge_enabled", False))
        self.base_url_var.set(self.settings.get("base_url", "http://127.0.0.1:1234/v1"))
        self.chat_model_var.set(self.settings.get("chat_model", "qwen2.5-vl-7b-instruct-abliterated"))
        self.embedding_model_var.set(self.settings.get("embedding_model", "text-embedding-nomic-embed-text-v1.5"))
        self.temperature_var.set(self.settings.get("temperature", 0.7))
        self.top_p_var.set(self.settings.get("top_p", 0.94))
        self.max_tokens_var.set(self.settings.get("max_tokens", 800))
        self.temperature_step_var.set(self.settings.get("temperature_step", 0.20))

        self.system_prompt_text.delete(1.0, tk.END)
        self.system_prompt_text.insert(tk.END, self.settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT))

        self.memory_extractor_prompt_text.delete(1.0, tk.END)
        self.memory_extractor_prompt_text.insert(tk.END, self.settings.get("memory_extractor_prompt",
                                                                           DEFAULT_MEMORY_EXTRACTOR_PROMPT))

        self.daydream_extractor_prompt_text.delete(1.0, tk.END)
        self.daydream_extractor_prompt_text.insert(tk.END, self.settings.get("daydream_extractor_prompt",
                                                                             DEFAULT_DAYDREAM_EXTRACTOR_PROMPT))

        self.ai_mode_var.set(self.settings.get("ai_mode", "Daydream"))

        # Memory settings
        self.daydream_cycle_limit_var.set(self.settings.get("daydream_cycle_limit", 15))
        self.max_inconclusive_attempts_var.set(self.settings.get("max_inconclusive_attempts", 3))
        self.max_retrieval_failures_var.set(self.settings.get("max_retrieval_failures", 3))
        self.concurrency_var.set(self.settings.get("concurrency", 4))

        thresholds = self.settings.get("consolidation_thresholds", {})
        default_thresholds = {"GOAL": 0.88, "IDENTITY": 0.87, "BELIEF": 0.87, "PERMISSION": 0.87, "FACT": 0.93,
                              "PREFERENCE": 0.93, "RULE": 0.93}
        for t, var in self.threshold_vars.items():
            var.set(thresholds.get(t, default_thresholds.get(t, 0.9)))

        # Load permissions
        perms = self.settings.get("permissions", {})
        for tool, var in self.permission_vars.items():
            var.set(perms.get(tool, True))

        self.faiss_index_type_var.set(self.settings.get("faiss_index_type", "IndexFlatIP"))
        self.faiss_nlist_var.set(self.settings.get("faiss_nlist", 100))

        # Load Backup Dir
        self.backup_dir_var.set(self.settings.get("backup_dir", "./data/backups"))

    def setup_permissions_tab(self, parent):
        """Setup tool permissions interface."""
        perms_box = ttk.LabelFrame(parent, text="Tool Permissions")
        perms_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.permission_vars = {}
        tools = [
            "SEARCH", "WIKI", "FIND_PAPER", "CALCULATOR", "CLOCK", 
            "SYSTEM_INFO", "PHYSICS", "SIMULATE_PHYSICS", "CAUSAL", "SIMULATE_ACTION", 
            "PREDICT", "READ_CHUNK", "DESCRIBE_IMAGE", "WRITE_FILE"
        ]
        
        # Create a grid of checkboxes
        for i, tool in enumerate(tools):
            var = tk.BooleanVar(value=True)
            self.permission_vars[tool] = var
            cb = ttk.Checkbutton(perms_box, text=tool, variable=var)
            row = i // 3
            col = i % 3
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)

    def setup_plugins_tab(self, parent):
        """Setup the plugins management interface."""
        if not hasattr(self, 'ai_core') or not self.ai_core.action_manager:
            ttk.Label(parent, text="AI Core not initialized.").pack(padx=10, pady=10)
            return

        plugins = self.ai_core.action_manager.get_plugins()
        
        if not plugins:
            ttk.Label(parent, text="No plugins loaded.").pack(padx=10, pady=10)
            return

        # Scrollable frame for plugins
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for name, info in plugins.items():
            frame = ttk.LabelFrame(scrollable_frame, text=name)
            frame.pack(fill=tk.X, padx=5, pady=5)
            
            var = tk.BooleanVar(value=info['enabled'])
            cb = ttk.Checkbutton(frame, text="Enabled", variable=var, 
                                 command=lambda n=name, v=var: self.ai_core.action_manager.toggle_plugin(n, v.get()))
            cb.pack(anchor=tk.W, padx=5, pady=2)
            
            ttk.Label(frame, text=info.get('description', 'No description')).pack(anchor=tk.W, padx=5, pady=2)

    def load_settings_from_file_dialog(self):
        """Load settings from a selected file via dialog"""
        file_path = filedialog.askopenfilename(
            title="Select settings file",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ],
            initialfile="settings.json"
        )

        if not file_path:
            return  # User cancelled the dialog

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)

            # Update current settings with loaded ones
            self.settings.update(loaded_settings)

            # Update the settings file path to the loaded file
            self.settings_file_path = file_path

            # Update UI with new settings
            self.load_settings_into_ui()

            # Apply the loaded theme
            theme_map = {
                "Cosmo": "cosmo",
                "Cyborg": "cyborg",
                "Darkly": "darkly"
            }
            theme_to_apply = theme_map.get(self.settings.get("theme", "Darkly"), self.settings.get("theme", "darkly"))
            self.style.theme_use(theme_to_apply)

            # Update connection state based on loaded settings
            if (self.settings.get("telegram_bridge_enabled", False) and
                    self.settings.get("bot_token") and
                    self.settings.get("chat_id")):
                self.telegram_bridge_enabled.set(True)
                self.connect()
            else:
                self.telegram_bridge_enabled.set(False)
                self.disconnect()

            messagebox.showinfo("Settings", f"Settings loaded successfully from {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings from {file_path}:\n{str(e)}")

    def save_settings_from_ui(self):
        """Save settings from UI fields"""
        try:
            # 1. Validation
            try:
                temp = self.temperature_var.get()
                if not (0.0 <= temp <= 2.0):
                    raise ValueError("Temperature must be between 0.0 and 2.0")

                top_p = self.top_p_var.get()
                if not (0.0 <= top_p <= 1.0):
                    raise ValueError("Top P must be between 0.0 and 1.0")

                tokens = self.max_tokens_var.get()
                if tokens <= 0:
                     raise ValueError("Max Tokens must be positive")

                concurrency = self.concurrency_var.get()
                if concurrency <= 0:
                    raise ValueError("Concurrency must be positive")
            except ValueError as ve:
                messagebox.showerror("Validation Error", str(ve))
                return

            self.settings["bot_token"] = self.bot_token_var.get()
            self.settings["chat_id"] = self.chat_id_var.get()
            self.settings["theme"] = self.theme_var.get().lower()  # Convert to lowercase for ttkbootstrap
            self.settings["telegram_bridge_enabled"] = self.telegram_bridge_enabled.get()
            self.settings["base_url"] = self.base_url_var.get()
            self.settings["chat_model"] = self.chat_model_var.get()
            self.settings["embedding_model"] = self.embedding_model_var.get()
            self.settings["temperature"] = self.temperature_var.get()
            self.settings["top_p"] = self.top_p_var.get()
            self.settings["max_tokens"] = self.max_tokens_var.get()
            self.settings["temperature_step"] = self.temperature_step_var.get()
            self.settings["system_prompt"] = self.system_prompt_text.get(1.0, tk.END).strip()
            self.settings["memory_extractor_prompt"] = self.memory_extractor_prompt_text.get(1.0, tk.END).strip()
            self.settings["daydream_extractor_prompt"] = self.daydream_extractor_prompt_text.get(1.0, tk.END).strip()
            self.settings["ai_mode"] = self.ai_mode_var.get()
            self.settings["backup_dir"] = self.backup_dir_var.get()

            # Memory settings
            self.settings["daydream_cycle_limit"] = self.daydream_cycle_limit_var.get()
            self.settings["max_inconclusive_attempts"] = self.max_inconclusive_attempts_var.get()
            self.settings["max_retrieval_failures"] = self.max_retrieval_failures_var.get()
            self.settings["concurrency"] = self.concurrency_var.get()

            thresholds = {t: var.get() for t, var in self.threshold_vars.items()}
            self.settings["consolidation_thresholds"] = thresholds

            # Permissions
            permissions = {t: var.get() for t, var in self.permission_vars.items()}
            self.settings["permissions"] = permissions
            
            self.settings["faiss_index_type"] = self.faiss_index_type_var.get()
            self.settings["faiss_nlist"] = self.faiss_nlist_var.get()

            self.save_settings()

            # Apply new theme (convert capitalized name to lowercase)
            theme_map = {
                "Cosmo": "cosmo",
                "Cyborg": "cyborg",
                "Darkly": "darkly"
            }
            theme_to_apply = theme_map.get(self.settings["theme"].capitalize(), self.settings["theme"])
            self.style.theme_use(theme_to_apply)

            # Update text widget colors
            if hasattr(self, 'apply_theme_colors'):
                self.apply_theme_colors()

            # Only attempt connection if bridge is enabled and both credentials are provided
            if (self.settings["telegram_bridge_enabled"] and
                    self.settings["bot_token"] and
                    self.settings["chat_id"]):
                self.connect()
            elif not self.settings["telegram_bridge_enabled"]:
                # If bridge is disabled, make sure we're disconnected
                self.disconnect()

            # Create a custom popup or toast if possible, otherwise use standard messagebox
            # but maybe less intrusive? Standard is fine for now.
            messagebox.showinfo("Success", "Settings saved successfully.")

        except tk.TclError as e:
            messagebox.showerror("Validation Error", f"Invalid input in settings: {e}\nPlease check numeric fields.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
