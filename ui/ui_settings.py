import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import ttkbootstrap as ttk
import json
import os
import logging
from ai_core.lm import DEFAULT_SYSTEM_PROMPT, DEFAULT_MEMORY_EXTRACTOR_PROMPT
from treeoflife.chokmah import DAYDREAM_EXTRACTOR_PROMPT as DEFAULT_DAYDREAM_EXTRACTOR_PROMPT

class SettingsUI:
    """Mixin for Settings UI tab"""

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

        # Model settings
        model_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(model_frame, text=" Model")

        # URL settings box
        url_box = ttk.LabelFrame(model_frame, text="API URLs")
        url_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(url_box, text="Base URL:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.base_url_var = tk.StringVar(value="http://127.0.0.1:1234/v1")
        base_url_entry = ttk.Entry(url_box, textvariable=self.base_url_var, width=50)
        base_url_entry.grid(row=0, column=1, padx=5, pady=5)

        # Model settings box (removed padding as it causes TclError)
        model_box = ttk.LabelFrame(model_frame, text="Model Names")
        model_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(model_box, text="Chat Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.chat_model_var = tk.StringVar(value="qwen2.5-vl-7b-instruct-abliterated")
        chat_model_entry = ttk.Entry(model_box, textvariable=self.chat_model_var, width=50)
        chat_model_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(model_box, text="Embedding Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.embedding_model_var = tk.StringVar(value="text-embedding-nomic-embed-text-v1.5")
        embedding_model_entry = ttk.Entry(model_box, textvariable=self.embedding_model_var, width=50)
        embedding_model_entry.grid(row=1, column=1, padx=5, pady=5)

        # Generation settings tab
        gen_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(gen_frame, text=" Generation")

        gen_box = ttk.LabelFrame(gen_frame, text="Generation Parameters")
        gen_box.pack(fill=tk.X, padx=5, pady=5)
        gen_box.columnconfigure(1, weight=1)

        # Temperature
        ttk.Label(gen_box, text="Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.temperature_var = tk.DoubleVar(value=0.7)
        temp_scale = ttk.Scale(gen_box, from_=0.0, to=2.0, variable=self.temperature_var,
                               command=lambda v: self.temperature_var.set(round(float(v), 2)))
        temp_scale.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        temp_entry = ttk.Entry(gen_box, textvariable=self.temperature_var, width=10)
        temp_entry.grid(row=0, column=2, padx=5, pady=5)

        # Top P
        ttk.Label(gen_box, text="Top P:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.top_p_var = tk.DoubleVar(value=0.94)
        top_p_scale = ttk.Scale(gen_box, from_=0.0, to=1.0, variable=self.top_p_var,
                                command=lambda v: self.top_p_var.set(round(float(v), 2)))
        top_p_scale.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        top_p_entry = ttk.Entry(gen_box, textvariable=self.top_p_var, width=10)
        top_p_entry.grid(row=1, column=2, padx=5, pady=5)

        # Max Tokens
        ttk.Label(gen_box, text="Max Tokens:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_tokens_var = tk.IntVar(value=800)
        max_tokens_entry = ttk.Entry(gen_box, textvariable=self.max_tokens_var, width=10)
        max_tokens_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Auto-Adjust Step
        ttk.Label(gen_box, text="Auto-Adjust Step:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.temperature_step_var = tk.DoubleVar(value=0.20)
        step_scale = ttk.Scale(gen_box, from_=0.01, to=0.50, variable=self.temperature_step_var,
                               command=lambda v: self.temperature_step_var.set(round(float(v), 2)))
        step_scale.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=5)
        step_entry = ttk.Entry(gen_box, textvariable=self.temperature_step_var, width=10)
        step_entry.grid(row=3, column=2, padx=5, pady=5)

        # Prompts settings tab
        prompts_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(prompts_frame, text=" Prompts")

        prompts_box = ttk.LabelFrame(prompts_frame, text="Prompts")
        prompts_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        prompts_box.columnconfigure(0, weight=1)
        prompts_box.rowconfigure(1, weight=1)
        prompts_box.rowconfigure(3, weight=1)

        ttk.Label(prompts_box, text="System Prompt:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.system_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=10, width=60)
        self.system_prompt_text.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(prompts_box, text="Memory Extractor Prompt:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.memory_extractor_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=15, width=60)
        self.memory_extractor_prompt_text.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(prompts_box, text="Daydream Extractor Prompt:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.daydream_extractor_prompt_text = scrolledtext.ScrolledText(prompts_box, wrap=tk.WORD, height=15, width=60)
        self.daydream_extractor_prompt_text.grid(row=5, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Memory settings tab
        memory_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(memory_frame, text=" Memory")

        # Cycles & Limits
        limits_box = ttk.LabelFrame(memory_frame, text="Cycles & Limits")
        limits_box.pack(fill=tk.NONE, anchor=tk.W, padx=5, pady=5)

        ttk.Label(limits_box, text="Daydream Cycles (Before Verification):").grid(row=0, column=0, sticky=tk.W, padx=5,
                                                                                  pady=5)
        self.daydream_cycle_limit_var = tk.IntVar(value=15)
        ttk.Entry(limits_box, textvariable=self.daydream_cycle_limit_var, width=10).grid(row=0, column=1, padx=5,
                                                                                         pady=5)

        ttk.Label(limits_box, text="Inconclusive Deletion Limit:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_inconclusive_attempts_var = tk.IntVar(value=3)
        ttk.Entry(limits_box, textvariable=self.max_inconclusive_attempts_var, width=10).grid(row=1, column=1, padx=5,
                                                                                              pady=5)

        ttk.Label(limits_box, text="Retrieval Failure Deletion Limit:").grid(row=2, column=0, sticky=tk.W, padx=5,
                                                                             pady=5)
        self.max_retrieval_failures_var = tk.IntVar(value=3)
        ttk.Entry(limits_box, textvariable=self.max_retrieval_failures_var, width=10).grid(row=2, column=1, padx=5,
                                                                                           pady=5)

        ttk.Label(limits_box, text="Verification Concurrency:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.concurrency_var = tk.IntVar(value=4)
        ttk.Entry(limits_box, textvariable=self.concurrency_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        # Consolidation Thresholds
        thresholds_box = ttk.LabelFrame(memory_frame, text="Consolidation Thresholds (0.0 - 1.0)")
        thresholds_box.pack(fill=tk.NONE, anchor=tk.W, padx=5, pady=5)

        self.threshold_vars = {}
        # Hierarchy order: PERMISSION -> RULE -> IDENTITY -> PREFERENCE -> GOAL -> FACT -> BELIEF
        types = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]

        # Create single column of entries
        for i, t in enumerate(types):
            ttk.Label(thresholds_box, text=f"{t}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.DoubleVar(value=0.9)
            self.threshold_vars[t] = var
            ttk.Entry(thresholds_box, textvariable=var, width=8).grid(row=i, column=1, padx=5, pady=5)

        # General settings tab
        general_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(general_frame, text=" General")

        # Startup settings
        startup_box = ttk.LabelFrame(general_frame, text="Startup Settings")
        startup_box.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(startup_box, text="Initial AI Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.ai_mode_var = tk.StringVar(value="Daydream")
        ai_mode_combo = ttk.Combobox(startup_box, textvariable=self.ai_mode_var, values=["Chat", "Daydream"],
                                     state="readonly", width=15)
        ai_mode_combo.grid(row=0, column=1, padx=5, pady=5)

        # Bridges settings
        bridges_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(bridges_frame, text=" Bridges")

        # Telegram bridge settings box
        telegram_box = ttk.LabelFrame(bridges_frame, text="Telegram Bridge Settings")
        telegram_box.pack(fill=tk.X, padx=5, pady=5, ipadx=5, ipady=5)

        # Telegram settings inside the box
        ttk.Label(telegram_box, text="Bot Token:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.bot_token_var = tk.StringVar()
        bot_token_entry = ttk.Entry(telegram_box, textvariable=self.bot_token_var, width=50)
        bot_token_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(telegram_box, text="Chat ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
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

        # Appearance settings
        appearance_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(appearance_frame, text="Appearance")

        ttk.Label(appearance_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.theme_var = tk.StringVar(value="Darkly")
        theme_combo = ttk.Combobox(appearance_frame, textvariable=self.theme_var, values=[
            "Cosmo", "Cyborg", "Darkly"
        ])
        theme_combo.grid(row=0, column=1, padx=5, pady=5)

        # Plugins tab
        plugins_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(plugins_frame, text=" Plugins")
        self.setup_plugins_tab(plugins_frame)

        # Permissions tab
        perms_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(perms_frame, text=" Permissions")
        self.setup_permissions_tab(perms_frame)

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

            messagebox.showinfo("Settings", "Settings saved successfully!")
        except tk.TclError as e:
            messagebox.showerror("Validation Error", f"Invalid input in settings: {e}\nPlease check numeric fields.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")