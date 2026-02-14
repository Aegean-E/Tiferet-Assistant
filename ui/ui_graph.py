import tkinter as tk
import ttkbootstrap as ttk
from tkinter import messagebox
import os
import webbrowser
import logging
import tkinter as tk

class GraphUI:
    """Mixin for Knowledge Graph Visualization"""

    def setup_graph_tab(self):
        """Setup graph interface"""
        # Controls
        controls_frame = ttk.Frame(self.graph_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        refresh_btn = ttk.Button(controls_frame, text="🔄 Generate Graph", command=self.generate_graph_viz, bootstyle="primary")
        refresh_btn.pack(side=tk.LEFT, padx=5)

        open_btn = ttk.Button(controls_frame, text="🌐 Open in Browser", command=self.open_graph_browser, bootstyle="info")
        open_btn.pack(side=tk.LEFT, padx=5)

        # Filter Frame
        filter_frame = ttk.LabelFrame(self.graph_frame, text="Filters")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Search Node:").pack(side=tk.LEFT, padx=5)
        self.graph_search_var = tk.StringVar()
        ttk.Entry(filter_frame, textvariable=self.graph_search_var).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Apply", command=self.generate_graph_viz, bootstyle="secondary").pack(side=tk.LEFT, padx=5)

        # Info Label
        self.graph_status = tk.StringVar(value="Ready to generate.")
        lbl = ttk.Label(self.graph_frame, textvariable=self.graph_status, bootstyle="secondary")
        lbl.pack(pady=20)

    def generate_graph_viz(self):
        """Generate HTML visualization using PyVis"""
        try:
            import networkx as nx
            from pyvis.network import Network
        except ImportError:
            messagebox.showerror("Missing Dependency", "Please install pyvis: pip install pyvis")
            return

        gml_path = "./data/knowledge_graph.gml"
        html_path = "./data/knowledge_graph.html"

        if not os.path.exists(gml_path):
            messagebox.showinfo("No Graph", "No Knowledge Graph (GML) found. Let Da'at build it first.")
            return

        try:
            self.graph_status.set("Loading GML...")
            G = nx.read_gml(gml_path)
            
            # Apply Filters
            search_term = self.graph_search_var.get().lower().strip()
            if search_term:
                nodes_to_keep = [n for n, data in G.nodes(data=True) if search_term in str(data.get('label', '')).lower()]
                if nodes_to_keep:
                    G = G.subgraph(nodes_to_keep)
                else:
                    self.graph_status.set(f"No nodes match '{search_term}'")
                    return
            
            self.graph_status.set(f"Generating visualization ({len(G.nodes)} nodes)...")
            net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
            net.from_nx(G)
            
            # Add search bar to HTML
            # net.show_buttons(filter_=['physics']) # Optional: show physics controls
            
            # Physics options
            net.force_atlas_2based()
            
            net.save_graph(html_path)
            self.graph_status.set(f"Graph generated: {html_path}")
            self.open_graph_browser()
            
        except Exception as e:
            self.graph_status.set(f"Error: {e}")
            logging.error(f"Graph Viz Error: {e}")

    def open_graph_browser(self):
        html_path = os.path.abspath("./data/knowledge_graph.html")
        if os.path.exists(html_path):
            webbrowser.open(f"file://{html_path}")