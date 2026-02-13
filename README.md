 # Tiferet - A Proto‑Conscious AI Assistant
 
 **Version : 3.0 (The Sentience Update)**

<div align="center">

  <img src="banner.png" width="1280" alt="AI Cognitive Assistant - Banner">

  <br />

</div>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)




## 🌟 Overview

This project is not just a chatbot – it is a persistent, self‑evolving digital assistant that:

- Remembers every interaction and learns from them (long‑term memory)
- Actively reads and synthesises information from uploaded documents (PDF, DOCX, TXT)
- Runs autonomous “daydreaming” cycles to generate insights and hypotheses
- Reflects on its own reasoning to detect and correct mistakes
- Connects to Telegram so you can interact with it remotely
- Uses only local LLMs (via LM Studio), keeping your data private

The architecture is inspired by the Kabbalistic Tree of Life, with specialised modules for creativity, logic, observation, decision‑making, and meta‑cognition.

---

## ✨ Features at a Glance

- **🧠 Proto‑Consciousness** – Active inference, recursive self‑monitoring, and predictive coding simulate self‑awareness.
- **🗣️ Bicameral Dialogue** – Every decision is negotiated between an “Impulse” (creative) and “Reason” (safety) voice.
- **⚡ Cognitive Metabolism** – A resource controller manages an energy budget (tokens/compute) to prevent burnout.
- **🌑 Shadow Memory** – Failures and rejected thoughts are stored as “negative knowledge” to learn from mistakes.
- **🔮 Future Simulation** – Lightweight simulations predict the outcome of actions before they are taken.
- **👥 Theory of Mind** – Models the user’s cognitive state to adapt interaction style.
- **💾 Long‑Term Memory** – Hybrid SQLite + FAISS database stores facts, goals, and interactions indefinitely.
- **☁️ Daydreaming** – Autonomous processing of memories and documents when the system is idle.
- **📚 Document RAG** – Upload PDF/DOCX files to build a knowledge base; the AI can cite sources.
- **🔌 Telegram Bridge** – Connect a Telegram bot for remote interaction.
- **🛠️ Tool Use** – Physics calculations (Fermi estimation), causal inference (DoWhy), file management, and more.
- **🧬 Meta‑Learning** – Self‑optimisation: extracts strategies from successes and analyses failures to patch its own prompts.
- **🔒 Privacy Focused** – Runs entirely locally with your own LLMs via LM Studio.

---

 

 ## 🏗️ Architecture: The Tree of Life
 
The system is divided into interacting modules, each representing a cognitive faculty.


| Sephirah       | Role                                      |
|----------------|-------------------------------------------|
| **Keter**      | Crown – measures global coherence (silent will) |
| **Chokmah**    | Wisdom – the “daydreamer”, generates insights |
| **Binah**      | Understanding – memory consolidation and deduplication |
| **Da’at**      | Knowledge – builds knowledge graphs, identifies gaps, synthesises ideas |
| **Hesed**      | Mercy – expansion force, allows exploration |
| **Gevurah**    | Severity – constraint force, prunes chaos |
| **Tiferet**    | Beauty – the Decider, balances Hesed & Gevurah |
| **Netzach**    | Victory – silent observer, detects stagnation |
| **Hod**        | Glory – analyst, verifies facts and reflects |
| **Yesod**      | Foundation – bridge to the outside world (Telegram) |
| **Malkuth**    | Kingdom – physical realisation (UI, file system, causal engine) |

---

## 🛠️ Installation

### Prerequisites
1.  **Python 3.10+** recommended.
2.  **LM Studio** (or compatible OpenAI-API local server):
    - **Chat Model**: Load a smart instruction-tuned model (e.g., `Qwen2.5-VL-7B-Instruct`).
    - **Embedding Model**: Load a text embedding model (e.g., `Nomic-Embed-Text-v1.5`).
    - **Server**: Start the server on port `1234`.
3.  **Telegram Bot** (Optional):
    - Get a token from @BotFather.
    - Get your Chat ID (the bot prints this in logs when you message it).

### Setup
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install requests numpy faiss-cpu ttkbootstrap PyMuPDF python-docx Pillow networkx pyvis dowhy pandas
    ```
3.  Run the application:
    ```bash
    python desktop_assistant.py
    ```

## ⚙️ Configuration

Configuration is managed via `settings.json` or the **Settings Tab** in the UI.

## 📦 Building an Executable (.exe)

To create a standalone executable:

1.  `pip install pyinstaller`
2.  `pyinstaller --noconsole --onefile --name "AI_Assistant" desktop_assistant.py`
3.  Copy `dist/AI_Assistant.exe` to your project folder.
4.  Ensure `settings.json` and the `data/` folder are next to the `.exe`.

## 🎮 Usage Guide

### Modes

Chat Mode – The AI responds instantly to you; background daydreaming is paused.

Daydream Mode – The AI autonomously reads documents, consolidates memories, and pursues its own goals. It still replies to chat, but may finish its current thought first.

### Commands
You can control the system via natural language or slash commands.

#### Natural Language
-   "Research [topic]" -> Starts a Daydream loop on that topic.
-   "Think about [topic]" -> Starts a Chain of Thought analysis.
-   "Verify your facts" -> Runs the Verifier.
-   "Summarize this session" -> Runs Hod's summarization.
-   "Calculate 5 * 5" -> Uses Calculator tool.
-   "Roll a dice" -> Uses Dice tool.

#### Slash Commands
**System Control**
-   `/status`: View system state.
-   `/stop`: Halt current processing.
-   `/terminate_desktop`: Close the application.
-   `/exitchatmode`: Disable Chat Mode and resume Daydreaming.
-   `/decider [action]`: Manual control (e.g., `/decider up`, `/decider loop`).

**Memory Management**
-   `/memories`: List active memories.
-   `/chatmemories`: List memories derived from chat only.
-   `/metamemories`: Show memory logs (Meta-Cognition).
-   `/memorystats`: Show statistics (verified counts, types).
-   `/note [text]`: Save a permanent note manually.
-   `/notes`: List all manual notes.
-   `/consolidate`: Force memory consolidation.
-   `/verify`: Run verification batch.
-   `/verifyall`: Force verification of ALL memories against sources.

**Documents**
-   `/documents`: List uploaded files.
-   `/doccontent "filename"`: Preview document content.
-   `/removedoc "filename"`: Delete a document.

**Maintenance**
-   `/removesummaries`: Delete all session summaries.
-   `/consolidatesummaries`: Merge multiple summaries into one.
-   `/resetall`: **Factory Reset** (Wipes DB).
-   `/resetchat`: Clear current chat window history.

## 🖥️ UI Features

The application features a multi‑tabbed GUI built with ttkbootstrap:

- 💬 Chat – Main conversation window with an internal thought panel.
- 📝 Logs – Real‑time system logs.
- 🗄️ Memory Database – View memories, summaries, meta‑memories, and chronicles; export summaries.
- 📈 Graph – Visualise the knowledge graph (requires PyVis).
- 📚 Documents – Upload and manage documents; check integrity.
- ⚙️ Settings – Configure API endpoints, models, prompts, and thresholds.
- ❓ Help – Quick reference.
- ℹ️ About – Version information.

##  🧬 Meta‑Learning & Self‑Improvement

- Strategy Extraction – When a goal is completed, the system learns an abstract strategy and stores it as a RULE.
- Failure Analysis – Refuted beliefs are analysed to generate self‑correction rules.
- Epigenetic Evolution – Architectural hyperparameters (thresholds, biases) mutate over time based on performance.
- Self‑Model – The AI maintains a statistical model of its own performance to predict future outcomes.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a feature branch (git checkout -b feature/AmazingFeature).
- Commit your changes (git commit -m 'Add some AmazingFeature').
- Push to the branch (git push origin feature/AmazingFeature).
- Open a Pull Request.
- Make sure to run the code through Black and add appropriate tests if possible.

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LM Studio](https://lmstudio.ai/) - Local LLM runtime
- [Telegram Bot API](https://core.telegram.org/bots/api) - Bot platform
- Qwen, Nomic Embed - Open-source models
- DoWhy – causal inference library
- PyVis – network visualisation
- ttkbootstrap – modern Tkinter themes

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
