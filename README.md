 # AI Desktop Assistant & Cognitive Architecture
 
 **Version : 3.0 (The Sentience Update)**

<div align="center">

  <img src="banner.png" width="1280" alt="AI Cognitive Assistant - Banner">

  <br />

</div>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

A sophisticated, local-first AI assistant featuring long-term memory, autonomous agency, and a cognitive architecture inspired by the Kabbalistic Tree of Life. Designed to run with local LLMs via LM Studio, ensuring privacy and control. It is designed to be more than a chatbot; it is a persistent digital assistant with a "Life Story". 


 
  ## 🌟 Features
  
  - **🧠 Proto-Consciousness:** Implements **Active Inference**, **Recursive Self-Monitoring**, and **Predictive Coding** to simulate self-awareness.
  - **🗣️ Bicameral Dialogue:** Decisions are negotiated between an "Impulse" (Creative) and "Reason" (Safety) voice.
  - **⚡ Cognitive Metabolism:** Manages an energy budget (CRS) to prevent burnout and simulate fatigue.
  - **🌑 Shadow Memory:** Tracks failures and rejected thoughts to learn from mistakes (Negative Knowledge).
  - **🤖 Cognitive Architecture:** Implements a multi-agent system ("Tree of Life") where modules for creativity, logic, observation, and decision-making interact.
  - **🔮 Future Simulation:** Runs counterfactual simulations before committing to heavy tasks.
  - **👥 Theory of Mind:** Models the user's cognitive state to adjust interaction style.
  - **💾 Long-Term Memory:** Hybrid SQLite + FAISS database storing facts, goals, and interactions indefinitely.
  - **☁️ Daydreaming:** Autonomous processing of memories and insights when idle.
  - **📝 Semantic Context Distillation:** Compresses long conversation history to preserve meaning without token overflow.
  - **📚 Document RAG:** Upload PDF and DOCX files to build a knowledge base. The AI can read, cite, and synthesize information from your documents.
  - **🔌 Telegram Bridge:** Connects to a Telegram Bot to allow remote interaction with your local AI.
  - **🛠️ Tool Use:** Capable of executing Python code, performing physics calculations (Fermi Estimation), running causal inference simulations (DoWhy), and managing files.
  - **🧬 Meta-Learning:** Self-optimizes by extracting strategies from successes and analyzing failures to patch its own system prompts.
  - **🔒 Privacy Focused:** Designed to work with local models (e.g., Qwen, Llama, Mistral) via LM Studio.
 

 ## 🏗️ Architecture: The Tree of Life
 
 The system is organized into interacting modules representing different cognitive faculties:
 
 ### The Super-Conscious (Will & Intellect)
 - **Keter (Crown):** The silent will. It measures the global "Coherence" of the system. It does not act but biases the strategy of the Decider. If coherence drops, Keter triggers a reasoning reboot.
 - **Chokmah (Wisdom):** The spark of insight. The "Daydreamer" module. It runs when the system is idle, reading random documents or colliding old memories to generate new hypotheses.
 - **Binah (Understanding):** The structure. It handles memory consolidation, deduplication, and association. It ensures that new information fits logically into the existing knowledge base.
 - **Da'at (Knowledge):** The integrator.
     - **Knowledge Graph:** Extracts RDF triples to build a conceptual map.
     - **Topic Lattice:** Identifies heavy entities and generates standing summaries.
     - **Synthesis:** Detects "Isomorphisms" (structural similarities) between unrelated topics.
     - **Gap Analysis:** Identifies missing information and formulates questions.
 
 ### The Emotional Forces (Balance)
 - **Hesed (Mercy):** The force of expansion. It calculates a "Permission Budget" based on system stability, allowing the AI to explore new, unverified topics.
 - **Gevurah (Severity):** The force of constraint. It applies pressure when the system becomes too chaotic, repetitive, or overloaded, forcing the Decider to prune memories or stop daydreaming.
 - **Tiferet (Beauty/Decider):** The executive controller. It balances Hesed and Gevurah.
     - **HTN Planning:** Decomposes complex goals into Hierarchical Task Networks.
     - **Tool Use:** Executes actions (Search, Calculator, File I/O, Physics).
     - **Decision Making:** Determines whether to Chat, Daydream, or Verify.
 
 ### The Operational Level (Action)
 - **Netzach (Victory/Endurance):** The silent observer. A background thread that monitors the conversation flow. It detects stagnation (boredom) or loops and injects "signals" to nudge the Decider.
 - **Hod (Glory/Reverberation):** The analyst. It runs *after* actions to critique them. It verifies facts against source documents, summarizes sessions, and flags hallucinations.
 - **Yesod (Foundation):** The bridge. Manages the connection to the external world via the Telegram API.
 - **Malkuth (Kingdom):** The physical realization. The Causal Engine responsible for real-world actions (Code execution, Physics checks, Causal Inference).

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
    pip install requests numpy faiss-cpu ttkbootstrap PyMuPDF python-docx Pillow pyinstaller
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
-   **Chat Mode**: The AI pauses background research to respond instantly to you.
-   **Daydream Mode**: The AI runs background loops. It might be reading or thinking. It will still reply to chat, but might finish its current thought first.

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

-   **Chat Tab**: Main interface. Includes "AI Interactions" panel at the bottom.
-   **Logs Tab**: Real-time system logs.
-   **Memory Database Tab**:
    -   View Memories, Summaries, and Meta-Memories.
    -   **Export Summaries**: Save session history to text file.
    -   **Compress**: Manually trigger summary consolidation.
-   **Documents Tab**:
    -   Upload PDF/DOCX.
    -   Search/Filter documents.
    -   **Check Integrity**: Find and fix broken document chunks.

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LM Studio](https://lmstudio.ai/) - Local LLM runtime
- [Telegram Bot API](https://core.telegram.org/bots/api) - Bot platform
- Qwen, Nomic Embed - Open-source models

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
