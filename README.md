 # AI Desktop Assistant & Cognitive Architecture (Tree of Life)
 
 **Version:** 2.3 (The Da'at Update)

<div align="center">

  <img src="banner.png" width="1280" alt="AI Cognitive Assistant - Banner">

  <br />

</div>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

A sophisticated, local-first AI assistant featuring long-term memory, autonomous agency, and a cognitive architecture inspired by the Kabbalistic Tree of Life. Designed to run with local LLMs via LM Studio, ensuring privacy and control. It is designed to be more than a chatbot; it is a persistent digital assistant with a "Life Story". 

 ## 🌟 Features
 
 - **🧠 Long-Term Memory:** Uses a hybrid SQLite + FAISS vector database to store facts, goals, and interactions indefinitely.
 - **🤖 Cognitive Architecture:** Implements a multi-agent system ("Tree of Life") where modules for creativity, logic, observation, and decision-making interact.
 - **📚 Document RAG:** Upload PDF and DOCX files to build a knowledge base. The AI can read, cite, and synthesize information from your documents.
 - **🔌 Telegram Bridge:** Connects to a Telegram Bot to allow remote interaction with your local AI.
 - **🛠️ Tool Use:** Capable of executing Python code, performing physics calculations (Fermi Estimation), running causal inference simulations (DoWhy), and managing files.
 - **☁️ Daydreaming:** When idle, the AI autonomously processes memories, consolidates knowledge, and generates new insights.
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
     - **Synthesis:** Detects "Isomorphisms" (structural similarities) between unrelated topics.
     - **Gap Analysis:** Identifies missing information and formulates questions.
 
 ### The Emotional Forces (Balance)
 - **Hesed (Mercy):** The force of expansion. It calculates a "Permission Budget" based on system stability, allowing the AI to explore new, unverified topics.
 - **Gevurah (Severity):** The force of constraint. It applies pressure when the system becomes too chaotic, repetitive, or overloaded, forcing the Decider to prune memories or stop daydreaming.
 - **Tiferet (Beauty/Decider):** The executive controller. It balances Hesed and Gevurah.
     - **HTN Planning:** Decomposes complex goals into Hierarchical Task Networks.
     - **Tool Use:** Executes actions (Search, Calculator, File I/O, Code, Physics).
     - **Decision Making:** Determines whether to Chat, Daydream, or Verify.
 
 ### The Operational Level (Action)
 - **Netzach (Victory/Endurance):** The silent observer. A background thread that monitors the conversation flow. It detects stagnation (boredom) or loops and injects "signals" to nudge the Decider.
 - **Hod (Glory/Reverberation):** The analyst. It runs *after* actions to critique them. It verifies facts against source documents, summarizes sessions, and flags hallucinations.
 - **Yesod (Foundation):** The bridge. Manages the connection to the external world via the Telegram API.
 - **Malkuth (Kingdom):** The physical realization. The Causal Engine responsible for real-world actions (Code execution, Physics checks, Causal Inference).
## 📂 Project Structure & File Descriptions

### Core Application
-   **`desktop_assistant.py`**: The main entry point. Initializes the UI (Tkinter), starts background threads (Daydreamer, Consolidator, Telegram Polling), and coordinates the components.
-   **`ui.py`**: Contains the `DesktopAssistantUI` mixin class. Handles all Tkinter widget setup, layout, and event binding to keep the main logic clean.
-   **`settings.json`**: Configuration file storing API keys, model settings, thresholds, and prompts.

### Cognitive Modules
-   **`decider.py`**: Implements the `Decider` class. Contains logic for task switching, tool execution, and strategic thinking chains.
-   **`continuousobserver.py`**: Implements `Netzach`. Uses a separate LLM call to monitor the "State of the World" and publish events to the `EventBus`.
-   **`hod.py`**: Implements `Hod`. Analyzes system logs and memories to ensure stability and coherence.
-   **`daydreaming.py`**: Implements `Daydreamer`. Handles the autonomous research loop and insight generation.

### Memory System
-   **`memory.py`**: The primary `MemoryStore`. Manages the SQLite database for long-term memories (`memories` table) and the FAISS vector index for semantic search.
-   **`meta_memory.py`**: The `MetaMemoryStore`. Tracks events *about* memories (creation, updates, conflicts) to enable self-reflection.
-   **`reasoning.py`**: The `ReasoningStore`. A transient buffer for working memory, hypotheses, and tool outputs. Items here expire after a TTL (Time-To-Live).
-   **`memory_arbiter.py`**: The `MemoryArbiter`. A gatekeeper that evaluates new information from the Reasoning layer against confidence thresholds and precedence rules before promoting it to Long-Term Memory.
-   **`memory_consolidator.py`**: The `MemoryConsolidator`. A background process that finds duplicate or related memories and links them (versioning) to prevent database bloat.

### Document Handling (RAG)
-   **`document_store_faiss.py`**: Manages the storage of document chunks and their embeddings. Uses FAISS for fast retrieval.
-   **`document_processor.py`**: Handles file ingestion. Extracts text from PDF (via PyMuPDF) and DOCX, cleans it, and splits it into semantic chunks with overlap.

### Infrastructure
-   **`lm.py`**: The interface for the Local LLM (LM Studio). Handles API requests, vision payloads, and memory extraction prompts.
-   **`telegram_api.py`**: Low-level wrapper for Telegram Bot API requests.
-   **`event_bus.py`**: A simple Publish/Subscribe system that allows agents (Netzach, Decider, Hod) to communicate asynchronously without tight coupling.

## ✨ Key Features

### 1. Multi-Layered Memory
-   **Chat Memory**: Short-term rolling context (last 20 messages).
-   **Long-Term Memory**: Permanent storage for Facts, Goals, and Identity.
-   **Meta-Memory**: Logs of *why* and *when* memories changed.
-   **Reasoning Store**: Temporary storage for thoughts and tool outputs (TTL-based).

### 2. Autonomous Daydreaming
When not chatting, the AI enters "Daydream Mode". It will:
-   Select a random document or memory.
-   Generate new insights or hypotheses.
-   Verify existing beliefs against documents.
-   Consolidate duplicate information.

### 3. Self-Correction & Verification
The system includes a **Verifier** loop. It periodically checks "BELIEF" and "FACT" memories against source documents. If a memory is unsupported or hallucinated, it is removed, and a correction is logged in Meta-Memory.

### 4. Multi-Modal Support
You can send images via the Desktop UI or Telegram. The system uses vision-capable models (like `qwen2.5-vl`) to analyze them.

### 5. Tools
The Decider can autonomously execute:
-   **Calculator**: For math operations.
-   **Clock**: To check current time.
-   **Dice**: For RNG.
-   **System Info**: To check host machine stats.

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

| Category | Setting | Description | Default |
| :--- | :--- | :--- | :--- |
| **API** | `bot_token` | Telegram Bot Token | "" |
| | `chat_id` | Your Telegram User ID | "" |
| | `base_url` | LLM Server URL | `http://127.0.0.1:1234/v1` |
| **Models** | `chat_model` | Model identifier string | `qwen2.5-vl...` |
| | `embedding_model` | Embedding identifier string | `nomic-embed...` |
| **System** | `ai_mode` | Startup mode ("Chat" or "Daydream") | "Daydream" |
| | `telegram_bridge_enabled` | Enable/Disable Telegram | `true` |
| | `theme` | UI Theme (Darkly, Cosmo, Cyborg) | "Darkly" |
| **Generation** | `temperature` | Creativity (0.0 - 2.0) | 0.7 |
| | `max_tokens` | Response length limit | 800 |
| | `system_prompt` | Core personality instructions | (See file) |
| **Memory** | `daydream_cycle_limit` | Cycles before pausing loop | 10 |
| | `consolidation_thresholds` | Similarity required to merge | (Dict) |

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
