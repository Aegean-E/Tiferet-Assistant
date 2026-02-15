import os

# Base Paths
SETTINGS_FILE_PATH = "./settings.json"
DATA_DIR = "./data"

# Data Subdirectories
TEMP_UPLOADS_DIR = os.path.join(DATA_DIR, "temp_uploads")
UPLOADED_DOCS_DIR = os.path.join(DATA_DIR, "uploaded_docs")
BACKUPS_DIR = os.path.join(DATA_DIR, "backups")

# Files
TEMP_VOICE_INPUT_FILE = os.path.join(DATA_DIR, "temp_voice_input.wav")

def ensure_data_directories():
    """Ensure that all data directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    os.makedirs(UPLOADED_DOCS_DIR, exist_ok=True)
    os.makedirs(BACKUPS_DIR, exist_ok=True)
