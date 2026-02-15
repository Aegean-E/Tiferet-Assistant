import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE_PATH = os.path.join(BASE_DIR, "settings.json")
DATA_DIR = os.path.join(BASE_DIR, "data")

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
