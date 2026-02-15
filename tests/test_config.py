import unittest
import os
import sys

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

class TestConfig(unittest.TestCase):
    def test_base_dir_is_absolute(self):
        """Test that BASE_DIR is an absolute path."""
        self.assertTrue(os.path.isabs(config.BASE_DIR))

    def test_paths_are_relative_to_base_dir(self):
        """Test that other paths are constructed relative to BASE_DIR."""
        self.assertTrue(config.SETTINGS_FILE_PATH.startswith(config.BASE_DIR))
        self.assertTrue(config.DATA_DIR.startswith(config.BASE_DIR))

    def test_data_dir_structure(self):
        """Test that data subdirectories are inside DATA_DIR."""
        self.assertTrue(config.TEMP_UPLOADS_DIR.startswith(config.DATA_DIR))
        self.assertTrue(config.UPLOADED_DOCS_DIR.startswith(config.DATA_DIR))
        self.assertTrue(config.BACKUPS_DIR.startswith(config.DATA_DIR))
        self.assertTrue(config.LOGS_DIR.startswith(config.DATA_DIR))

if __name__ == '__main__':
    unittest.main()
