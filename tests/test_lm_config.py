import unittest
import sys
import os

# Ensure repo root is in path
sys.path.append(os.getcwd())

import ai_core.lm

class TestLMConfig(unittest.TestCase):
    def test_configure_lm_defaults(self):
        # Reset to empty/default first
        ai_core.lm._settings.clear()
        settings = {}
        ai_core.lm.configure_lm(settings)

        # Check defaults
        self.assertEqual(ai_core.lm.EMBEDDING_CACHE_DB, "./data/embedding_cache.sqlite3")
        self.assertEqual(ai_core.lm.LM_STUDIO_BASE_URL, "http://127.0.0.1:1234/v1")

    def test_configure_lm_custom(self):
        # Custom settings
        settings = {
            "embedding_cache_db": "./data/custom_cache.sqlite3",
            "base_url": "http://localhost:5000",
            "chat_model": "custom-model"
        }
        ai_core.lm.configure_lm(settings)

        # Check updates
        self.assertEqual(ai_core.lm.EMBEDDING_CACHE_DB, "./data/custom_cache.sqlite3")
        self.assertEqual(ai_core.lm.LM_STUDIO_BASE_URL, "http://localhost:5000")
        self.assertEqual(ai_core.lm.CHAT_MODEL, "custom-model")

if __name__ == '__main__':
    unittest.main()
