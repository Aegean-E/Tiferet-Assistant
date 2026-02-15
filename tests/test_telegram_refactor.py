import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock UI libraries before importing desktop_assistant
sys.modules['tkinter'] = MagicMock()
sys.modules['tkinter.filedialog'] = MagicMock()
sys.modules['tkinter.messagebox'] = MagicMock()
sys.modules['ttkbootstrap'] = MagicMock()
sys.modules['ttkbootstrap.constants'] = MagicMock()

# Also mock imports used in ai_core if needed, but let's see.
# ai_core might use heavy libraries like faiss, numpy.
sys.modules['faiss'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['pyvis'] = MagicMock()
sys.modules['pyvis.network'] = MagicMock()
sys.modules['dowhy'] = MagicMock()
sys.modules['fitz'] = MagicMock()  # PyMuPDF
sys.modules['docx'] = MagicMock()  # python-docx

# Now import the app
from desktop_assistant import DesktopAssistantApp

class TestTelegramRefactor(unittest.TestCase):
    def setUp(self):
        # Create a mock for the root window
        self.mock_root = MagicMock()

        # Patch AI Controller, AICore, etc to avoid heavy initialization
        self.patcher_core = patch('desktop_assistant.AICore')
        self.mock_core_cls = self.patcher_core.start()
        self.mock_core = self.mock_core_cls.return_value

        self.patcher_controller = patch('desktop_assistant.AIController')
        self.mock_controller_cls = self.patcher_controller.start()

        self.patcher_bridge = patch('desktop_assistant.TelegramBridge')
        self.mock_bridge_cls = self.patcher_bridge.start()
        self.mock_bridge = self.mock_bridge_cls.return_value

        # Instantiate app
        # Since __init__ does a lot, we might need to mock open() for settings.json too
        with patch('builtins.open', unittest.mock.mock_open(read_data='{}')):
            with patch('os.path.exists', return_value=True):
                 with patch('os.replace'):  # Mock os.replace used in save_settings
                     # Also need to mock logging handlers to avoid "no handlers found" or errors
                     with patch('logging.getLogger'):
                         self.app = DesktopAssistantApp(self.mock_root)

        # Manually set the bridge on the app instance
        self.app.telegram_bridge = self.mock_bridge
        self.app.connected = True

    def tearDown(self):
        self.patcher_core.stop()
        self.patcher_controller.stop()
        self.patcher_bridge.stop()

    def test_handle_telegram_photo(self):
        # Setup mock return values
        file_id = "test_photo_id"
        file_path = "photos/test_photo.jpg"
        self.mock_bridge.get_file_info.return_value = {"file_path": file_path}

        # Helper for add_chat_message which is called in the method
        self.app.add_chat_message = MagicMock()
        self.app.process_message_thread = MagicMock()

        msg = {
            "photo": {"file_id": file_id},
            "caption": "Test caption",
            "chat_id": 12345,
            "from": "User"
        }

        # Call the method
        self.app.handle_telegram_photo(msg)

        # Verify get_file_info called
        self.mock_bridge.get_file_info.assert_called_with(file_id)

        # Verify download_file called
        # The exact temp path depends on implementation (timestamp or ID)
        # But we know it should end with .jpg and contain the ID
        args, _ = self.mock_bridge.download_file.call_args
        telegram_path_arg, local_path_arg = args

        self.assertEqual(telegram_path_arg, file_path)
        self.assertIn(file_id, local_path_arg)
        self.assertTrue(local_path_arg.endswith('.jpg'))

    def test_handle_telegram_document(self):
        file_id = "test_doc_id"
        file_name = "test_doc.pdf"
        file_path = "docs/test_doc.pdf"
        self.mock_bridge.get_file_info.return_value = {"file_path": file_path}

        self.app.document_store = MagicMock()
        self.app.document_store.compute_file_hash.return_value = "hash123"
        self.app.document_store.document_exists.return_value = False

        self.app.document_processor = MagicMock()
        self.app.document_processor.process_document.return_value = ([], 1, "pdf")

        msg = {
            "document": {
                "file_id": file_id,
                "file_name": file_name,
                "file_size": 1000
            },
            "chat_id": 12345
        }

        # Mock os.makedirs and os.path.join if needed, but os.makedirs is safe to run if we mock os.path.join?
        # Actually, let's just let os.makedirs run or mock it if it tries to create real dirs.
        # But we don't want real dirs created in tests.
        with patch('os.makedirs') as mock_makedirs:
            with patch('os.remove'): # handle_telegram_document removes the file at the end
                self.app.handle_telegram_document(msg)

        self.mock_bridge.get_file_info.assert_called_with(file_id)

        args, _ = self.mock_bridge.download_file.call_args
        telegram_path_arg, local_path_arg = args

        self.assertEqual(telegram_path_arg, file_path)
        self.assertTrue(local_path_arg.endswith(file_name))

    def test_handle_telegram_voice(self):
        file_id = "test_voice_id"
        file_path = "voice/test_voice.ogg"
        self.mock_bridge.get_file_info.return_value = {"file_path": file_path}

        # Mock transcribe_audio
        with patch('desktop_assistant.transcribe_audio', return_value="Transcribed text"):
            with patch('os.makedirs') as mock_makedirs:
                 with patch('os.remove'):
                    self.app.add_chat_message = MagicMock()
                    self.app.process_message_thread = MagicMock()

                    msg = {
                        "voice": {"file_id": file_id},
                        "chat_id": 12345,
                        "from": "User"
                    }

                    self.app.handle_telegram_voice(msg)

        self.mock_bridge.get_file_info.assert_called_with(file_id)

        args, _ = self.mock_bridge.download_file.call_args
        telegram_path_arg, local_path_arg = args

        self.assertEqual(telegram_path_arg, file_path)
        self.assertIn("voice_", local_path_arg)
        self.assertTrue(local_path_arg.endswith('.ogg'))

if __name__ == '__main__':
    unittest.main()
