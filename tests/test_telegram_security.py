import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock imports that might be missing or problematic
sys.modules['pyaudio'] = MagicMock()
sys.modules['wave'] = MagicMock()

# Import after mocking
from desktop_assistant import DesktopAssistantApp

class TestTelegramSecurity(unittest.TestCase):
    def setUp(self):
        self.root = MagicMock()

        # Mocking tkinter variables
        patcher_bool = patch('tkinter.BooleanVar', return_value=MagicMock())
        patcher_string = patch('tkinter.StringVar', return_value=MagicMock())
        patcher_int = patch('tkinter.IntVar', return_value=MagicMock())

        self.MockBooleanVar = patcher_bool.start()
        self.MockStringVar = patcher_string.start()
        self.MockIntVar = patcher_int.start()

        self.addCleanup(patcher_bool.stop)
        self.addCleanup(patcher_string.stop)
        self.addCleanup(patcher_int.stop)

    @patch('desktop_assistant.AICore')
    @patch('desktop_assistant.AIController')
    @patch('desktop_assistant.TelegramBridge')
    @patch('desktop_assistant.DesktopAssistantUI.setup_ui')
    @patch('desktop_assistant.DesktopAssistantUI.load_settings_into_ui')
    @patch('desktop_assistant.DesktopAssistantApp.redirect_logging')
    @patch('desktop_assistant.DesktopAssistantApp.start_settings_watcher')
    @patch('desktop_assistant.DesktopAssistantApp.refresh_chat_list')
    @patch('desktop_assistant.DesktopAssistantApp.refresh_database_view')
    @patch('desktop_assistant.DesktopAssistantApp.refresh_documents')
    @patch('desktop_assistant.DesktopAssistantApp.clear_chat_display')
    @patch('desktop_assistant.DesktopAssistantApp.connect')
    @patch('desktop_assistant.DesktopAssistantApp.disconnect')
    @patch('ttkbootstrap.Style')
    def test_handle_telegram_text_security(self, MockStyle, mock_disconnect, mock_connect, mock_clear, mock_docs, mock_db, mock_chat, mock_watcher, mock_redirect, mock_load_ui, mock_setup_ui, MockBridge, MockController, MockCore):
        """Test that handle_telegram_text respects authorized chat_id"""
        # Create app instance
        app = DesktopAssistantApp(self.root)

        # Setup authorized chat_id
        authorized_chat_id = "123456789"
        app.settings["chat_id"] = authorized_chat_id

        # Mock internal methods to verify execution
        app.process_message_thread = MagicMock()
        app.add_chat_message = MagicMock()
        app.handle_disrupt_command = MagicMock()

        # 1. Test UNAUTHORIZED access
        unauthorized_chat_id = 999999999
        msg = {
            "text": "Secret command",
            "chat_id": unauthorized_chat_id,
            "from": "Hacker",
            "date": 1234567890
        }

        app.handle_telegram_text(msg)

        # Assert that it was NOT processed
        app.process_message_thread.assert_not_called()
        app.add_chat_message.assert_not_called()

        # 2. Test AUTHORIZED access
        authorized_msg = {
            "text": "Hello",
            "chat_id": int(authorized_chat_id),
            "from": "Owner",
            "date": 1234567890
        }

        app.process_message_thread.reset_mock()
        app.handle_telegram_text(authorized_msg)

        # Assert that it WAS processed
        app.process_message_thread.assert_called_once() # Should be called
        # Check arguments passed to thread
        args, _ = app.process_message_thread.call_args
        self.assertEqual(args[0], "Hello")

    @patch('desktop_assistant.AICore')
    @patch('desktop_assistant.AIController')
    @patch('desktop_assistant.TelegramBridge')
    @patch('desktop_assistant.DesktopAssistantUI.setup_ui')
    @patch('desktop_assistant.DesktopAssistantUI.load_settings_into_ui')
    @patch('desktop_assistant.DesktopAssistantApp.redirect_logging')
    @patch('desktop_assistant.DesktopAssistantApp.start_settings_watcher')
    @patch('desktop_assistant.DesktopAssistantApp.refresh_chat_list')
    @patch('desktop_assistant.DesktopAssistantApp.refresh_database_view')
    @patch('desktop_assistant.DesktopAssistantApp.refresh_documents')
    @patch('desktop_assistant.DesktopAssistantApp.clear_chat_display')
    @patch('desktop_assistant.DesktopAssistantApp.connect')
    @patch('desktop_assistant.DesktopAssistantApp.disconnect')
    @patch('ttkbootstrap.Style')
    def test_other_handlers_security(self, MockStyle, mock_disconnect, mock_connect, mock_clear, mock_docs, mock_db, mock_chat, mock_watcher, mock_redirect, mock_load_ui, mock_setup_ui, MockBridge, MockController, MockCore):
        """Test security for document, photo, and voice handlers"""
        app = DesktopAssistantApp(self.root)
        app.settings["chat_id"] = "123456789"

        # Mock internal methods
        app.telegram_bridge = MagicMock()
        app.document_processor = MagicMock()
        app.document_store = MagicMock()
        app.process_message_thread = MagicMock()

        unauthorized_chat_id = 999

        # Document Handler
        msg_doc = {"document": {"file_id": "123"}, "chat_id": unauthorized_chat_id}
        app.handle_telegram_document(msg_doc)
        app.telegram_bridge.get_file_info.assert_not_called()

        # Photo Handler
        msg_photo = {"photo": {"file_id": "123"}, "chat_id": unauthorized_chat_id}
        app.handle_telegram_photo(msg_photo)
        app.telegram_bridge.get_file_info.assert_not_called()

        # Voice Handler
        msg_voice = {"voice": {"file_id": "123"}, "chat_id": unauthorized_chat_id}
        app.handle_telegram_voice(msg_voice)
        app.telegram_bridge.get_file_info.assert_not_called()

if __name__ == '__main__':
    unittest.main()
