import requests
import os
from datetime import datetime
import time
from typing import List, Dict, Callable
import logging

BASE_URL = "https://api.telegram.org/bot"

class TelegramBridge:
    """Handles communication with Telegram API"""

    def __init__(self, bot_token: str, chat_id: int, log_fn: Callable[[str], None] = logging.info):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.log = log_fn
        self.is_connected = False
        self.last_update_id = None
        self.session = requests.Session()

    def _send_message_chunk(self, text):
        url = f"{BASE_URL}{self.bot_token}/sendMessage"
        response = self.session.post(url, json={
            "chat_id": self.chat_id,
            "text": text
        }, timeout=30)
        response.raise_for_status()

    def send_message(self, text: str) -> bool:
        """Send message to Telegram"""
        try:
            if not text: return False
            limit = 3072
            for i in range(0, len(text), limit):
                chunk = text[i:i + limit]
                self._send_message_chunk(chunk)
            return True
        except Exception as e:
            self.log(f"❌ Telegram send error: {e}")
            return False

    def get_messages(self) -> List[Dict]:
        """Get new messages from Telegram"""
        try:
            # Use offset + 1 to confirm processed messages to Telegram
            offset = self.last_update_id + 1 if self.last_update_id is not None else None
            
            url = f"{BASE_URL}{self.bot_token}/getUpdates"
            params = {"timeout": 40}
            if offset is not None: params["offset"] = offset
            
            response = self.session.get(url, params=params, timeout=45)
            response.raise_for_status()
            updates = response.json()
            
            messages = []

            for update in updates.get("result", []):
                # Update last_update_id to the current update's ID
                self.last_update_id = update["update_id"]

                if "message" in update:
                    msg = update["message"]
                    
                    # Base message data
                    message_data = {
                        "id": update["update_id"],
                        "chat_id": msg.get("chat", {}).get("id"),
                        "from": msg.get("from", {}).get("first_name", "Unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "date": msg.get("date", int(time.time())), # Capture Telegram timestamp
                        "type": "unknown"
                    }

                    if "text" in msg:
                        message_data["type"] = "text"
                        message_data["text"] = msg["text"]
                    elif "document" in msg:
                        message_data["type"] = "document"
                        message_data["document"] = msg["document"]
                    elif "photo" in msg:
                        message_data["type"] = "photo"
                        # Telegram sends multiple sizes; take the last one (highest quality)
                        message_data["photo"] = msg["photo"][-1]
                        message_data["caption"] = msg.get("caption", "")
                    elif "voice" in msg:
                        message_data["type"] = "voice"
                        message_data["voice"] = msg["voice"]
                    
                    if message_data["type"] != "unknown":
                        messages.append(message_data)
            return messages
        except Exception as e:
            self.log(f"❌ Telegram receive error: {e}")
            return []

    def get_file_info(self, file_id: str) -> Dict:
        """Get file metadata from Telegram"""
        url = f"{BASE_URL}{self.bot_token}/getFile"
        response = self.session.get(url, params={"file_id": file_id}, timeout=30)
        response.raise_for_status()
        return response.json()['result']

    def download_file(self, file_path: str, save_path: str) -> bool:
        """Download file from Telegram"""
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        response = self.session.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    def listen(self, 
               on_text: Callable[[Dict], None], 
               on_document: Callable[[Dict], None],
               on_photo: Callable[[Dict], None],
               on_voice: Callable[[Dict], None],
               running_check: Callable[[], bool],
               start_timestamp: float = 0):
        """
        Poll for messages and dispatch to callbacks.
        Runs until running_check() returns False.
        """
        self.log("🔌 Telegram Bridge: Listening for messages...")
        while running_check():
            try:
                messages = self.get_messages()
                for msg in messages:
                    # Ignore old messages
                    if msg.get("date", 0) < start_timestamp:
                        self.log(f"⚠️ Ignoring old message from {msg.get('from')}: {msg.get('text', 'doc')}")
                        continue

                    msg_type = msg.get("type")
                    if msg_type == "text":
                        on_text(msg)
                    elif msg_type == "document":
                        on_document(msg)
                    elif msg_type == "photo":
                        on_photo(msg)
                    elif msg_type == "voice":
                        on_voice(msg)
                
                time.sleep(1.0)
            except Exception as e:
                self.log(f"Error polling messages: {e}")
                time.sleep(1)
        self.log("🔌 Telegram Bridge: Stopped listening.")
