import requests
import os
from datetime import datetime
import time
from typing import List, Dict, Callable

BASE_URL = "https://api.telegram.org/bot"

def get_updates(bot_token, offset=None, timeout=40):
    url = f"{BASE_URL}{bot_token}/getUpdates"

    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset

    response = requests.get(url, params=params, timeout=timeout + 5)
    response.raise_for_status()

    return response.json()

def send_message(bot_token, chat_id, text):
    url = f"{BASE_URL}{bot_token}/sendMessage"

    response = requests.post(url, json={
        "chat_id": chat_id,
        "text": text
    }, timeout=30)

    response.raise_for_status()

def send_long_message(bot_token, chat_id, text, limit=3072):
    """
    Splits long messages into chunks safe for Telegram (4096 char limit).
    """
    if not text:
        return

    for i in range(0, len(text), limit):
        chunk = text[i:i + limit]
        send_message(bot_token, chat_id, chunk)

def get_file(bot_token, file_id):
    """
    Get file metadata from Telegram.
    
    Returns:
        dict with 'file_path', 'file_size', etc.
    """
    url = f"{BASE_URL}{bot_token}/getFile"

    response = requests.get(url, params={"file_id": file_id}, timeout=30)
    response.raise_for_status()

    return response.json()['result']

def download_file(bot_token, file_path, save_path):
    """
    Download file from Telegram servers.
    
    Args:
        file_path: Telegram file_path from getFile
        save_path: Local path to save file
    
    Returns:
        True if successful
    """
    url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return True

class TelegramBridge:
    """Handles communication with Telegram API"""

    def __init__(self, bot_token: str, chat_id: int):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.is_connected = False
        self.last_update_id = None

    def send_message(self, text: str) -> bool:
        """Send message to Telegram"""
        try:
            send_long_message(self.bot_token, self.chat_id, text)
            return True
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
            return False

    def get_messages(self) -> List[Dict]:
        """Get new messages from Telegram"""
        try:
            # Use offset + 1 to confirm processed messages to Telegram
            offset = self.last_update_id + 1 if self.last_update_id is not None else None
            updates = get_updates(self.bot_token, offset=offset)
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
                    
                    if message_data["type"] != "unknown":
                        messages.append(message_data)
            return messages
        except Exception as e:
            print(f"❌ Telegram receive error: {e}")
            return []

    def get_file_info(self, file_id: str) -> Dict:
        """Get file metadata from Telegram"""
        return get_file(self.bot_token, file_id)

    def download_file(self, file_path: str, save_path: str) -> bool:
        """Download file from Telegram"""
        return download_file(self.bot_token, file_path, save_path)

    def listen(self, 
               on_text: Callable[[Dict], None], 
               on_document: Callable[[Dict], None],
               on_photo: Callable[[Dict], None],
               running_check: Callable[[], bool],
               start_timestamp: float = 0):
        """
        Poll for messages and dispatch to callbacks.
        Runs until running_check() returns False.
        """
        print("🔌 Telegram Bridge: Listening for messages...")
        while running_check():
            try:
                messages = self.get_messages()
                for msg in messages:
                    # Ignore old messages
                    if msg.get("date", 0) < start_timestamp:
                        print(f"⚠️ Ignoring old message from {msg.get('from')}: {msg.get('text', 'doc')}")
                        continue

                    msg_type = msg.get("type")
                    if msg_type == "text":
                        on_text(msg)
                    elif msg_type == "document":
                        on_document(msg)
                    elif msg_type == "photo":
                        on_photo(msg)
                
                time.sleep(0.01)
            except Exception as e:
                print(f"Error polling messages: {e}")
                time.sleep(1)
        print("🔌 Telegram Bridge: Stopped listening.")
