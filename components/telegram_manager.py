import threading
import logging
import tkinter as tk
from tkinter import messagebox
import time
import os
from bridges.telegram_api import TelegramBridge
from ai_core.lm import transcribe_audio

class TelegramManager:
    def __init__(self, app):
        self.app = app
        self.bridge = None
        self.connected = False
        self.status_sent = False # Track if status has been sent to avoid spam

    def toggle_connection(self):
        """Toggle connection to Telegram"""
        # Toggle the setting
        with self.app.settings_lock:
            new_state = not self.app.telegram_bridge_enabled.get()
            self.app.telegram_bridge_enabled.set(new_state)
            # Save the new state
            self.app.settings["telegram_bridge_enabled"] = new_state
            self.app.save_settings()

        # Update connection based on new state
        if new_state:
            # Only connect if both credentials are provided
            bot_token = self.app.bot_token_var.get().strip()
            chat_id_str = self.app.chat_id_var.get().strip()
            if bot_token and chat_id_str:
                try:
                    int(chat_id_str)  # Validate chat ID is numeric
                    self.connect()
                except ValueError:
                    messagebox.showerror("Connection Error", "Chat ID must be a valid number")
                    self.app.telegram_bridge_enabled.set(False)
                    with self.app.settings_lock:
                        self.app.settings["telegram_bridge_enabled"] = False
                        self.app.save_settings()
            else:
                messagebox.showerror("Connection Error", "Please enter both Bot Token and Chat ID in Settings")
                self.app.telegram_bridge_enabled.set(False)
                with self.app.settings_lock:
                    self.app.settings["telegram_bridge_enabled"] = False
                    self.app.save_settings()
        else:
            self.disconnect()

    def connect(self):
        """Connect to Telegram (Non-blocking)"""
        if self.is_connected():
            return  # Already connected

        bot_token = self.app.bot_token_var.get().strip()
        chat_id_str = self.app.chat_id_var.get().strip()

        if not bot_token or not chat_id_str:
            return

        def connection_worker():
            try:
                chat_id = int(chat_id_str)
                bridge = TelegramBridge(bot_token, chat_id, log_fn=self.app.log_to_main)

                # Test connection (Blocks thread, but not UI)
                if bridge.send_message("✅ Connected to Desktop Assistant"):
                    self.bridge = bridge
                    self.connected = True

                    # Update UI on main thread
                    if hasattr(self.app, 'connect_button'):
                        self.app.root.after(0, lambda: self.app.connect_button.config(text="Connected", bootstyle="success"))
                    self.app.root.after(0, lambda: self.app.status_var.set("Connected to Telegram"))

                    # Start message polling
                    threading.Thread(
                        target=self.bridge.listen,
                        kwargs={
                            "on_text": self.handle_telegram_text,
                            "on_document": lambda m: threading.Thread(target=self.handle_telegram_document, args=(m,), daemon=True).start(),
                            "on_photo": self.handle_telegram_photo,
                            "on_voice": lambda m: threading.Thread(target=self.handle_telegram_voice, args=(m,), daemon=True).start(),
                            "running_check": lambda: self.is_connected() and self.app.settings.get("telegram_bridge_enabled", False),
                            "start_timestamp": self.app.start_time
                        },
                        daemon=True
                    ).start()
                else:
                    raise Exception("Failed to send test message")

            except Exception as e:
                logging.error(f"Telegram connection error: {e}")
                if self.app.settings.get("telegram_bridge_enabled", False):
                    self.app.root.after(0, lambda: messagebox.showerror("Connection Error", f"Failed to connect: {e}"))
                self.app.root.after(0, self.disconnect)

        threading.Thread(target=connection_worker, daemon=True).start()

    def disconnect(self):
        """Disconnect from Telegram"""
        self.connected = False
        self.bridge = None
        if hasattr(self.app, 'connect_button'):
            self.app.connect_button.config(text="Connect", bootstyle="secondary")
        if hasattr(self.app, 'status_var'):
            self.app.status_var.set("Disconnected from Telegram")

    def is_connected(self):
        """Check if connected to Telegram"""
        return self.connected and self.bridge is not None

    def send_telegram_status(self, message: str):
        """Send a status update to Telegram if connected"""
        if self.is_connected() and self.app.settings.get("telegram_bridge_enabled", False):
             # Suppress repetitive status messages until user interacts
             if self.status_sent:
                 return

             if self.bridge.send_message(message):
                 if "finished" in message.lower():
                     self.status_sent = True

    def send_message(self, message):
         if self.is_connected():
             return self.bridge.send_message(message)
         return False

    def close(self):
        if self.bridge:
            self.bridge.close()

    def handle_disrupt_command(self, chat_id):
        """Handle /disrupt command from Telegram to stop processing immediately"""
        logging.info("🛑 Disrupt command received from Telegram.")
        if self.bridge:
            self.bridge.send_message("🛑 Disrupting current process...")

        if self.app.controller:
            self.app.controller.stop_processing_flag = True

        if hasattr(self.app, 'ai_core') and self.app.ai_core and self.app.ai_core.decider:
            self.app.ai_core.decider.report_forced_stop()

        def reset_flag():
            time.sleep(1.5)
            if self.app.controller:
                self.app.controller.stop_processing_flag = False
            logging.info("▶️ Decider ready for next turn (Cooldown active).")
            if self.bridge:
                self.bridge.send_message("▶️ Process disrupted. Decider is in cooldown.")

        threading.Thread(target=reset_flag, daemon=True).start()

    def handle_telegram_text(self, msg):
        """Handle text message from Telegram"""
        # Reset status suppression on interaction
        self.status_sent = False

        # Check for disrupt command OR implicit disrupt on any message
        text_content = msg.get("text", "").strip().lower()
        is_explicit_disrupt = text_content == "/disrupt"

        if is_explicit_disrupt:
            self.handle_disrupt_command(msg["chat_id"])
            return

        # Show in UI
        self.app.root.after(0, lambda m=msg: self.app.add_chat_message(m["from"], m["text"], "incoming"))
        # Process logic
        threading.Thread(
            target=self.app.process_message_thread,
            args=(msg["text"], False, msg["chat_id"]), # Use actual chat_id from msg
            daemon=True
        ).start()

    def handle_telegram_photo(self, msg):
        """Handle photo from Telegram"""
        try:
            file_id = msg["photo"]["file_id"]
            caption = msg.get("caption", "") or "Analyze this image."

            # Download to temp
            temp_path = f"./data/temp_img_{file_id}.jpg"
            file_data = self.bridge.get_file_info(file_id)
            self.bridge.download_file(file_data["file_path"], temp_path)

            self.app.root.after(0, lambda m=msg, c=caption, p=temp_path: self.app.add_chat_message(m["from"], c, "incoming", image_path=p))

            threading.Thread(
                target=self.app.process_message_thread,
                args=(caption, False, msg["chat_id"], temp_path), # Use actual chat_id from msg
                daemon=True
            ).start()
        except Exception as e:
            logging.error(f"Error handling photo: {e}")

    def handle_telegram_voice(self, msg):
        """Handle voice message from Telegram"""
        try:
            file_id = msg["voice"]["file_id"]
            chat_id = msg["chat_id"]

            self.app.root.after(0, lambda: self.app.status_var.set("🎙️ Receiving voice message..."))

            file_data = self.bridge.get_file_info(file_id)
            telegram_file_path = file_data["file_path"]

            temp_dir = "./data/temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            local_file_path = os.path.join(temp_dir, f"voice_{file_id}.ogg")

            self.bridge.download_file(telegram_file_path, local_file_path)

            self.app.root.after(0, lambda: self.app.status_var.set("📝 Transcribing voice..."))
            text = transcribe_audio(local_file_path)

            if text and not text.startswith("[Error"):
                self.app.root.after(0, lambda: self.app.add_chat_message(msg["from"], f"🎙️ {text}", "incoming"))
                threading.Thread(
                    target=self.app.process_message_thread,
                    args=(text, False, chat_id),
                    daemon=True
                ).start()
            else:
                self.bridge.send_message(f"⚠️ Sorry, I couldn't transcribe that voice message: {text}")

            if os.path.exists(local_file_path):
                os.remove(local_file_path)

        except Exception as e:
            logging.error(f"Error handling Telegram voice: {e}")
            if self.bridge:
                self.bridge.send_message(f"❌ Error processing voice message: {str(e)}")
        finally:
            self.app.root.after(0, lambda: self.app.status_var.set("Ready"))

    def handle_telegram_document(self, msg):
        """Handle document upload from Telegram"""
        try:
            file_info = msg["document"]
            file_id = file_info["file_id"]
            file_name = file_info.get("file_name", "unknown_file")
            chat_id = msg["chat_id"]

            # Check supported types
            if not file_name.lower().endswith(('.pdf', '.docx')):
                self.bridge.send_message(f"⚠️ Unsupported file type: {file_name}. Please send PDF or DOCX.")
                return

            self.bridge.send_message(f"📄 Received {file_name}, processing...")

            # Get file path from Telegram
            file_data = self.bridge.get_file_info(file_id)
            telegram_file_path = file_data["file_path"]

            # Download
            local_dir = "./data/uploaded_docs"
            os.makedirs(local_dir, exist_ok=True)
            local_file_path = os.path.join(local_dir, file_name)

            self.bridge.download_file(telegram_file_path, local_file_path)

            # Ingest using common logic (delegated to app)
            result = self.app._ingest_document(local_file_path, upload_source="telegram", original_filename=file_name)

            if result['status'] == 'duplicate':
                self.bridge.send_message(f"⚠️ Document '{file_name}' already exists in database. Skipping...")
            elif result['status'] == 'success':
                self.bridge.send_message(f"✅ Successfully added '{file_name}' to database ({result['chunks_count']} chunks).")
                self.app.root.after(0, self.app.refresh_documents)

        except Exception as e:
            logging.error(f"Error handling Telegram document: {e}")
            if self.bridge:
                self.bridge.send_message(f"❌ Error processing document: {str(e)}")
        finally:
            if 'local_file_path' in locals() and os.path.exists(local_file_path):
                os.remove(local_file_path)
