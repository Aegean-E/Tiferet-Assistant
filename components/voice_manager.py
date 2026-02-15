import threading
import wave
import os
import logging
import tkinter as tk
from tkinter import messagebox
from ai_core.lm import transcribe_audio
import config

# Audio recording
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    pyaudio = None

class VoiceManager:
    def __init__(self, app):
        self.app = app
        self.is_recording = False
        self.audio_frames = []

    def toggle_recording(self):
        """Toggle voice recording state."""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "PyAudio not installed. Cannot record.")
            return

        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        if hasattr(self.app, 'voice_btn'):
            self.app.voice_btn.config(text="⏹️", bootstyle="danger")
        self.audio_frames = []

        # Start blinking effect
        self._blink_recording_indicator()

        def record_loop():
            try:
                chunk = 1024
                format = pyaudio.paInt16
                channels = 1
                rate = 44100

                p = pyaudio.PyAudio()
                stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

                while self.is_recording:
                    data = stream.read(chunk)
                    self.audio_frames.append(data)

                stream.stop_stream()
                stream.close()
                p.terminate()

                # Save to file
                temp_wav = config.TEMP_VOICE_INPUT_FILE
                wf = wave.open(temp_wav, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(format))
                wf.setframerate(rate)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()

                # Transcribe
                self.app.root.after(0, lambda: self.app.status_var.set("Transcribing voice..."))
                text = transcribe_audio(temp_wav)

                # Insert into entry
                self.app.root.after(0, lambda: self.app.message_entry.insert(tk.END, text + " "))
                self.app.root.after(0, lambda: self.app.status_var.set("Ready"))

                # Cleanup
                try: os.remove(temp_wav)
                except: pass
            except Exception as e:
                logging.error(f"Voice recording error: {e}")
                self.app.root.after(0, lambda: messagebox.showerror("Recording Error", f"Failed to record audio: {e}"))
                self.app.root.after(0, self.stop_recording)

        threading.Thread(target=record_loop, daemon=True).start()

    def _blink_recording_indicator(self):
        """Blink the recording button to indicate activity."""
        if not self.is_recording:
            return

        if hasattr(self.app, 'voice_btn'):
            current_text = self.app.voice_btn.cget("text")
            # Toggle between Stop icon and Red Circle
            new_text = "🔴" if current_text == "⏹️" else "⏹️"
            self.app.voice_btn.config(text=new_text)

        self.app.root.after(500, self._blink_recording_indicator)

    def stop_recording(self):
        self.is_recording = False
        if hasattr(self.app, 'voice_btn'):
            self.app.voice_btn.config(text="🎤", bootstyle="default", style="Big.Link.TButton")
