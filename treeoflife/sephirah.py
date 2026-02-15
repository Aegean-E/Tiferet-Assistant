from abc import ABC
from typing import Callable, Dict, Optional, Any
import logging

class Sephirah(ABC):
    """
    Abstract Base Class for all Sephirot (Cognitive Modules).
    Enforces structural coherence across the Tree of Life.
    """
    def __init__(self, name: str, description: str, log_fn: Callable[[str], None] = logging.info, event_bus: Optional[Any] = None):
        self.name = name
        self.description = description
        self.log = log_fn
        self.event_bus = event_bus
        self.log_event("Initialized.")

    def log_event(self, message: str):
        """Standardized logging format."""
        if self.log:
            # Emoji prefix based on Sephirah name for quick visual identification
            emoji = self._get_emoji()
            self.log(f"{emoji} {self.name}: {message}")

    def _get_emoji(self) -> str:
        """Return the emoji associated with the Sephirah."""
        emojis = {
            "Keter": "👑",
            "Chokmah": "☁️",
            "Binah": "🧠",
            "Daat": "✨",
            "Hesed": "🌊",
            "Gevurah": "🔥",
            "Tiferet": "🤖",
            "Netzach": "👁️",
            "Hod": "🔮",
            "Yesod": "📝",
            "Malkuth": "🌍"
        }
        return emojis.get(self.name, "🔹")

    def shutdown(self):
        """
        Graceful shutdown hook. Override if resources need cleanup.
        """
        self.log_event("Shutting down...")

    def status(self) -> Dict[str, Any]:
        """
        Return current operational status. Override to provide specific metrics.
        """
        return {"name": self.name, "active": True}
