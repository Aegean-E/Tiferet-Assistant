import re
from typing import Set, List, Optional

class MemoryQualityFilter:
    """
    Strategy for filtering out low-quality memory candidates.
    Extracts logic previously embedded in ai_core/lm.py.
    """

    ERROR_PATTERNS = [
        "client error", "bad request", "local model error",
        "encountered an error", "400 client error", "500 server error",
        "generation failed", "context length"
    ]

    PURE_GREETINGS = {
        "hi", "hello", "hey", "greetings", "welcome", "howdy", "nice to meet you"
    }

    PROTECTED_TYPES = {
        "IDENTITY", "PERMISSION", "RULE", "GOAL", "BELIEF", "PREFERENCE", "REFUTED_BELIEF"
    }

    FILLER_PHRASES = {
        "what brings you here",
        "how can i help",
        "how are you",
        "what would you like",
        "can i assist",
        "is there anything",
    }

    DOCUMENT_ARTIFACTS = [
        "all rights reserved", "copyright", "page", "login", "sign up", "menu", "search"
    ]

    GENERIC_GOAL_PATTERNS = [
        "goal is to help",
        "goal is to assist",
        "goal is to support",
        "here to help",
        "here to assist",
        "help with a variety",
        "help with any",
        "assist with any",
        "available to help",
    ]

    PASSIVE_RESEARCH_PATTERNS = [
        "future investigation", "future research", "further research",
        "further investigation", "further studies", "additional studies",
        "comprehensive education", "therapeutic approaches need",
        "there is a need", "this finding may offer", "needs to be", "should focus on"
    ]

    CONTEXT_SPECIFIC_KEYWORDS = [
        "academic", "professional", "university", "medical", "research",
        "study", "studies", "work", "question", "topic", "information"
    ]

    def is_low_quality(self, text: str, mem_type: str = None) -> bool:
        """
        Main entry point to check if a memory candidate is low quality.
        """
        text_lower = text.lower().strip()
        mem_type_upper = mem_type.upper() if mem_type else None

        if self._is_question(text_lower):
            return True

        if self._is_error(text_lower):
            return True

        if self._is_pure_greeting(text_lower):
            return True

        if self._is_too_short(text_lower, mem_type_upper):
            return True

        if self._is_filler(text_lower):
            return True

        if self._is_document_artifact(text_lower):
            return True

        if mem_type_upper == "GOAL" and self._is_generic_goal(text_lower):
            return True

        return False

    def _is_question(self, text: str) -> bool:
        return text.endswith("?")

    def _is_error(self, text: str) -> bool:
        return any(p in text for p in self.ERROR_PATTERNS)

    def _is_pure_greeting(self, text: str) -> bool:
        # Check if text is exactly a greeting or starts with a greeting word as the only word
        words = text.split()
        if not words:
            return False

        is_exact = text in self.PURE_GREETINGS
        is_single_word_greeting = words[0] in self.PURE_GREETINGS and len(words) == 1

        return is_exact or is_single_word_greeting

    def _is_too_short(self, text: str, mem_type: Optional[str]) -> bool:
        word_count = len(text.split())

        # Identity facts are exempt from length check
        is_identity_fact = (
            "name is" in text or
            "lives in" in text or
            "works at" in text or
            "i am" in text or
            "user is" in text
        )

        if word_count < 3:
            # If no memory type is provided, we skip length checks (legacy behavior)
            if not mem_type:
                return False

            is_protected = mem_type in self.PROTECTED_TYPES
            if not is_protected and not is_identity_fact:
                return True

        return False

    def _is_filler(self, text: str) -> bool:
        return any(phrase in text for phrase in self.FILLER_PHRASES)

    def _is_document_artifact(self, text: str) -> bool:
        if len(text) < 30:
            return any(a in text for a in self.DOCUMENT_ARTIFACTS)
        return False

    def _is_generic_goal(self, text: str) -> bool:
        # Check explicit generic patterns
        if any(pattern in text for pattern in self.GENERIC_GOAL_PATTERNS):
            return True

        # Check for context-specific goals (e.g. "help with academic research")
        if "help with" in text or "assist with" in text:
            if any(keyword in text for keyword in self.CONTEXT_SPECIFIC_KEYWORDS):
                return True

        # Check for passive research recommendations
        if any(text.startswith(p) for p in self.PASSIVE_RESEARCH_PATTERNS):
            if "assistant" not in text and " i " not in text:
                return True

        # "The goal is to..." referring to external goal
        if text.startswith("the goal is to") and "assistant" not in text:
            return True

        return False
