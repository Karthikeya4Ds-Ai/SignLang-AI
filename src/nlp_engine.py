"""
Smart NLP & Sentence Reconstruction Engine
Converts raw, rolling frame predictions into coherent words and sentences
with debounce filtering, letter accumulation, auto-spacing, and undo support.
"""

from collections import deque, Counter
import time
from typing import List, Optional, Callable, Tuple


class SignSentenceEngine:
    """
    State machine that manages token debounce consensus, fingerspelling word assembly,
    and sentence reconstruction.
    """

    def __init__(
        self,
        debounce_window: int = 10,
        consensus_ratio: float = 0.70,
        idle_duration_for_space: float = 1.8,
        on_token_callback: Optional[Callable[[str, str], None]] = None,
        on_sentence_callback: Optional[Callable[[str], None]] = None
    ):
        self.debounce_window = debounce_window
        self.consensus_ratio = consensus_ratio
        self.idle_duration_for_space = idle_duration_for_space
        self.on_token_callback = on_token_callback
        self.on_sentence_callback = on_sentence_callback

        self._prediction_buffer = deque(maxlen=debounce_window)
        self._current_word = ""
        self._sentence_tokens: List[str] = []
        self._sentence_history: List[str] = []

        self._last_confirmed_token: Optional[str] = None
        self._last_token_timestamp = 0.0
        self._idle_start_time: Optional[float] = None
        self._space_added_for_current_idle = False

    def update(self, predicted_class: str, confidence: float) -> Tuple[Optional[str], str]:
        """
        Receives a single frame prediction.

        Returns:
            Tuple of (confirmed_token_if_any, current_full_sentence_string)
        """
        now = time.time()
        self._prediction_buffer.append(predicted_class)

        # Require a full buffer for consensus
        if len(self._prediction_buffer) < self.debounce_window:
            return None, self.get_full_display_text()

        # Compute consensus (mode of recent buffer)
        counts = Counter(self._prediction_buffer)
        most_common_class, count = counts.most_common(1)[0]
        consensus_fraction = count / self.debounce_window

        if consensus_fraction < self.consensus_ratio:
            return None, self.get_full_display_text()

        confirmed_class = most_common_class

        # Handle resting / _idle_ state
        if confirmed_class == "_idle_":
            if self._idle_start_time is None:
                self._idle_start_time = now
            elif (now - self._idle_start_time) >= self.idle_duration_for_space:
                # If idle has lasted sufficiently and we have an ongoing word, flush to sentence
                if self._current_word and not self._space_added_for_current_idle:
                    self._commit_current_word()
                    self._space_added_for_current_idle = True
            self._last_confirmed_token = "_idle_"
            return None, self.get_full_display_text()

        # Reset idle timer when active gesture appears
        self._idle_start_time = None
        self._space_added_for_current_idle = False

        # Prevent duplicate firing if user keeps holding the same sign
        # Only accept if different from last confirmed token OR enough time elapsed (1.2s debounce)
        if confirmed_class == self._last_confirmed_token and (now - self._last_token_timestamp) < 1.2:
            return None, self.get_full_display_text()

        self._last_confirmed_token = confirmed_class
        self._last_token_timestamp = now

        # Process the new confirmed gesture
        self._dispatch_token(confirmed_class)

        return confirmed_class, self.get_full_display_text()

    def _dispatch_token(self, token: str):
        """Processes a single verified token into words or dynamic sentences."""
        token = token.strip()
        if not token or token == "_idle_":
            return

        # Case 1: Single letter fingerspelling (A-Z)
        if len(token) == 1 and token.isalpha():
            self._current_word += token.upper()
            if self.on_token_callback:
                self.on_token_callback(token.upper(), "letter")

        # Case 2: Number (0-9)
        elif len(token) == 1 and token.isdigit():
            self._current_word += token
            if self.on_token_callback:
                self.on_token_callback(token, "digit")

        # Case 3: Dynamic gesture (e.g. Hello, Water, Food, Goodbye, etc.)
        else:
            # If we had letters accumulated, commit that word first
            if self._current_word:
                self._commit_current_word()

            self._sentence_tokens.append(token)
            if self.on_token_callback:
                self.on_token_callback(token, "word")

    def _commit_current_word(self):
        """Commits the ongoing fingerspelled word into the sentence tokens."""
        if self._current_word:
            self._sentence_tokens.append(self._current_word)
            word = self._current_word
            self._current_word = ""
            if self.on_token_callback:
                self.on_token_callback(word, "word_completed")

    def add_space(self):
        """Manually commits current word with a space delimiter."""
        self._commit_current_word()

    def backspace(self):
        """Deletes the last character or word."""
        if self._current_word:
            self._current_word = self._current_word[:-1]
        elif self._sentence_tokens:
            self._sentence_tokens.pop()

    def clear(self):
        """Clears the active sentence and uncommitted word."""
        full_text = self.get_full_display_text()
        if full_text:
            self._sentence_history.append(full_text)
            if self.on_sentence_callback:
                self.on_sentence_callback(full_text)
        self._sentence_tokens.clear()
        self._current_word = ""
        self._prediction_buffer.clear()
        self._last_confirmed_token = None

    def get_full_display_text(self) -> str:
        """Returns the full sentence with ongoing uncommitted word appended."""
        words = list(self._sentence_tokens)
        if self._current_word:
            words.append(self._current_word + "_")  # Visual cursor indicator
        return " ".join(words)

    def get_completed_sentence(self) -> str:
        """Returns the finalized sentence string."""
        words = list(self._sentence_tokens)
        if self._current_word:
            words.append(self._current_word)
        return " ".join(words)

    def get_history(self) -> List[str]:
        """Returns previous archived sentences."""
        return list(self._sentence_history)
