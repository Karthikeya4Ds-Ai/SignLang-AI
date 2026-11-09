"""
Asynchronous, Non-Blocking Text-to-Speech (TTS) Engine
Uses pyttsx3 in a background daemon thread so speech synthesis never locks
the real-time OpenCV camera loop or drops video frames.
"""

import threading
import queue
import time
from typing import Optional, List, Dict, Any


class AsyncTTSEngine:
    """Thread-safe, non-blocking offline TTS engine."""

    def __init__(self, rate: int = 155, volume: float = 1.0, voice_index: int = 0, enabled: bool = True):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self._enabled = enabled
        self._muted = False

        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_spoken_text = ""
        self._last_spoken_time = 0.0

        if self._enabled:
            self._start_worker()

    def _start_worker(self):
        """Starts the background worker thread."""
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="SignLangTTSThread")
        self._thread.start()

    def _worker_loop(self):
        """Dedicated loop running in its own thread with its own COM/pyttsx3 instance."""
        engine = None
        try:
            import pyttsx3
            # Initialize COM and engine inside this specific thread
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)

            voices = engine.getProperty('voices')
            if voices and 0 <= self.voice_index < len(voices):
                engine.setProperty('voice', voices[self.voice_index].id)
        except Exception as e:
            print(f"[TTS Warning] Could not initialize pyttsx3 engine: {e}. Audio output disabled.")
            engine = None

        while not self._stop_event.is_set():
            try:
                # Wait for utterance with a 0.2s timeout
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None or self._stop_event.is_set():
                break

            if not self._muted and engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[TTS Error] Speech playback error: {e}")

            self._queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speak(self, text: str, min_interval: float = 1.0):
        """
        Submits text to the non-blocking speech queue.
        Filters out rapid duplicates within min_interval seconds.
        """
        text = str(text).strip()
        if not text or self._muted or not self._enabled:
            return

        now = time.time()
        if text.lower() == self._last_spoken_text.lower() and (now - self._last_spoken_time) < min_interval:
            return

        self._last_spoken_text = text
        self._last_spoken_time = now
        self._queue.put(text)

    def toggle_mute(self) -> bool:
        """Toggles audio mute state. Returns the new muted status."""
        self._muted = not self._muted
        return self._muted

    def set_muted(self, muted: bool):
        """Sets audio mute state."""
        self._muted = muted

    def is_muted(self) -> bool:
        """Returns True if audio is muted."""
        return self._muted

    def clear_queue(self):
        """Flushes any pending unspoken utterances."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def stop(self):
        """Gracefully halts the background thread."""
        self._stop_event.set()
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
