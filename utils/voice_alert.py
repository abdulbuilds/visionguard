"""
=============================================================================
  voice_alert.py  —  Thread-safe edge-tts + pygame voice alert engine
=============================================================================
"""
import threading
import queue
import time
import asyncio
import uuid
import os
from typing import Optional

try:
    import edge_tts
    import pygame
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

FINAL_LABELS = {
    0: "Speed limit 5 km/h",
    1: "Speed limit 15 km/h",
    2: "Speed limit 100 km/h",
    3: "Speed limit 120 km/h",
    4: "Traffic signals",
    5: "Zebra crossing",
    6: "Stop",
    7: "Speed limit 20 km/h",
    8: "Speed limit 30 km/h",
    9: "Speed limit 40 km/h",
    10: "Speed limit 50 km/h",
    11: "Speed limit 60 km/h",
    12: "Speed limit 70 km/h",
    13: "Speed limit 80 km/h",
    14: "End of speed limit 80 km/h"
}

MESSAGES = {
    "Speed limit 5 km/h": "Speed limit 5 kilometers per hour. Please slow down.",
    "Speed limit 15 km/h": "Speed limit 15 kilometers per hour. Please slow down.",
    "Speed limit 100 km/h": "Speed limit 100 kilometers per hour. Stay within limit.",
    "Speed limit 120 km/h": "Speed limit 120 kilometers per hour. Stay within limit.",
    "Traffic signals": "Traffic signals ahead. Be prepared to stop.",
    "Zebra crossing": "Zebra crossing ahead. Watch out for pedestrians.",
    "Stop": "Stop sign ahead. Please stop your vehicle.",
    "Speed limit 20 km/h": "Speed limit 20 kilometers per hour. Please slow down.",
    "Speed limit 30 km/h": "Speed limit 30 kilometers per hour. Please slow down.",
    "Speed limit 40 km/h": "Speed limit 40 kilometers per hour. Please slow down.",
    "Speed limit 50 km/h": "Speed limit 50 kilometers per hour. Reduce your speed.",
    "Speed limit 60 km/h": "Speed limit 60 kilometers per hour. Stay within limit.",
    "Speed limit 70 km/h": "Speed limit 70 kilometers per hour. Stay within limit.",
    "Speed limit 80 km/h": "Speed limit 80 kilometers per hour. Stay within limit.",
    "End of speed limit 80 km/h": "End of speed limit 80 kilometers per hour.",
}

class VoiceAlertEngine:
    """
    Singleton-style TTS engine that runs in a dedicated background thread.
    Uses edge-tts and pygame to play audio locally.
    """
    
    DEBOUNCE_SECONDS: float = 3.0

    def __init__(self, rate: int = 155, volume: float = 1.0):
        # rate and volume kept for backward compatibility with app.py
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._last_spoken: dict[str, float] = {}   # class_name → timestamp
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._voice = "en-US-JennyNeural"

    def is_available(self) -> bool:
        return _TTS_AVAILABLE

    def start(self) -> None:
        if not _TTS_AVAILABLE:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

    def speak(self, class_name: str) -> bool:
        if not _TTS_AVAILABLE or not self._running:
            return False

        now = time.time()
        last = self._last_spoken.get(class_name, 0.0)
        if now - last < self.DEBOUNCE_SECONDS:
            return False

        self._last_spoken[class_name] = now
        
        # Determine the exact message
        msg = MESSAGES.get(class_name, f"{class_name.replace('_', ' ')} detected. Please follow traffic rules.")
        self._queue.put(msg)
        return True

    def speak_all(self, detections: list[dict]) -> list[str]:
        announced = []
        seen = set()
        for d in detections:
            name = d.get("class_name", "")
            if name and name not in seen:
                seen.add(name)
                if self.speak(name):
                    announced.append(name)
        return announced

    def reset_debounce(self) -> None:
        self._last_spoken.clear()

    def _worker(self) -> None:
        # Initialize pygame mixer once
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"pygame init failed: {e}")
            
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break
                
            try:
                # asyncio.run blocks until the async function completes, ensuring sequential playback
                asyncio.run(self._generate_and_play(text))
            except Exception as e:
                print(f"Edge-TTS playback error: {e}")

    async def _generate_and_play(self, text: str):
        audio_file = f"temp_audio_{uuid.uuid4().hex}.mp3"
        try:
            tts = edge_tts.Communicate(text, voice=self._voice)
            await tts.save(audio_file)
            
            if pygame.mixer.get_init():
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Wait for audio to finish playing so next audio doesn't overlap
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Failed to generate or play audio: {e}")
        finally:
            if pygame.mixer.get_init():
                try:
                    pygame.mixer.music.unload()
                except AttributeError:
                    pass
            
            # Clean up the temporary file
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except OSError:
                    pass
