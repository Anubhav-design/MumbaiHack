"""
Text-to-Speech Engine for Hospital Sign Language System
Converts detected gestures to spoken words
FIXED VERSION - Reliable speech for every gesture
"""

import pyttsx3
import threading
import queue
from typing import Optional
import time


class TextToSpeech:
    """
    Text-to-Speech engine using pyttsx3.
    Uses a dedicated thread with proper engine management.
    """
    
    def __init__(self, rate: int = 150, volume: float = 1.0, voice_gender: str = "female"):
        """
        Initialize the TTS engine.
        
        Args:
            rate: Speech rate (words per minute), default 150
            volume: Volume level (0.0 to 1.0)
            voice_gender: 'male' or 'female' voice preference
        """
        self.rate = rate
        self.volume = volume
        self.voice_gender = voice_gender
        
        # Speech queue for non-blocking operation
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.is_running = True
        
        # Cooldown to prevent repeated speech
        self.last_speech_time = 0
        self.cooldown_seconds = 1.5  # Reduced cooldown for better responsiveness
        self.last_message = ""
        
        # Start speech worker thread
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        
        print("[TTS] Text-to-Speech engine initialized")
    
    def _create_engine(self):
        """Create a fresh TTS engine instance."""
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        engine.setProperty('volume', self.volume)
        
        # Set voice
        voices = engine.getProperty('voices')
        if voices and len(voices) > 1:
            # Usually index 1 is female on Windows
            voice_idx = 1 if self.voice_gender == "female" else 0
            engine.setProperty('voice', voices[voice_idx].id)
        
        return engine
    
    def _speech_worker(self):
        """Background thread that processes speech queue."""
        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    continue
                    
                self.is_speaking = True
                print(f"[TTS] Speaking: {text}")
                
                # Create fresh engine for each speech (more reliable)
                try:
                    engine = self._create_engine()
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"[TTS] Engine error: {e}")
                
                self.is_speaking = False
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Worker error: {e}")
                self.is_speaking = False
    
    def speak(self, text: str, force: bool = False) -> bool:
        """
        Add text to the speech queue.
        
        Args:
            text: Text to speak
            force: If True, ignore cooldown and speak immediately
            
        Returns:
            True if text was queued, False if skipped due to cooldown
        """
        if not text:
            return False
            
        current_time = time.time()
        
        # Check cooldown (unless forced)
        if not force:
            # Only apply cooldown for the SAME message
            if text == self.last_message:
                if current_time - self.last_speech_time < self.cooldown_seconds:
                    return False
        
        # Don't queue if already speaking
        if self.is_speaking and not force:
            return False
        
        self.last_message = text
        self.last_speech_time = current_time
        self.speech_queue.put(text)
        return True
    
    def speak_now(self, text: str):
        """
        Speak immediately, clearing any pending speech.
        
        Args:
            text: Text to speak immediately
        """
        # Clear the queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
        
        self.speech_queue.put(text)
    
    def speak_emergency(self, text: str):
        """
        Speak emergency message immediately.
        
        Args:
            text: Emergency text to speak
        """
        self.speak_now(text)
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)."""
        self.rate = rate
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
    
    def set_cooldown(self, seconds: float):
        """Set cooldown between repeated messages."""
        self.cooldown_seconds = seconds
    
    def get_available_voices(self) -> list:
        """Get list of available voice names."""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.stop()
            return [voice.name for voice in voices] if voices else []
        except:
            return []
    
    def stop(self):
        """Stop the TTS engine and cleanup."""
        self.is_running = False
        self.speech_queue.put(None)  # Signal to stop
        
        # Wait for thread to finish
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=2.0)
        
        print("[TTS] Engine stopped")
    
    def is_busy(self) -> bool:
        """Check if TTS is currently speaking."""
        return self.is_speaking or not self.speech_queue.empty()


class HospitalTTS(TextToSpeech):
    """
    Specialized TTS for hospital environment.
    Includes pre-defined phrases and emergency handling.
    """
    
    def __init__(self):
        super().__init__(rate=150, volume=1.0, voice_gender="female")
        
        # Hospital-specific phrases
        self.greetings = [
            "Hospital sign language system is ready.",
            "I am here to help you communicate.",
            "Please show your hand gesture in the camera."
        ]
        
        self.emergency_prefix = "Attention! Emergency! "
        self.acknowledgments = [
            "Message received.",
            "I understand.",
            "Noted."
        ]
        
        # Track last spoken gesture to avoid repetition
        self.last_gesture_id = None
        self.last_gesture_time = 0
        self.gesture_cooldown = 2.0  # Seconds between same gesture speech
    
    def speak_greeting(self):
        """Speak a random greeting."""
        import random
        greeting = random.choice(self.greetings)
        self.speak(greeting, force=True)
    
    def speak_gesture(self, gesture_text: str, gesture_id: int = None, is_emergency: bool = False) -> bool:
        """
        Speak the detected gesture.
        
        Args:
            gesture_text: Text associated with the gesture
            gesture_id: Unique ID of the gesture (for cooldown tracking)
            is_emergency: If True, treat as emergency
            
        Returns:
            True if spoken, False if skipped
        """
        current_time = time.time()
        
        # Check gesture-specific cooldown
        if gesture_id is not None and gesture_id == self.last_gesture_id:
            if current_time - self.last_gesture_time < self.gesture_cooldown:
                return False
        
        # Update tracking
        self.last_gesture_id = gesture_id
        self.last_gesture_time = current_time
        
        if is_emergency:
            full_text = self.emergency_prefix + gesture_text
            self.speak_emergency(full_text)
            return True
        else:
            return self.speak(gesture_text, force=True)
    
    def speak_acknowledgment(self):
        """Speak a random acknowledgment."""
        import random
        ack = random.choice(self.acknowledgments)
        self.speak(ack)
    
    def announce_calibration(self, progress: int):
        """Announce calibration progress."""
        if progress == 0:
            self.speak("Calibrating camera. Please wait.", force=True)
        elif progress == 100:
            self.speak("Calibration complete. You can now show gestures.", force=True)


# Test the TTS
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Hospital Text-to-Speech System")
    print("=" * 50)
    
    tts = HospitalTTS()
    
    print("\nAvailable voices:")
    for i, voice in enumerate(tts.get_available_voices()):
        print(f"  {i}: {voice}")
    
    print("\n--- Test 1: Greeting ---")
    tts.speak_greeting()
    time.sleep(4)
    
    print("\n--- Test 2: Normal gesture ---")
    tts.speak_gesture("I need water please.", gesture_id=9)
    time.sleep(3)
    
    print("\n--- Test 3: Different gesture ---")
    tts.speak_gesture("Yes, I understand.", gesture_id=1)
    time.sleep(3)
    
    print("\n--- Test 4: Emergency ---")
    tts.speak_gesture("Patient needs immediate help!", gesture_id=0, is_emergency=True)
    time.sleep(4)
    
    tts.stop()
    print("\n✅ TTS test complete!")
