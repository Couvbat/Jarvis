"""Audio input/output handler with Voice Activity Detection."""

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad
from collections import deque
from typing import Optional, Generator
from loguru import logger
from config import settings
from wake_word_detector import create_wake_word_detector, WakeWordDetector


class AudioHandler:
    """Handles audio recording and playback with VAD support."""
    
    def __init__(self):
        self.sample_rate = settings.sample_rate
        self.channels = settings.channels
        self.chunk_size = settings.chunk_size
        self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3, 2 is moderate
        self.wake_word_detector: Optional[WakeWordDetector] = None
        
    def record_until_silence(
        self, 
        silence_threshold: float = 1.0,
        max_duration: float = 30.0
    ) -> np.ndarray:
        """
        Record audio until silence is detected.
        
        Args:
            silence_threshold: Seconds of silence before stopping
            max_duration: Maximum recording duration in seconds
            
        Returns:
            Audio data as numpy array
        """
        logger.info("Starting audio recording...")
        
        frames = []
        silence_frames = 0
        silence_frame_count = int(silence_threshold * self.sample_rate / self.chunk_size)
        max_frames = int(max_duration * self.sample_rate / self.chunk_size)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                blocksize=self.chunk_size
            ) as stream:
                logger.info("Listening... (speak now)")
                
                for _ in range(max_frames):
                    audio_chunk, _ = stream.read(self.chunk_size)
                    frames.append(audio_chunk.copy())
                    
                    # Check for voice activity
                    # VAD requires 16-bit PCM, 8kHz, 16kHz, 32kHz, or 48kHz
                    audio_bytes = audio_chunk.tobytes()
                    
                    try:
                        is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
                        
                        if is_speech:
                            silence_frames = 0
                        else:
                            silence_frames += 1
                            
                        # Stop if we've had enough silence
                        if silence_frames >= silence_frame_count and len(frames) > 10:
                            logger.info("Silence detected, stopping recording")
                            break
                            
                    except Exception as e:
                        # VAD can fail with certain audio, continue anyway
                        logger.debug(f"VAD error: {e}")
                        
        except Exception as e:
            logger.error(f"Recording error: {e}")
            raise
        
        # Concatenate all frames
        if not frames:
            return np.array([], dtype=np.int16)
            
        audio_data = np.concatenate(frames, axis=0)
        logger.info(f"Recording complete: {len(audio_data) / self.sample_rate:.2f}s")
        
        return audio_data
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """
        Play audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate (uses default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        logger.info(f"Playing audio: {len(audio_data) / sample_rate:.2f}s")
        
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()
            logger.debug("Playback complete")
        except Exception as e:
            logger.error(f"Playback error: {e}")
            raise
    
    def save_audio(self, audio_data: np.ndarray, filename: str):
        """Save audio data to file."""
        try:
            sf.write(filename, audio_data, self.sample_rate)
            logger.info(f"Audio saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise
    
    def load_audio(self, filename: str) -> tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            audio_data, sample_rate = sf.read(filename, dtype='int16')
            logger.info(f"Audio loaded from {filename}")
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def initialize_wake_word(
        self,
        access_key: Optional[str] = None,
        keyword: str = "jarvis",
        sensitivity: float = 0.5
    ):
        """
        Initialize wake word detection.
        
        Args:
            access_key: Picovoice access key
            keyword: Wake word keyword
            sensitivity: Detection sensitivity (0.0-1.0)
        """
        try:
            self.wake_word_detector = create_wake_word_detector(
                access_key=access_key,
                keyword=keyword,
                sensitivity=sensitivity,
                sample_rate=self.sample_rate,
                fallback_to_simple=True
            )
            
            if self.wake_word_detector:
                self.wake_word_detector.initialize()
                logger.info("Wake word detection ready")
            else:
                logger.warning("Wake word detection not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize wake word detection: {e}")
            self.wake_word_detector = None
    
    def listen_for_wake_word(self, timeout: Optional[float] = None) -> bool:
        """
        Listen for wake word.
        
        Args:
            timeout: Maximum time to listen (None = indefinite)
            
        Returns:
            True if wake word detected, False otherwise
        """
        if self.wake_word_detector is None:
            logger.warning("Wake word detector not initialized")
            return False
        
        try:
            return self.wake_word_detector.listen_for_wake_word(timeout)
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False
    
    def cleanup_wake_word(self):
        """Cleanup wake word detector resources."""
        if self.wake_word_detector:
            try:
                self.wake_word_detector.cleanup()
                logger.debug("Wake word detector cleaned up")
            except Exception as e:
                logger.error(f"Wake word cleanup error: {e}")
