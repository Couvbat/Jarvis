"""
Wake word detection module using Porcupine.

Provides hands-free activation by listening for "Hey Jarvis" or custom wake words.
"""

import struct
from typing import Optional
from loguru import logger
import sounddevice as sd
import numpy as np

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    logger.warning("pvporcupine not installed - wake word detection unavailable")


class WakeWordDetector:
    """Wake word detector using Picovoice Porcupine."""
    
    def __init__(
        self,
        access_key: str,
        keyword: str = "jarvis",
        sensitivity: float = 0.5,
        sample_rate: int = 16000
    ):
        """
        Initialize wake word detector.
        
        Args:
            access_key: Picovoice access key (get from https://console.picovoice.ai/)
            keyword: Wake word keyword (jarvis, computer, hey google, etc.)
            sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)
            sample_rate: Audio sample rate (must be 16000 for Porcupine)
        
        Raises:
            ImportError: If pvporcupine is not installed
            ValueError: If access key is invalid
        """
        if not PORCUPINE_AVAILABLE:
            raise ImportError(
                "pvporcupine not installed. Install with: pip install pvporcupine\n"
                "Get access key from: https://console.picovoice.ai/"
            )
        
        if not access_key:
            raise ValueError(
                "Porcupine access key required. Get one from: https://console.picovoice.ai/\n"
                "Set in .env: PORCUPINE_ACCESS_KEY=your_key_here"
            )
        
        self.access_key = access_key
        self.keyword = keyword
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self.porcupine = None
        self._initialized = False
        
        logger.info(f"Wake word detector initialized (keyword: {keyword}, sensitivity: {sensitivity})")
    
    def initialize(self):
        """Initialize Porcupine wake word engine."""
        if self._initialized:
            logger.debug("Wake word detector already initialized")
            return
        
        try:
            # Create Porcupine instance
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[self.keyword],
                sensitivities=[self.sensitivity]
            )
            
            # Verify sample rate
            if self.porcupine.sample_rate != self.sample_rate:
                logger.warning(
                    f"Sample rate mismatch: expected {self.sample_rate}, "
                    f"Porcupine requires {self.porcupine.sample_rate}"
                )
                self.sample_rate = self.porcupine.sample_rate
            
            self._initialized = True
            logger.info(
                f"Porcupine initialized (version: {self.porcupine.version}, "
                f"frame_length: {self.porcupine.frame_length})"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise
    
    def cleanup(self):
        """Cleanup Porcupine resources."""
        if self.porcupine is not None:
            self.porcupine.delete()
            self.porcupine = None
            self._initialized = False
            logger.debug("Porcupine resources released")
    
    def listen_for_wake_word(self, timeout: Optional[float] = None) -> bool:
        """
        Listen for wake word detection.
        
        Args:
            timeout: Maximum time to listen in seconds (None = indefinite)
            
        Returns:
            True if wake word detected, False if timeout
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("Wake word detector not initialized. Call initialize() first.")
        
        logger.info(f"Listening for wake word '{self.keyword}'...")
        
        try:
            # Open audio stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.porcupine.frame_length
            )
            
            stream.start()
            
            frames_processed = 0
            max_frames = None
            if timeout:
                max_frames = int(timeout * self.sample_rate / self.porcupine.frame_length)
            
            detected = False
            
            while True:
                # Read audio frame
                audio_frame, overflowed = stream.read(self.porcupine.frame_length)
                
                if overflowed:
                    logger.warning("Audio buffer overflow")
                
                # Convert to required format
                pcm = struct.unpack_from(
                    f'{self.porcupine.frame_length}h',
                    audio_frame.tobytes()
                )
                
                # Process frame
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    logger.info(f"Wake word detected! (keyword index: {keyword_index})")
                    detected = True
                    break
                
                frames_processed += 1
                
                # Check timeout
                if max_frames and frames_processed >= max_frames:
                    logger.debug(f"Wake word detection timed out after {timeout}s")
                    break
            
            stream.stop()
            stream.close()
            
            return detected
            
        except KeyboardInterrupt:
            logger.info("Wake word detection interrupted")
            return False
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


class SimpleWakeWordDetector:
    """
    Simple wake word detector using audio energy threshold.
    
    This is a fallback when Porcupine is not available. It doesn't actually
    detect specific words, but detects any loud sound/voice as a trigger.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 1000.0
    ):
        """
        Initialize simple wake word detector.
        
        Args:
            sample_rate: Audio sample rate
            energy_threshold: Energy threshold for detection
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self._initialized = True
        
        logger.warning(
            "Using simple energy-based wake word detection (fallback). "
            "For true wake word detection, install pvporcupine."
        )
    
    def initialize(self):
        """Initialize (no-op for simple detector)."""
        pass
    
    def cleanup(self):
        """Cleanup (no-op for simple detector)."""
        pass
    
    def listen_for_wake_word(self, timeout: Optional[float] = None) -> bool:
        """
        Listen for any loud sound as wake word.
        
        Args:
            timeout: Maximum time to listen in seconds
            
        Returns:
            True if loud sound detected, False if timeout
        """
        logger.info("Listening for any voice/sound (simple detection)...")
        
        try:
            duration = timeout if timeout else 60.0  # Max 60s if no timeout
            
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            
            # Monitor for loud sound
            chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                
                # Calculate energy
                energy = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                
                if energy > self.energy_threshold:
                    logger.info(f"Sound detected (energy: {energy:.0f})")
                    sd.stop()
                    return True
                
                # Check for interruption
                if sd.get_stream().stopped:
                    break
            
            logger.debug("No wake word detected (timeout)")
            return False
            
        except KeyboardInterrupt:
            logger.info("Wake word detection interrupted")
            sd.stop()
            return False
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False


def create_wake_word_detector(
    access_key: Optional[str] = None,
    keyword: str = "jarvis",
    sensitivity: float = 0.5,
    sample_rate: int = 16000,
    fallback_to_simple: bool = True
) -> Optional[WakeWordDetector]:
    """
    Factory function to create wake word detector with automatic fallback.
    
    Args:
        access_key: Picovoice access key (required for Porcupine)
        keyword: Wake word keyword
        sensitivity: Detection sensitivity (0.0-1.0)
        sample_rate: Audio sample rate
        fallback_to_simple: Use simple detector if Porcupine unavailable
        
    Returns:
        WakeWordDetector instance or None if unavailable
    """
    # Try Porcupine first
    if PORCUPINE_AVAILABLE and access_key:
        try:
            detector = WakeWordDetector(
                access_key=access_key,
                keyword=keyword,
                sensitivity=sensitivity,
                sample_rate=sample_rate
            )
            logger.info("Using Porcupine wake word detection")
            return detector
        except Exception as e:
            logger.warning(f"Failed to create Porcupine detector: {e}")
    
    # Fallback to simple detector
    if fallback_to_simple:
        logger.info("Using simple energy-based wake word detection")
        return SimpleWakeWordDetector(sample_rate=sample_rate)
    
    logger.error("No wake word detector available")
    return None
