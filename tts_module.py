"""Text-to-Speech module using Piper."""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from loguru import logger
from config import settings
import soundfile as sf
from error_recovery import retry_with_backoff, RetryExhaustedError, degraded_mode


class TTSModule:
    """Text-to-Speech service using Piper."""
    
    def __init__(self):
        self.model_name = settings.piper_model
        self.speaker_id = settings.piper_speaker_id
        self.piper_binary = self._find_piper_binary()
        self.model_path = None
        
    def _find_piper_binary(self) -> str:
        """Locate the piper binary."""
        # Common installation locations
        possible_paths = [
            "piper",  # System PATH
            str(Path.home() / ".local" / "bin" / "piper"),
            "/usr/local/bin/piper",
            "./piper/piper"  # Local installation
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    logger.info(f"Found Piper binary: {path}")
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        logger.warning("Piper binary not found in common locations")
        return "piper"  # Fallback to PATH
    
    def _find_model_path(self) -> Path:
        """Find the Piper model file."""
        # Common model locations
        model_dirs = [
            Path.home() / ".local" / "share" / "piper" / "models",
            Path("/usr/share/piper/models"),
            Path("./piper/models")
        ]
        
        model_filename = f"{self.model_name}.onnx"
        
        for model_dir in model_dirs:
            model_path = model_dir / model_filename
            if model_path.exists():
                logger.info(f"Found model: {model_path}")
                return model_path
        
        logger.warning(f"Model not found: {model_filename}")
        return Path(model_filename)  # Return filename, let piper try to find it
    
    def initialize(self):
        """Initialize the TTS module."""
        logger.info(f"Initializing TTS with model: {self.model_name}")
        
        # Find model
        self.model_path = self._find_model_path()
        
        # Test piper
        try:
            result = subprocess.run(
                [self.piper_binary, "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("Piper TTS initialized successfully")
            else:
                logger.warning("Piper binary found but may not be working correctly")
        except Exception as e:
            logger.error(f"Piper initialization error: {e}")
            logger.info("Install Piper: https://github.com/rhasspy/piper")
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Convert text to speech with retry logic and fallback.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array (int16)
        """
        if not text or text.strip() == "":
            logger.warning("Empty text provided to TTS")
            return np.array([], dtype=np.int16)
        
        logger.info(f"Synthesizing: {text[:50]}...")
        
        # Try with retry logic if enabled
        if settings.enable_retry_logic:
            try:
                return self._synthesize_with_retry(text)
            except RetryExhaustedError as e:
                logger.error(f"All TTS retry attempts failed: {e}")
                degraded_mode.mark_degraded('tts')
                # Try fallback TTS
                try:
                    return self._fallback_synthesize(text)
                except Exception as fallback_error:
                    logger.error(f"Fallback TTS also failed: {fallback_error}")
                    # Return empty array - caller will handle text output
                    return np.array([], dtype=np.int16)
        else:
            return self._synthesize_once(text)
    
    @retry_with_backoff(
        max_attempts=2,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(Exception,)
    )
    def _synthesize_with_retry(self, text: str) -> np.ndarray:
        """Internal method with retry decorator."""
        return self._synthesize_once(text)
    
    def _synthesize_once(self, text: str) -> np.ndarray:
        """Single synthesis attempt without retry."""
        try:
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as text_file:
                text_file.write(text)
                text_file_path = text_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_file_path = wav_file.name
            
            # Build piper command
            cmd = [
                self.piper_binary,
                "--model", str(self.model_path),
                "--output_file", wav_file_path
            ]
            
            # Add speaker ID if specified
            if self.speaker_id > 0:
                cmd.extend(["--speaker", str(self.speaker_id)])
            
            # Run piper
            with open(text_file_path, 'r') as stdin_file:
                result = subprocess.run(
                    cmd,
                    stdin=stdin_file,
                    capture_output=True,
                    timeout=30
                )
            
            # Clean up text file
            Path(text_file_path).unlink()
            
            if result.returncode != 0:
                logger.error(f"Piper error: {result.stderr.decode()}")
                raise RuntimeError(f"Piper synthesis failed: {result.stderr.decode()}")
            
            # Load generated audio
            audio_data, sample_rate = sf.read(wav_file_path, dtype='int16')
            
            # Clean up wav file
            Path(wav_file_path).unlink()
            
            logger.info(f"Synthesized {len(audio_data) / sample_rate:.2f}s of audio")
            
            # Mark as recovered if it was degraded
            if degraded_mode.is_degraded('tts'):
                degraded_mode.mark_recovered('tts')
            
            return audio_data
            
        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out")
            raise RuntimeError("TTS synthesis timed out")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise
    
    def _fallback_synthesize(self, text: str) -> np.ndarray:
        """
        Fallback TTS using espeak (basic but always available on Linux).
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array
        """
        try:
            logger.info("Using fallback TTS (espeak)")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_file_path = wav_file.name
            
            # Run espeak
            cmd = [
                "espeak",
                "-w", wav_file_path,
                text
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"espeak failed: {result.stderr.decode()}")
            
            # Load generated audio
            audio_data, sample_rate = sf.read(wav_file_path, dtype='int16')
            
            # Clean up
            Path(wav_file_path).unlink()
            
            logger.info("Fallback TTS synthesis completed")
            return audio_data
            
        except FileNotFoundError:
            logger.error("Fallback TTS (espeak) not available")
            raise
        except Exception as e:
            logger.error(f"Fallback TTS error: {e}")
            raise
    
    def synthesize_to_file(self, text: str, output_file: str):
        """
        Synthesize text and save to file.
        
        Args:
            text: Text to synthesize
            output_file: Output audio file path
        """
        logger.info(f"Synthesizing to file: {output_file}")
        
        try:
            cmd = [
                self.piper_binary,
                "--model", str(self.model_path),
                "--output_file", output_file
            ]
            
            if self.speaker_id > 0:
                cmd.extend(["--speaker", str(self.speaker_id)])
            
            # Run piper with text as stdin
            result = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Piper error: {result.stderr.decode()}")
                raise RuntimeError(f"Piper synthesis failed")
            
            logger.info(f"Audio saved to {output_file}")
            
        except Exception as e:
            logger.error(f"TTS to file error: {e}")
            raise
