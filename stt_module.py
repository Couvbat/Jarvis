"""Speech-to-Text module using faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger
from config import settings
import tempfile
import soundfile as sf
from error_recovery import retry_with_backoff, RetryExhaustedError, degraded_mode


class STTModule:
    """Speech-to-Text service using faster-whisper."""
    
    def __init__(self):
        self.model_name = settings.whisper_model
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type
        self.language = settings.whisper_language
        self.model = None
        
    def initialize(self):
        """Load the Whisper model."""
        if self.model is not None:
            logger.info("STT model already loaded")
            return
            
        logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
        
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("STT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            raise
    
    def set_language(self, language: str):
        """
        Change the transcription language.
        
        Args:
            language: Language code ('en', 'fr', 'auto', etc.)
        """
        self.language = language
        logger.info(f"Language changed to: {language}")
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text with retry logic.
        
        Args:
            audio_data: Audio samples as numpy array (int16)
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("STT model not initialized. Call initialize() first.")
        
        logger.info("Transcribing audio...")
        
        # Try with retry logic if enabled
        if settings.enable_retry_logic:
            try:
                return self._transcribe_with_retry(audio_data, sample_rate)
            except RetryExhaustedError as e:
                logger.error(f"All STT retry attempts failed: {e}")
                degraded_mode.mark_degraded('stt')
                # Try fallback STT if available
                if settings.enable_fallback_stt:
                    try:
                        return self._fallback_transcribe(audio_data, sample_rate)
                    except Exception as fallback_error:
                        logger.error(f"Fallback STT also failed: {fallback_error}")
                raise RuntimeError("Speech recognition is currently unavailable")
        else:
            return self._transcribe_once(audio_data)
    
    @retry_with_backoff(
        max_attempts=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(Exception,)
    )
    def _transcribe_with_retry(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Internal method with retry decorator."""
        return self._transcribe_once(audio_data)
    
    def _transcribe_once(self, audio_data: np.ndarray) -> str:
        """Single transcription attempt without retry."""
        try:
            # faster-whisper expects float32 audio normalized to [-1, 1]
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_float,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine all segments
            transcription = " ".join([segment.text for segment in segments])
            transcription = transcription.strip()
            
            logger.info(f"Transcription: {transcription}")
            logger.debug(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            # Mark as recovered if it was degraded
            if degraded_mode.is_degraded('stt'):
                degraded_mode.mark_recovered('stt')
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def _fallback_transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Fallback transcription using openai-whisper.
        
        This is a slower but more robust alternative when faster-whisper fails.
        """
        try:
            import whisper
            logger.info("Using fallback STT (openai-whisper)")
            
            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Load model (cached after first load)
            if not hasattr(self, '_fallback_model'):
                self._fallback_model = whisper.load_model(self.model_name)
            
            # Transcribe
            result = self._fallback_model.transcribe(
                audio_float,
                language=self.language if self.language != 'auto' else None
            )
            
            transcription = result['text'].strip()
            logger.info(f"Fallback transcription: {transcription}")
            return transcription
            
        except ImportError:
            logger.error("Fallback STT not available (openai-whisper not installed)")
            raise
        except Exception as e:
            logger.error(f"Fallback transcription error: {e}")
            raise
    
    def transcribe_file(self, audio_file: str) -> str:
        """
        Transcribe audio from a file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("STT model not initialized. Call initialize() first.")
        
        logger.info(f"Transcribing file: {audio_file}")
        
        try:
            segments, info = self.model.transcribe(
                audio_file,
                language=self.language,
                beam_size=5,
                vad_filter=True
            )
            
            transcription = " ".join([segment.text for segment in segments])
            transcription = transcription.strip()
            
            logger.info(f"Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
