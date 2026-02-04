"""Speech-to-Text module using faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger
from config import settings
import tempfile
import soundfile as sf


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
        Transcribe audio to text.
        
        Args:
            audio_data: Audio samples as numpy array (int16)
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("STT model not initialized. Call initialize() first.")
        
        logger.info("Transcribing audio...")
        
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
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
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
