"""Speech-to-Text module using faster-whisper."""

from typing import Any, Dict, Optional

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger
from config import settings


class STTModule:
    """Speech-to-Text service using faster-whisper."""
    
    def __init__(self):
        self.model_name = settings.whisper_model
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type
        self.language = settings.whisper_language
        self.model = None
        #: What the last transcription sounded like, whatever was asked for.
        self.detected_language: Optional[str] = None
        self.detected_language_probability: float = 0.0
        
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

    def _language_argument(self) -> Optional[str]:
        """What to hand the model.

        "auto" is how the README spells automatic detection, but faster-whisper
        wants None for that; the literal string is not a valid code and the
        model rejects it.
        """
        if not self.language or self.language.strip().lower() in ("auto", ""):
            return None
        return self.language

    @staticmethod
    def _join(segments) -> str:
        """Assemble segments into one line.

        Whisper segments already start with a space, so joining on one doubled
        every gap - visible in the prompt and in the transcript panel.
        """
        text = "".join(segment.text for segment in segments)
        return " ".join(text.split())
    
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
                language=self._language_argument(),
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            transcription = self._join(segments)
            
            logger.info(f"Transcription: {transcription}")
            logger.debug(
                f"Detected language: {info.language} "
                f"(probability: {info.language_probability:.2f})"
            )
            
            self.detected_language = info.language
            self.detected_language_probability = info.language_probability
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

    def transcribe_with_info(
        self, audio_data: np.ndarray, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Transcribe, and say which language it heard.
        
        The detected language was logged and then thrown away, so nothing
        could act on a speaker switching language mid-conversation.
        
        Returns:
            Dict with 'text', 'language' and 'language_probability'
        """
        text = self.transcribe(audio_data, sample_rate)
        return {
            "text": text,
            "language": self.detected_language,
            "language_probability": self.detected_language_probability,
        }
    
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
                language=self._language_argument(),
                beam_size=5,
                vad_filter=True
            )
            
            transcription = self._join(segments)
            self.detected_language = info.language
            
            logger.info(f"Transcription: {transcription}")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
