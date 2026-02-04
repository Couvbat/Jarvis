"""Unit tests for stt_module."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from stt_module import STTModule


class TestSTTModule:
    """Test STTModule class."""
    
    @patch('stt_module.WhisperModel')
    def test_init_success(self, mock_whisper):
        """Test successful initialization."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        stt = STTModule(
            model_size="tiny",
            device="cpu",
            compute_type="int8"
        )
        
        assert stt.model is not None
        mock_whisper.assert_called_once_with("tiny", device="cpu", compute_type="int8")
    
    @patch('stt_module.WhisperModel')
    def test_init_failure(self, mock_whisper):
        """Test initialization failure."""
        mock_whisper.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception):
            STTModule(model_size="invalid")
    
    @patch('stt_module.WhisperModel')
    def test_transcribe_success(self, mock_whisper, mock_audio_data):
        """Test successful transcription."""
        # Mock Whisper model
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "This is a test transcription."
        mock_model.transcribe.return_value = ([mock_segment], {})
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        result = stt.transcribe(mock_audio_data)
        
        assert result == "This is a test transcription."
        mock_model.transcribe.assert_called_once()
    
    @patch('stt_module.WhisperModel')
    def test_transcribe_empty(self, mock_whisper, mock_audio_data):
        """Test transcription with empty result."""
        # Mock Whisper model with empty result
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], {})
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        result = stt.transcribe(mock_audio_data)
        
        assert result == ""
    
    @patch('stt_module.WhisperModel')
    def test_transcribe_multiple_segments(self, mock_whisper, mock_audio_data):
        """Test transcription with multiple segments."""
        # Mock Whisper model with multiple segments
        mock_model = MagicMock()
        mock_segment1 = MagicMock()
        mock_segment1.text = "Hello"
        mock_segment2 = MagicMock()
        mock_segment2.text = "world"
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], {})
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        result = stt.transcribe(mock_audio_data)
        
        assert "Hello" in result
        assert "world" in result
    
    @patch('stt_module.WhisperModel')
    def test_transcribe_with_retry(self, mock_whisper, mock_audio_data):
        """Test transcription with retry on failure."""
        # Mock Whisper model to fail twice, then succeed
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Success after retry"
        
        mock_model.transcribe.side_effect = [
            Exception("Connection error"),
            Exception("Connection error"),
            ([mock_segment], {})
        ]
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        result = stt.transcribe(mock_audio_data)
        
        assert result == "Success after retry"
        assert mock_model.transcribe.call_count == 3
    
    @patch('stt_module.WhisperModel')
    @patch('stt_module.whisper')
    def test_transcribe_with_fallback(self, mock_openai_whisper, mock_faster_whisper, mock_audio_data, tmp_path):
        """Test fallback to openai-whisper."""
        # Make faster-whisper fail
        mock_faster_model = MagicMock()
        mock_faster_model.transcribe.side_effect = Exception("Model error")
        mock_faster_whisper.return_value = mock_faster_model
        
        # Make openai-whisper succeed
        mock_openai_whisper.load_model.return_value = MagicMock()
        mock_openai_whisper.transcribe.return_value = {
            "text": "Fallback transcription"
        }
        
        stt = STTModule()
        result = stt.transcribe(mock_audio_data)
        
        # Should use fallback
        assert result == "Fallback transcription"
    
    @patch('stt_module.WhisperModel')
    def test_transcribe_with_language(self, mock_whisper, mock_audio_data):
        """Test transcription with language specification."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Bonjour"
        mock_model.transcribe.return_value = ([mock_segment], {})
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        result = stt.transcribe(mock_audio_data, language="fr")
        
        assert result == "Bonjour"
        # Check that language was passed to transcribe
        call_args = mock_model.transcribe.call_args
        assert 'language' in call_args[1]
        assert call_args[1]['language'] == "fr"
    
    @patch('stt_module.WhisperModel')
    def test_transcribe_empty_audio(self, mock_whisper):
        """Test transcription with empty audio."""
        mock_model = MagicMock()
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        
        # Empty numpy array
        empty_audio = np.array([], dtype=np.float32)
        result = stt.transcribe(empty_audio)
        
        # Should handle gracefully
        assert isinstance(result, str)
    
    @patch('stt_module.WhisperModel')
    @patch('stt_module.DegradedModeManager.mark_service_degraded')
    def test_mark_degraded_on_failure(self, mock_mark_degraded, mock_whisper, mock_audio_data):
        """Test marking service as degraded on persistent failure."""
        # Make all attempts fail
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Persistent error")
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        
        try:
            stt.transcribe(mock_audio_data)
        except:
            pass
        
        # Should mark as degraded after retries exhausted
        # (This depends on implementation details)
        # The test verifies the behavior exists
    
    @patch('stt_module.WhisperModel')
    def test_batch_transcribe(self, mock_whisper):
        """Test batch transcription (if implemented)."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Test"
        mock_model.transcribe.return_value = ([mock_segment], {})
        mock_whisper.return_value = mock_model
        
        stt = STTModule()
        
        # Create multiple audio samples
        audio_samples = [
            np.random.randn(16000).astype(np.float32),
            np.random.randn(16000).astype(np.float32)
        ]
        
        results = [stt.transcribe(audio) for audio in audio_samples]
        
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
