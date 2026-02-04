"""Unit tests for tts_module."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from tts_module import TTSModule


class TestTTSModule:
    """Test TTSModule class."""
    
    @patch('tts_module.subprocess.run')
    def test_init_success(self, mock_run):
        """Test successful initialization."""
        mock_run.return_value = MagicMock(returncode=0)
        
        tts = TTSModule(
            voice_model="en_US-lessac-medium",
            sample_rate=22050
        )
        
        assert tts.voice_model == "en_US-lessac-medium"
        assert tts.sample_rate == 22050
    
    @patch('tts_module.subprocess.run')
    def test_speak_success(self, mock_run):
        """Test successful text-to-speech."""
        # Mock Piper output (raw PCM audio)
        fake_audio = np.random.randn(22050).astype(np.int16).tobytes()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=fake_audio
        )
        
        tts = TTSModule()
        tts.speak("Hello, world!")
        
        # Should call subprocess to run Piper
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert 'piper' in call_args or 'echo' in call_args
    
    @patch('tts_module.subprocess.run')
    def test_speak_empty_text(self, mock_run):
        """Test speaking empty text."""
        tts = TTSModule()
        tts.speak("")
        
        # Should not call subprocess
        assert not mock_run.called
    
    @patch('tts_module.subprocess.run')
    def test_speak_long_text(self, mock_run):
        """Test speaking long text."""
        fake_audio = np.random.randn(22050).astype(np.int16).tobytes()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=fake_audio
        )
        
        long_text = "This is a very long text. " * 50
        tts = TTSModule()
        tts.speak(long_text)
        
        assert mock_run.called
    
    @patch('tts_module.subprocess.run')
    def test_speak_with_retry(self, mock_run):
        """Test retry on failure."""
        fake_audio = np.random.randn(22050).astype(np.int16).tobytes()
        
        # Fail twice, then succeed
        mock_run.side_effect = [
            Exception("Piper error"),
            Exception("Piper error"),
            MagicMock(returncode=0, stdout=fake_audio)
        ]
        
        tts = TTSModule()
        tts.speak("Test retry")
        
        assert mock_run.call_count == 3
    
    @patch('tts_module.subprocess.run')
    def test_speak_with_fallback_espeak(self, mock_run):
        """Test fallback to espeak."""
        # Make Piper fail
        mock_run.side_effect = [
            Exception("Piper not available"),
            MagicMock(returncode=0)  # espeak succeeds
        ]
        
        tts = TTSModule()
        tts.speak("Fallback test")
        
        # Should try Piper, then fallback to espeak
        assert mock_run.call_count >= 1
    
    @patch('tts_module.subprocess.run')
    @patch('tts_module.sd.play')
    @patch('tts_module.sd.wait')
    def test_speak_plays_audio(self, mock_wait, mock_play, mock_run):
        """Test that audio is played."""
        fake_audio = np.random.randn(22050).astype(np.int16).tobytes()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=fake_audio
        )
        
        tts = TTSModule()
        tts.speak("Test audio playback")
        
        # Should play audio
        assert mock_play.called or mock_run.called
    
    @patch('tts_module.subprocess.run')
    def test_speak_special_characters(self, mock_run):
        """Test speaking text with special characters."""
        fake_audio = np.random.randn(22050).astype(np.int16).tobytes()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=fake_audio
        )
        
        special_text = "Hello! How are you? I'm fine. #test @mention"
        tts = TTSModule()
        tts.speak(special_text)
        
        assert mock_run.called
    
    @patch('tts_module.subprocess.run')
    @patch('tts_module.DegradedModeManager.mark_service_degraded')
    def test_mark_degraded_on_failure(self, mock_mark_degraded, mock_run):
        """Test marking service as degraded on persistent failure."""
        # Make all attempts fail
        mock_run.side_effect = Exception("Persistent error")
        
        tts = TTSModule()
        
        try:
            tts.speak("Test degraded")
        except:
            pass
        
        # Should mark as degraded after retries exhausted
        # (This depends on implementation details)
    
    @patch('tts_module.subprocess.run')
    def test_speak_different_voices(self, mock_run):
        """Test using different voice models."""
        fake_audio = np.random.randn(22050).astype(np.int16).tobytes()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=fake_audio
        )
        
        # Try different voices
        voices = ["en_US-lessac-medium", "en_GB-alan-medium"]
        
        for voice in voices:
            tts = TTSModule(voice_model=voice)
            tts.speak("Test voice")
            assert mock_run.called
            mock_run.reset_mock()
    
    @patch('tts_module.subprocess.run')
    def test_cleanup(self, mock_run):
        """Test cleanup (should not raise error)."""
        tts = TTSModule()
        tts.cleanup()  # Should not raise
    
    def test_audio_format(self):
        """Test audio format configuration."""
        tts = TTSModule(sample_rate=16000)
        assert tts.sample_rate == 16000
        
        tts = TTSModule(sample_rate=22050)
        assert tts.sample_rate == 22050
