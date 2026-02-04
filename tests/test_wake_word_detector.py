"""Unit tests for wake_word_detector."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
from wake_word_detector import WakeWordDetector, SimpleWakeWordDetector, get_wake_word_detector


class TestWakeWordDetector:
    """Test WakeWordDetector class with Porcupine."""
    
    @patch('wake_word_detector.pvporcupine.create')
    def test_init_success(self, mock_create):
        """Test successful initialization."""
        mock_porcupine = MagicMock()
        mock_porcupine.sample_rate = 16000
        mock_porcupine.frame_length = 512
        mock_create.return_value = mock_porcupine
        
        detector = WakeWordDetector(
            access_key="test_key",
            keywords=["jarvis"],
            sensitivities=[0.5]
        )
        
        assert detector.porcupine is not None
        assert detector.sample_rate == 16000
        assert detector.frame_length == 512
    
    @patch('wake_word_detector.pvporcupine.create')
    def test_init_failure(self, mock_create):
        """Test initialization failure."""
        mock_create.side_effect = Exception("Invalid access key")
        
        with pytest.raises(Exception):
            WakeWordDetector(
                access_key="invalid_key",
                keywords=["jarvis"]
            )
    
    @patch('wake_word_detector.pvporcupine.create')
    @patch('wake_word_detector.sd.InputStream')
    def test_listen_detection(self, mock_stream, mock_create):
        """Test wake word detection."""
        # Setup Porcupine mock
        mock_porcupine = MagicMock()
        mock_porcupine.sample_rate = 16000
        mock_porcupine.frame_length = 512
        mock_porcupine.process.side_effect = [-1, -1, 0]  # Detect on 3rd frame
        mock_create.return_value = mock_porcupine
        
        # Setup audio stream mock
        mock_audio_stream = MagicMock()
        mock_audio_stream.__enter__ = MagicMock(return_value=mock_audio_stream)
        mock_audio_stream.__exit__ = MagicMock(return_value=False)
        mock_audio_stream.read.return_value = (np.zeros(512, dtype=np.int16), None)
        mock_stream.return_value = mock_audio_stream
        
        detector = WakeWordDetector(
            access_key="test_key",
            keywords=["jarvis"]
        )
        
        detected = detector.listen_for_wake_word(timeout=1.0)
        
        assert detected is True
        assert detector.porcupine.process.call_count == 3
    
    @patch('wake_word_detector.pvporcupine.create')
    @patch('wake_word_detector.sd.InputStream')
    def test_listen_timeout(self, mock_stream, mock_create):
        """Test timeout without detection."""
        # Setup Porcupine mock
        mock_porcupine = MagicMock()
        mock_porcupine.sample_rate = 16000
        mock_porcupine.frame_length = 512
        mock_porcupine.process.return_value = -1  # Never detect
        mock_create.return_value = mock_porcupine
        
        # Setup audio stream mock
        mock_audio_stream = MagicMock()
        mock_audio_stream.__enter__ = MagicMock(return_value=mock_audio_stream)
        mock_audio_stream.__exit__ = MagicMock(return_value=False)
        mock_audio_stream.read.return_value = (np.zeros(512, dtype=np.int16), None)
        mock_stream.return_value = mock_audio_stream
        
        detector = WakeWordDetector(
            access_key="test_key",
            keywords=["jarvis"]
        )
        
        detected = detector.listen_for_wake_word(timeout=0.1)
        
        assert detected is False
    
    @patch('wake_word_detector.pvporcupine.create')
    def test_cleanup(self, mock_create):
        """Test cleanup of resources."""
        mock_porcupine = MagicMock()
        mock_porcupine.sample_rate = 16000
        mock_porcupine.frame_length = 512
        mock_create.return_value = mock_porcupine
        
        detector = WakeWordDetector(
            access_key="test_key",
            keywords=["jarvis"]
        )
        
        detector.cleanup()
        
        mock_porcupine.delete.assert_called_once()


class TestSimpleWakeWordDetector:
    """Test SimpleWakeWordDetector fallback."""
    
    def test_init(self):
        """Test initialization."""
        detector = SimpleWakeWordDetector(
            energy_threshold=2000,
            duration_threshold=0.3
        )
        
        assert detector.energy_threshold == 2000
        assert detector.duration_threshold == 0.3
        assert detector.sample_rate == 16000
    
    @patch('wake_word_detector.sd.InputStream')
    def test_listen_detection(self, mock_stream):
        """Test detection with loud audio."""
        detector = SimpleWakeWordDetector(
            energy_threshold=1000,
            duration_threshold=0.1
        )
        
        # Create loud audio that exceeds threshold
        loud_audio = np.ones(1024, dtype=np.int16) * 5000
        
        mock_audio_stream = MagicMock()
        mock_audio_stream.__enter__ = MagicMock(return_value=mock_audio_stream)
        mock_audio_stream.__exit__ = MagicMock(return_value=False)
        mock_audio_stream.read.return_value = (loud_audio, None)
        mock_stream.return_value = mock_audio_stream
        
        detected = detector.listen_for_wake_word(timeout=0.5)
        
        assert detected is True
    
    @patch('wake_word_detector.sd.InputStream')
    def test_listen_no_detection(self, mock_stream):
        """Test no detection with quiet audio."""
        detector = SimpleWakeWordDetector(
            energy_threshold=5000,
            duration_threshold=0.1
        )
        
        # Create quiet audio below threshold
        quiet_audio = np.ones(1024, dtype=np.int16) * 100
        
        mock_audio_stream = MagicMock()
        mock_audio_stream.__enter__ = MagicMock(return_value=mock_audio_stream)
        mock_audio_stream.__exit__ = MagicMock(return_value=False)
        mock_audio_stream.read.return_value = (quiet_audio, None)
        mock_stream.return_value = mock_audio_stream
        
        detected = detector.listen_for_wake_word(timeout=0.2)
        
        assert detected is False
    
    def test_cleanup(self):
        """Test cleanup (should not raise error)."""
        detector = SimpleWakeWordDetector()
        detector.cleanup()  # Should not raise


class TestFactoryFunction:
    """Test get_wake_word_detector factory function."""
    
    @patch('wake_word_detector.pvporcupine.create')
    def test_create_porcupine_detector(self, mock_create):
        """Test creating Porcupine detector."""
        mock_porcupine = MagicMock()
        mock_porcupine.sample_rate = 16000
        mock_porcupine.frame_length = 512
        mock_create.return_value = mock_porcupine
        
        detector = get_wake_word_detector(
            access_key="test_key",
            keywords=["jarvis"]
        )
        
        assert isinstance(detector, WakeWordDetector)
    
    @patch('wake_word_detector.pvporcupine.create')
    def test_fallback_to_simple(self, mock_create):
        """Test fallback to SimpleWakeWordDetector."""
        mock_create.side_effect = Exception("Porcupine not available")
        
        detector = get_wake_word_detector(
            access_key="invalid_key",
            keywords=["jarvis"],
            fallback_to_simple=True
        )
        
        assert isinstance(detector, SimpleWakeWordDetector)
    
    @patch('wake_word_detector.pvporcupine.create')
    def test_no_fallback_raises(self, mock_create):
        """Test no fallback raises exception."""
        mock_create.side_effect = Exception("Porcupine not available")
        
        with pytest.raises(Exception):
            get_wake_word_detector(
                access_key="invalid_key",
                keywords=["jarvis"],
                fallback_to_simple=False
            )
