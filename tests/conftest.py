"""
Pytest configuration and fixtures for Jarvis test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    import numpy as np
    # 1 second of 16kHz audio
    return np.random.randint(-32768, 32767, 16000, dtype=np.int16)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from unittest.mock import MagicMock
    settings = MagicMock()
    settings.sample_rate = 16000
    settings.channels = 1
    settings.chunk_size = 1024
    settings.whisper_model = "base"
    settings.whisper_device = "cpu"
    settings.whisper_compute_type = "int8"
    settings.whisper_language = "en"
    settings.ollama_host = "http://localhost:11434"
    settings.ollama_model = "llama3.1:8b"
    settings.llm_temperature = 0.7
    settings.llm_max_tokens = 1000
    settings.piper_model = "en_US-lessac-medium"
    settings.piper_speaker_id = 0
    settings.max_conversation_history = 10
    settings.enable_retry_logic = True
    settings.max_retry_attempts = 3
    settings.retry_backoff_factor = 2.0
    settings.enable_fallback_stt = True
    settings.enable_offline_mode = True
    settings.offline_response_cache_size = 50
    settings.enable_conversation_persistence = True
    settings.load_previous_context = False
    settings.enable_wake_word = False
    return settings


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "message": {
            "role": "assistant",
            "content": "This is a test response.",
            "tool_calls": []
        }
    }


@pytest.fixture
def sample_conversation_messages():
    """Sample conversation messages for testing."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "timestamp": "2026-02-04T10:00:00",
            "metadata": {}
        },
        {
            "role": "user",
            "content": "Hello, how are you?",
            "timestamp": "2026-02-04T10:01:00",
            "metadata": {}
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you!",
            "timestamp": "2026-02-04T10:01:05",
            "metadata": {}
        }
    ]


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    mock = MagicMock()
    mock.transcribe.return_value = (
        [MagicMock(text="This is a test transcription.")],
        MagicMock(language="en", language_probability=0.99)
    )
    return mock


@pytest.fixture(autouse=True)
def reset_degraded_mode():
    """Reset degraded mode state before each test."""
    from error_recovery import degraded_mode
    degraded_mode.degraded_services.clear()
    degraded_mode.offline_mode = False
    yield
    degraded_mode.degraded_services.clear()
    degraded_mode.offline_mode = False
