"""Pytest configuration: dependency stubs, environment isolation, fixtures."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# Environment isolation.
#
# ``config.Settings`` reads a ``.env`` file from the current working directory
# and instantiates a module-level singleton at import time.  Real environment
# variables outrank dotenv in pydantic-settings, so pinning them here makes the
# suite independent of whatever ``.env`` a developer happens to have.
# --------------------------------------------------------------------------- #
_TEST_ENV = {
    "SAMPLE_RATE": "16000",
    "CHANNELS": "1",
    "CHUNK_SIZE": "1024",
    "WHISPER_MODEL": "base",
    "WHISPER_DEVICE": "cpu",
    "WHISPER_COMPUTE_TYPE": "int8",
    "WHISPER_LANGUAGE": "en",
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_MODEL": "llama3.1:8b",
    "LLM_TEMPERATURE": "0.7",
    "LLM_MAX_TOKENS": "1000",
    "PIPER_MODEL": "en_US-lessac-medium",
    "PIPER_SPEAKER_ID": "0",
    "MAX_CONVERSATION_HISTORY": "10",
    "ALLOWED_DIRECTORIES": "/tmp",
    "COMMAND_WHITELIST": "ls,cat,mkdir,touch,rm,echo,code,firefox,nautilus",
    "LOG_LEVEL": "INFO",
}
os.environ.update(_TEST_ENV)

from tests import _stubs  # noqa: E402

STUBS = _stubs.install()

from loguru import logger  # noqa: E402

logger.remove()  # keep test output readable


@pytest.fixture(autouse=True)
def reset_stubs():
    """Give every test a clean slate of stub state."""
    for module in STUBS.values():
        module.reset()
    yield
    for module in STUBS.values():
        module.reset()


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every Jarvis env var so ``Settings()`` shows its declared defaults."""
    for key in _TEST_ENV:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)


@pytest.fixture
def fake_sd():
    """The stubbed ``sounddevice`` module."""
    return STUBS["sounddevice"]


@pytest.fixture
def fake_vad():
    """The stubbed ``webrtcvad`` module."""
    return STUBS["webrtcvad"]


@pytest.fixture
def fake_whisper():
    """The stubbed ``faster_whisper`` module."""
    return STUBS["faster_whisper"]


@pytest.fixture
def fake_ollama():
    """The stubbed ``ollama`` module."""
    return STUBS["ollama"]


@pytest.fixture
def settings(monkeypatch):
    """The global settings singleton, safe to mutate for the test's duration."""
    from config import settings as _settings

    original = {key: getattr(_settings, key) for key in type(_settings).model_fields}
    yield _settings
    for key, value in original.items():
        setattr(_settings, key, value)


@pytest.fixture
def sandbox(tmp_path, settings):
    """A temp directory registered as the only allowed directory."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    settings.allowed_directories = str(sandbox_dir)
    return sandbox_dir


@pytest.fixture
def approve_all():
    """Confirmation stub that approves without remembering."""
    calls = []

    def confirm(decision):
        calls.append(decision)
        return (True, False)

    confirm.calls = calls
    return confirm


@pytest.fixture
def approve_and_remember():
    """Confirmation stub that approves and asks to stop being asked."""
    calls = []

    def confirm(decision):
        calls.append(decision)
        return (True, True)

    confirm.calls = calls
    return confirm


@pytest.fixture
def deny_all():
    """Confirmation stub that refuses everything."""
    calls = []

    def confirm(decision):
        calls.append(decision)
        return (False, False)

    confirm.calls = calls
    return confirm
