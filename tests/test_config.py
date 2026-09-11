"""Tests for configuration parsing (config.py)."""

from pathlib import Path

import pytest

from config import Settings, settings


class TestDefaults:
    def test_singleton_is_exported(self):
        assert isinstance(settings, Settings)

    def test_declared_defaults(self, clean_env):
        fresh = Settings(_env_file=None)
        assert fresh.sample_rate == 16000
        assert fresh.channels == 1
        assert fresh.chunk_size == 1024
        assert fresh.whisper_model == "base"
        assert fresh.ollama_model == "llama3.1:8b"
        assert fresh.max_conversation_history == 10

    def test_default_stt_and_tts_languages_disagree(self, clean_env):
        """Out of the box STT listens in French while TTS speaks English.

        Not a crash, but it means a default install answers French speech with
        an English voice.  Pinned here so the mismatch is a deliberate choice
        rather than an accident.
        """
        fresh = Settings(_env_file=None)
        assert fresh.whisper_language == "fr"
        assert fresh.piper_model.startswith("en_")


class TestEnvironmentOverrides:
    def test_env_vars_are_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("whisper_model", "small")
        assert Settings(_env_file=None).whisper_model == "small"

    def test_numeric_coercion(self, monkeypatch):
        monkeypatch.setenv("SAMPLE_RATE", "48000")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.25")
        fresh = Settings(_env_file=None)
        assert fresh.sample_rate == 48000
        assert isinstance(fresh.sample_rate, int)
        assert fresh.llm_temperature == pytest.approx(0.25)

    def test_invalid_numeric_value_is_rejected(self, monkeypatch):
        from pydantic import ValidationError

        monkeypatch.setenv("SAMPLE_RATE", "not-a-number")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_dotenv_file_is_read(self, tmp_path, monkeypatch):
        env_file = tmp_path / "custom.env"
        env_file.write_text("OLLAMA_MODEL=mistral:7b\n")
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        assert Settings(_env_file=str(env_file)).ollama_model == "mistral:7b"


class TestDerivedProperties:
    def test_allowed_dirs_list_parses_and_strips(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_DIRECTORIES", "/home, /tmp ,/srv/data")
        dirs = Settings(_env_file=None).allowed_dirs_list
        assert dirs == [Path("/home"), Path("/tmp"), Path("/srv/data")]
        assert all(isinstance(item, Path) for item in dirs)

    def test_command_whitelist_parses_and_strips(self, monkeypatch):
        monkeypatch.setenv("COMMAND_WHITELIST", "ls, cat ,echo")
        assert Settings(_env_file=None).command_whitelist_list == ["ls", "cat", "echo"]

    def test_empty_allowed_directories_yields_cwd_not_empty_list(self, monkeypatch):
        """``""`` splits to ``[""]`` which becomes ``Path(".")`` - the CWD.

        An operator who empties the setting expecting "deny everything" gets
        "allow the working directory" instead.
        """
        monkeypatch.setenv("ALLOWED_DIRECTORIES", "")
        assert Settings(_env_file=None).allowed_dirs_list == [Path(".")]

    def test_empty_command_whitelist_yields_one_empty_entry(self, monkeypatch):
        monkeypatch.setenv("COMMAND_WHITELIST", "")
        assert Settings(_env_file=None).command_whitelist_list == [""]
