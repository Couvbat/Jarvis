"""Configuration management for Jarvis voice assistant."""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    
    # STT settings
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "fr"
    
    # LLM settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # TTS settings
    piper_model: str = "en_US-lessac-medium"
    piper_speaker_id: int = 0
    
    # System settings
    max_conversation_history: int = 10
    allowed_directories: str = "/home,/tmp"
    command_whitelist: str = "ls,cat,mkdir,touch,rm,echo,code,firefox,nautilus"
    
    # Logging
    log_level: str = "INFO"
    
    @property
    def allowed_dirs_list(self) -> List[Path]:
        """Parse allowed directories into Path objects."""
        return [Path(d.strip()) for d in self.allowed_directories.split(',')]
    
    @property
    def command_whitelist_list(self) -> List[str]:
        """Parse command whitelist into list."""
        return [c.strip() for c in self.command_whitelist.split(',')]


# Global settings instance
settings = Settings()
