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
    # 20 ms at 16 kHz. Silence is only re-checked once per captured block, so
    # smaller blocks make end-of-speech detection more responsive. The VAD
    # itself carves its own frames, so any value works.
    chunk_size: int = 320
    
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
    # Ollama's context window defaults to a few thousand tokens whatever the
    # model's native size, and tool schemas are re-sent every turn. Raising
    # this costs RAM; leaving it low silently truncates the conversation.
    llm_num_ctx: int = 8192
    # Above this many tools, only the ones relevant to the turn are offered:
    # a small model stops picking correctly past a dozen or so.
    tool_selection_threshold: int = 20
    tool_selection_top_k: int = 15
    
    # TTS settings
    piper_model: str = "en_US-lessac-medium"
    piper_speaker_id: int = 0
    
    # System settings
    max_conversation_history: int = 10
    # Deliberately narrow. "/home" would put ~/.ssh, ~/.aws and every dotfile
    # in a household inside the sandbox; widen it knowingly, not by default.
    allowed_directories: str = "~/Documents,~/Downloads,/tmp"
    # Glob patterns refused even inside an allowed directory, matched against
    # each path component below the allowed root.
    denied_patterns: str = (
        ".ssh,.aws,.gnupg,.kube,.docker,.env,.env.*,*_history,"
        "*.pem,*.key,id_rsa*,credentials,.netrc,.npmrc,.pypirc"
    )
    # Launchers only by default. A whitelisted CLI tool such as rm, cat or
    # mkdir reaches the whole disk: it never passes through the path sandbox.
    # File operations belong to the sandboxed fs__ tools.
    command_whitelist: str = "code,firefox,nautilus,gedit,xdg-open"
    # Commands launched detached, without waiting for them or reading output.
    gui_applications: str = "code,firefox,nautilus,gedit,xdg-open"
    
    # Web access
    allow_private_network_fetch: bool = False
    max_fetch_bytes: int = 2_000_000
    fetch_timeout: int = 10
    
    # MCP servers
    mcp_config_path: str = "mcp_servers.json"
    mcp_connect_timeout: float = 15.0
    mcp_call_timeout: float = 60.0
    
    # Persistent state
    data_dir: str = "~/.local/share/jarvis"
    
    # Logging
    log_level: str = "INFO"
    
    @property
    def allowed_dirs_list(self) -> List[Path]:
        """Parse allowed directories into Path objects."""
        return [Path(d.strip()) for d in self.allowed_directories.split(',')]
    
    @property
    def approvals_path(self) -> Path:
        """Where standing approvals are kept."""
        return Path(self.data_dir).expanduser() / "approvals.db"
    
    @property
    def gui_applications_list(self) -> List[str]:
        """Parse GUI application list."""
        return [g.strip() for g in self.gui_applications.split(',') if g.strip()]
    
    @property
    def denied_patterns_list(self) -> List[str]:
        """Parse denied path patterns into a list."""
        return [p.strip() for p in self.denied_patterns.split(',') if p.strip()]
    
    @property
    def command_whitelist_list(self) -> List[str]:
        """Parse command whitelist into list."""
        return [c.strip() for c in self.command_whitelist.split(',')]


# Global settings instance
settings = Settings()
