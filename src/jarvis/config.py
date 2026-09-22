"""Configuration management for Jarvis voice assistant."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

    # Audio settings
    # Off by default: a microphone in the same room as a speaker hears the
    # speaker, so without echo cancellation Jarvis interrupts itself. With
    # headphones, or a mic that cancels echo in hardware, turn this on.
    barge_in: bool = False
    barge_in_min_speech_ms: int = 200
    # Hands-free activation. Off by default: it needs an optional dependency
    # (`pip install "jarvis-assistant[wakeword]"`) and a model download, and
    # without them Jarvis should still start and listen as it always has.
    wake_word: bool = False
    # A pretrained openWakeWord model name. "hey_jarvis" is one of them,
    # which is convenient; "alexa", "hey_mycroft" and "hey_rhasspy" also
    # exist, as does the path to a model you trained yourself.
    wake_word_model: str = "hey_jarvis"
    # Raise it if the room sets it off, lower it if it misses you.
    wake_word_threshold: float = 0.5
    # Seconds of audio kept after the phrase, so a command said in the same
    # breath survives the handover to the recorder.
    wake_word_tail: float = 0.5
    # After an answer, listen this long without needing the phrase again.
    # Saying it before every turn of an exchange is the thing people stop
    # doing. 0 requires it every time.
    wake_word_follow_up: float = 8.0
    sample_rate: int = 16000
    channels: int = 1
    # Which microphone to use: a name, an index, or empty for the system
    # default. `python -c "import sounddevice; print(sounddevice.query_devices())"`
    # lists them.
    audio_input_device: str = ""
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
    # A second Ollama to fall back to when the first one cannot be reached:
    # typically a self-hosted box on the LAN holding the big model, and a
    # small model on this machine for when the LAN is not there. Leave the
    # host empty to run with one provider.
    ollama_fallback_host: str = ""
    ollama_fallback_model: str = ""
    # A small fallback model picks badly from a large toolbox, so it is
    # offered fewer tools than the primary. 0 means no cap.
    ollama_fallback_max_tools: int = 8
    # More than two providers, or per-provider context windows, go in this
    # file; when it exists it replaces the three settings above.
    llm_providers_path: str = "llm_providers.json"
    # A provider probe runs before the first word is spoken, so it has to be
    # quick. Raise it if a remote host is slow to answer.
    llm_probe_timeout: float = 5.0
    # How long to stay on a fallback before looking for the preferred
    # provider again, so a brief network blink does not strand the session
    # on the small model.
    llm_provider_recheck_seconds: float = 60.0
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

    # Document search (RAG)
    # Embeddings come from Ollama rather than sentence-transformers: it is
    # already running, so this adds no Python dependency and no torch, and a
    # self-hosted provider computes them too. `ollama pull nomic-embed-text`.
    rag_embed_model: str = "nomic-embed-text"
    # Characters per chunk, and how much each one repeats of the last, so an
    # answer straddling a boundary is still found from either side.
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    # Inputs per embedding request while indexing.
    rag_batch_size: int = 16
    # Passages returned by one search.
    rag_top_k: int = 5
    # Similarity below which a passage is not worth returning. Nearest-k
    # always returns something, so without a floor a question the documents
    # say nothing about comes back with the five least-irrelevant passages,
    # which the model then summarises confidently. The right value depends on
    # the embedding model - each has its own baseline for unrelated text - so
    # the default is off. To calibrate: ask about something you know is not
    # in your notes and look at the similarities in the answer.
    rag_min_similarity: float = 0.0
    # Only text is indexed; a PDF reader would be a dependency and a separate
    # decision.
    rag_extensions: str = ".md,.txt,.rst,.org,.markdown,.text"
    rag_max_file_bytes: int = 2_000_000

    # MCP servers
    mcp_config_path: str = "mcp_servers.json"
    mcp_connect_timeout: float = 15.0
    mcp_call_timeout: float = 60.0

    # Persistent state
    data_dir: str = "~/.local/share/jarvis"
    # Conversations are written to disk in the clear, on this machine.
    conversation_history: bool = True

    # Logging
    log_level: str = "INFO"

    @property
    def allowed_dirs_list(self) -> list[Path]:
        """Parse allowed directories into Path objects."""
        return [Path(d.strip()) for d in self.allowed_directories.split(',')]

    @property
    def approvals_path(self) -> Path:
        """Where standing approvals are kept."""
        return Path(self.data_dir).expanduser() / "approvals.db"

    @property
    def conversations_path(self) -> Path:
        """Where conversation history is kept."""
        return Path(self.data_dir).expanduser() / "conversations.db"

    @property
    def documents_path(self) -> Path:
        """Where the document index is kept."""
        return Path(self.data_dir).expanduser() / "documents.db"

    @property
    def rag_extensions_list(self) -> list[str]:
        """Parse the indexable file extensions into a list."""
        return [e.strip().lower() for e in self.rag_extensions.split(',') if e.strip()]

    @property
    def gui_applications_list(self) -> list[str]:
        """Parse GUI application list."""
        return [g.strip() for g in self.gui_applications.split(',') if g.strip()]

    @property
    def denied_patterns_list(self) -> list[str]:
        """Parse denied path patterns into a list."""
        return [p.strip() for p in self.denied_patterns.split(',') if p.strip()]

    @property
    def command_whitelist_list(self) -> list[str]:
        """Parse command whitelist into list."""
        return [c.strip() for c in self.command_whitelist.split(',')]


# Global settings instance
settings = Settings()
