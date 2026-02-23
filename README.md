# Jarvis - Local Voice Assistant

A privacy-focused, local-first voice assistant for Linux that runs entirely on your machine. Jarvis can understand speech, process commands using AI, perform system operations, and respond with natural speech.

## Features

### Core Features

- 🧠 **Local AI**: Powered by Ollama with support for various LLM models (Llama 3.1, Mistral, etc.)
- 🎤 **Speech-to-Text**: Uses OpenAI Whisper (via faster-whisper) for accurate voice recognition
- 🔊 **Text-to-Speech**: Uses Piper TTS for natural voice synthesis
- 🎯 **Voice Activity Detection**: Intelligent listening with automatic silence detection
- 🛠️ **System Operations**:
  - File management (create, read, delete files and directories)
  - Web page fetching and information retrieval
  - Application launching
- 🔒 **Security**: Sandboxed execution with whitelisted commands and directory restrictions
- 💬 **Conversation Memory**: Maintains context across multiple interactions

### Phase 1 Enhancements (New!)

- 🎙️ **Wake Word Detection**: Hands-free activation with "Hey Jarvis" using Porcupine
- 💾 **Conversation Persistence**: Save and restore conversations across sessions with SQLite
- 🔄 **Error Recovery**: Automatic retry with fallback mechanisms for robust operation
  - STT fallback to openai-whisper if faster-whisper fails
  - TTS fallback to espeak if Piper fails
  - LLM offline cache for common responses
  - Web fetch retry with exponential backoff
- 🏥 **Health Checks**: Automatic service monitoring on startup

## Architecture

```
Audio Input → STT (Whisper) → LLM (Ollama) → Action Executor → TTS (Piper) → Audio Output
                ↑                   ↓                                  ↓
          Wake Word          File Ops | Web Fetch | App Launch    Fallback TTS
                                       ↓
                               Conversation DB
```

## Requirements

- **OS**: Linux (tested on x86_64 and arm64)
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~10GB for models
- **Optional**: NVIDIA GPU with CUDA for faster inference

## Installation

### 1. Clone the Repository

```bash
cd /home/jules/Dev/other/Jarvis
```

### 2. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg

# Fedora
sudo dnf install portaudio-devel python3-pyaudio ffmpeg

# Arch
sudo pacman -S portaudio python-pyaudio ffmpeg
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Install Piper TTS

```bash
# Automated installation
python setup_piper.py

# Or manual installation:
# Download from: https://github.com/rhasspy/piper/releases
# Extract and place binary in ./piper/piper
```

### 5. Install and Configure Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (choose one)
ollama pull llama3.1:8b      # Recommended for 16GB RAM
ollama pull mistral:7b       # Alternative
ollama pull llama3.1:8b-q4_0 # Quantized version for 8GB RAM
```

### 6. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit configuration
nano .env
```

Key settings to configure:

- `WHISPER_MODEL`: `tiny`, `base`, `small`, `medium`, or `large` (base recommended)
- `OLLAMA_MODEL`: Model name you pulled (e.g., `llama3.1:8b`)
- `ALLOWED_DIRECTORIES`: Directories where file operations are permitted
- `COMMAND_WHITELIST`: Whitelisted commands for system operations

### 7. Enable Phase 1 Features (Optional)

Edit `.env` to enable new features:

```env
# Enable conversation persistence
ENABLE_CONVERSATION_PERSISTENCE=True
LOAD_PREVIOUS_CONTEXT=True

# Enable error recovery
ENABLE_RETRY_LOGIC=True
ENABLE_FALLBACK_STT=True

# Enable wake word (requires Picovoice access key)
ENABLE_WAKE_WORD=True
PORCUPINE_ACCESS_KEY=your_key_here  # Get from https://console.picovoice.ai/
WAKE_WORD_KEYWORD=jarvis
WAKE_WORD_SENSITIVITY=0.5
```

**Get Picovoice Access Key:**

1. Visit https://console.picovoice.ai/
2. Sign up for a free account
3. Create a new access key
4. Copy the key to your `.env` file

## Usage

### Default Mode (Voice + TUI)

```bash
python main.py
```

The Terminal UI launches by default, providing a rich interface with:

- **Header**: Current status and language
- **Conversation Panel**: Complete chat history with timestamps
- **Actions Panel**: Real-time log of system operations and tools
- **Help Panel**: Quick reference for commands

**TUI Features:**
- Say "show options" or "open settings" to access the interactive options menu
- Real-time status updates
- Color-coded actions (green=success, red=error, yellow=info)
- Persistent display of conversation context

Speak naturally after the "Listening..." prompt. The assistant will:

1. Record your voice until silence is detected
2. Transcribe using Whisper
3. Process with the LLM
4. Execute any requested actions
5. Respond with synthesized speech

**With Wake Word Enabled:**

1. Wait for Jarvis to say "Listening for wake word..."
2. Say "Hey Jarvis" (or your configured wake word)
3. Give your command immediately after detection
4. Jarvis processes and responds
5. Returns to wake word listening

### Voice Mode without TUI

For voice mode with plain log output instead of the Terminal UI:

```bash
python main.py --no-tui
```

### Text Mode

For testing without audio I/O:

```bash
python main.py --text
```

### Example Commands

**English:**

- "Create a file called notes.txt in my home directory"
- "What's on the website example.com?"
- "List the files in my Documents folder"
- "Open Firefox"
- "Delete the file test.txt from tmp"
- "What's the weather?" (with web search)

**French:**

- "Crée un fichier appelé notes.txt dans mon répertoire personnel"
- "Qu'est-ce qu'il y a sur le site exemple.com?"
- "Liste les fichiers dans mon dossier Documents"
- "Ouvre Firefox"
- "Supprime le fichier test.txt de tmp"

### Language Switching

**Switch to French:**

- "Switch to French"
- "Parle français"
- "En français"

**Switch to English:**

- "Switch to English"
- "Parle anglais"
- "In English"

You can also set the default language in `.env` with `WHISPER_LANGUAGE=fr` or `WHISPER_LANGUAGE=en`.

### Options Menu (TUI Mode)

When using TUI mode (`python main.py --tui`), you can access an interactive options menu:

**Voice Command:**
- Say "show options", "open settings", or "show settings"

**Text Mode:**
- Type `/options`

The options menu allows you to:
- 🌐 **Switch Language** - Toggle between English and French
- 🗑️ **Clear Conversation** - Reset the current conversation history
- 📋 **Clear Actions Log** - Clear the actions panel
- 💾 **Save Current Session** - Manually save the conversation
- ℹ️ **View System Info** - Check status of all components (Ollama, Whisper, Piper, Audio, Wake Word)

All changes take effect immediately without restarting Jarvis.

### Exit Commands

Say or type: "exit", "quit", "goodbye", "stop" (English) or "au revoir", "arrête" (French)

## Configuration

Edit [.env](.env) to customize:

### Audio Settings

- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `CHANNELS`: Audio channels (default: 1 for mono)

### STT Settings

- `WHISPER_MODEL`: Model size (`tiny`, `base`, `small`, `medium`, `large`)
  - `tiny`: Fastest, least accurate (~75 MB)
  - `base`: Good balance (~142 MB) - **Recommended**
  - `small`: Better accuracy (~466 MB)
  - `medium`: High accuracy (~1.5 GB)
  - `large`: Best accuracy (~2.9 GB)
- `WHISPER_DEVICE`: `cpu` or `cuda` (for NVIDIA GPUs)
- `WHISPER_COMPUTE_TYPE`: `int8` (CPU) or `float16` (GPU)
- `WHISPER_LANGUAGE`: Language code (`en` for English, `fr` for French, or `auto` for auto-detection)

### LLM Settings

- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (e.g., `llama3.1:8b`)
- `LLM_TEMPERATURE`: Response creativity (0.0-1.0, default: 0.7)
- `LLM_MAX_TOKENS`: Maximum response length (default: 1000)

### TTS Settings

- `PIPER_MODEL`: Voice model (default: `en_US-lessac-medium`)
- `PIPER_SPEAKER_ID`: Voice variant (0-based index)

### Security Settings

- `ALLOWED_DIRECTORIES`: Comma-separated paths where file operations are allowed
- `COMMAND_WHITELIST`: Comma-separated list of allowed commands

### Phase 1 Settings (New!)

#### Error Recovery

- `ENABLE_RETRY_LOGIC`: Enable automatic retry with backoff (default: `True`)
- `MAX_RETRY_ATTEMPTS`: Maximum retry attempts (default: `3`)
- `RETRY_BACKOFF_FACTOR`: Backoff multiplier between retries (default: `2.0`)
- `ENABLE_FALLBACK_STT`: Enable fallback to openai-whisper (default: `True`)
- `ENABLE_OFFLINE_MODE`: Enable offline response caching (default: `True`)
- `OFFLINE_RESPONSE_CACHE_SIZE`: Number of responses to cache (default: `50`)

#### Conversation Persistence

- `ENABLE_CONVERSATION_PERSISTENCE`: Save conversations to database (default: `True`)
- `CONVERSATION_DB_PATH`: Path to SQLite database (default: `~/.local/share/jarvis/conversations.db`)
- `LOAD_PREVIOUS_CONTEXT`: Load last conversation on startup (default: `True`)

#### Wake Word Detection

- `ENABLE_WAKE_WORD`: Enable hands-free wake word activation (default: `False`)
- `PORCUPINE_ACCESS_KEY`: Picovoice access key from https://console.picovoice.ai/ (required if enabled)
- `WAKE_WORD_KEYWORD`: Wake word to detect (default: `jarvis`, options: `computer`, `hey google`, etc.)
- `WAKE_WORD_SENSITIVITY`: Detection sensitivity 0.0-1.0 (default: `0.5`)

**Note**: Wake word detection requires a free Picovoice account. Visit https://console.picovoice.ai/ to get your access key.

## Project Structure

```
Jarvis/
├── main.py                   # Main orchestration loop
├── config.py                 # Configuration management
├── audio_handler.py          # Audio I/O, VAD, and wake word
├── stt_module.py             # Speech-to-text with retry/fallback
├── llm_module.py             # LLM integration with offline cache
├── action_executor.py        # System operations with retry
├── tts_module.py             # Text-to-speech with fallback
├── error_recovery.py         # Retry and fallback utilities (Phase 1)
├── persistence_module.py     # Conversation storage backend (Phase 1)
├── wake_word_detector.py     # Wake word detection (Phase 1)
├── tui.py                    # Terminal UI
├── whitelist_manager.py      # Security whitelist management
├── setup_piper.py            # Piper installation script
├── requirements.txt          # Python dependencies
├── .env.example              # Example configuration
├── .env                      # Your configuration (create this)
└── piper/                    # Piper binary and models
```

## Troubleshooting

### Audio Issues

**No microphone input:**

```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test recording
python -c "import sounddevice as sd; import numpy as np; print('Recording...'); audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1); sd.wait(); print('Done')"
```

**Permission denied:**

```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Log out and back in
```

### Whisper Issues

**Model download fails:**

```bash
# Manually download models
python -c "from faster_whisper import WhisperModel; model = WhisperModel('base')"
```

**Out of memory:**

- Use a smaller model (`tiny` or `base`)
- Set `WHISPER_COMPUTE_TYPE=int8`

### Ollama Issues

**Connection refused:**

```bash
# Start Ollama service
ollama serve

# Or check if running
ps aux | grep ollama
```

**Model not found:**

```bash
# List installed models
ollama list

# Pull required model
ollama pull llama3.1:8b
```

### Piper Issues

**Binary not found:**

```bash
# Re-run setup
python setup_piper.py

# Or set explicit path in tts_module.py
```

**Voice sounds robotic:**

- Try a different model (e.g., `en_US-amy-medium`)
- Download from: https://huggingface.co/rhasspy/piper-voices

## Performance Optimization

### For Limited Hardware (8GB RAM)

```env
WHISPER_MODEL=tiny
OLLAMA_MODEL=llama3.1:8b-q4_0
WHISPER_COMPUTE_TYPE=int8
```

### For Better Quality (16GB+ RAM)

```env
WHISPER_MODEL=small
OLLAMA_MODEL=llama3.1:8b
WHISPER_COMPUTE_TYPE=int8
```

### With NVIDIA GPU

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

Pull GPU-optimized Ollama models and ensure CUDA is installed.

## Security Considerations

Jarvis includes several security measures:

1. **Directory Whitelisting**: File operations only in `ALLOWED_DIRECTORIES`
2. **Command Whitelisting**: Only whitelisted commands can be executed
3. **Action Confirmation**: All file operations, web requests, and app launches require user approval
4. **Smart Whitelist**: Approve once, auto-approve future identical actions
   - File operations: Whitelisted by directory
   - Web requests: Whitelisted by domain
   - Applications: Whitelisted by exact command
5. **No Shell Injection**: Uses subprocess with explicit arguments (no `shell=True`)
6. **Path Validation**: Resolves and validates all paths before operations
7. **Timeout Protection**: All operations have timeouts

**Whitelist Storage**: Approved actions are stored in `command_whitelist.json` for persistence.

**Confirmation Options**:

- `y` - Execute this action once
- `a` - Execute and add to whitelist for future auto-approval
- `n` - Cancel the action

**Important**: Review and customize security settings in `.env` before use.

## Extending Jarvis

### Adding New Tools

Edit [llm_module.py](llm_module.py) to add tool definitions:

```python
TOOLS = [
    # ... existing tools ...
    {
        "type": "function",
        "function": {
            "name": "your_tool_name",
            "description": "What your tool does",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param1"]
            }
        }
    }
]
```

Then implement in [action_executor.py](action_executor.py):

```python
def your_tool_name(self, param1: str) -> str:
    """Your tool implementation."""
    # ... your code ...
    return "Result"
```

## Testing

Jarvis includes a comprehensive test suite to ensure reliability and correctness.

### Running Tests

**Run all tests with coverage:**

```bash
./run_tests.sh
```

**Run only unit tests:**

```bash
./run_tests.sh --unit
```

**Run tests without coverage:**

```bash
./run_tests.sh --no-coverage
```

**Verbose output:**

```bash
./run_tests.sh -v
```

**Run specific test file:**

```bash
pytest tests/test_error_recovery.py -v
```

**Run tests by marker:**

```bash
pytest -m unit                    # Only unit tests
pytest -m integration             # Only integration tests
pytest -m "not slow"              # Skip slow tests
pytest -m requires_ollama         # Tests requiring Ollama
```

### Test Markers

- `unit` - Unit tests (fast, isolated, mocked dependencies)
- `integration` - Integration tests (slower, real dependencies)
- `slow` - Tests that take significant time
- `requires_audio` - Tests requiring audio hardware
- `requires_ollama` - Tests requiring Ollama server
- `requires_porcupine` - Tests requiring Porcupine access key

### Coverage Report

After running tests with coverage, open the HTML report:

```bash
xdg-open htmlcov/index.html
```

Target coverage: 80%+ for all modules

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_error_recovery.py         # Error recovery tests
├── test_persistence_module.py     # Persistence backend tests
├── test_wake_word_detector.py     # Wake word detection tests
├── test_llm_module.py             # LLM and conversation tests
├── test_stt_module.py             # Speech-to-text tests
└── test_tts_module.py             # Text-to-speech tests
```

## Contributing

Contributions are welcome! Areas for improvement:

- Multi-language support
- Plugin architecture
- Web UI
- Home automation integration
- Voice cloning for personalized TTS
- Additional test coverage

**Before submitting a PR:**

1. Run the test suite: `./run_tests.sh`
2. Ensure coverage remains above 80%
3. Add tests for new features
4. Update documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Piper](https://github.com/rhasspy/piper) - Fast neural TTS
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) - Voice activity detection

## Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Note**: This is a local-first assistant. All processing happens on your machine - no data is sent to external servers.
