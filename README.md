# Jarvis - Local Voice Assistant

A privacy-focused, local-first voice assistant for Linux that runs entirely on your machine. Jarvis can understand speech, process commands using AI, perform system operations, and respond with natural speech.

## Features

- 🧠 **Local AI**: Powered by Ollama with support for various LLM models (Llama 3.1, Mistral, etc.)
- 🎤 **Speech-to-Text**: Uses OpenAI Whisper (via faster-whisper) for accurate voice recognition
- 🔊 **Text-to-Speech**: Uses Piper TTS for natural voice synthesis
- 🎯 **Voice Activity Detection**: Intelligent listening with automatic silence detection
- 👋 **Wake word** (optional): hands-free activation on "hey Jarvis", locally, with no API key
- 🛠️ **System Operations**: 
  - File management (create, read, delete files and directories)
  - Web page fetching and information retrieval
  - Application launching
- 🔒 **Security**: Sandboxed execution with whitelisted commands and directory restrictions
- 💬 **Conversation Memory**: Maintains context across multiple interactions
- 📚 **Document search** (optional): ask questions about your own notes and files, indexed locally
- ⏰ **Reminders**: "remind me to call the plumber tomorrow at nine", kept across restarts
- 👁️ **Vision** (optional): ask what is in a screenshot or a photo on your disk

## Architecture

```
Audio Input → STT (Whisper) → LLM (Ollama) → Action Executor → TTS (Piper) → Audio Output
                                     ↓
                            File Ops | Web Fetch | App Launch
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
git clone <repository-url> Jarvis
cd Jarvis
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

# Install Jarvis and its dependencies
pip install -e .
```

That puts two commands on the PATH: `jarvis` and `jarvis-setup-piper`. Add
the development extras with `pip install -e ".[dev]"`.

### 4. Install Piper TTS

```bash
# English voice (default)
jarvis-setup-piper

# French voice - WHISPER_LANGUAGE defaults to fr, so you probably want this
jarvis-setup-piper --voice fr_FR-siwis-medium

jarvis-setup-piper --list-voices
```

Downloads are checked against pinned SHA-256 digests before anything is
extracted or marked executable, and archives are extracted with the member
filter that refuses paths escaping the target directory.

### 5. Install and Configure Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (choose one)
ollama pull llama3.1:8b      # Recommended for 16GB RAM
ollama pull mistral:7b       # Alternative
ollama pull llama3.1:8b-q4_0 # Quantized version for 8GB RAM
```

Ollama does not have to run on this machine. If you self-host one on a NAS or
a server, point `OLLAMA_HOST` at it and set a small local model as a fallback
for when that box is unreachable - see
[Remote and fallback providers](#remote-and-fallback-providers).

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

## Usage

### Voice Mode (Default)

```bash
jarvis
```

Speak naturally after the "Listening..." prompt. The assistant will:
1. Record your voice until silence is detected
2. Transcribe using Whisper
3. Process with the LLM
4. Execute any requested actions
5. Respond with synthesized speech

### Voice Mode with Terminal UI

For a rich terminal interface showing chat history and actions:

```bash
jarvis --tui
```

The TUI displays:
- **Header**: Current status and language
- **Conversation Panel**: Complete chat history with timestamps
- **Actions Panel**: Real-time log of system operations and tools
- **Help Panel**: Quick reference for commands

### Text Mode

For testing without audio I/O:

```bash
jarvis --text
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
- `LLM_NUM_CTX`: Context window (default: 8192). Ollama defaults to a few
  thousand tokens whatever the model's native size, and the tool schemas are
  re-sent every turn.

### Looking at images

```bash
ollama pull llava
# .env
VISION=true
```

*"What's in ~/Downloads/screenshot.png?"* — the image is read from disk and
put to a multimodal model, and the answer comes back as a tool result.

It is a **separate model from the one answering your questions**. Making
vision conditional on swapping the chat model would mean choosing between
"can see" and "can think": a text-only 70B on the NAS stays where it is, and
the picture goes to `llava` wherever that is pulled.

The same path sandbox applies — a photo is a file like any other, and
`~/.ssh` is no more readable for being asked about in pictures. Descriptions
count as **untrusted content**, because a screenshot of a web page is a web
page: text in an image can say "ignore your previous instructions", and
visual prompt injection is not hypothetical.

Screenshot *capture* is not included: it needs a display-server-specific
binary (`grim`, `spectacle`, `scrot`…) and it is a different privacy
question from reading a file you named. Take the screenshot yourself and ask
about the file.

### Reminders

*"Remind me to call the plumber tomorrow at nine."* *"What have I got
scheduled?"* *"Cancel the second one."*

Reminders are stored on disk, so they survive a restart, and one that came
due while Jarvis was off is delivered the next time it starts — prefixed
"while you were away", because a reminder that arrives late without saying so
arrives wrong.

Two things worth knowing:

- **No date-parsing library.** `dateparser` exists to turn "demain à 9h" into
  a timestamp, but there is already a language model in the loop whose whole
  job is understanding what was said. The current time goes into its
  instructions, it hands over a timestamp, and the tool *checks* it: a
  reminder in the past, or a year out, is handed back so the model can try
  again rather than setting the wrong time. If your model keeps getting the
  arithmetic wrong, "in 30 minutes" takes a different path that is much
  harder to get wrong.
- **They fire between turns, never during one.** Speaking over a recording
  would put Jarvis's own voice into the microphone — the problem barge-in
  exists for. So a reminder interrupts the wait for the wake word, or arrives
  when the current answer finishes. A few seconds late; never talking over
  you.

Set `REMINDERS=false` to drop the three tools entirely.

### Searching your own documents

Index your notes once, and Jarvis can answer from them instead of guessing:

```bash
ollama pull nomic-embed-text
jarvis-index ~/Documents          # or a single file, or several paths
jarvis-index --status             # what is indexed, and with which model
```

Then: *"what did I write about the boiler?"*, *"what's the plumber's
number?"*, *"summarise my notes from the meeting on Tuesday"*. Answers name
the files they came from, so you can check them.

Re-run `jarvis-index` whenever your notes change — it only re-reads files
whose size or timestamp moved, and drops files you deleted. `--force`
re-embeds everything.

Design notes worth knowing:

- **Embeddings come from Ollama**, not `sentence-transformers`. That would
  mean torch: a multi-gigabyte install, on a machine already running a
  language model and a speech recogniser, to do something the model server
  does natively. This adds no Python dependency at all, and your self-hosted
  provider computes the embeddings when it is the one answering.
- **The search is a tool the model calls**, not context stuffed into every
  turn. "What time is it" should not drag your notes into the prompt.
- **The path sandbox applies.** `jarvis-index ~/.ssh` is refused, and so is
  any file matching `DENIED_PATTERNS` inside a directory you did allow —
  indexing a file puts its contents one similarity match away from the
  prompt.
- **Retrieved passages count as untrusted**, which `fs__read` does not. With
  `fs__read` you named the file; here a similarity score chose it, and your
  index may hold a README from a cloned repo or a downloaded document that
  says "ignore your instructions". So a write *after* a document search is
  confirmed at the keyboard. Searching again is not re-escalated — that is
  what the origin rule is for.
- **Text only.** PDFs would mean another dependency and a separate decision;
  `RAG_EXTENSIONS` says what is indexed.

If a question your notes say nothing about comes back with vaguely related
passages, set `RAG_MIN_SIMILARITY`. Nearest-k always returns *something*, so
a floor is the only thing that turns "here are five irrelevant passages" into
"nothing matched". The right value depends on the embedding model, so it is
off by default; the similarity of each passage is printed in the answer,
which is how you find yours.

### Hands-free activation

Without a wake word Jarvis records, transcribes and answers everything it
hears. That is fine for a session you started deliberately and wrong for
something that sits on a desk all day.

```bash
pip install "jarvis-assistant[wakeword]"
python -c "import openwakeword.utils; openwakeword.utils.download_models()"
```

Then in `.env`:

```bash
WAKE_WORD=true
WAKE_WORD_MODEL=hey_jarvis     # also: alexa, hey_mycroft, hey_rhasspy
WAKE_WORD_THRESHOLD=0.5        # up if the room sets it off, down if it misses you
WAKE_WORD_FOLLOW_UP=8          # seconds a follow-up needs no wake word
```

The detector is [openWakeWord](https://github.com/dscripka/openWakeWord),
which runs locally and happens to ship a pretrained `hey_jarvis` model.
Porcupine, the usual suggestion, needs a Picovoice API key — a key check
against a remote service is exactly what "fully local" rules out, however
good the detector is.

Two things it does that a naive version does not:

- **The command in the same breath survives.** People say "hey Jarvis quelle
  heure est-il", not "hey Jarvis", pause, "quelle heure est-il". The audio
  after the phrase is carried into the recording (`WAKE_WORD_TAIL`).
- **A follow-up does not need the phrase again.** For `WAKE_WORD_FOLLOW_UP`
  seconds after an answer, Jarvis just listens. Saying the wake word before
  every turn is the thing people stop doing. Set it to `0` to require it
  every time.

If the package or the models are missing, Jarvis says so once and listens the
way it always has, rather than refusing to start. Needs `SAMPLE_RATE=16000`,
which is the default — the models mean nothing at any other rate.

### Remote and fallback providers

A self-hosted Ollama on the LAN and a small model on this machine are not the
same thing, and which one is reachable changes through the day. Jarvis takes a
list of providers, most preferred first: the first one that answers **and** has
its model pulled serves the turn.

```bash
# .env - the big model lives on the NAS, a small one here for when it doesn't
OLLAMA_HOST=http://nas.local:11434
OLLAMA_MODEL=qwen3:32b
OLLAMA_FALLBACK_HOST=http://localhost:11434
OLLAMA_FALLBACK_MODEL=llama3.2:3b
OLLAMA_FALLBACK_MAX_TOOLS=8
```

What this buys you:

- **The NAS being off does not stop the session.** The startup line names the
  provider that is answering and why the other one is not.
- **Coming back is automatic.** After `LLM_PROVIDER_RECHECK_SECONDS` (60 by
  default) Jarvis looks for the preferred provider again, so one blink of the
  network does not strand the session on the small model.
- **The small model gets a smaller toolbox.** A 3B model picks badly from a
  32B's tool list, so `OLLAMA_FALLBACK_MAX_TOOLS` caps what it is offered.
- **Failing over mid-answer does not repeat speech.** Once a fragment has been
  spoken the turn is committed to that provider: you hear an apology, not the
  same sentence twice in two different voices.

A provider is skipped only for being unreachable or missing its model. An
answer that looks wrong is never a reason to swap models mid-conversation.

Three or more providers, or per-provider context windows, go in a file:

```bash
cp llm_providers.example.json llm_providers.json
```

When `llm_providers.json` exists it replaces the `OLLAMA_*` settings above.

### TTS Settings
- `PIPER_MODEL`: Voice model (default: `en_US-lessac-medium`)
- `PIPER_SPEAKER_ID`: Voice variant (0-based index)

### Security Settings
- `ALLOWED_DIRECTORIES`: Comma-separated paths where file operations are allowed
- `COMMAND_WHITELIST`: Comma-separated list of allowed commands

## Project Structure

```
Jarvis/
├── src/jarvis/                 # The package
│   ├── __main__.py             #   `python -m jarvis`, and the `jarvis` command
│   ├── main.py                 #   orchestration loop, CLI, confirmation surface
│   ├── config.py               #   configuration (pydantic-settings)
│   ├── audio_handler.py        #   audio I/O and VAD
│   ├── stt_module.py           #   speech-to-text (Whisper)
│   ├── llm_module.py           #   conversation, streaming, tool-call loop
│   ├── llm_providers.py        #   ordered providers, probing and failover
│   ├── tts_module.py           #   text-to-speech (Piper)
│   ├── tui.py                  #   Rich terminal interface
│   ├── text_utils.py           #   shared normalisation and tokenisation
│   ├── conversation_store.py   #   conversation history (SQLite)
│   ├── reminders.py            #   scheduled reminders (SQLite)
│   ├── vision.py               #   multimodal model for images
│   ├── setup_piper.py          #   Piper installer (`jarvis-setup-piper`)
│   ├── speech/                 #   sentence chunking, TTS queue, barge-in, wake word
│   ├── rag/                    #   document index: chunking, embeddings, search
│   ├── tools/                  # Tool layer
│   │   ├── schema.py           #   tool specs, risk levels, MCP conversion
│   │   ├── registry.py         #   registration and dispatch
│   │   ├── selection.py        #   which tools to offer this turn
│   │   ├── builtin.py          #   assembling the built-in tools
│   │   ├── local/              #   filesystem (CRUD), web, applications
│   │   └── mcp/                #   MCP client: config, connections, adapter
│   └── policy/                 # What a tool call is allowed to do
│       ├── paths.py            #   sandbox and denied patterns
│       ├── engine.py           #   auto / confirm / refuse decisions
│       ├── taint.py            #   untrusted-content tracking
│       └── store.py            #   persistent approvals (SQLite)
├── tests/                      # Test suite (see tests/README.md)
├── pyproject.toml              # Package metadata, dependencies, entry points
├── requirements-test.txt       # Test-only dependencies (no native deps)
├── .env.example                # Example configuration
├── llm_providers.example.json  # Example provider list (optional)
├── mcp_servers.example.json    # Example MCP server list (optional)
├── .env                        # Your configuration (create this)
└── piper/                      # Piper binary and models (created by setup)
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

### Reminder Issues

**It sets the wrong time:** small models are worse at date arithmetic than at
language. Ask for "in 30 minutes" rather than "at half past" and see whether
that is reliable; the two go down different paths. A time in the past or more
than a year out is refused and handed back, so watch for the model retrying.

**A reminder never arrived:** Jarvis has to be running. They fire between
turns, so one due mid-answer waits for the answer to finish. Anything older
than a week is not delivered at all — waking up to a backlog would be worse —
but it stays visible in the list until cancelled.

### Document Search Issues

**"no documents are indexed yet":** run `jarvis-index ~/Documents`.

**"could not embed":** `ollama pull nomic-embed-text`. Pointing
`RAG_EMBED_MODEL` at a chat model instead of an embedding model is caught and
reported rather than indexed as garbage.

**"this index was built with X":** you changed `RAG_EMBED_MODEL`. Vectors from
two models are not comparable, so this is refused rather than averaged over —
re-index, or set it back.

**Answers cite the wrong passages:** raise `RAG_MIN_SIMILARITY` (see above),
or lower `RAG_CHUNK_SIZE` so each passage is about one thing.

### Wake Word Issues

**It never hears me:** lower `WAKE_WORD_THRESHOLD`, and check the microphone
is the one you think it is (`AUDIO_INPUT_DEVICE`). `hey_jarvis` is trained on
English pronunciation; a strong accent may need a lower threshold or a model
you train yourself.

**It goes off on its own:** raise the threshold. Check the follow-up window
too — inside it Jarvis is listening on purpose.

**"Wake word disabled" at startup:** the message says which of the two steps
is missing, installing the package or downloading the models. Jarvis carries
on listening without it.

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

**Answers got worse / it used the wrong model:** the startup line names the
provider that is serving the session, and a fallback is logged as
`Falling back from ... to ...`. A remote provider that is up but slow to
answer its model listing looks unreachable; raise `LLM_PROBE_TIMEOUT`.

### Piper Issues

**Binary not found:**
```bash
# Re-run setup
jarvis-setup-piper

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

## Development

### Running the tests

```bash
pip install -r requirements-test.txt   # no PortAudio, no models, no Ollama needed
pytest
pytest --cov --cov-report=term-missing
ruff check .
```

The suite stubs every native and network dependency (`sounddevice`,
`webrtcvad`, `faster-whisper`, `ollama`), so it runs on a bare machine in
about a second. MCP tests drive a real server over the SDK's in-memory
transport. See [tests/README.md](tests/README.md).

`requirements-test.txt` is deliberately not the `[test]` extra: installing
Jarvis itself pulls in the four packages the suite stubs, and the point is to
need none of them. With Jarvis installed (`pip install -e .`) the suite runs
against the installed package; from a bare checkout it falls back to `src/`.

For the checks the suite cannot make - real transcription quality, a physical
microphone, measured latency, a remote Ollama, third-party MCP servers - see
[TESTING.md](TESTING.md).

### Connecting MCP servers

Copy [mcp_servers.example.json](mcp_servers.example.json) to
`mcp_servers.json`. The `mcpServers` object is the shape other MCP hosts use,
so an existing configuration works unchanged; Jarvis adds `enabled` and
`trust` (`confirm` by default, `trusted`, or `readonly`).

A server's own annotations can only make one of its tools look *more*
dangerous, never safer — they are written by the server. What relaxes a tool
is the trust level you set.

### Choosing a model

Which local model calls tools well changes faster than any recommendation, and
it depends on the machine. Measure it:

```bash
python tests/eval/run_tool_calling.py --model llama3.1:8b --compare
```

See [tests/eval/README.md](tests/eval/README.md).

### Project status

The goal is a fully local voice assistant — STT and TTS on-device — with tool
calling, CRUD filesystem access, and MCP connections to external services.

- [ARCHITECTURE.md](ARCHITECTURE.md) — the target design and the decisions behind it
- [AUDIT.md](AUDIT.md) — what is implemented, what is not, and the known bugs
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — the phased path to the target
- [FEATURES_IDEA.md](FEATURES_IDEA.md) — longer-term feature ideas

Known bugs are each pinned by an `xfail(strict=True)` test carrying their
`BUG-xx` identifier. Run `pytest -rx` to list them.

## Contributing

Contributions are welcome! Areas for improvement:

- Wake word detection (e.g., "Hey Jarvis")
- Multi-language support
- Plugin architecture
- Web UI
- Home automation integration
- Voice cloning for personalized TTS

## License

MIT - see [`LICENSE`](LICENSE).

Jarvis bundles nothing: Piper, its voices, the Whisper weights and the models
Ollama serves are downloaded separately and carry their own licences.

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
