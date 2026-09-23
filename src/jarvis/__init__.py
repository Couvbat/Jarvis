"""Jarvis - a local voice assistant with tool calling.

Everything runs on this machine: speech in through faster-whisper, the model
through Ollama (local or self-hosted on the LAN), speech out through Piper.
Nothing is sent anywhere except the MCP servers and the URLs you approve.

Importing this package is deliberately cheap: the submodules pull in numpy,
sounddevice and the MCP SDK, so they are imported where they are used rather
than re-exported here. `python -m jarvis` or the `jarvis` command is the way
in.
"""

__version__ = "0.4.0"
