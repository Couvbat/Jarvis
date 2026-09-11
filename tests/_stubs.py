"""In-process stubs for the native / network dependencies Jarvis relies on.

The real ``sounddevice``, ``webrtcvad``, ``faster_whisper`` and ``ollama``
packages need PortAudio, a C toolchain, multi-gigabyte model files or a running
Ollama server.  None of that belongs in a unit test run, so the stubs below are
installed into ``sys.modules`` by ``conftest.py`` before the Jarvis modules are
imported.

They are deliberately *faithful* rather than permissive: ``webrtcvad`` rejects
frame sizes the real library rejects, ``ollama`` returns the object shape the
real client returns.  A stub that accepts everything would hide the very bugs
these tests exist to catch.
"""

import sys
import types
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# sounddevice
# --------------------------------------------------------------------------- #

class FakeInputStream:
    """Replays a scripted list of int16 blocks, then silence forever."""

    #: blocks handed to the next stream instance, set by tests
    scripted_blocks: List[np.ndarray] = []
    #: every stream instance created, for assertions on constructor kwargs
    instances: List["FakeInputStream"] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.blocks = list(FakeInputStream.scripted_blocks)
        self.read_calls = 0
        self.closed = False
        FakeInputStream.instances.append(self)

    def __enter__(self) -> "FakeInputStream":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.closed = True
        return False

    def read(self, frames: int) -> Tuple[np.ndarray, bool]:
        self.read_calls += 1
        if self.blocks:
            block = self.blocks.pop(0)
        else:
            channels = self.kwargs.get("channels", 1)
            block = np.zeros((frames, channels), dtype=np.int16)
        return block, False


class _SoundDeviceModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("sounddevice")
        self.InputStream = FakeInputStream
        self.played: List[Tuple[np.ndarray, int]] = []
        self.wait_calls = 0
        self.play_error: Optional[Exception] = None

    def play(self, data: np.ndarray, samplerate: int, **kwargs: Any) -> None:
        if self.play_error is not None:
            raise self.play_error
        self.played.append((np.asarray(data), samplerate))

    def wait(self) -> None:
        self.wait_calls += 1

    def query_devices(self) -> list:
        return [{"name": "fake-mic", "max_input_channels": 1}]

    def reset(self) -> None:
        self.played.clear()
        self.wait_calls = 0
        self.play_error = None
        FakeInputStream.scripted_blocks = []
        FakeInputStream.instances = []


# --------------------------------------------------------------------------- #
# webrtcvad
# --------------------------------------------------------------------------- #

#: frame durations (ms) the real webrtcvad accepts
VALID_FRAME_MS = (10, 20, 30)
#: sample rates (Hz) the real webrtcvad accepts
VALID_RATES = (8000, 16000, 32000, 48000)


class FakeVad:
    """Mirrors the real ``webrtcvad.Vad`` contract, including its rejections.

    The real library raises for any frame that is not exactly 10, 20 or 30 ms
    of 16-bit mono PCM at 8/16/32/48 kHz.  Reproducing that is the whole point:
    Jarvis feeds it 64 ms blocks.
    """

    #: decides whether a well-formed frame counts as speech
    speech_decider: Callable[[bytes], bool] = staticmethod(lambda buf: True)

    def __init__(self, aggressiveness: int = 0):
        if not 0 <= aggressiveness <= 3:
            raise ValueError("Invalid aggressiveness")
        self.aggressiveness = aggressiveness

    def set_mode(self, mode: int) -> None:
        self.aggressiveness = mode

    def is_speech(self, buf: bytes, sample_rate: int) -> bool:
        if sample_rate not in VALID_RATES:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        samples = len(buf) // 2  # 16-bit mono
        frame_ms = samples * 1000 / sample_rate
        if frame_ms not in VALID_FRAME_MS:
            raise ValueError("Error while processing frame")
        return type(self).speech_decider(buf)


class _WebrtcVadModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("webrtcvad")
        self.Vad = FakeVad

    def reset(self) -> None:
        FakeVad.speech_decider = staticmethod(lambda buf: True)


# --------------------------------------------------------------------------- #
# faster_whisper
# --------------------------------------------------------------------------- #

class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeTranscriptionInfo:
    def __init__(self, language: str = "en", language_probability: float = 0.99):
        self.language = language
        self.language_probability = language_probability


class FakeWhisperModel:
    """Records how it was constructed and called; returns scripted segments."""

    #: text the next transcribe() call splits into segments
    scripted_text: str = "hello world"
    #: raised by transcribe() when set
    transcribe_error: Optional[Exception] = None
    #: raised by the constructor when set
    load_error: Optional[Exception] = None
    instances: List["FakeWhisperModel"] = []

    def __init__(self, model_size_or_path: str, device: str = "cpu",
                 compute_type: str = "default", **kwargs: Any):
        if FakeWhisperModel.load_error is not None:
            raise FakeWhisperModel.load_error
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.kwargs = kwargs
        self.transcribe_calls: List[dict] = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio: Any, **kwargs: Any):
        if FakeWhisperModel.transcribe_error is not None:
            raise FakeWhisperModel.transcribe_error
        # The real model rejects an unknown language code outright.
        language = kwargs.get("language")
        if language is not None and language not in (
            "en", "fr", "es", "de", "it", "pt", "nl", "ja", "zh",
        ):
            raise ValueError(f"Invalid language code: {language}")
        self.transcribe_calls.append({"audio": audio, **kwargs})
        segments = [FakeSegment(f" {word}") for word in
                    FakeWhisperModel.scripted_text.split()]
        return iter(segments), FakeTranscriptionInfo()

    @classmethod
    def reset(cls) -> None:
        cls.scripted_text = "hello world"
        cls.transcribe_error = None
        cls.load_error = None
        cls.instances = []


class _FasterWhisperModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("faster_whisper")
        self.WhisperModel = FakeWhisperModel

    def reset(self) -> None:
        FakeWhisperModel.reset()


# --------------------------------------------------------------------------- #
# ollama
# --------------------------------------------------------------------------- #

class SubscriptableModel:
    """Stand-in for ollama's ``SubscriptableBaseModel`` response objects.

    The modern ollama client returns *pydantic* models that support mapping
    access (``response["message"]``, ``.get(...)``) as well as attribute
    access.  Crucially they are **not** ``dict`` subclasses, so
    ``json.dumps`` refuses them - which is exactly the behaviour Jarvis trips
    over.  A plain dict stub would hide that.
    """

    def __init__(self, **fields: Any):
        self.__dict__["_fields"] = dict(fields)

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._fields[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __iter__(self):
        return iter(self._fields)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.__dict__["_fields"][item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self._fields.get(key, default)

    def keys(self):
        return self._fields.keys()

    def model_dump(self) -> dict:
        return dict(self._fields)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SubscriptableModel):
            return self._fields == other._fields
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self._fields!r})"


class ToolCall(SubscriptableModel):
    """A tool call as the real client returns it: mapping-like, not JSON-safe."""


def make_chat_response(content: str = "", tool_calls: Optional[list] = None,
                       as_model: bool = True) -> Any:
    """Build a chat response in either the modern (model) or legacy (dict) shape."""
    message = {"content": content, "tool_calls": tool_calls or []}
    if not as_model:
        return {"message": message}
    return SubscriptableModel(message=SubscriptableModel(**message))


def make_tool_call(name: str, arguments: Optional[dict] = None,
                   as_model: bool = True) -> Any:
    """Build a tool call in either the modern (model) or legacy (dict) shape."""
    if not as_model:
        return {"function": {"name": name, "arguments": arguments or {}}}
    return ToolCall(function=SubscriptableModel(name=name,
                                               arguments=arguments or {}))


class _OllamaModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("ollama")
        self.calls: List[dict] = []
        self.responses: List[Any] = []
        self.error: Optional[Exception] = None

    def chat(self, **kwargs: Any) -> Any:
        # The real client serialises the payload immediately; snapshot the
        # message list so later mutations of the caller's history do not
        # rewrite what this call was given.
        recorded = dict(kwargs)
        if isinstance(recorded.get("messages"), list):
            recorded["messages"] = [dict(message) for message in recorded["messages"]]
        self.calls.append(recorded)
        if self.error is not None:
            raise self.error
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return {"message": {"content": "stub response", "tool_calls": []}}

    def reset(self) -> None:
        self.calls.clear()
        self.responses.clear()
        self.error = None


# --------------------------------------------------------------------------- #
# installation
# --------------------------------------------------------------------------- #

def install() -> dict:
    """Install every stub into ``sys.modules`` and return them by name."""
    modules = {
        "sounddevice": _SoundDeviceModule(),
        "webrtcvad": _WebrtcVadModule(),
        "faster_whisper": _FasterWhisperModule(),
        "ollama": _OllamaModule(),
    }
    for name, module in modules.items():
        sys.modules[name] = module
    return modules
