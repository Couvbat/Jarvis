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
import zlib
from collections.abc import Callable
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# sounddevice
# --------------------------------------------------------------------------- #

class FakeInputStream:
    """Replays a scripted list of int16 blocks, then silence forever."""

    #: blocks handed to the next stream instance, set by tests
    scripted_blocks: list[np.ndarray] = []
    #: every stream instance created, for assertions on constructor kwargs
    instances: list["FakeInputStream"] = []

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

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
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
        self.played: list[tuple[np.ndarray, int]] = []
        self.wait_calls = 0
        self.play_error: Exception | None = None

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
    transcribe_error: Exception | None = None
    #: raised by the constructor when set
    load_error: Exception | None = None
    instances: list["FakeWhisperModel"] = []

    def __init__(self, model_size_or_path: str, device: str = "cpu",
                 compute_type: str = "default", **kwargs: Any):
        if FakeWhisperModel.load_error is not None:
            raise FakeWhisperModel.load_error
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.kwargs = kwargs
        self.transcribe_calls: list[dict] = []
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


def make_chat_response(content: str = "", tool_calls: list | None = None,
                       as_model: bool = True) -> Any:
    """Build a chat response in either the modern (model) or legacy (dict) shape."""
    message = {"content": content, "tool_calls": tool_calls or []}
    if not as_model:
        return {"message": message}
    return SubscriptableModel(message=SubscriptableModel(**message))


def make_tool_call(name: str, arguments: dict | None = None,
                   as_model: bool = True) -> Any:
    """Build a tool call in either the modern (model) or legacy (dict) shape."""
    if not as_model:
        return {"function": {"name": name, "arguments": arguments or {}}}
    return ToolCall(function=SubscriptableModel(name=name,
                                               arguments=arguments or {}))


class _AsyncStream:
    """Delivers a scripted response the way the real client streams it.

    The content is handed over in small slices, because a caller that only
    works when the whole answer arrives at once is not actually streaming.
    """

    def __init__(self, response: Any, slice_size: int = 7):
        self.response = response
        self.slice_size = slice_size

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        message = self.response.get("message", {}) or {}
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        for start in range(0, len(content), self.slice_size):
            yield SubscriptableModel(
                message=SubscriptableModel(
                    content=content[start:start + self.slice_size], tool_calls=[]
                )
            )

        yield SubscriptableModel(
            message=SubscriptableModel(content="", tool_calls=tool_calls)
        )


class StreamThenFail:
    """A scripted response that streams part of an answer and then breaks.

    A server that dies mid-generation is not the same failure as one that was
    never there: some of the answer has already been spoken, and that cannot
    be taken back by trying another provider.
    """

    def __init__(self, content: str, error: Exception):
        self.content = content
        self.error = error


class _FailingStream:
    """Yields ``response.content`` in slices, then raises."""

    def __init__(self, failure: StreamThenFail, slice_size: int = 7):
        self.failure = failure
        self.slice_size = slice_size

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        content = self.failure.content
        for start in range(0, len(content), self.slice_size):
            yield SubscriptableModel(
                message=SubscriptableModel(
                    content=content[start:start + self.slice_size], tool_calls=[]
                )
            )
        raise self.failure.error


class _AsyncClient:
    """Stand-in for ``ollama.AsyncClient``.

    Records the host it was built with, so a test can check that the
    configured OLLAMA_HOST actually reaches the client.  Behaviour can be
    scripted per host, because a pool of providers is only interesting when
    one of them answers and another does not.
    """

    def __init__(self, host: str | None = None, **kwargs: Any):
        self.host = host
        _OLLAMA_MODULE["instance"].hosts.append(host)

    async def chat(self, **kwargs: Any) -> Any:
        module = _OLLAMA_MODULE["instance"]
        module.chat_hosts.append(self.host)

        error = module.host_chat_errors.get(self.host)
        if error is not None:
            raise error

        response = module.chat(**kwargs)
        if isinstance(response, StreamThenFail):
            return _FailingStream(response)
        if kwargs.get("stream"):
            return _AsyncStream(response)
        return response

    async def embed(self, model: str = "", input: Any = "", **kwargs: Any) -> Any:
        """Mirrors ``AsyncClient.embed``: one vector per input, in a response
        object rather than a bare list."""
        module = _OLLAMA_MODULE["instance"]
        texts = [input] if isinstance(input, str) else list(input)
        module.embed_calls.append({"host": self.host, "model": model,
                                   "input": texts})

        error = module.host_embed_errors.get(self.host, module.embed_error)
        if error is not None:
            raise error

        return SubscriptableModel(
            embeddings=[module.embedding_for(text) for text in texts]
        )

    async def list(self) -> Any:
        module = _OLLAMA_MODULE["instance"]
        module.list_hosts.append(self.host)

        error = module.host_list_errors.get(self.host, module.list_error)
        if error is not None:
            raise error
        return SubscriptableModel(
            models=[
                SubscriptableModel(model=name)
                for name in module.host_models.get(self.host, module.models)
            ]
        )


#: Lets _AsyncClient reach the module object that owns the scripted responses.
_OLLAMA_MODULE: dict = {}


class _OllamaModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("ollama")
        self.calls: list[dict] = []
        self.responses: list[Any] = []
        self.hosts: list[str | None] = []
        self.error: Exception | None = None
        self.list_error: Exception | None = None
        self.models: list[str] = ["llama3.1:8b"]
        #: hosts a chat / list call was made against, in order
        self.chat_hosts: list[str | None] = []
        self.list_hosts: list[str | None] = []
        #: per-host overrides, for pools where one provider is down
        self.host_models: dict[str | None, list[str]] = {}
        self.host_list_errors: dict[str | None, Exception] = {}
        self.host_chat_errors: dict[str | None, Exception] = {}
        #: embedding calls made, and how to answer them
        self.embed_calls: list[dict] = []
        self.embed_error: Exception | None = None
        self.host_embed_errors: dict[str | None, Exception] = {}
        self.embed_dimensions = 8
        #: set to return a fixed vector regardless of the text
        self.embedding_override: list[float] | None = None
        self.AsyncClient = _AsyncClient
        self.StreamThenFail = StreamThenFail
        _OLLAMA_MODULE["instance"] = self

    def embedding_for(self, text: str) -> list:
        """A deterministic vector for one text.

        Derived from the characters, so texts that share words land near each
        other and a nearest-neighbour test measures something rather than
        asserting on noise.
        """
        if self.embedding_override is not None:
            return list(self.embedding_override)

        vector = [0.0] * self.embed_dimensions
        for word in str(text).lower().split():
            # crc32, not hash(): PYTHONHASHSEED randomises str hashing per
            # process, and a doubling stub is no use if it answers differently
            # on every run.
            vector[zlib.crc32(word.encode()) % self.embed_dimensions] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

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
        self.hosts.clear()
        self.chat_hosts.clear()
        self.list_hosts.clear()
        self.host_models.clear()
        self.host_list_errors.clear()
        self.host_chat_errors.clear()
        self.embed_calls.clear()
        self.host_embed_errors.clear()
        self.embed_error = None
        self.embed_dimensions = 8
        self.embedding_override = None
        self.error = None
        self.list_error = None
        self.models = ["llama3.1:8b"]


# --------------------------------------------------------------------------- #
# openwakeword
# --------------------------------------------------------------------------- #

class FakeWakeModel:
    """Mirrors ``openwakeword.model.Model``, including what it refuses.

    The real model rejects anything that is not an ndarray, and answers with a
    score per loaded model rather than a bare float.  It also accumulates
    audio internally, which is why Jarvis carves exact 1280-sample frames: the
    sizes it was fed are recorded here so a test can hold that to account.
    """

    #: scores handed out in order; the last one repeats once exhausted
    scripted_scores: list[float] = []
    #: every model instance built, for assertions on constructor kwargs
    instances: list["FakeWakeModel"] = []

    def __init__(self, wakeword_models: list | None = None, **kwargs: Any):
        if not wakeword_models:
            raise ValueError("no wakeword models specified")
        self.wakeword_models = list(wakeword_models)
        self.kwargs = kwargs
        self.scores = list(FakeWakeModel.scripted_scores)
        self.frame_sizes: list[int] = []
        self.resets = 0
        FakeWakeModel.instances.append(self)

    def predict(self, x: Any, **kwargs: Any) -> dict:
        if not isinstance(x, np.ndarray):
            raise ValueError(
                "The input audio data (x) must by a Numpy array, instead "
                f"received an object of type {type(x)}."
            )
        self.frame_sizes.append(len(x))
        score = self.scores.pop(0) if self.scores else 0.0
        return {self.wakeword_models[0]: score}

    def reset(self) -> None:
        self.resets += 1


class _OpenWakeWordModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("openwakeword")
        self.Model = FakeWakeModel
        #: raised by Model(...) when set, standing in for missing model files
        self.load_error: Exception | None = None

        model_module = types.ModuleType("openwakeword.model")
        model_module.Model = self._build
        sys.modules["openwakeword.model"] = model_module
        self.model = model_module

    def _build(self, *args: Any, **kwargs: Any) -> FakeWakeModel:
        if self.load_error is not None:
            raise self.load_error
        return FakeWakeModel(*args, **kwargs)

    def reset(self) -> None:
        self.load_error = None
        FakeWakeModel.scripted_scores = []
        FakeWakeModel.instances = []


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
        "openwakeword": _OpenWakeWordModule(),
    }
    for name, module in modules.items():
        sys.modules[name] = module
    return modules
