"""Tests for looking at images (jarvis/vision.py, tools/local/images.py).

A description of a picture is untrusted content for the same reason a fetched
web page is: a screenshot of a web page *is* a web page, and text in an image
can say "ignore your previous instructions". That, and the path sandbox, are
what these tests are mostly about - the model call itself is three lines.
"""

import pytest

from jarvis.policy.paths import PathPolicy
from jarvis.tools.local.images import ORIGIN, build_tools
from jarvis.tools.schema import Risk
from jarvis.vision import VisionModel, VisionUnavailable

#: A one-pixel PNG, so the bytes on disk are a real image rather than a lie
#: the test would have to keep consistent.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6360000002000100ffff03000006"
    "0005a4cbd80000000049454e44ae426082"
)


@pytest.fixture
def picture(sandbox):
    path = sandbox / "photo.png"
    path.write_bytes(PNG)
    return path


@pytest.fixture
def tool(sandbox, fake_ollama):
    paths = PathPolicy([sandbox])
    return build_tools(paths, VisionModel(model="llava"))[0]


class TestReadingTheFile:
    def model(self, **kwargs):
        return VisionModel(model="llava", **kwargs)

    def test_an_image_is_read(self, picture):
        assert self.model().read(picture) == PNG

    def test_a_non_image_is_refused_before_it_is_read(self, sandbox):
        """Sending a megabyte of ZIP to a model that cannot use it wastes the
        time and produces confident nonsense."""
        path = sandbox / "archive.zip"
        path.write_bytes(b"PK\x03\x04not an image")
        with pytest.raises(ValueError) as raised:
            self.model().read(path)
        assert ".png" in str(raised.value)

    def test_an_oversized_image_is_refused(self, picture):
        with pytest.raises(ValueError) as raised:
            self.model(max_bytes=10).read(picture)
        assert "limit" in str(raised.value)

    def test_an_empty_file_is_refused(self, sandbox):
        path = sandbox / "empty.png"
        path.write_bytes(b"")
        with pytest.raises(ValueError):
            self.model().read(path)

    def test_a_missing_file_is_refused(self, sandbox):
        with pytest.raises(ValueError):
            self.model().read(sandbox / "absent.png")

    @pytest.mark.parametrize("suffix", [".png", ".jpg", ".JPEG", ".webp"])
    def test_the_usual_formats_are_accepted(self, sandbox, suffix):
        path = sandbox / f"photo{suffix}"
        path.write_bytes(PNG)
        assert self.model().read(path) == PNG


class TestAskingTheModel:
    async def test_it_asks_the_vision_model(self, fake_ollama):
        fake_ollama.responses.append(
            {"message": {"content": "A cat on a sofa.", "tool_calls": []}}
        )
        answer = await VisionModel(model="llava").describe(PNG, "what is this?")
        assert answer == "A cat on a sofa."
        assert fake_ollama.calls[-1]["model"] == "llava"

    async def test_the_image_reaches_the_model(self, fake_ollama):
        fake_ollama.responses.append(
            {"message": {"content": "Something.", "tool_calls": []}}
        )
        await VisionModel(model="llava").describe(PNG)
        assert fake_ollama.calls[-1]["messages"][0]["images"] == [PNG]

    async def test_no_question_still_asks_something(self, fake_ollama):
        """An empty prompt gets an empty answer out of most models."""
        fake_ollama.responses.append(
            {"message": {"content": "A photo.", "tool_calls": []}}
        )
        await VisionModel(model="llava").describe(PNG, "   ")
        assert fake_ollama.calls[-1]["messages"][0]["content"].strip()

    async def test_the_chat_model_is_not_used(self, fake_ollama, settings):
        """The point of a separate model: a text-only 70B on the NAS should
        keep answering the questions."""
        settings.ollama_model = "llama3.1:70b"
        fake_ollama.responses.append(
            {"message": {"content": "A cat.", "tool_calls": []}}
        )
        await VisionModel(model="llava").describe(PNG)
        assert fake_ollama.calls[-1]["model"] == "llava"

    async def test_it_falls_back_like_a_chat_turn(self, fake_ollama, settings):
        settings.ollama_host = "http://nas:11434"
        settings.ollama_fallback_host = "http://localhost:11434"
        fake_ollama.host_models["http://nas:11434"] = ["llama3.1:70b"]
        fake_ollama.host_models["http://localhost:11434"] = ["llama3.2:3b"]
        fake_ollama.host_chat_errors["http://nas:11434"] = ConnectionError("down")
        fake_ollama.responses.append(
            {"message": {"content": "A cat.", "tool_calls": []}}
        )

        assert await VisionModel(model="llava").describe(PNG) == "A cat."
        assert fake_ollama.chat_hosts[-1] == "http://localhost:11434"

    async def test_no_provider_names_the_pull_command(self, fake_ollama):
        fake_ollama.error = ConnectionError("refused")
        with pytest.raises(VisionUnavailable) as raised:
            await VisionModel(model="llava").describe(PNG)
        assert "ollama pull llava" in str(raised.value)

    async def test_a_text_model_asked_to_look_is_caught(self, fake_ollama):
        """Pointing VISION_MODEL at llama3.1 is an easy mistake."""
        fake_ollama.responses.append(
            {"message": {"content": "", "tool_calls": []}}
        )
        with pytest.raises(VisionUnavailable):
            await VisionModel(model="llava").describe(PNG)


class TestTheTool:
    def test_it_describes_an_image(self, tool, picture, fake_ollama):
        fake_ollama.responses.append(
            {"message": {"content": "A cat on a sofa.", "tool_calls": []}}
        )
        result = tool.handler(path=str(picture), question="what is this?")
        assert result.ok is True
        assert "A cat on a sofa." in result.content

    def test_the_answer_names_the_file(self, tool, picture, fake_ollama):
        fake_ollama.responses.append(
            {"message": {"content": "A cat.", "tool_calls": []}}
        )
        assert "photo.png" in tool.handler(path=str(picture)).content

    def test_the_description_is_untrusted(self, tool, picture, fake_ollama):
        """A screenshot of a web page is a web page. Text in an image can say
        "ignore your previous instructions", and it is read out to a model."""
        fake_ollama.responses.append(
            {"message": {"content": "A cat.", "tool_calls": []}}
        )
        result = tool.handler(path=str(picture))
        assert result.untrusted is True
        assert result.origin == ORIGIN

    def test_two_images_share_one_origin(self, tool, sandbox, fake_ollama):
        """Looking at a second photo is not the exfiltration shape, and a
        prompt per photo is how people learn to stop reading prompts."""
        for name in ("one.png", "two.png"):
            (sandbox / name).write_bytes(PNG)
        fake_ollama.responses.extend([
            {"message": {"content": "First.", "tool_calls": []}},
            {"message": {"content": "Second.", "tool_calls": []}},
        ])
        first = tool.handler(path=str(sandbox / "one.png"))
        second = tool.handler(path=str(sandbox / "two.png"))
        assert first.origin == second.origin

    def test_it_only_reads(self, tool):
        assert tool.risk is Risk.READ_ONLY
        assert tool.egress is False

    def test_the_sandbox_applies(self, tool, tmp_path):
        """A photo is a file like any other, and ~/.ssh is no more readable
        for being asked about in pictures."""
        outside = tmp_path / "elsewhere.png"
        outside.write_bytes(PNG)
        result = tool.handler(path=str(outside))
        assert result.ok is False
        assert "outside the allowed directories" in result.content

    def test_a_refusal_costs_no_confirmation(self, tool, tmp_path):
        """The precheck runs before the user is asked anything."""
        assert tool.precheck({"path": str(tmp_path / "elsewhere.png")})

    def test_an_allowed_path_passes_the_precheck(self, tool, picture):
        assert tool.precheck({"path": str(picture)}) is None

    def test_no_path_at_all_is_refused(self, tool):
        assert tool.precheck({}) == "no image path given"
        assert tool.handler(path="").ok is False

    def test_a_missing_file_is_reported(self, tool, sandbox):
        assert tool.handler(path=str(sandbox / "absent.png")).ok is False

    def test_a_non_image_is_reported(self, tool, sandbox):
        path = sandbox / "notes.txt"
        path.write_text("not a picture")
        assert tool.handler(path=str(path)).ok is False

    def test_an_unreachable_model_is_reported_not_raised(
        self, tool, picture, fake_ollama
    ):
        fake_ollama.error = ConnectionError("refused")
        result = tool.handler(path=str(picture))
        assert result.ok is False
        assert "ollama pull" in result.content

    def test_an_unexpected_failure_is_reported(
        self, tool, picture, fake_ollama, monkeypatch
    ):
        def explode(*args, **kwargs):
            raise MemoryError("out of memory")

        monkeypatch.setattr(tool.handler.__self__.model, "describe", explode)
        result = tool.handler(path=str(picture))
        assert result.ok is False
        assert "could not look at" in result.content


class TestRegistration:
    def test_it_is_off_by_default(self, settings, sandbox):
        """A separate model to pull; a text-only setup should not be offered
        a tool that cannot work."""
        from jarvis.tools.builtin import build_default_registry

        assert not any(
            name.startswith("vision__") for name in build_default_registry().names()
        )

    def test_turning_it_on_registers_the_tool(self, settings, sandbox, fake_ollama):
        from jarvis.tools.builtin import build_default_registry

        settings.vision = True
        assert "vision__describe" in build_default_registry().names()
