"""Looking at an image, with a second model.

Separate from the chat model on purpose. The model answering questions may be
a text-only 70B on a NAS, and making vision conditional on swapping it would
mean choosing between "can see" and "can think". Ollama serves multimodal
models alongside text ones (`ollama pull llava`), so this asks a different
model the visual question and hands the answer back as a tool result - which
also means the description passes through the same policy and taint machinery
as anything else read off the disk.

That last part matters more than it looks. A picture can contain text, and
text in a picture can say "ignore your previous instructions". Visual prompt
injection is not hypothetical, and a description is untrusted content for the
same reason a fetched web page is.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from jarvis.config import settings
from jarvis.llm_providers import ProviderPool

#: What Ollama's multimodal models accept. Anything else is refused before a
#: megabyte of it is read off the disk and sent to a model that cannot use it.
SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")

DEFAULT_QUESTION = "Describe this image briefly and plainly."


class VisionUnavailable(RuntimeError):
    """No provider could answer - unreachable, or the model is not pulled."""


class VisionModel:
    """Asks a multimodal model about one image."""

    def __init__(
        self,
        model: str | None = None,
        providers: ProviderPool | None = None,
        max_bytes: int | None = None,
    ):
        self.model = model or settings.vision_model
        self.providers = providers if providers is not None else ProviderPool()
        self.max_bytes = max_bytes or settings.vision_max_bytes

    def read(self, path: Path) -> bytes:
        """The image bytes, or a reason it cannot be looked at."""
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"'{path.suffix or path.name}' is not an image format these "
                f"models read ({', '.join(SUPPORTED_SUFFIXES)})"
            )
        try:
            size = path.stat().st_size
        except OSError as e:
            raise ValueError(f"could not read {path}: {e}") from e

        if size > self.max_bytes:
            raise ValueError(
                f"{path.name} is {size // 1024} kB, over the "
                f"{self.max_bytes // 1024} kB limit"
            )
        if size == 0:
            raise ValueError(f"{path.name} is empty")

        try:
            return path.read_bytes()
        except OSError as e:
            raise ValueError(f"could not read {path}: {e}") from e

    async def describe(self, image: bytes, question: str = "") -> str:
        """Ask the vision model about one image.

        Providers are tried in order, as a chat turn is: the NAS being off
        should fall back to whatever is here, not fail the call.
        """
        prompt = (question or "").strip() or DEFAULT_QUESTION
        last_error: Exception | None = None

        for provider in await self.providers.candidates():
            try:
                answer = await self._ask(provider, image, prompt)
            except Exception as e:
                logger.warning(f"Vision failed on {provider.describe()}: {e}")
                self.providers.report_failure(provider, e)
                last_error = e
                continue

            self.providers.report_success(provider)
            return answer

        raise VisionUnavailable(
            f"no provider could run '{self.model}': {last_error}. "
            f"Pull it with: ollama pull {self.model}"
        )

    async def _ask(self, provider, image: bytes, prompt: str) -> str:
        client = self.providers.client_for(provider)
        response = await client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image]}],
            options={"temperature": settings.llm_temperature},
        )

        message = (
            response.get("message") if hasattr(response, "get")
            else getattr(response, "message", None)
        ) or {}
        content = (
            message.get("content") if hasattr(message, "get")
            else getattr(message, "content", "")
        )
        text = str(content or "").strip()

        if not text:
            raise VisionUnavailable(
                f"'{self.model}' returned nothing - is it a vision model?"
            )
        return text
