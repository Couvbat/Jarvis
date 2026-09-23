"""Asking about an image on disk.

The same path sandbox as the file tools, for the same reason: this reads a
file and puts what is in it in front of a model. A photo is a file like any
other, and ~/.ssh is no more readable for being asked about in pictures.

The result is untrusted content. A picture can contain text, and text in a
picture can say "ignore your previous instructions" - a screenshot of a web
page is a web page. It shares one origin with the other images, so looking at
a second one does not escalate again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from jarvis.policy.paths import PathPolicy
from jarvis.tools.schema import Risk, ToolResult, ToolSpec, namespaced
from jarvis.vision import VisionModel, VisionUnavailable

NAMESPACE = "vision"

#: One origin for every image: describing a second one is not the
#: exfiltration shape, and escalating it would be a prompt per photo.
ORIGIN = "an image on this machine"


class ImageTools:
    """The vision side of the toolbox."""

    def __init__(self, paths: PathPolicy, model: VisionModel):
        self.paths = paths
        self.model = model

    def precheck(self, arguments: dict[str, Any]) -> str | None:
        """Refuse before asking: a prompt spent on a refusal teaches people
        to wave prompts through."""
        raw = str(arguments.get("path") or "").strip()
        if not raw:
            return "no image path given"
        verdict = self.paths.check(raw)
        return None if verdict else verdict.reason

    def describe(self, path: str = "", question: str = "", **_: Any) -> ToolResult:
        """Look at an image and answer a question about it."""
        verdict = self.paths.check(str(path or "").strip())
        if not verdict:
            return ToolResult.error(verdict.reason)

        target = Path(verdict.path)
        if not target.is_file():
            return ToolResult.error(f"no such image: {target}")

        try:
            image = self.model.read(target)
        except ValueError as e:
            return ToolResult.error(str(e))

        try:
            # The handlers are synchronous - the registry runs them in a
            # worker thread - so this cannot simply await.
            answer = asyncio.run(self.model.describe(image, question))
        except VisionUnavailable as e:
            return ToolResult.error(str(e))
        except Exception as e:
            logger.error(f"Vision failed on {target}: {e}")
            return ToolResult.error(f"could not look at {target.name}: {e}")

        logger.info(f"Described {target}")
        return ToolResult(
            f"{target}:\n{answer}", untrusted=True, origin=ORIGIN
        )


def build_tools(paths: PathPolicy, model: VisionModel) -> list[ToolSpec]:
    """Build the vision tools."""
    tools = ImageTools(paths, model)
    return [
        ToolSpec(
            name=namespaced(NAMESPACE, "describe"),
            description=(
                "Look at an image file and answer a question about it, or "
                "describe it. Use this when the user asks what is in a "
                "picture, screenshot or photo on their machine."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the image file",
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "What to ask about it; omit for a plain "
                            "description"
                        ),
                    },
                },
                "required": ["path"],
            },
            handler=tools.describe,
            # Reads a file and changes nothing.
            risk=Risk.READ_ONLY,
            precheck=tools.precheck,
            scope_for=lambda arguments: str(
                Path(str(arguments.get("path") or ".")).parent
            ),
        ),
    ]
