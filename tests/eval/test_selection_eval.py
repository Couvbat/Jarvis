"""How often does tool selection put the right tool in front of the model?

This runs in CI: it needs no model and no network, only the labelled cases.
A tool that is not offered cannot be chosen at all, so recall is the number
that matters - precision only costs context.
"""

from __future__ import annotations

import pytest

from jarvis.policy.paths import PathPolicy
from jarvis.tools.local import apps, filesystem, web
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.schema import Risk, ToolResult, ToolSpec
from jarvis.tools.selection import ToolSelector
from tests.eval.cases import CASES, EXTRA_TOOLS

#: Floors, not targets. They exist to catch a regression in selection, so they
#: sit a little below what the current implementation achieves.
MIN_RECALL_AT_TOP_K = 0.95
MIN_RECALL_AT_5 = 0.85
MIN_RECALL_AT_1 = 0.55


@pytest.fixture(scope="module")
def registry():
    instance = ToolRegistry()
    instance.register_all(filesystem.build_tools(PathPolicy(["/tmp"])))
    instance.register_all(web.build_tools())
    instance.register_all(apps.build_tools(["firefox", "code"]))
    for name, description in EXTRA_TOOLS:
        instance.register(ToolSpec(
            name=name,
            description=description,
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kwargs: ToolResult("ok"),
            risk=Risk.WRITE,
        ))
    return instance


def recall_at(registry, k: int):
    """Fraction of cases whose expected tool is in the first k offered."""
    selector = ToolSelector(threshold=1, top_k=k)
    hits, misses = 0, []

    for case in CASES:
        selector.reset()
        chosen = selector.select(registry, case.utterance, case.context) or []
        if case.expected_tool in chosen[:k]:
            hits += 1
        else:
            misses.append((case.utterance, case.expected_tool, chosen[:k]))

    return hits / len(CASES), misses


def test_the_tool_set_is_big_enough_to_need_selecting(registry):
    """Below the threshold the selector is a no-op, so the numbers below would
    mean nothing."""
    assert len(registry.names()) > ToolSelector().threshold


def test_recall_at_top_k(registry):
    recall, misses = recall_at(registry, ToolSelector().top_k)
    assert recall >= MIN_RECALL_AT_TOP_K, f"recall {recall:.0%}; missed {misses}"


def test_recall_at_five(registry):
    recall, misses = recall_at(registry, 5)
    assert recall >= MIN_RECALL_AT_5, f"recall {recall:.0%}; missed {misses}"


def test_recall_at_one(registry):
    """Not a requirement - the model sees the whole shortlist - but a drop
    here means the ranking got worse."""
    recall, misses = recall_at(registry, 1)
    assert recall >= MIN_RECALL_AT_1, f"recall {recall:.0%}; missed {misses}"


def test_both_languages_are_served(registry):
    """English descriptions with French speech is the case naive matching
    fails outright, so it is measured separately."""
    selector = ToolSelector(threshold=1, top_k=5)
    for language in ("fr", "en"):
        cases = [case for case in CASES if case.language == language]
        hits = 0
        for case in cases:
            selector.reset()
            chosen = selector.select(registry, case.utterance, case.context) or []
            hits += case.expected_tool in chosen
        assert hits / len(cases) >= MIN_RECALL_AT_5, f"{language}: {hits}/{len(cases)}"


def test_selection_is_cheaper_than_offering_everything(registry):
    from jarvis.tools.selection import estimate_schema_tokens

    everything = estimate_schema_tokens(registry.describe())
    selected = estimate_schema_tokens(
        registry.describe(ToolSelector(threshold=1).select(registry, "lis un fichier"))
    )
    assert selected < everything
