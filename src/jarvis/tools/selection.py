"""Choosing which tools to put in front of the model this turn.

This exists because of a constraint specific to running locally. A modest MCP
setup exposes thirty or forty tools; their schemas are re-sent on every turn,
and past a dozen or so a small model stops picking correctly no matter how
much context it has. Offering fewer, better-chosen tools is worth more than
offering all of them.

Scoring is lexical: term overlap between what the user said and each tool's
name and description, weighted by how discriminative each term is across the
whole tool set. Tool descriptions are short and keyword-dense, and so are
spoken requests, which is the case lexical matching handles well. An embedding
backend would score better on paraphrase, at the cost of pulling in a deep
learning stack for the sake of ranking forty short strings - a trade worth
making on evidence from tests/eval, not in advance.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from loguru import logger

from jarvis.text_utils import expand, tokenise
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.schema import ToolSpec

#: A name match says far more than a description match.
NAME_WEIGHT = 3.0
DESCRIPTION_WEIGHT = 1.0
#: Argument names are a weak signal, but they do carry one.
PARAMETER_WEIGHT = 0.5
#: How much matching several distinct query terms is worth. Without this, one
#: strong name match beats two description matches, so "create a shopping list
#: file" ranks every *list* tool above the one that writes files.
COVERAGE_WEIGHT = 1.0


def _spec_terms(spec: ToolSpec) -> Counter:
    """Weighted terms describing one tool."""
    terms: Counter = Counter()
    for term in tokenise(spec.name):
        terms[term] += NAME_WEIGHT
    for term in tokenise(spec.description):
        terms[term] += DESCRIPTION_WEIGHT
    for parameter in (spec.input_schema.get("properties") or {}):
        for term in tokenise(str(parameter)):
            terms[term] += PARAMETER_WEIGHT
    return terms


@dataclass
class ToolSelector:
    """Picks the tools worth offering for a given utterance."""

    #: Below this many tools, everything is offered and none of this runs.
    #: Set generously: offering a slightly crowded toolbox beats offering a
    #: confidently wrong shortlist, and selection is only lexical.
    threshold: int = 20
    #: How many scored tools to offer above the threshold. Recall matters more
    #: than precision here - a tool that is missing cannot be chosen at all.
    top_k: int = 15
    #: Tools used recently are kept available: a follow-up like "and delete
    #: it" carries no keywords of its own.
    recent_size: int = 4

    _recent: list[str] = field(default_factory=list)
    _index: dict[str, Counter] = field(default_factory=dict)
    _idf: dict[str, float] = field(default_factory=dict)
    _indexed_names: tuple = ()

    # -- index ------------------------------------------------------------ #

    def _reindex(self, specs: Sequence[ToolSpec]) -> None:
        """Rebuild term weights. Cheap, and only when the tool set changes."""
        names = tuple(spec.name for spec in specs)
        if names == self._indexed_names:
            return

        self._index = {spec.name: _spec_terms(spec) for spec in specs}
        document_frequency: Counter = Counter()
        for terms in self._index.values():
            document_frequency.update(terms.keys())

        total = max(1, len(self._index))
        # A term on every tool ("file" across ten file tools) discriminates
        # nothing; one on a single tool is the whole signal.
        self._idf = {
            term: math.log(1 + total / count)
            for term, count in document_frequency.items()
        }
        self._indexed_names = names
        logger.debug(f"Indexed {len(self._index)} tools for selection")

    # -- scoring ---------------------------------------------------------- #

    def score(self, spec_name: str, query_terms: Iterable[str]) -> float:
        terms = self._index.get(spec_name)
        if not terms:
            return 0.0

        total = 0.0
        matched = set()
        for term in set(query_terms):
            weight = terms.get(term, 0.0)
            if weight:
                total += weight * self._idf.get(term, 0.0)
                matched.add(term)

        if not matched:
            return 0.0
        # Breadth of match, not just depth: a tool the request touches in two
        # places is usually the one meant.
        return total * (1 + COVERAGE_WEIGHT * (len(matched) - 1))

    def note_used(self, name: str) -> None:
        """Remember a tool the model just used."""
        if name in self._recent:
            self._recent.remove(name)
        self._recent.append(name)
        del self._recent[: max(0, len(self._recent) - self.recent_size)]

    def reset(self) -> None:
        self._recent = []

    # -- selection -------------------------------------------------------- #

    def select(
        self,
        registry: ToolRegistry,
        utterance: str = "",
        context: Sequence[str] | None = None,
    ) -> list[str] | None:
        """Tool names to offer, or None to offer everything.

        ``context`` is recent conversation text: "and delete it" only makes
        sense next to the turn that named the thing.
        """
        specs = registry.specs()
        if len(specs) <= self.threshold:
            return None

        self._reindex(specs)

        query_terms = tokenise(utterance)
        for line in (context or []):
            query_terms.extend(tokenise(line))
        # Tool descriptions are English; the user may not be.
        query_terms = expand(query_terms)

        scored = sorted(
            ((self.score(spec.name, query_terms), spec.name) for spec in specs),
            key=lambda pair: (-pair[0], pair[1]),
        )

        chosen: list[str] = []
        # Recently used tools first: a follow-up needs them and may not name
        # them, and they cost nothing when they would have scored anyway.
        for name in reversed(self._recent):
            if registry.get(name) is not None and name not in chosen:
                chosen.append(name)

        for score, name in scored:
            if len(chosen) >= self.top_k:
                break
            if name not in chosen and score > 0:
                chosen.append(name)

        # Never hand the model an empty toolbox because nothing matched: fall
        # back to registration order, which puts the general-purpose local
        # tools first.
        for spec in specs:
            if len(chosen) >= self.top_k:
                break
            if spec.name not in chosen:
                chosen.append(spec.name)

        logger.debug(f"Offering {len(chosen)} of {len(specs)} tools: {chosen}")
        return chosen


def estimate_schema_tokens(schemas: Sequence[dict]) -> int:
    """Roughly how much context the tool schemas will cost.

    Four characters per token is crude, but the point is to notice when the
    schemas alone are crowding out the conversation, and for that it is enough.
    """
    import json

    return sum(len(json.dumps(schema)) for schema in schemas) // 4
