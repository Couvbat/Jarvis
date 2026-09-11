"""Tool descriptions, risk levels and schema conversion.

A :class:`ToolSpec` is the single description of a capability, whatever
implements it: a local function today, an MCP server tomorrow. The registry
dispatches on it, the policy engine decides on it, and the model sees it as a
function schema. Keeping one description means a capability cannot be exposed
to the model without also being visible to the policy engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Optional


#: Separates the provider from the tool name: ``fs__read``, ``git__commit``.
#: MCP servers get a namespace each, so two servers can both expose "search".
NAMESPACE_SEPARATOR = "__"


class Risk(IntEnum):
    """How much damage a call could do. Ordered, so risks can be combined."""

    #: Observes without changing anything.
    READ_ONLY = 0
    #: Creates or modifies, recoverably.
    WRITE = 1
    #: Removes or overwrites; the previous state is gone.
    DESTRUCTIVE = 2


def namespaced(namespace: str, name: str) -> str:
    """Build a fully qualified tool name."""
    return f"{namespace}{NAMESPACE_SEPARATOR}{name}"


def split_name(name: str) -> tuple[Optional[str], str]:
    """Split a qualified tool name into (namespace, bare name)."""
    if NAMESPACE_SEPARATOR not in name:
        return None, name
    namespace, _, bare = name.partition(NAMESPACE_SEPARATOR)
    return namespace, bare


@dataclass
class ToolResult:
    """What a tool hands back.

    ``content`` goes to the model. ``untrusted`` tells the taint tracker that
    this content came from somewhere the user does not control, so anything the
    model does next on the strength of it deserves a second look.
    """

    content: str
    ok: bool = True
    untrusted: bool = False

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        """A failure the model should see and can act on."""
        return cls(content=f"Error: {message}", ok=False)


@dataclass(frozen=True)
class ToolSpec:
    """Everything the registry, the policy engine and the model need."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., ToolResult]

    #: Baseline risk. Defaults to the worst case: a capability that has not
    #: said what it does is not assumed to be harmless.
    risk: Risk = Risk.DESTRUCTIVE

    #: True when calling this sends data off the machine.
    egress: bool = False

    #: Refines the risk from the actual arguments. Overwriting an existing
    #: file is destructive; creating a new one is not, and only the arguments
    #: can tell the two apart. It may only raise the baseline, never lower it.
    risk_for: Optional[Callable[[Dict[str, Any]], Risk]] = None

    #: Claims made by the provider (e.g. MCP tool annotations). Recorded, not
    #: believed: how much weight they carry is the policy engine's call.
    hints: Dict[str, Any] = field(default_factory=dict)

    #: Cheap, deterministic reasons this call can never succeed - a path
    #: outside the sandbox, a command that is not whitelisted. Returns the
    #: reason, or None. The policy engine runs it before asking the user
    #: anything: a confirmation prompt spends the user's attention, and
    #: spending it on something that will be refused anyway trains them to
    #: wave prompts through. Anything needing I/O belongs in the handler.
    precheck: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None

    #: What an approval for a call should cover, derived from its arguments -
    #: a directory for a file write, a domain for a fetch. The tool knows this;
    #: the policy engine does not. Without one, approvals cover exactly the
    #: arguments given, which is the narrowest and safest default.
    scope_for: Optional[Callable[[Dict[str, Any]], str]] = None

    def assess(self, arguments: Dict[str, Any]) -> Risk:
        """The risk of calling this tool with these arguments."""
        if self.risk_for is None:
            return self.risk
        try:
            return max(self.risk, self.risk_for(arguments))
        except Exception:
            # A tool that cannot work out its own risk gets the worst case.
            return Risk.DESTRUCTIVE


def to_ollama_schema(spec: ToolSpec) -> Dict[str, Any]:
    """Render a spec as the function schema Ollama expects."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def from_mcp_tool(tool: Any, namespace: str, handler: Callable[..., ToolResult]) -> ToolSpec:
    """Build a spec from an MCP tool description.

    Duck-typed on purpose: it needs ``name``, ``description`` and
    ``input_schema``, which is what ``mcp.types.Tool`` exposes, so this is
    testable without a server and without importing the MCP SDK.

    The risk is always the worst case here. MCP annotations are declared by
    the server, which is third-party code; they are carried in ``hints`` for
    the policy engine to weigh against how much the user trusts that server.
    """
    annotations = getattr(tool, "annotations", None)
    hints = {}
    for hint in ("read_only_hint", "destructive_hint", "idempotent_hint", "open_world_hint"):
        value = getattr(annotations, hint, None)
        if value is not None:
            hints[hint] = value

    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)

    return ToolSpec(
        name=namespaced(namespace, tool.name),
        description=getattr(tool, "description", "") or "",
        input_schema=schema or {"type": "object", "properties": {}},
        handler=handler,
        risk=Risk.DESTRUCTIVE,
        egress=True,  # a server can be anywhere; assume the data leaves
        hints=hints,
    )
