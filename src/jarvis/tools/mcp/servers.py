"""Reading the MCP server configuration.

The file format follows the ``mcpServers`` object other MCP hosts use, so an
existing configuration can be dropped in unchanged. Two keys are Jarvis's own:
``enabled`` and ``trust``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class Trust(str, Enum):
    """How far the user trusts a server, declared by them and not by it."""

    #: Read-only tools run without asking; writes are confirmed once.
    TRUSTED = "trusted"
    #: Every call is confirmed. The default: a server is third-party code.
    CONFIRM = "confirm"
    #: Only tools the server marks read-only are exposed at all.
    READONLY = "readonly"

    @classmethod
    def parse(cls, value: Any) -> Trust:
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            logger.warning(f"Unknown trust level {value!r}; falling back to 'confirm'")
            return cls.CONFIRM


class Transport(str, Enum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass(frozen=True)
class ServerConfig:
    """One configured MCP server."""

    name: str
    transport: Transport
    trust: Trust = Trust.CONFIRM
    enabled: bool = True

    # stdio
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    # http / sse
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


class ConfigError(ValueError):
    """The configuration file cannot be used as written."""


def _parse_entry(name: str, entry: dict[str, Any]) -> ServerConfig:
    if not isinstance(entry, dict):
        raise ConfigError(f"server '{name}': expected an object")

    declared = entry.get("transport")
    command = entry.get("command")
    url = entry.get("url")

    if declared:
        try:
            transport = Transport(str(declared).strip().lower())
        except ValueError:
            raise ConfigError(
                f"server '{name}': unknown transport {declared!r}"
            ) from None
    elif command:
        transport = Transport.STDIO
    elif url:
        transport = Transport.HTTP
    else:
        raise ConfigError(f"server '{name}': needs either 'command' or 'url'")

    if transport is Transport.STDIO and not command:
        raise ConfigError(f"server '{name}': stdio transport needs 'command'")
    if transport in (Transport.HTTP, Transport.SSE) and not url:
        raise ConfigError(f"server '{name}': {transport.value} transport needs 'url'")

    return ServerConfig(
        name=name,
        transport=transport,
        trust=Trust.parse(entry.get("trust", Trust.CONFIRM.value)),
        enabled=bool(entry.get("enabled", True)),
        command=command,
        args=[str(a) for a in entry.get("args", [])],
        env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
        cwd=entry.get("cwd"),
        url=url,
        headers={str(k): str(v) for k, v in (entry.get("headers") or {}).items()},
        timeout=float(entry.get("timeout", 30.0)),
    )


def load_servers(path: str | Path) -> list[ServerConfig]:
    """Read the server list. A missing file simply means no MCP servers.

    A malformed *entry* is skipped with a warning rather than taking the whole
    file down: one bad server should not cost the user the others.
    """
    config_path = Path(path).expanduser()
    if not config_path.exists():
        logger.debug(f"No MCP configuration at {config_path}")
        return []

    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Could not read {config_path}: {e}")
        return []

    entries = raw.get("mcpServers", raw) if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        logger.error(f"{config_path}: expected an 'mcpServers' object")
        return []

    servers: list[ServerConfig] = []
    for name, entry in entries.items():
        try:
            servers.append(_parse_entry(str(name), entry))
        except ConfigError as e:
            logger.error(f"Skipping MCP server: {e}")

    logger.info(f"Loaded {len(servers)} MCP server(s) from {config_path}")
    return servers
