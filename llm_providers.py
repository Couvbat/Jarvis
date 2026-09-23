"""Where the model actually runs, in order of preference.

A self-hosted Ollama on the LAN and a small model on this machine are not the
same thing, and which one is reachable changes through the day. So providers
are a list: the first one that answers and has its model wins, and Jarvis goes
back to a preferred one as soon as it returns - otherwise the first time the
NAS blinks you are stuck on the small model for the rest of the session.

A provider is only skipped for being *unreachable* or missing its model. A
model that answers badly is not a failover condition: silently swapping models
mid-conversation because an answer looked wrong would be worse than the bad
answer.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ollama
from loguru import logger

from config import settings


@dataclass(frozen=True)
class ProviderConfig:
    """One Ollama endpoint and the model to ask it for."""

    name: str
    host: str
    model: str
    #: Caps the tools offered when this provider is in use. A 3B fallback
    #: chokes on a toolbox a 70B handles, and the point of falling back is to
    #: keep working, not to keep the same shape.
    max_tools: Optional[int] = None
    #: Context window for this model, if it differs from the global setting.
    num_ctx: Optional[int] = None

    def describe(self) -> str:
        return f"{self.name} ({self.model} at {self.host})"


@dataclass
class ProviderState:
    """What was last seen of a provider."""

    reachable: bool = False
    has_model: bool = False
    detail: str = "not probed"
    checked_at: float = 0.0

    @property
    def usable(self) -> bool:
        return self.reachable and self.has_model


def _parse_provider(index: int, entry: Dict[str, Any]) -> Optional[ProviderConfig]:
    host = str(entry.get("host") or "").strip()
    model = str(entry.get("model") or "").strip()
    if not host or not model:
        logger.error(f"LLM provider #{index}: needs both 'host' and 'model'")
        return None

    max_tools = entry.get("max_tools")
    num_ctx = entry.get("num_ctx")
    return ProviderConfig(
        name=str(entry.get("name") or f"provider{index}"),
        host=host,
        model=model,
        max_tools=int(max_tools) if max_tools else None,
        num_ctx=int(num_ctx) if num_ctx else None,
    )


def load_providers(path: Optional[Union[str, Path]] = None) -> List[ProviderConfig]:
    """The configured providers, most preferred first.

    A JSON file wins when it exists. Otherwise the settings are read, which
    covers the common shape - one remote, one local fallback - without asking
    anyone to write a file for two entries.
    """
    config_path = Path(path or settings.llm_providers_path).expanduser()

    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Could not read {config_path}: {e}")
            raw = None

        entries = raw.get("providers") if isinstance(raw, dict) else raw
        if isinstance(entries, list):
            providers = [
                provider
                for index, entry in enumerate(entries)
                if isinstance(entry, dict)
                and (provider := _parse_provider(index, entry)) is not None
            ]
            if providers:
                logger.info(
                    f"Loaded {len(providers)} LLM provider(s) from {config_path}"
                )
                return providers
            logger.error(f"{config_path} defines no usable provider; using settings")
        else:
            logger.error(f"{config_path}: expected a 'providers' list; using settings")

    providers = [ProviderConfig(
        name="primary",
        host=settings.ollama_host,
        model=settings.ollama_model,
    )]

    if settings.ollama_fallback_host or settings.ollama_fallback_model:
        providers.append(ProviderConfig(
            name="fallback",
            host=settings.ollama_fallback_host or settings.ollama_host,
            model=settings.ollama_fallback_model or settings.ollama_model,
            max_tools=settings.ollama_fallback_max_tools or None,
        ))

    return providers


class ProviderPool:
    """Keeps track of which providers are up, and hands out the best one."""

    def __init__(
        self,
        providers: Optional[List[ProviderConfig]] = None,
        probe_timeout: Optional[float] = None,
        recheck_seconds: Optional[float] = None,
        client_factory=None,
    ):
        self.providers = providers if providers is not None else load_providers()
        self.probe_timeout = (
            probe_timeout if probe_timeout is not None else settings.llm_probe_timeout
        )
        self.recheck_seconds = (
            recheck_seconds if recheck_seconds is not None
            else settings.llm_provider_recheck_seconds
        )
        self._factory = client_factory or (lambda host: ollama.AsyncClient(host=host))
        self._clients: Dict[str, Any] = {}
        self.state: Dict[str, ProviderState] = {
            provider.name: ProviderState() for provider in self.providers
        }
        self.active: Optional[ProviderConfig] = None
        #: the provider the user was last told about. Kept apart from
        #: ``active``, which a failure clears: otherwise the one message worth
        #: printing - that the model quietly changed - is the one that never
        #: prints.
        self._announced: Optional[ProviderConfig] = None
        self._last_full_probe = 0.0

    # -- clients ---------------------------------------------------------- #

    def client_for(self, provider: ProviderConfig):
        """The client for one provider, built once and kept."""
        if provider.name not in self._clients:
            self._clients[provider.name] = self._factory(provider.host)
        return self._clients[provider.name]

    @property
    def preferred(self) -> Optional[ProviderConfig]:
        return self.providers[0] if self.providers else None

    def describe(self) -> Dict[str, str]:
        """What is known about each provider, for the logs and the UI."""
        return {name: state.detail for name, state in self.state.items()}

    # -- probing ---------------------------------------------------------- #

    async def probe(self, provider: ProviderConfig) -> ProviderState:
        """Ask a provider whether it is there and has the model."""
        state = self.state.setdefault(provider.name, ProviderState())
        state.checked_at = time.monotonic()

        try:
            listing = await asyncio.wait_for(
                self.client_for(provider).list(), timeout=self.probe_timeout
            )
        except Exception as e:
            state.reachable = False
            state.has_model = False
            state.detail = f"unreachable: {type(e).__name__}"
            return state

        state.reachable = True
        state.has_model = self._lists_model(listing, provider.model)
        state.detail = (
            "ready" if state.has_model
            else f"reachable, but '{provider.model}' is not pulled"
        )
        return state

    @staticmethod
    def _lists_model(listing: Any, model: str) -> bool:
        """Whether a listing includes a model, ignoring the tag."""
        models = listing.get("models") if hasattr(listing, "get") else None
        if not models:
            # An empty or unreadable listing is not proof of absence; assume
            # the model is there rather than skipping a working provider.
            return True

        wanted = model.split(":")[0]
        for entry in models:
            name = str(
                (entry.get("model") if hasattr(entry, "get") else None)
                or (entry.get("name") if hasattr(entry, "get") else None)
                or ""
            )
            if name == model or name.split(":")[0] == wanted:
                return True
        return False

    async def refresh(self, force: bool = False) -> Optional[ProviderConfig]:
        """Probe in order and settle on the first usable provider.

        Probing stops at the first one that works: there is no point asking
        the laptop whether it is up when the NAS already answered.
        """
        if not self.providers:
            return None

        now = time.monotonic()
        if not force and self.active is not None:
            on_preferred = self.active is self.preferred
            recently = now - self._last_full_probe < self.recheck_seconds
            if on_preferred or recently:
                return self.active

        self._last_full_probe = now

        for provider in self.providers:
            state = await self.probe(provider)
            if state.usable:
                self.active = provider
                self._announce(provider)
                return provider

        self.active = None
        logger.error(f"No LLM provider is usable: {self.describe()}")
        return None

    def _announce(self, chosen: ProviderConfig) -> None:
        """Say once, and only on a change, which model is now answering."""
        previous, self._announced = self._announced, chosen
        if previous is chosen:
            return

        if previous is None:
            logger.info(f"Using LLM provider {chosen.describe()}")
        elif chosen is self.preferred:
            logger.info(f"Back on the preferred provider {chosen.describe()}")
        else:
            logger.warning(
                f"Falling back from {previous.describe()} to {chosen.describe()}"
            )

    # -- using ------------------------------------------------------------ #

    async def candidates(self) -> List[ProviderConfig]:
        """Providers to try for one turn, best first.

        The active one leads; the rest follow in configured order so a turn
        can still fail over without waiting for the next probe.
        """
        await self.refresh()
        ordered = [self.active] if self.active is not None else []
        ordered.extend(p for p in self.providers if p is not self.active)
        return ordered

    def report_failure(self, provider: ProviderConfig, error: Exception) -> None:
        """Record that a provider just failed a real call."""
        state = self.state.setdefault(provider.name, ProviderState())
        state.reachable = False
        state.has_model = False
        state.detail = f"failed: {type(error).__name__}"
        state.checked_at = time.monotonic()
        if self.active is provider:
            # Force the next turn to look again rather than trusting this one.
            self.active = None
            self._last_full_probe = 0.0
        logger.warning(f"{provider.describe()} failed: {error}")

    def report_success(self, provider: ProviderConfig) -> None:
        """Record that a provider just served a turn."""
        state = self.state.setdefault(provider.name, ProviderState())
        state.reachable = True
        state.has_model = True
        state.detail = "ready"
        state.checked_at = time.monotonic()
        self.active = provider
        # A turn can settle on a provider by failing over mid-flight, without
        # any probe having run; that is exactly the change worth announcing.
        self._announce(provider)
