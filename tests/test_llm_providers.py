"""Tests for ordered LLM providers and failover (llm_providers.py).

The shape being tested is a self-hosted Ollama on the LAN holding the big
model, with a small model on this machine for when the LAN is not there. What
matters is that Jarvis keeps answering when the first one goes away, *and*
comes back to it when it returns - a fallback that sticks for the rest of the
session is barely better than no fallback.
"""

import json
from pathlib import Path

import pytest

from llm_providers import (
    ProviderConfig,
    ProviderPool,
    ProviderState,
    load_providers,
)

NAS = "http://nas:11434"
LOCAL = "http://localhost:11434"


def pool(*providers, **kwargs):
    """A pool over explicit providers, with probing fast enough for a test."""
    kwargs.setdefault("probe_timeout", 1.0)
    kwargs.setdefault("recheck_seconds", 0.0)
    return ProviderPool(list(providers), **kwargs)


@pytest.fixture
def log_lines():
    """Everything Jarvis logs during the test, as plain strings."""
    from loguru import logger

    lines: list = []
    sink = logger.add(lines.append, level="DEBUG")
    yield lines
    logger.remove(sink)


def nas(**kwargs):
    return ProviderConfig(name="nas", host=NAS, model="llama3.1:70b", **kwargs)


def local(**kwargs):
    return ProviderConfig(name="local", host=LOCAL, model="llama3.2:3b", **kwargs)


class TestLoadFromSettings:
    """Two providers is the common case; it should not need a config file."""

    def test_one_provider_by_default(self, settings):
        settings.ollama_host = NAS
        settings.ollama_model = "llama3.1:70b"
        settings.ollama_fallback_host = ""
        settings.ollama_fallback_model = ""

        providers = load_providers()
        assert [p.host for p in providers] == [NAS]
        assert providers[0].model == "llama3.1:70b"
        assert providers[0].max_tools is None

    def test_a_fallback_host_adds_a_second_provider(self, settings):
        settings.ollama_host = NAS
        settings.ollama_model = "llama3.1:70b"
        settings.ollama_fallback_host = LOCAL
        settings.ollama_fallback_model = "llama3.2:3b"

        providers = load_providers()
        assert [(p.host, p.model) for p in providers] == [
            (NAS, "llama3.1:70b"),
            (LOCAL, "llama3.2:3b"),
        ]

    def test_the_preferred_provider_comes_first(self, settings):
        """Order is preference: the remote box is asked before the laptop."""
        settings.ollama_host = NAS
        settings.ollama_fallback_host = LOCAL
        assert load_providers()[0].host == NAS

    def test_a_fallback_model_alone_reuses_the_same_host(self, settings):
        """A smaller model on the same server is a legitimate fallback."""
        settings.ollama_host = LOCAL
        settings.ollama_model = "llama3.1:8b"
        settings.ollama_fallback_host = ""
        settings.ollama_fallback_model = "llama3.2:1b"

        providers = load_providers()
        assert [(p.host, p.model) for p in providers] == [
            (LOCAL, "llama3.1:8b"),
            (LOCAL, "llama3.2:1b"),
        ]

    def test_the_fallback_gets_a_smaller_toolbox(self, settings):
        """A 3B model picks badly from a 70B's tool list."""
        settings.ollama_fallback_host = LOCAL
        settings.ollama_fallback_max_tools = 6
        assert load_providers()[1].max_tools == 6

    def test_a_zero_cap_means_no_cap(self, settings):
        settings.ollama_fallback_host = LOCAL
        settings.ollama_fallback_max_tools = 0
        assert load_providers()[1].max_tools is None


class TestLoadFromFile:
    """More than two providers, or per-provider context windows, go in a file."""

    def write(self, tmp_path, payload):
        path = tmp_path / "llm_providers.json"
        path.write_text(json.dumps(payload))
        return path

    def test_a_file_replaces_the_settings(self, tmp_path, settings):
        settings.ollama_host = "http://ignored:11434"
        path = self.write(tmp_path, {"providers": [
            {"name": "nas", "host": NAS, "model": "qwen3:32b"},
            {"name": "laptop", "host": LOCAL, "model": "qwen3:4b", "max_tools": 5},
        ]})

        providers = load_providers(path)
        assert [p.name for p in providers] == ["nas", "laptop"]
        assert providers[1].max_tools == 5

    def test_a_bare_list_is_accepted(self, tmp_path):
        path = self.write(tmp_path, [{"host": NAS, "model": "qwen3:32b"}])
        assert load_providers(path)[0].host == NAS

    def test_per_provider_context_windows(self, tmp_path):
        path = self.write(tmp_path, {"providers": [
            {"host": NAS, "model": "qwen3:32b", "num_ctx": 32768},
        ]})
        assert load_providers(path)[0].num_ctx == 32768

    def test_a_provider_missing_its_model_is_skipped(self, tmp_path):
        """One bad entry must not cost the whole file."""
        path = self.write(tmp_path, {"providers": [
            {"host": NAS},
            {"host": LOCAL, "model": "qwen3:4b"},
        ]})
        assert [p.host for p in load_providers(path)] == [LOCAL]

    def test_unnamed_providers_still_get_distinct_names(self, tmp_path):
        """Names key the health table, so two blanks would collide."""
        path = self.write(tmp_path, {"providers": [
            {"host": NAS, "model": "a"},
            {"host": LOCAL, "model": "b"},
        ]})
        names = [p.name for p in load_providers(path)]
        assert len(set(names)) == 2

    def test_malformed_json_falls_back_to_the_settings(self, tmp_path, settings):
        settings.ollama_host = LOCAL
        path = tmp_path / "llm_providers.json"
        path.write_text("{ not json")
        assert [p.host for p in load_providers(path)] == [LOCAL]

    def test_an_empty_provider_list_falls_back_to_the_settings(self, tmp_path, settings):
        settings.ollama_host = LOCAL
        path = self.write(tmp_path, {"providers": []})
        assert [p.host for p in load_providers(path)] == [LOCAL]

    def test_a_missing_file_is_not_an_error(self, tmp_path, settings):
        settings.ollama_host = LOCAL
        assert load_providers(tmp_path / "absent.json")[0].host == LOCAL


class TestProbing:
    async def test_a_reachable_provider_with_its_model_is_ready(
        self, fake_ollama
    ):
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]
        state = await pool(nas()).probe(nas())
        assert state.usable is True
        assert state.detail == "ready"

    async def test_an_unreachable_provider_says_so(self, fake_ollama):
        fake_ollama.host_list_errors[NAS] = ConnectionError("no route to host")
        state = await pool(nas()).probe(nas())
        assert state.usable is False
        assert "unreachable" in state.detail

    async def test_a_missing_model_is_named(self, fake_ollama):
        fake_ollama.host_models[NAS] = ["mistral:7b"]
        state = await pool(nas()).probe(nas())
        assert state.reachable is True
        assert state.has_model is False
        assert "llama3.1:70b" in state.detail

    async def test_a_tag_difference_still_matches(self, fake_ollama):
        """"llama3.1" and "llama3.1:70b" are the same pull to a user."""
        fake_ollama.host_models[NAS] = ["llama3.1:latest"]
        assert (await pool(nas()).probe(nas())).has_model is True

    async def test_an_empty_listing_is_not_proof_of_absence(self, fake_ollama):
        """Skipping a server that answered, because its listing came back
        unreadable, falls back for no reason."""
        fake_ollama.host_models[NAS] = []
        assert (await pool(nas()).probe(nas())).usable is True

    async def test_a_hanging_provider_does_not_hold_up_startup(self, fake_ollama):
        """A NAS that is asleep answers nothing at all; the probe has to give
        up on its own, or nothing is ever spoken."""
        import asyncio

        class _Hang:
            async def list(self):
                await asyncio.sleep(30)

        group = ProviderPool(
            [nas()], probe_timeout=0.01, client_factory=lambda host: _Hang()
        )
        state = await asyncio.wait_for(group.probe(nas()), timeout=5)
        assert state.usable is False


class TestChoosingAProvider:
    async def test_the_preferred_provider_wins_when_it_is_up(self, fake_ollama):
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        group = pool(nas(), local())
        assert (await group.refresh()).host == NAS

    async def test_probing_stops_at_the_first_usable_provider(self, fake_ollama):
        """No point asking the laptop whether it is up when the NAS answered."""
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        await pool(nas(), local()).refresh()
        assert fake_ollama.list_hosts == [NAS]

    async def test_an_unreachable_preferred_provider_falls_back(self, fake_ollama):
        fake_ollama.host_list_errors[NAS] = ConnectionError("no route to host")
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        assert (await pool(nas(), local()).refresh()).host == LOCAL

    async def test_a_provider_without_its_model_is_skipped(self, fake_ollama):
        """Reachable is not enough: a server without the model cannot answer."""
        fake_ollama.host_models[NAS] = ["mistral:7b"]
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        assert (await pool(nas(), local()).refresh()).host == LOCAL

    async def test_nothing_usable_reports_nothing(self, fake_ollama):
        fake_ollama.host_list_errors[NAS] = ConnectionError("down")
        fake_ollama.host_list_errors[LOCAL] = ConnectionError("down")

        group = pool(nas(), local())
        assert await group.refresh() is None
        assert group.active is None

    async def test_an_empty_pool_is_survivable(self):
        assert await pool().refresh() is None

    async def test_the_health_table_covers_every_provider(self, fake_ollama):
        fake_ollama.host_list_errors[NAS] = ConnectionError("down")
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        group = pool(nas(), local())
        await group.refresh()
        report = group.describe()
        assert "unreachable" in report["nas"]
        assert report["local"] == "ready"


class TestStickinessAndRecovery:
    async def test_the_preferred_provider_is_not_re_probed_every_turn(
        self, fake_ollama
    ):
        """Probing before every answer would add a round trip to each turn."""
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]

        group = pool(nas(), local())
        await group.refresh()
        await group.refresh()
        assert fake_ollama.list_hosts == [NAS]

    async def test_a_fallback_is_kept_for_a_while(self, fake_ollama):
        """Re-probing a NAS that is off, once per turn, costs a timeout each
        time; it is only worth doing occasionally."""
        fake_ollama.host_list_errors[NAS] = ConnectionError("down")
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        group = pool(nas(), local(), recheck_seconds=3600)
        assert (await group.refresh()).host == LOCAL
        fake_ollama.list_hosts.clear()

        assert (await group.refresh()).host == LOCAL
        assert fake_ollama.list_hosts == []

    async def test_the_preferred_provider_is_taken_back_when_it_returns(
        self, fake_ollama
    ):
        """Otherwise the first time the NAS blinks you are stuck on the small
        model for the rest of the session."""
        fake_ollama.host_list_errors[NAS] = ConnectionError("down")
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        group = pool(nas(), local())
        assert (await group.refresh()).host == LOCAL

        del fake_ollama.host_list_errors[NAS]
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]
        assert (await group.refresh()).host == NAS

    async def test_falling_back_is_announced(self, fake_ollama, log_lines):
        """A session that quietly changes model is a session whose answers get
        worse for no visible reason."""
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        group = pool(nas(), local())
        await group.refresh()

        # What actually happens when the NAS goes away mid-session: the call
        # fails and the turn is finished by the next provider.
        group.report_failure(group.providers[0], ConnectionError("dropped"))
        group.report_success(group.providers[1])

        assert any("Falling back" in line for line in log_lines)

    async def test_coming_back_is_announced(self, fake_ollama, log_lines):
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]

        group = pool(nas(), local())
        group.report_success(group.providers[1])
        log_lines.clear()

        await group.refresh(force=True)
        assert any("Back on the preferred" in line for line in log_lines)

    async def test_the_serving_provider_is_announced_once(self, fake_ollama):
        """Once per change, not once per turn: a line before every answer is
        noise nobody reads."""
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]

        group = pool(nas(), local())
        await group.refresh()
        group.report_success(group.providers[0])
        group.report_success(group.providers[0])
        assert group._announced is group.providers[0]

    async def test_a_forced_refresh_looks_again_immediately(self, fake_ollama):
        """Startup asks for the truth, not for a cached answer."""
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]

        group = pool(nas(), recheck_seconds=3600)
        await group.refresh()
        fake_ollama.list_hosts.clear()

        await group.refresh(force=True)
        assert fake_ollama.list_hosts == [NAS]

    async def test_a_reported_failure_forces_a_fresh_look(self, fake_ollama):
        """A provider that just failed a real call is not trusted by a probe
        result from a minute ago."""
        fake_ollama.host_models[NAS] = ["llama3.1:70b"]
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        group = pool(nas(), local(), recheck_seconds=3600)
        await group.refresh()

        fake_ollama.host_list_errors[NAS] = ConnectionError("dropped")
        group.report_failure(group.providers[0], ConnectionError("dropped"))
        assert group.active is None
        assert (await group.refresh()).host == LOCAL

    def test_a_reported_failure_is_recorded_in_the_health_table(self):
        group = pool(nas(), local())
        group.report_failure(group.providers[0], TimeoutError("timed out"))
        assert "TimeoutError" in group.describe()["nas"]

    def test_a_reported_success_makes_a_provider_active(self):
        group = pool(nas(), local())
        group.report_success(group.providers[1])
        assert group.active is group.providers[1]
        assert group.describe()["local"] == "ready"

    def test_a_failure_on_an_inactive_provider_leaves_the_active_one_alone(self):
        group = pool(nas(), local())
        group.report_success(group.providers[0])
        group.report_failure(group.providers[1], ConnectionError("down"))
        assert group.active is group.providers[0]


class TestCandidates:
    async def test_the_active_provider_leads(self, fake_ollama):
        fake_ollama.host_list_errors[NAS] = ConnectionError("down")
        fake_ollama.host_models[LOCAL] = ["llama3.2:3b"]

        group = pool(nas(), local(), recheck_seconds=3600)
        candidates = await group.candidates()
        assert [p.host for p in candidates] == [LOCAL, NAS]

    async def test_every_provider_is_still_offered(self, fake_ollama):
        """A probe failure is not a verdict: the chat may work where list did
        not, so a turn is allowed to try them all."""
        fake_ollama.host_list_errors[NAS] = ConnectionError("down")
        fake_ollama.host_list_errors[LOCAL] = ConnectionError("down")

        group = pool(nas(), local())
        assert [p.host for p in await group.candidates()] == [NAS, LOCAL]


class TestClients:
    def test_a_client_is_built_once_per_provider(self, fake_ollama):
        group = pool(nas(), local())
        first = group.client_for(group.providers[0])
        assert group.client_for(group.providers[0]) is first
        assert fake_ollama.hosts == [NAS]

    def test_each_provider_gets_its_own_host(self, fake_ollama):
        group = pool(nas(), local())
        group.client_for(group.providers[0])
        group.client_for(group.providers[1])
        assert fake_ollama.hosts == [NAS, LOCAL]


class TestDescriptions:
    def test_a_provider_describes_its_model_and_host(self):
        assert "llama3.1:70b" in nas().describe()
        assert NAS in nas().describe()

    def test_an_unprobed_state_says_so(self):
        assert ProviderState().usable is False
        assert ProviderState().detail == "not probed"


@pytest.mark.parametrize("listing,model,expected", [
    ({"models": [{"model": "llama3.1:8b"}]}, "llama3.1:8b", True),
    ({"models": [{"model": "llama3.1:latest"}]}, "llama3.1:8b", True),
    ({"models": [{"name": "llama3.1:8b"}]}, "llama3.1:8b", True),
    ({"models": [{"model": "mistral:7b"}]}, "llama3.1:8b", False),
    ({"models": []}, "llama3.1:8b", True),
    ({}, "llama3.1:8b", True),
])
def test_model_matching(listing, model, expected):
    assert ProviderPool._lists_model(listing, model) is expected


class TestShippedExample:
    """The example file is the documentation people copy; if it stops parsing
    the docs are wrong and nobody finds out."""

    path = Path(__file__).resolve().parent.parent / "llm_providers.example.json"

    def test_it_parses(self):
        providers = load_providers(self.path)
        assert [p.name for p in providers] == ["nas", "laptop"]

    def test_it_shows_the_shape_worth_copying(self):
        """A remote box with the big model first, a small local one behind it
        with a smaller toolbox."""
        remote, fallback = load_providers(self.path)
        assert remote.num_ctx and remote.num_ctx > (fallback.num_ctx or 0)
        assert fallback.max_tools
