"""Tests for approvals, taint tracking and the decision engine."""

import pytest

from policy.engine import Decision, PolicyEngine, Surface, summarise
from policy.store import ApprovalStore
from policy.taint import TaintState
from tools.schema import Risk, ToolResult, ToolSpec


def make_spec(name="fs__write", risk=Risk.WRITE, **overrides):
    defaults = dict(
        name=name,
        description="does something",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **kwargs: ToolResult("ok"),
        risk=risk,
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


@pytest.fixture
def store():
    instance = ApprovalStore()
    yield instance
    instance.close()


@pytest.fixture
def engine(store):
    return PolicyEngine(store)


@pytest.fixture
def tainted():
    state = TaintState()
    state.observe("web__fetch", ToolResult("page text", untrusted=True))
    return state


# --------------------------------------------------------------------------- #
# Taint
# --------------------------------------------------------------------------- #

class TestTaint:
    def test_starts_clean(self):
        assert TaintState().tainted is False

    def test_a_trusted_result_does_not_taint(self):
        state = TaintState()
        state.observe("fs__read", ToolResult("local file"))
        assert state.tainted is False

    def test_an_untrusted_result_taints(self):
        state = TaintState()
        state.observe("web__fetch", ToolResult("page", untrusted=True))
        assert state.tainted is True

    def test_the_source_is_recorded(self, tainted):
        assert tainted.sources == ["web__fetch"]

    def test_the_origin_is_recorded_when_given(self):
        state = TaintState()
        state.observe("web__fetch", ToolResult("p", untrusted=True, origin="a.test"))
        assert state.origins == ["a.test"]

    def test_the_tool_name_stands_in_for_a_missing_origin(self):
        state = TaintState()
        state.observe("web__fetch", ToolResult("p", untrusted=True))
        assert state.origins == ["web__fetch"]

    def test_origins_are_not_duplicated(self):
        state = TaintState()
        for _ in range(3):
            state.observe("web__fetch", ToolResult("p", untrusted=True, origin="a.test"))
        assert state.origins == ["a.test"]

    def test_knows_origin(self):
        state = TaintState()
        state.observe("web__fetch", ToolResult("p", untrusted=True, origin="a.test"))
        assert state.knows_origin("a.test") is True
        assert state.knows_origin("b.test") is False
        assert state.knows_origin(None) is False

    def test_reset_clears_origins(self, tainted):
        tainted.reset()
        assert tainted.origins == []

    def test_sources_are_not_duplicated(self):
        state = TaintState()
        for _ in range(3):
            state.observe("web__fetch", ToolResult("page", untrusted=True))
        assert state.sources == ["web__fetch"]

    def test_taint_is_sticky_within_a_turn(self, tainted):
        tainted.observe("fs__read", ToolResult("local file"))
        assert tainted.tainted is True

    def test_reset_clears_it(self, tainted):
        tainted.reset()
        assert tainted.tainted is False
        assert tainted.sources == []

    def test_describe_explains_why(self, tainted):
        assert "web__fetch" in tainted.describe()

    def test_describe_is_empty_when_clean(self):
        assert TaintState().describe() == ""


# --------------------------------------------------------------------------- #
# Approval store
# --------------------------------------------------------------------------- #

class TestApprovalStore:
    def test_nothing_is_approved_initially(self, store):
        assert store.is_approved("fs__write", "/tmp", Risk.WRITE) is False

    def test_approve_then_check(self, store):
        store.approve("fs__write", "/tmp", Risk.WRITE)
        assert store.is_approved("fs__write", "/tmp", Risk.WRITE) is True

    def test_approval_is_scoped(self, store):
        store.approve("fs__write", "/tmp/a", Risk.WRITE)
        assert store.is_approved("fs__write", "/tmp/b", Risk.WRITE) is False

    def test_approval_is_per_tool(self, store):
        store.approve("fs__write", "/tmp", Risk.WRITE)
        assert store.is_approved("fs__delete", "/tmp", Risk.WRITE) is False

    def test_a_lower_approval_does_not_cover_a_higher_risk(self, store):
        """Saying yes to a write is not saying yes to a later overwrite."""
        store.approve("fs__write", "/tmp", Risk.WRITE)
        assert store.is_approved("fs__write", "/tmp", Risk.DESTRUCTIVE) is False

    def test_a_higher_approval_covers_a_lower_risk(self, store):
        store.approve("fs__write", "/tmp", Risk.DESTRUCTIVE)
        assert store.is_approved("fs__write", "/tmp", Risk.READ_ONLY) is True

    def test_reapproving_widens_but_never_narrows(self, store):
        store.approve("fs__write", "/tmp", Risk.DESTRUCTIVE)
        store.approve("fs__write", "/tmp", Risk.READ_ONLY)
        assert store.is_approved("fs__write", "/tmp", Risk.DESTRUCTIVE) is True

    def test_revoke(self, store):
        store.approve("fs__write", "/tmp", Risk.WRITE)
        assert store.revoke("fs__write", "/tmp") is True
        assert store.is_approved("fs__write", "/tmp", Risk.WRITE) is False

    def test_revoking_nothing_reports_nothing(self, store):
        assert store.revoke("fs__write", "/tmp") is False

    def test_get_returns_the_record(self, store):
        store.approve("fs__write", "/tmp", Risk.WRITE)
        approval = store.get("fs__write", "/tmp")
        assert approval.risk is Risk.WRITE
        assert approval.granted_at

    def test_all_lists_approvals(self, store):
        store.approve("fs__write", "/tmp", Risk.WRITE)
        store.approve("web__fetch", "example.com", Risk.READ_ONLY)
        assert len(store.all()) == 2

    def test_clear(self, store):
        store.approve("fs__write", "/tmp", Risk.WRITE)
        assert store.clear() == 1
        assert store.all() == []

    def test_approvals_survive_a_restart(self, tmp_path):
        path = tmp_path / "approvals.db"
        first = ApprovalStore(path)
        first.approve("fs__write", "/tmp", Risk.WRITE)
        first.close()

        second = ApprovalStore(path)
        assert second.is_approved("fs__write", "/tmp", Risk.WRITE) is True
        second.close()

    def test_the_parent_directory_is_created(self, tmp_path):
        store = ApprovalStore(tmp_path / "nested" / "dir" / "approvals.db")
        store.approve("fs__write", "/tmp", Risk.WRITE)
        assert (tmp_path / "nested" / "dir" / "approvals.db").exists()
        store.close()


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #

class TestReadOnly:
    def test_first_use_asks_by_voice(self, engine):
        """The README promises every file operation is confirmed; reads
        used to run without asking at all."""
        decision = engine.evaluate(make_spec("fs__read", Risk.READ_ONLY), {"path": "/tmp/a"})
        assert decision.surface is Surface.VOICE

    def test_once_approved_it_runs_silently(self, engine):
        spec = make_spec("fs__read", Risk.READ_ONLY, scope_for=lambda a: "/tmp")
        first = engine.evaluate(spec, {"path": "/tmp/a"})
        engine.remember(first)
        assert engine.evaluate(spec, {"path": "/tmp/b"}).is_automatic is True

    def test_approval_does_not_leak_to_another_scope(self, engine):
        spec = make_spec("fs__read", Risk.READ_ONLY, scope_for=lambda a: a["path"])
        engine.remember(engine.evaluate(spec, {"path": "/tmp/a"}))
        assert engine.evaluate(spec, {"path": "/tmp/b"}).needs_confirmation is True


class TestWrite:
    def test_asks_by_voice(self, engine):
        decision = engine.evaluate(make_spec("fs__write", Risk.WRITE), {"path": "/tmp/a"})
        assert decision.surface is Surface.VOICE

    def test_can_be_approved_for_next_time(self, engine):
        spec = make_spec("fs__write", Risk.WRITE, scope_for=lambda a: "/tmp")
        engine.remember(engine.evaluate(spec, {"path": "/tmp/a"}))
        assert engine.evaluate(spec, {"path": "/tmp/b"}).is_automatic is True


class TestDestructive:
    def test_requires_the_keyboard(self, engine):
        """Speech recognition mishears, and ambient talk can supply a "yes"."""
        decision = engine.evaluate(make_spec("fs__delete", Risk.DESTRUCTIVE), {"path": "/tmp/a"})
        assert decision.surface is Surface.TERMINAL

    def test_is_never_stored_as_an_approval(self, engine, store):
        spec = make_spec("fs__delete", Risk.DESTRUCTIVE, scope_for=lambda a: "/tmp")
        engine.remember(engine.evaluate(spec, {"path": "/tmp/a"}))
        assert store.all() == []

    def test_asks_again_every_time(self, engine):
        spec = make_spec("fs__delete", Risk.DESTRUCTIVE, scope_for=lambda a: "/tmp")
        for _ in range(3):
            decision = engine.evaluate(spec, {"path": "/tmp/a"})
            engine.remember(decision)
            assert decision.surface is Surface.TERMINAL

    def test_an_argument_derived_escalation_is_honoured(self, engine):
        """Overwriting an existing file is destructive even for a write tool."""
        spec = make_spec(
            "fs__write", Risk.WRITE,
            risk_for=lambda a: Risk.DESTRUCTIVE if a.get("mode") == "overwrite" else Risk.WRITE,
            scope_for=lambda a: "/tmp",
        )
        engine.remember(engine.evaluate(spec, {"path": "/tmp/a", "mode": "create"}))
        escalated = engine.evaluate(spec, {"path": "/tmp/a", "mode": "overwrite"})
        assert escalated.surface is Surface.TERMINAL


class TestEgress:
    def test_asks_by_voice_when_untainted(self, engine):
        spec = make_spec("web__fetch", Risk.READ_ONLY, egress=True)
        decision = engine.evaluate(spec, {"url": "https://example.com"})
        assert decision.surface is Surface.VOICE
        assert "off the machine" in decision.reason

    def test_can_be_approved_per_domain(self, engine):
        spec = make_spec(
            "web__fetch", Risk.READ_ONLY, egress=True,
            scope_for=lambda a: "example.com",
        )
        engine.remember(engine.evaluate(spec, {"url": "https://example.com/a"}))
        assert engine.evaluate(spec, {"url": "https://example.com/b"}).is_automatic is True


class TestTaintEscalation:
    def test_a_write_is_escalated_to_the_keyboard(self, engine, tainted):
        decision = engine.evaluate(
            make_spec("fs__write", Risk.WRITE), {"path": "/tmp/a"}, tainted
        )
        assert decision.surface is Surface.TERMINAL
        assert "web__fetch" in decision.reason

    def test_egress_is_escalated_even_when_read_only(self, engine, tainted):
        """Reading a page then fetching another URL is the exfiltration shape."""
        spec = make_spec("web__fetch", Risk.READ_ONLY, egress=True)
        assert engine.evaluate(spec, {"url": "https://evil.test/?d=x"}, tainted).surface \
            is Surface.TERMINAL

    def test_a_standing_approval_does_not_survive_taint(self, engine, tainted):
        spec = make_spec("fs__write", Risk.WRITE, scope_for=lambda a: "/tmp")
        engine.remember(engine.evaluate(spec, {"path": "/tmp/a"}))
        assert engine.evaluate(spec, {"path": "/tmp/a"}).is_automatic is True
        assert engine.evaluate(spec, {"path": "/tmp/a"}, tainted).surface is Surface.TERMINAL

    def test_a_purely_local_read_is_not_escalated(self, engine, tainted):
        """Escalating everything would make confirmation fatigue the attack."""
        spec = make_spec("fs__read", Risk.READ_ONLY, scope_for=lambda a: "/tmp")
        engine.remember(engine.evaluate(spec, {"path": "/tmp/a"}))
        assert engine.evaluate(spec, {"path": "/tmp/a"}, tainted).is_automatic is True

    def test_a_clean_turn_is_not_escalated(self, engine):
        decision = engine.evaluate(
            make_spec("fs__write", Risk.WRITE), {"path": "/tmp/a"}, TaintState()
        )
        assert decision.surface is Surface.VOICE


class TestPreapproved:
    """Some authorisation is given in configuration, not at a prompt."""

    def test_a_preapproved_read_runs_without_asking(self, engine):
        spec = make_spec("mcp__peek", Risk.READ_ONLY, preapproved=True)
        assert engine.evaluate(spec, {}).is_automatic is True

    def test_a_preapproved_write_runs_without_asking(self, engine):
        spec = make_spec("mcp__put", Risk.WRITE, preapproved=True)
        assert engine.evaluate(spec, {}).is_automatic is True

    def test_it_never_waves_through_a_destructive_call(self, engine):
        spec = make_spec("mcp__wipe", Risk.DESTRUCTIVE, preapproved=True)
        assert engine.evaluate(spec, {}).surface is Surface.TERMINAL

    def test_it_does_not_survive_taint(self, engine, tainted):
        """Configured trust is not trust in whatever a web page just said."""
        spec = make_spec("mcp__put", Risk.WRITE, preapproved=True, egress=True)
        assert engine.evaluate(spec, {}, tainted).surface is Surface.TERMINAL

    def test_it_never_waves_through_a_refused_call(self, engine):
        spec = make_spec(preapproved=True, precheck=lambda a: "server is down")
        assert engine.evaluate(spec, {}).allowed is False

    def test_the_reason_says_where_the_authorisation_came_from(self, engine):
        spec = make_spec("mcp__peek", Risk.READ_ONLY, preapproved=True)
        assert "configuration" in engine.evaluate(spec, {}).reason


class TestPrecheck:
    """A call that cannot succeed is refused before the user is asked.

    A confirmation prompt spends the user's attention; spending it on
    something that will be refused anyway teaches them to wave prompts
    through, which is the failure mode the whole design is guarding against.
    """

    def test_a_refused_call_is_not_allowed(self, engine):
        spec = make_spec(precheck=lambda a: "outside the allowed directories")
        decision = engine.evaluate(spec, {"path": "/etc/passwd"})
        assert decision.allowed is False
        assert "outside the allowed directories" in decision.reason

    def test_a_refused_call_asks_for_nothing(self, engine):
        spec = make_spec(precheck=lambda a: "nope")
        decision = engine.evaluate(spec, {"path": "/etc/passwd"})
        assert decision.needs_confirmation is False
        assert decision.surface is Surface.NONE

    def test_a_refused_destructive_call_is_not_escalated(self, engine):
        """Refusal wins over the keyboard prompt: nothing to confirm."""
        spec = make_spec(risk=Risk.DESTRUCTIVE, precheck=lambda a: "nope")
        assert engine.evaluate(spec, {"path": "/etc"}).allowed is False

    def test_passing_the_precheck_proceeds_normally(self, engine):
        spec = make_spec(precheck=lambda a: None)
        decision = engine.evaluate(spec, {"path": "/tmp/a"})
        assert decision.allowed is True
        assert decision.surface is Surface.VOICE

    def test_a_precheck_that_raises_refuses_rather_than_crashing(self, engine):
        def explode(arguments):
            raise RuntimeError("cannot tell")

        decision = engine.evaluate(make_spec(precheck=explode), {"path": "/tmp/a"})
        assert decision.allowed is False

    def test_a_refused_call_is_never_stored_as_an_approval(self, engine, store):
        spec = make_spec(precheck=lambda a: "nope")
        decision = engine.evaluate(spec, {"path": "/etc/passwd"})
        if decision.allowed:  # pragma: no cover - guarded by the test above
            engine.remember(decision)
        assert store.all() == []


class TestOriginAwareEscalation:
    """Taint escalation asks where the content would go, not just whether the
    turn is dirty.

    Reading a second note from the server that just answered is not the
    exfiltration shape. Escalating it would turn every multi-step workflow
    into a wall of prompts, which is how people learn to stop reading them.
    """

    def egress_spec(self, name="mcp__read", origin="notes"):
        return make_spec(
            name, Risk.READ_ONLY, egress=True, origin_for=lambda a: origin
        )

    def tainted_by(self, origin):
        state = TaintState()
        state.observe("source", ToolResult("content", untrusted=True, origin=origin))
        return state

    def test_reading_more_from_the_same_source_is_not_escalated(self, engine):
        spec = self.egress_spec(origin="notes")
        decision = engine.evaluate(spec, {}, self.tainted_by("notes"))
        assert decision.surface is not Surface.TERMINAL

    def test_reaching_a_different_source_is_escalated(self, engine):
        spec = self.egress_spec(origin="evil.test")
        decision = engine.evaluate(spec, {}, self.tainted_by("example.com"))
        assert decision.surface is Surface.TERMINAL
        assert "evil.test" in decision.reason

    def test_a_tool_that_cannot_name_its_origin_is_escalated(self, engine):
        """Unknown destination is treated as a new one."""
        spec = make_spec("mcp__do", Risk.READ_ONLY, egress=True)
        assert engine.evaluate(spec, {}, self.tainted_by("notes")).surface \
            is Surface.TERMINAL

    def test_a_failing_origin_function_is_escalated(self, engine):
        def explode(arguments):
            raise RuntimeError("cannot tell")

        spec = make_spec("mcp__do", Risk.READ_ONLY, egress=True, origin_for=explode)
        assert engine.evaluate(spec, {}, self.tainted_by("notes")).surface \
            is Surface.TERMINAL

    def test_a_write_is_escalated_whatever_the_origin(self, engine):
        """Origin only softens egress; writing after untrusted content never
        gets a pass."""
        spec = make_spec("mcp__put", Risk.WRITE, egress=True, origin_for=lambda a: "notes")
        assert engine.evaluate(spec, {}, self.tainted_by("notes")).surface \
            is Surface.TERMINAL

    def test_several_origins_accumulate(self, engine):
        state = self.tainted_by("notes")
        state.observe("web__fetch", ToolResult("page", untrusted=True, origin="a.test"))
        assert engine.evaluate(self.egress_spec(origin="a.test"), {}, state).surface \
            is not Surface.TERMINAL
        assert engine.evaluate(self.egress_spec(origin="b.test"), {}, state).surface \
            is Surface.TERMINAL

    def test_a_preapproved_tool_still_loses_it_across_origins(self, engine):
        spec = make_spec(
            "mcp__read", Risk.READ_ONLY, egress=True,
            preapproved=True, origin_for=lambda a: "elsewhere",
        )
        assert engine.evaluate(spec, {}, self.tainted_by("notes")).surface \
            is Surface.TERMINAL


class TestScope:
    def test_a_tool_without_a_scope_grants_only_these_arguments(self, engine):
        """An unfamiliar tool - an MCP one, say - gets the narrowest grant."""
        spec = make_spec("mcp_x__do", Risk.WRITE)
        engine.remember(engine.evaluate(spec, {"target": "a"}))
        assert engine.evaluate(spec, {"target": "a"}).is_automatic is True
        assert engine.evaluate(spec, {"target": "b"}).needs_confirmation is True

    def test_argument_order_does_not_change_the_scope(self, engine):
        spec = make_spec("mcp_x__do", Risk.WRITE)
        first = engine.evaluate(spec, {"a": 1, "b": 2})
        second = engine.evaluate(spec, {"b": 2, "a": 1})
        assert first.scope == second.scope

    def test_a_failing_scope_function_falls_back(self, engine):
        def explode(arguments):
            raise RuntimeError("cannot tell")

        decision = engine.evaluate(make_spec(scope_for=explode), {"path": "/tmp/a"})
        assert decision.scope  # a usable fallback, not a crash


class TestSummary:
    def test_arguments_are_shown(self):
        """"delete /home/me/Documents, recursive=true" is a decision someone
        can make; "call fs__delete" is not."""
        rendered = summarise("fs__delete", {"path": "/home/me/Documents", "recursive": True})
        assert "/home/me/Documents" in rendered and "recursive=True" in rendered

    def test_long_values_are_trimmed(self):
        rendered = summarise("fs__write", {"content": "x" * 500})
        assert len(rendered) < 200

    def test_a_call_without_arguments_is_just_the_name(self):
        assert summarise("app__status", {}) == "app__status"

    def test_the_decision_carries_the_summary(self, engine):
        decision = engine.evaluate(make_spec(), {"path": "/tmp/secret.txt"})
        assert "/tmp/secret.txt" in decision.summary


class TestDecisionShape:
    def test_automatic_is_not_a_confirmation(self, engine):
        spec = make_spec("fs__read", Risk.READ_ONLY, scope_for=lambda a: "/tmp")
        engine.remember(engine.evaluate(spec, {"path": "/tmp/a"}))
        decision = engine.evaluate(spec, {"path": "/tmp/a"})
        assert decision.is_automatic is True
        assert decision.needs_confirmation is False

    def test_every_decision_explains_itself(self, engine):
        for risk in Risk:
            decision = engine.evaluate(make_spec(risk=risk), {"path": "/tmp/a"})
            assert decision.reason

    def test_no_arguments_is_handled(self, engine):
        assert isinstance(engine.evaluate(make_spec()), Decision)

    def test_the_engine_defaults_to_its_own_store(self):
        assert PolicyEngine().evaluate(make_spec(), {"path": "/tmp/a"}).needs_confirmation
