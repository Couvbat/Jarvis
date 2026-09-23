"""Tests for reminders (jarvis/reminders.py, tools/local/schedule.py).

The interesting cases are all about time: a reminder set for the past, one
that came due while Jarvis was off, one from last month that should not
suddenly arrive. The tool takes a timestamp rather than parsing language,
because the model in the loop is better at that than a regex would be - so
what is tested here is that a *wrong* timestamp is caught and handed back.
"""

from datetime import datetime, timedelta, timezone

import pytest

from jarvis.reminders import (
    MAX_HORIZON,
    MAX_LATENESS,
    ReminderStore,
    resolve_due_time,
)
from jarvis.tools.local.schedule import build_tools
from jarvis.tools.schema import Risk


@pytest.fixture
def store(tmp_path):
    return ReminderStore(tmp_path / "reminders.db")


def soon(minutes=30):
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


def ago(minutes=30):
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


class TestResolvingTheTime:
    """The model converts what was said; this checks what it hands over."""

    def test_minutes_from_now(self):
        due = resolve_due_time(in_minutes=30)
        assert timedelta(minutes=29) < due - datetime.now().astimezone() <= timedelta(
            minutes=30
        )

    def test_an_iso_timestamp(self):
        target = (datetime.now().astimezone() + timedelta(days=1)).replace(
            microsecond=0
        )
        assert resolve_due_time(when=target.isoformat()) == target

    def test_a_naive_timestamp_means_local_time(self):
        """A model writing "2026-09-23T09:00" means nine o'clock here, not
        nine o'clock UTC."""
        target = datetime.now().astimezone() + timedelta(days=1)
        naive = target.replace(tzinfo=None, microsecond=0)
        assert resolve_due_time(when=naive.isoformat()).utcoffset() == (
            target.utcoffset()
        )

    def test_a_trailing_z_is_understood(self):
        target = (datetime.now(timezone.utc) + timedelta(hours=2)).replace(
            microsecond=0
        )
        resolved = resolve_due_time(when=target.isoformat().replace("+00:00", "Z"))
        assert resolved == target

    def test_the_past_is_refused(self):
        """Better to say so than to set a reminder that fires instantly."""
        with pytest.raises(ValueError) as raised:
            resolve_due_time(when=ago(60).isoformat())
        assert "passed" in str(raised.value)

    def test_zero_minutes_is_refused(self):
        with pytest.raises(ValueError):
            resolve_due_time(in_minutes=0)

    def test_negative_minutes_are_refused(self):
        with pytest.raises(ValueError):
            resolve_due_time(in_minutes=-10)

    def test_the_far_future_is_refused(self):
        """A model that computed "in 5 minutes" as the year 3000 should be
        told, not obeyed."""
        far = datetime.now(timezone.utc) + MAX_HORIZON + timedelta(days=1)
        with pytest.raises(ValueError) as raised:
            resolve_due_time(when=far.isoformat())
        assert "year away" in str(raised.value)

    def test_unparseable_text_says_what_would_work(self):
        with pytest.raises(ValueError) as raised:
            resolve_due_time(when="demain vers neuf heures")
        assert "YYYY-MM-DD" in str(raised.value)

    def test_nonsense_minutes_are_refused(self):
        with pytest.raises(ValueError):
            resolve_due_time(in_minutes="soon")

    def test_saying_nothing_at_all_is_refused(self):
        with pytest.raises(ValueError) as raised:
            resolve_due_time()
        assert "say when" in str(raised.value)

    def test_minutes_win_over_a_timestamp(self):
        """Given both, the number is the one a model gets right."""
        due = resolve_due_time(when=soon(600).isoformat(), in_minutes=5)
        assert due - datetime.now().astimezone() < timedelta(minutes=6)


class TestStore:
    def test_a_reminder_comes_back(self, store):
        store.add("call the plumber", soon())
        assert [r.text for r in store.pending()] == ["call the plumber"]

    def test_pending_is_soonest_first(self, store):
        store.add("later", soon(120))
        store.add("sooner", soon(10))
        assert [r.text for r in store.pending()] == ["sooner", "later"]

    def test_nothing_is_due_before_its_time(self, store):
        store.add("later", soon(120))
        assert store.due() == []

    def test_a_passed_time_is_due(self, store):
        store.add("now", ago(1))
        assert [r.text for r in store.due()] == ["now"]

    def test_a_fired_reminder_stops_being_pending(self, store):
        reminder = store.add("once", ago(1))
        store.mark_fired(reminder.id)
        assert store.pending() == []
        assert store.due() == []

    def test_cancelling_removes_it(self, store):
        reminder = store.add("never mind", soon())
        assert store.cancel(reminder.id) is True
        assert store.pending() == []

    def test_cancelling_twice_says_no(self, store):
        reminder = store.add("once", soon())
        store.cancel(reminder.id)
        assert store.cancel(reminder.id) is False

    def test_cancelling_something_that_never_existed(self, store):
        assert store.cancel(999) is False

    def test_a_fired_reminder_cannot_be_cancelled(self, store):
        """There is nothing left to cancel, and saying so is better than
        pretending."""
        reminder = store.add("done", ago(1))
        store.mark_fired(reminder.id)
        assert store.cancel(reminder.id) is False

    def test_a_very_old_reminder_is_not_delivered(self, store):
        """Waking up to a month of backlog would be worse than losing it."""
        store.add("ancient history", ago(MAX_LATENESS.total_seconds() / 60 + 60))
        assert store.due() == []
        assert len(store.pending()) == 1

    def test_one_set_on_friday_still_arrives_on_monday(self, store):
        store.add("weekend", ago(60 * 24 * 3))
        assert [r.text for r in store.due()] == ["weekend"]

    def test_it_survives_a_restart(self, tmp_path):
        """The whole point: a reminder for tomorrow has to outlive the
        session that set it."""
        path = tmp_path / "reminders.db"
        ReminderStore(path).add("tomorrow", soon(60 * 24))
        assert [r.text for r in ReminderStore(path).pending()] == ["tomorrow"]

    def test_the_next_due_time_is_reported(self, store):
        store.add("later", soon(120))
        store.add("sooner", soon(10))
        assert store.next_due() is not None
        assert store.next_due() < soon(60)

    def test_nothing_scheduled_has_no_next_time(self, store):
        assert store.next_due() is None


class TestDescription:
    def test_today_reads_as_today(self, store):
        reminder = store.add("soon", soon(60))
        assert "today at" in reminder.describe()

    def test_tomorrow_reads_as_tomorrow(self, store):
        target = datetime.now().astimezone() + timedelta(days=1)
        reminder = store.add("tomorrow", target.replace(hour=9, minute=0))
        assert "tomorrow at" in reminder.describe()

    def test_further_out_names_the_day(self, store):
        reminder = store.add("next week", soon(60 * 24 * 6))
        described = reminder.describe()
        assert "today" not in described and "tomorrow" not in described

    def test_it_reads_the_text_back(self, store):
        assert "call the plumber" in store.add(
            "call the plumber", soon()
        ).describe()


class TestTools:
    def tools(self, store):
        return {spec.name: spec for spec in build_tools(store)}

    def test_setting_one(self, store):
        result = self.tools(store)["remind__set"].handler(
            text="call the plumber", in_minutes=30
        )
        assert result.ok is True
        assert "call the plumber" in result.content
        assert len(store.pending()) == 1

    def test_setting_one_with_a_timestamp(self, store):
        result = self.tools(store)["remind__set"].handler(
            text="meeting", when=soon(120).isoformat()
        )
        assert result.ok is True

    def test_a_reminder_needs_something_to_say(self, store):
        assert self.tools(store)["remind__set"].handler(
            text="  ", in_minutes=30
        ).ok is False

    def test_a_bad_time_is_handed_back_to_be_corrected(self, store):
        """So the model can try again instead of setting the wrong time."""
        result = self.tools(store)["remind__set"].handler(
            text="something", when=ago(60).isoformat()
        )
        assert result.ok is False
        assert "passed" in result.content
        assert store.pending() == []

    def test_a_storage_failure_is_reported_not_raised(self, store, monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(store, "add", refuse)
        result = self.tools(store)["remind__set"].handler(
            text="something", in_minutes=5
        )
        assert result.ok is False

    def test_listing_when_there_is_nothing(self, store):
        assert "no reminders" in self.tools(store)["remind__list"].handler().content

    def test_listing_shows_numbers_to_cancel_by(self, store):
        reminder = store.add("call the plumber", soon())
        content = self.tools(store)["remind__list"].handler().content
        assert f"{reminder.id}." in content
        assert "call the plumber" in content

    def test_a_long_list_is_not_read_out_in_full(self, store):
        """Twenty-five reminders spoken aloud is not a list, it is a
        punishment."""
        for index in range(25):
            store.add(f"thing {index}", soon(index + 1))
        content = self.tools(store)["remind__list"].handler().content
        assert "and 5 more" in content

    def test_cancelling_by_number(self, store):
        reminder = store.add("never mind", soon())
        result = self.tools(store)["remind__cancel"].handler(
            reminder_id=reminder.id
        )
        assert result.ok is True
        assert store.pending() == []

    def test_cancelling_something_that_is_not_there(self, store):
        assert self.tools(store)["remind__cancel"].handler(reminder_id=42).ok is False

    def test_cancelling_without_a_number(self, store):
        result = self.tools(store)["remind__cancel"].handler(reminder_id="the first")
        assert result.ok is False
        assert "number" in result.content

    def test_a_listing_failure_is_reported(self, store, monkeypatch):
        def refuse():
            raise OSError("database is locked")

        monkeypatch.setattr(store, "pending", refuse)
        assert self.tools(store)["remind__list"].handler().ok is False


class TestRisk:
    """These are the only tools that do something *later*, which changes what
    the user should be asked."""

    def tools(self, store):
        return {spec.name: spec for spec in build_tools(store)}

    def test_setting_one_is_a_write(self, store):
        """Nothing on disk is at stake and it can be cancelled, but it is a
        commitment made on the user's behalf."""
        assert self.tools(store)["remind__set"].risk is Risk.WRITE

    def test_listing_needs_no_permission(self, store):
        """The user's own reminders, set through this same assistant."""
        spec = self.tools(store)["remind__list"]
        assert spec.risk is Risk.READ_ONLY
        assert spec.preapproved is True

    def test_cancelling_is_destructive(self, store):
        """A cancelled reminder is a promise quietly dropped, and there is no
        way to notice until the moment it should have fired."""
        assert self.tools(store)["remind__cancel"].risk is Risk.DESTRUCTIVE

    def test_none_of_them_leave_the_machine(self, store):
        assert not any(spec.egress for spec in build_tools(store))
