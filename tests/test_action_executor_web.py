"""Web fetching in ActionExecutor.fetch_web_page."""

import pytest
import requests
import responses


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


class TestFetching:
    def test_extracts_visible_text(self, executor, mocked_responses):
        mocked_responses.get(
            "https://example.com",
            body="<html><body><h1>Title</h1><p>Body text</p></body></html>",
            content_type="text/html",
        )
        result = executor.fetch_web_page("https://example.com")
        assert "Title" in result and "Body text" in result

    def test_strips_script_and_style(self, executor, mocked_responses):
        mocked_responses.get(
            "https://example.com",
            body=(
                "<html><head><style>.a{color:red}</style></head>"
                "<body><script>alert('xss')</script><p>Visible</p></body></html>"
            ),
            content_type="text/html",
        )
        result = executor.fetch_web_page("https://example.com")
        assert "Visible" in result
        assert "alert" not in result
        assert "color:red" not in result

    def test_sends_a_user_agent(self, executor, mocked_responses):
        mocked_responses.get("https://example.com", body="<p>ok</p>")
        executor.fetch_web_page("https://example.com")
        assert "Mozilla" in mocked_responses.calls[0].request.headers["User-Agent"]

    def test_long_page_is_truncated(self, executor, mocked_responses):
        mocked_responses.get("https://example.com", body="<p>" + "word " * 2000 + "</p>")
        result = executor.fetch_web_page("https://example.com")
        assert "(truncated)" in result

    def test_http_error_is_reported_not_raised(self, executor, mocked_responses):
        mocked_responses.get("https://example.com/missing", status=404)
        result = executor.fetch_web_page("https://example.com/missing")
        assert result.startswith("Error fetching web page")

    def test_connection_error_is_reported_not_raised(self, executor, mocked_responses):
        mocked_responses.get(
            "https://example.com", body=requests.exceptions.ConnectionError("boom")
        )
        result = executor.fetch_web_page("https://example.com")
        assert result.startswith("Error fetching web page")

    def test_timeout_is_reported_not_raised(self, executor, mocked_responses):
        mocked_responses.get("https://example.com", body=requests.exceptions.Timeout())
        result = executor.fetch_web_page("https://example.com")
        assert result.startswith("Error fetching web page")

    def test_a_timeout_is_always_passed(self, executor, monkeypatch):
        captured = {}

        def fake_get(url, **kwargs):
            captured.update(kwargs)
            raise requests.exceptions.Timeout()

        monkeypatch.setattr("action_executor.requests.get", fake_get)
        executor.fetch_web_page("https://example.com")
        assert captured.get("timeout") is not None


class TestConfirmation:
    def test_requires_confirmation_keyed_by_domain(
        self, executor, mocked_responses, approve_all
    ):
        mocked_responses.get("https://example.com/page", body="<p>ok</p>")
        executor.fetch_web_page("https://example.com/page")
        assert approve_all.calls[0][1] == "example.com"

    def test_denied_confirmation_makes_no_request(
        self, executor, mocked_responses, deny_all
    ):
        executor.confirmation_callback = deny_all
        result = executor.fetch_web_page("https://example.com")
        assert "cancelled by user" in result
        assert len(mocked_responses.calls) == 0

    def test_whitelisting_a_domain_covers_other_paths(
        self, executor, mocked_responses, approve_and_whitelist
    ):
        executor.confirmation_callback = approve_and_whitelist
        mocked_responses.get("https://example.com/a", body="<p>a</p>")
        mocked_responses.get("https://example.com/b", body="<p>b</p>")
        executor.fetch_web_page("https://example.com/a")
        executor.fetch_web_page("https://example.com/b")
        assert len(approve_and_whitelist.calls) == 1

    def test_whitelisting_a_domain_does_not_cover_others(
        self, executor, mocked_responses, approve_and_whitelist
    ):
        executor.confirmation_callback = approve_and_whitelist
        mocked_responses.get("https://example.com/a", body="<p>a</p>")
        mocked_responses.get("https://evil.test/b", body="<p>b</p>")
        executor.fetch_web_page("https://example.com/a")
        executor.fetch_web_page("https://evil.test/b")
        assert len(approve_and_whitelist.calls) == 2


# --------------------------------------------------------------------------- #
# Known gaps
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason="BUG-04: no URL scheme validation")
def test_non_http_scheme_should_be_rejected(executor):
    """``file://`` and friends should never reach ``requests``."""
    result = executor.fetch_web_page("file:///etc/passwd")
    assert "Error: unsupported URL scheme" in result


@pytest.mark.xfail(strict=True, reason="BUG-04: no SSRF guard on loopback/link-local")
def test_loopback_url_should_be_rejected(executor, mocked_responses):
    """A self-hosted box runs Ollama, dashboards and cloud metadata endpoints.

    Nothing stops the model from asking Jarvis to fetch them and read the
    answer back out loud.
    """
    mocked_responses.get("http://127.0.0.1:11434/api/tags", body="{}")
    result = executor.fetch_web_page("http://127.0.0.1:11434/api/tags")
    assert "Error" in result
    assert len(mocked_responses.calls) == 0


@pytest.mark.xfail(strict=True, reason="BUG-04: no response size cap before parsing")
def test_oversized_response_should_be_refused_before_parsing(executor, monkeypatch):
    """``requests.get`` downloads the whole body before the 2000-char trim."""
    seen = {}

    def fake_get(url, **kwargs):
        seen["stream"] = kwargs.get("stream")
        raise requests.exceptions.Timeout()

    monkeypatch.setattr("action_executor.requests.get", fake_get)
    executor.fetch_web_page("https://example.com")
    assert seen.get("stream") is True
