"""Tests for the web tool and its guards (tools/local/web.py)."""

import pytest
import requests
import responses

from jarvis.tools.local.web import WebTools, build_tools
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.schema import Risk


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        yield mock


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host to a public address unless a test says otherwise."""
    def fake_getaddrinfo(host, port, *args, **kwargs):
        mapping = {
            "localhost": "127.0.0.1",
            "metadata.internal": "169.254.169.254",
            "intranet.lan": "192.168.1.10",
        }
        return [(2, 1, 6, "", (mapping.get(host, "93.184.216.34"), 0))]

    monkeypatch.setattr("jarvis.tools.local.web.socket.getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def web(public_dns):
    return WebTools()


class TestFetching:
    def test_extracts_visible_text(self, web, mocked_responses):
        mocked_responses.get(
            "https://example.com",
            body="<html><body><h1>Title</h1><p>Body text</p></body></html>",
        )
        result = web.fetch("https://example.com")
        assert result.ok is True
        assert "Title" in result.content and "Body text" in result.content

    def test_strips_script_style_and_noscript(self, web, mocked_responses):
        mocked_responses.get("https://example.com", body=(
            "<html><head><style>.a{color:red}</style></head>"
            "<body><script>alert('x')</script><noscript>js off</noscript>"
            "<p>Visible</p></body></html>"
        ))
        content = web.fetch("https://example.com").content
        assert "Visible" in content
        assert "alert" not in content and "color:red" not in content
        assert "js off" not in content

    def test_results_are_marked_untrusted(self, web, mocked_responses):
        """Whoever wrote the page is not the user."""
        mocked_responses.get("https://example.com", body="<p>hi</p>")
        assert web.fetch("https://example.com").untrusted is True

    def test_sends_a_user_agent(self, web, mocked_responses):
        mocked_responses.get("https://example.com", body="<p>ok</p>")
        web.fetch("https://example.com")
        assert "Mozilla" in mocked_responses.calls[0].request.headers["User-Agent"]

    def test_http_errors_are_reported(self, web, mocked_responses):
        mocked_responses.get("https://example.com/missing", status=404)
        assert web.fetch("https://example.com/missing").ok is False

    def test_connection_errors_are_reported(self, web, mocked_responses):
        mocked_responses.get(
            "https://example.com", body=requests.exceptions.ConnectionError("boom")
        )
        assert web.fetch("https://example.com").ok is False

    def test_timeouts_are_reported(self, web, mocked_responses):
        mocked_responses.get("https://example.com", body=requests.exceptions.Timeout())
        assert web.fetch("https://example.com").ok is False

    def test_an_empty_url_is_reported(self, web):
        assert web.fetch("").ok is False


class TestSchemeValidation:
    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com",
        "data:text/html,<p>x</p>",
    ])
    def test_non_http_schemes_are_refused(self, web, url, mocked_responses):
        result = web.fetch(url)
        assert result.ok is False
        assert "unsupported URL scheme" in result.content
        assert len(mocked_responses.calls) == 0

    def test_a_url_without_a_host_is_refused(self, web):
        assert web.fetch("http:///nohost").ok is False


class TestSsrfGuard:
    @pytest.mark.parametrize("url", [
        "http://localhost:11434/api/tags",
        "http://metadata.internal/latest/meta-data/",
        "http://intranet.lan/admin",
    ])
    def test_internal_destinations_are_refused(self, web, url, mocked_responses):
        """Ollama, dashboards and cloud metadata all live on these addresses."""
        result = web.fetch(url)
        assert result.ok is False
        assert "local network" in result.content
        assert len(mocked_responses.calls) == 0

    @pytest.mark.parametrize("literal", [
        "http://127.0.0.1/", "http://10.0.0.5/", "http://192.168.1.1/",
        "http://169.254.169.254/", "http://[::1]/",
    ])
    def test_internal_ip_literals_are_refused(self, literal, monkeypatch, mocked_responses):
        def resolve_literally(host, port, *args, **kwargs):
            return [(2, 1, 6, "", (host.strip("[]"), 0))]

        monkeypatch.setattr("jarvis.tools.local.web.socket.getaddrinfo", resolve_literally)
        assert WebTools().fetch(literal).ok is False

    def test_the_guard_can_be_turned_off_deliberately(self, mocked_responses, public_dns):
        mocked_responses.get("http://localhost:11434/api/tags", body="{}")
        permissive = WebTools(allow_private_network=True)
        assert permissive.fetch("http://localhost:11434/api/tags").ok is True

    def test_an_unresolvable_host_is_refused(self, monkeypatch, mocked_responses):
        import socket as real_socket

        def fail(host, port, *args, **kwargs):
            raise real_socket.gaierror("no such host")

        monkeypatch.setattr("jarvis.tools.local.web.socket.getaddrinfo", fail)
        assert "could not resolve" in WebTools().fetch("https://nope.invalid").content


class TestRedirects:
    def test_redirects_are_followed(self, web, mocked_responses):
        mocked_responses.get(
            "https://example.com/a", status=302,
            headers={"Location": "https://example.com/b"},
        )
        mocked_responses.get("https://example.com/b", body="<p>arrived</p>")
        assert "arrived" in web.fetch("https://example.com/a").content

    def test_a_redirect_into_the_local_network_is_refused(self, web, mocked_responses):
        """Letting requests follow redirects would land the internal request
        before anyone had a chance to look at it."""
        mocked_responses.get(
            "https://example.com/a", status=302,
            headers={"Location": "http://localhost:11434/api/tags"},
        )
        result = web.fetch("https://example.com/a")
        assert result.ok is False
        assert "local network" in result.content

    def test_a_redirect_to_another_scheme_is_refused(self, web, mocked_responses):
        mocked_responses.get(
            "https://example.com/a", status=302,
            headers={"Location": "file:///etc/passwd"},
        )
        assert web.fetch("https://example.com/a").ok is False

    def test_a_redirect_loop_is_stopped(self, web, mocked_responses):
        mocked_responses.get(
            "https://example.com/a", status=302,
            headers={"Location": "https://example.com/a"},
        )
        result = web.fetch("https://example.com/a")
        assert result.ok is False
        assert "too many redirects" in result.content

    def test_a_redirect_without_a_target_is_reported(self, web, mocked_responses):
        mocked_responses.get("https://example.com/a", status=302)
        assert web.fetch("https://example.com/a").ok is False


class TestSizeCap:
    def test_a_large_body_is_truncated(self, mocked_responses, public_dns):
        small = WebTools(max_bytes=2048)
        mocked_responses.get("https://example.com", body="<p>" + "word " * 5000 + "</p>")
        result = small.fetch("https://example.com")
        assert result.ok is True
        assert "(truncated)" in result.content

    def test_a_declared_oversize_body_is_refused_before_reading(
        self, mocked_responses, public_dns
    ):
        small = WebTools(max_bytes=1000)
        mocked_responses.get(
            "https://example.com", body="x" * 50, headers={"Content-Length": "999999"},
        )
        result = small.fetch("https://example.com")
        assert result.ok is False
        assert "larger than" in result.content

    def test_extracted_text_is_capped_for_the_model(self, web, mocked_responses):
        mocked_responses.get("https://example.com", body="<p>" + "word " * 3000 + "</p>")
        assert len(web.fetch("https://example.com").content) < 5000


class TestToolSpec:
    def test_registered_and_namespaced(self, public_dns):
        registry = ToolRegistry()
        registry.register_all(build_tools())
        assert registry.names() == ["web__fetch"]

    def test_marked_as_egress(self):
        assert build_tools()[0].egress is True

    def test_reads_nothing_locally(self):
        assert build_tools()[0].risk is Risk.READ_ONLY

    def test_a_bad_scheme_is_refused_without_any_io(self):
        """The precheck runs during policy evaluation, so it must not resolve
        names or open sockets."""
        precheck = build_tools()[0].precheck
        assert "unsupported URL scheme" in precheck({"url": "file:///etc/passwd"})

    def test_an_internal_ip_literal_is_refused_without_dns(self):
        precheck = build_tools()[0].precheck
        assert "local network" in precheck({"url": "http://127.0.0.1:11434/"})

    def test_a_hostname_is_left_to_the_handler(self):
        """Resolving a name needs I/O, so it is not a precheck concern."""
        assert build_tools()[0].precheck({"url": "https://example.com"}) is None

    def test_an_empty_url_is_refused_up_front(self):
        assert build_tools()[0].precheck({"url": ""})

    async def test_dispatch_through_the_registry(self, mocked_responses, public_dns):
        registry = ToolRegistry()
        registry.register_all(build_tools())
        mocked_responses.get("https://example.com", body="<p>hello</p>")
        result = await registry.call("web__fetch", {"url": "https://example.com"})
        assert "hello" in result.content
