"""Tests for MCP server configuration (tools/mcp/servers.py)."""

import json

from jarvis.tools.mcp.servers import ServerConfig, Transport, Trust, load_servers


def write_config(tmp_path, servers):
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


class TestTrust:
    def test_known_levels(self):
        assert Trust.parse("trusted") is Trust.TRUSTED
        assert Trust.parse("CONFIRM") is Trust.CONFIRM
        assert Trust.parse(" readonly ") is Trust.READONLY

    def test_an_unknown_level_falls_back_to_confirming(self):
        """Never resolve an unreadable trust setting in the server's favour."""
        assert Trust.parse("yolo") is Trust.CONFIRM

    def test_a_missing_level_defaults_to_confirming(self):
        assert ServerConfig("x", Transport.STDIO).trust is Trust.CONFIRM


class TestLoading:
    def test_a_missing_file_means_no_servers(self, tmp_path):
        assert load_servers(tmp_path / "absent.json") == []

    def test_unreadable_json_means_no_servers(self, tmp_path):
        path = tmp_path / "mcp_servers.json"
        path.write_text("{ not json")
        assert load_servers(path) == []

    def test_a_stdio_server(self, tmp_path):
        path = write_config(tmp_path, {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/docs"],
                "env": {"TOKEN": "abc"},
            }
        })
        server = load_servers(path)[0]
        assert server.name == "filesystem"
        assert server.transport is Transport.STDIO
        assert server.command == "npx"
        assert server.args[-1] == "/docs"
        assert server.env == {"TOKEN": "abc"}

    def test_transport_is_inferred_from_the_keys(self, tmp_path):
        path = write_config(tmp_path, {
            "local": {"command": "run-me"},
            "remote": {"url": "https://example.com/mcp"},
        })
        by_name = {s.name: s for s in load_servers(path)}
        assert by_name["local"].transport is Transport.STDIO
        assert by_name["remote"].transport is Transport.HTTP

    def test_an_explicit_transport_wins(self, tmp_path):
        path = write_config(tmp_path, {
            "remote": {"url": "https://example.com/sse", "transport": "sse"},
        })
        assert load_servers(path)[0].transport is Transport.SSE

    def test_the_bare_object_form_is_accepted(self, tmp_path):
        """Some configurations omit the mcpServers wrapper."""
        path = tmp_path / "mcp_servers.json"
        path.write_text(json.dumps({"local": {"command": "run-me"}}))
        assert load_servers(path)[0].name == "local"

    def test_enabled_and_trust_are_read(self, tmp_path):
        path = write_config(tmp_path, {
            "git": {"command": "git-mcp", "enabled": False, "trust": "trusted"},
        })
        server = load_servers(path)[0]
        assert server.enabled is False
        assert server.trust is Trust.TRUSTED

    def test_servers_are_enabled_by_default(self, tmp_path):
        path = write_config(tmp_path, {"git": {"command": "git-mcp"}})
        assert load_servers(path)[0].enabled is True

    def test_headers_and_timeout_for_http(self, tmp_path):
        path = write_config(tmp_path, {
            "remote": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer x"},
                "timeout": 5,
            }
        })
        server = load_servers(path)[0]
        assert server.headers == {"Authorization": "Bearer x"}
        assert server.timeout == 5.0


class TestMalformedEntries:
    def test_a_bad_entry_is_skipped_not_fatal(self, tmp_path):
        """One broken server should not cost the user the others."""
        path = write_config(tmp_path, {
            "broken": {"nothing": "useful"},
            "good": {"command": "run-me"},
        })
        names = [s.name for s in load_servers(path)]
        assert names == ["good"]

    def test_an_unknown_transport_is_skipped(self, tmp_path):
        path = write_config(tmp_path, {"x": {"url": "u", "transport": "carrier-pigeon"}})
        assert load_servers(path) == []

    def test_stdio_without_a_command_is_skipped(self, tmp_path):
        path = write_config(tmp_path, {"x": {"transport": "stdio", "args": ["a"]}})
        assert load_servers(path) == []

    def test_http_without_a_url_is_skipped(self, tmp_path):
        path = write_config(tmp_path, {"x": {"transport": "http"}})
        assert load_servers(path) == []

    def test_a_non_object_entry_is_skipped(self, tmp_path):
        path = write_config(tmp_path, {"x": "just a string"})
        assert load_servers(path) == []

    def test_a_top_level_list_is_rejected(self, tmp_path):
        path = tmp_path / "mcp_servers.json"
        path.write_text(json.dumps(["not", "an", "object"]))
        assert load_servers(path) == []
