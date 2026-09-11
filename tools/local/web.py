"""Fetching web pages, with the guards a local assistant needs.

Anything this tool returns is content the user did not write, so results are
flagged ``untrusted``: the taint tracker uses that to require confirmation for
whatever the model wants to do next on the strength of it.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger

from tools.schema import Risk, ToolResult, ToolSpec, namespaced

NAMESPACE = "web"

ALLOWED_SCHEMES = ("http", "https")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
MAX_REDIRECTS = 5
#: Characters of extracted text handed to the model.
MAX_TEXT_CHARS = 4000
CHUNK_SIZE = 16384


def _address_is_internal(address: str) -> bool:
    """True for anything that is not a public internet address."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:  # pragma: no cover - getaddrinfo returns valid literals
        return True
    return any((
        ip.is_loopback, ip.is_private, ip.is_link_local, ip.is_reserved,
        ip.is_multicast, ip.is_unspecified,
    ))


class WebTools:
    """Web access, gated on scheme, destination address and response size."""

    def __init__(
        self,
        allow_private_network: bool = False,
        max_bytes: int = 2_000_000,
        timeout: int = 10,
    ):
        self.allow_private_network = allow_private_network
        self.max_bytes = max_bytes
        self.timeout = timeout

    def _check_url(self, url: str) -> Tuple[bool, str]:
        """Validate one URL before any request is made to it."""
        try:
            parsed = urlparse(url)
        except ValueError as e:
            return False, f"unusable URL: {e}"

        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return False, (
                f"unsupported URL scheme '{parsed.scheme}'; "
                f"only {' and '.join(ALLOWED_SCHEMES)} are allowed"
            )
        if not parsed.hostname:
            return False, "URL has no host"

        if self.allow_private_network:
            return True, ""

        try:
            infos = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as e:
            return False, f"could not resolve '{parsed.hostname}': {e}"

        for info in infos:
            address = info[4][0]
            if _address_is_internal(address):
                # A self-hosted box runs Ollama, dashboards and cloud metadata
                # endpoints; nothing else stops the model reading them aloud.
                return False, (
                    f"'{parsed.hostname}' resolves to {address}, which is on the "
                    f"local network; set ALLOW_PRIVATE_NETWORK_FETCH=true to permit this"
                )

        return True, ""

    def _read_capped(self, response: requests.Response) -> Tuple[str, bool]:
        """Read at most ``max_bytes`` of the body."""
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return "", True

        chunks: List[bytes] = []
        total = 0
        truncated = False
        for chunk in response.iter_content(CHUNK_SIZE):
            chunks.append(chunk)
            total += len(chunk)
            if total >= self.max_bytes:
                truncated = True
                break

        encoding = response.encoding or "utf-8"
        return b"".join(chunks).decode(encoding, errors="replace"), truncated

    def fetch(self, url: Any) -> ToolResult:
        """Fetch a web page and return its visible text."""
        if not url or not str(url).strip():
            return ToolResult.error("url is required")
        current = str(url).strip()

        for _ in range(MAX_REDIRECTS + 1):
            allowed, reason = self._check_url(current)
            if not allowed:
                return ToolResult.error(reason)

            try:
                response = requests.get(
                    current,
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout,
                    stream=True,
                    # Redirects are followed by hand so every hop goes through
                    # the same checks; letting requests follow them would land
                    # the first internal request before anyone looked.
                    allow_redirects=False,
                )
            except requests.RequestException as e:
                return ToolResult.error(f"could not fetch {current}: {e}")

            # Checked on the status code, not response.is_redirect: that is
            # False when Location is missing, which would otherwise fall
            # through and return an empty page as a success.
            if 300 <= response.status_code < 400:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    return ToolResult.error(
                        f"{current} answered {response.status_code} with no Location"
                    )
                current = urljoin(current, location)
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                response.close()
                return ToolResult.error(f"could not fetch {current}: {e}")

            with response:
                body, oversized = self._read_capped(response)

            if oversized and not body:
                return ToolResult.error(
                    f"{current} is larger than the {self.max_bytes} byte limit"
                )

            text = self._extract_text(body)
            if len(text) > MAX_TEXT_CHARS:
                text = text[:MAX_TEXT_CHARS] + "\n… (truncated)"
            elif oversized:
                text += "\n… (truncated)"

            logger.info(f"Fetched {len(text)} characters from {current}")
            # Untrusted: whoever wrote this page is not the user.
            return ToolResult(f"Content from {current}:\n{text}", untrusted=True)

        return ToolResult.error(f"too many redirects starting from {url}")

    @staticmethod
    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        lines = (line.strip() for line in soup.get_text().splitlines())
        return " ".join(line for line in lines if line)


def _url_precheck(allow_private_network: bool):
    """Refuse an unusable URL without doing any I/O.

    Only the checks that need no network: the scheme, and a host that is
    already an address literal. Resolving a name is left to the handler, so
    policy evaluation stays fast and side-effect free.
    """
    def check(arguments: dict) -> Optional[str]:
        raw = str(arguments.get("url") or "").strip()
        if not raw:
            return "url is required"
        try:
            parsed = urlparse(raw)
        except ValueError as e:
            return f"unusable URL: {e}"

        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return (
                f"unsupported URL scheme '{parsed.scheme}'; "
                f"only {' and '.join(ALLOWED_SCHEMES)} are allowed"
            )
        if not parsed.hostname:
            return "URL has no host"

        if not allow_private_network:
            try:
                ipaddress.ip_address(parsed.hostname)
            except ValueError:
                return None  # a name; the handler resolves it
            if _address_is_internal(parsed.hostname):
                return (
                    f"'{parsed.hostname}' is on the local network; set "
                    f"ALLOW_PRIVATE_NETWORK_FETCH=true to permit this"
                )
        return None

    return check


def _domain_scope(arguments: dict) -> str:
    """Approvals for a fetch cover the site, not the individual page."""
    try:
        return urlparse(str(arguments.get("url") or "")).netloc or "(no host)"
    except ValueError:
        return "(unparseable)"


def build_tools(
    allow_private_network: bool = False,
    max_bytes: int = 2_000_000,
    timeout: int = 10,
) -> List[ToolSpec]:
    """Build the web tools."""
    web = WebTools(allow_private_network, max_bytes, timeout)
    return [
        ToolSpec(
            name=namespaced(NAMESPACE, "fetch"),
            description="Fetch a web page and return its visible text content",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The http(s) URL to fetch"},
                },
                "required": ["url"],
            },
            handler=web.fetch,
            # Reads nothing local, but reaches off the machine, which is what
            # makes it an exfiltration channel worth tracking.
            risk=Risk.READ_ONLY,
            egress=True,
            scope_for=_domain_scope,
            precheck=_url_precheck(allow_private_network),
        ),
    ]
