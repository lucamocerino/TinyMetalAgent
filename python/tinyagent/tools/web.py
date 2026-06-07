"""Network tools: web_fetch (urllib) and web_search (pluggable provider).

Only these explicit tools touch the network; the LLM itself stays local. The model
inference never makes a network call.
"""
from __future__ import annotations

import html
import os
import re
import urllib.request
from typing import Any, Dict, List

from . import ToolContext, ToolSpec
from .files import ToolError

_USER_AGENT = "tinyagent/0.1 (+local Qwen coding agent)"
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n\s*\n\s*")


def _html_to_text(raw: str) -> str:
    raw = _SCRIPT_STYLE_RE.sub(" ", raw)
    raw = _TAG_RE.sub("", raw)
    raw = html.unescape(raw)
    raw = _WS_RE.sub("\n\n", raw)
    return raw.strip()


def web_fetch(args: Dict[str, Any], ctx: ToolContext) -> str:
    url = str(args["url"])
    if not url.startswith(("http://", "https://")):
        raise ToolError("url must start with http:// or https://")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:  # noqa: S310 - explicit scheme check above
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read(2_000_000).decode(charset, errors="replace")
            ctype = resp.headers.get_content_type()
    except Exception as exc:  # noqa: BLE001 - surface network errors to the model
        raise ToolError(f"fetch failed: {exc}")
    if "html" in (ctype or ""):
        return _html_to_text(body)
    return body


def web_search(args: Dict[str, Any], ctx: ToolContext) -> str:
    query = str(args["query"])
    provider = os.environ.get("TINYAGENT_SEARCH_URL")
    if not provider:
        raise ToolError(
            "web_search is not configured (set TINYAGENT_SEARCH_URL to a search "
            "endpoint that accepts {query}). Use web_fetch with a known URL instead."
        )
    url = provider.replace("{query}", urllib.request.quote(query))
    return web_fetch({"url": url}, ctx)


SPECS: List[ToolSpec] = [
    ToolSpec(
        name="web_fetch",
        description="Fetch a URL and return its readable text (HTML is stripped).",
        effect="network",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        handler=web_fetch,
    ),
    ToolSpec(
        name="web_search",
        description="Search the web for a query (requires a configured provider).",
        effect="network",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=web_search,
    ),
]
