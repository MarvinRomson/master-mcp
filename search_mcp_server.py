"""Search MCP server.

This MCP indexes a curated subset of Master-MCP tools from
`schemas/search_tools.json` and provides two search methods:

1) Regex full-text search
2) BM25-ranked search

The index is built in-memory on startup and reloaded on every call so edits to
the JSON are picked up without restarting (small file, simple + robust).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions


BASE_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = BASE_DIR / "schemas"
SEARCH_TOOLS_PATH = SCHEMA_DIR / "search_tools.json"


def _is_codeexec_mode() -> bool:
    # Flag mirrors the schema editor naming.
    # In practice, CodeExec tools are produced by tcf.py, but we keep the
    # switch simple and explicit.
    return os.environ.get("MASTER_MCP_CODEEXEC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _split_tool_name(tool_name: str) -> tuple[str, str]:
    """Split MCP tool name into (server_name, function_name) using last '-'."""
    idx = tool_name.rfind("-")
    if idx == -1:
        return "general", tool_name
    return tool_name[:idx], tool_name[idx + 1 :]


def _load_curated_tools() -> list[dict[str, Any]]:
    """Loads tool list from schemas/search_tools.json.

    Expected format: {"modified": [...], "modified_hash": "..."}
    We ignore the hash here; this is a local file.
    """
    if not SEARCH_TOOLS_PATH.exists():
        return []
    data = json.loads(SEARCH_TOOLS_PATH.read_text(encoding="utf-8"))
    tools = data.get("modified")
    if not isinstance(tools, list):
        return []
    return tools


def _tool_to_text(tool: dict[str, Any]) -> str:
    """Canonical text for full-text matching/ranking."""
    name = str(tool.get("name", ""))
    desc = str(tool.get("description", ""))
    input_schema = tool.get("inputSchema")
    output_schema = tool.get("outputSchema")
    return "\n".join(
        [
            f"name: {name}",
            f"description: {desc}",
            f"inputSchema: {json.dumps(input_schema, ensure_ascii=False, sort_keys=True)}",
            f"outputSchema: {json.dumps(output_schema, ensure_ascii=False, sort_keys=True)}",
        ]
    ).lower()


_TOKEN_RE = re.compile(r"[a-z0-9_\-]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class _Doc:
    tool: dict[str, Any]
    text: str
    tokens: tuple[str, ...]
    tf: Counter[str]


class _Bm25Index:
    def __init__(self, tools: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        docs: list[_Doc] = []
        df: Counter[str] = Counter()
        total_len = 0

        for t in tools:
            text = _tool_to_text(t)
            tokens = tuple(_tokenize(text))
            tf = Counter(tokens)
            if tf:
                df.update(set(tf.keys()))
            total_len += len(tokens)
            docs.append(_Doc(tool=t, text=text, tokens=tokens, tf=tf))

        self.docs = docs
        self.df = df
        self.N = len(docs)
        self.avgdl = (total_len / self.N) if self.N else 0.0

    def idf(self, term: str) -> float:
        # BM25+ style IDF (classic Okapi with +0.5)
        n_q = self.df.get(term, 0)
        return math.log(1.0 + (self.N - n_q + 0.5) / (n_q + 0.5))

    def score(self, query_tokens: Iterable[str], doc: _Doc) -> float:
        score = 0.0
        dl = len(doc.tokens)
        if dl == 0 or self.N == 0 or self.avgdl == 0:
            return 0.0

        for term in query_tokens:
            f = doc.tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf(term)
            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            score += idf * (f * (self.k1 + 1.0)) / denom
        return score

    def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored: list[tuple[float, _Doc]] = []
        for d in self.docs:
            s = self.score(q_tokens, d)
            if s > 0:
                scored.append((s, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for s, d in scored[: max(1, top_k)]:
            name = str(d.tool.get("name") or "")
            input_schema = d.tool.get("inputSchema")
            out.append(
                {
                    "name": name,
                    "description": d.tool.get("description"),
                    "inputSchema": input_schema,
                    "score": s,
                }
            )
        return out


def _regex_search(tools: list[dict[str, Any]], pattern: str, *, flags: str = "i", top_k: int = 50):
    re_flags = 0
    if "i" in flags:
        re_flags |= re.IGNORECASE
    if "m" in flags:
        re_flags |= re.MULTILINE
    if "s" in flags:
        re_flags |= re.DOTALL

    compiled = re.compile(pattern, re_flags)
    matches: list[dict[str, Any]] = []
    for t in tools:
        text = _tool_to_text(t)
        m = compiled.search(text)
        if not m:
            continue
        # Provide a small snippet around the first match
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        snippet = text[start:end]
        matches.append(
            {
                "name": t.get("name"),
                "description": t.get("description"),
                "inputSchema": t.get("inputSchema"),
                "snippet": snippet,
            }
        )
        if len(matches) >= max(1, top_k):
            break
    return matches


server = Server("search-mcp-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_regex_tools",
            description=(
                "Regex full-text search over curated Master-MCP tools (schemas/search_tools.json). "
                "Returns tool name/description/inputSchema; in CodeExec mode returns server/function/parameters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "flags": {
                        "type": "string",
                        "description": "Regex flags: i (ignorecase), m (multiline), s (dotall)",
                        "default": "i",
                    },
                    "top_k": {"type": "integer", "description": "Max results", "default": 50},
                },
                "required": ["pattern"],
            },
        ),
        types.Tool(
            name="search_bm25_tools",
            description=(
                "BM25 ranked search over curated Master-MCP tools (schemas/search_tools.json). "
                "Returns tool name/description/inputSchema; in CodeExec mode returns server/function/parameters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        ),
    ]


def _format_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format results depending on CodeExec mode.

    Normal mode: keep `name` as the MCP tool name (already namespaced/prefixed).
    CodeExec mode: return server_name + function_name + parameters schema.
    """

    if not _is_codeexec_mode():
        return results

    formatted: list[dict[str, Any]] = []
    for r in results:
        tool_name = str(r.get("name") or "")
        server_name, function_name = _split_tool_name(tool_name)
        formatted.append(
            {
                "server_name": server_name,
                "function_name": function_name,
                "description": r.get("description"),
                # For CodeExec, we expose the callable parameters schema (properties/required)
                "parameters": (r.get("inputSchema") or {}).get("properties", {})
                if isinstance(r.get("inputSchema"), dict)
                else {},
                "required": (r.get("inputSchema") or {}).get("required", [])
                if isinstance(r.get("inputSchema"), dict)
                else [],
                # Keep original tool name for traceability
                "tool": tool_name,
                **({"score": r["score"]} if "score" in r else {}),
                **({"snippet": r["snippet"]} if "snippet" in r else {}),
            }
        )
    return formatted


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]):
    tools = _load_curated_tools()

    if name == "search_regex_tools":
        pattern = str(arguments.get("pattern", ""))
        flags = str(arguments.get("flags", "i"))
        top_k = int(arguments.get("top_k", 50))
        res = _regex_search(tools, pattern, flags=flags, top_k=top_k)
        res = _format_results(res)
        return {"results": res, "count": len(res), "codeexec": _is_codeexec_mode()}

    if name == "search_bm25_tools":
        query = str(arguments.get("query", ""))
        top_k = int(arguments.get("top_k", 10))
        idx = _Bm25Index(tools)
        res = idx.search(query, top_k=top_k)
        res = _format_results(res)
        return {"results": res, "count": len(res), "codeexec": _is_codeexec_mode()}

    raise ValueError(f"Unknown tool: {name}")


async def server_run() -> None:
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="Search-MCP-server",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    # Ensure schemas dir exists for first run.
    os.makedirs(SCHEMA_DIR, exist_ok=True)
    asyncio.run(server_run())