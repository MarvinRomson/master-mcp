"""Tiny demo for Search MCP internals (no MCP runtime required).

Run:
  python demo_search_mcp.py
"""

from search_mcp_server import _Bm25Index, _regex_search


def main() -> None:
    tools = [
        {"name": "chart-generate", "description": "Generate a chart", "inputSchema": {}, "outputSchema": {}},
        {"name": "weather-forecast", "description": "Get weather forecast", "inputSchema": {}, "outputSchema": {}},
        {"name": "search_regex_tools", "description": "Regex full-text search", "inputSchema": {}, "outputSchema": {}},
    ]

    print("Regex search for 'weather':")
    print(_regex_search(tools, r"weather"))

    print("\nBM25 search for 'regex search':")
    idx = _Bm25Index(tools)
    print(idx.search("regex search", top_k=5))


if __name__ == "__main__":
    main()

