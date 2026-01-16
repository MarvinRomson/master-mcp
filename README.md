![Master-MCP](docs/gemini-gen-cover.png)

# Master-MCP

**One MCP endpoint. All your tools. Fully editable.**

Master-MCP is a lightweight **proxy MCP server** that connects to multiple MCP servers and re-exports their tools through a **single MCP endpoint**.

It’s designed for agentic workflows where tool catalogs can grow large, messy, and hard to manage.

## Why Master-MCP

- **Unify MCPs**: plug in many MCP servers, expose a single tool list.
- **Namespacing**: tools are automatically renamed to `<server>-<tool>`.
- **Schema Editor UI**: edit tool descriptions/params, hide tools, and curate the catalog.
- **Search MCP (optional)**: regex + BM25 search over tools.
- **CodeExec support**: generate code-friendly wrappers for Python/TypeScript.

## Quick start

```bash
pip install -r requirements.txt
python master_mcp_client.py
```

Then open the schema editor:

```bash
open http://127.0.0.1:8001/
```

## Documentation

- **Full documentation & architecture**: see **[FULL_README.md](FULL_README.md)**
- Configure external MCPs in `mcps.json`

## Links

- Blogpost: https://towardsdatascience.com/master-mcp-as-a-way-to-keep-mcps-useful-in-agentic-pipelines/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk


---

## Roadmap / Ideas

- Code execution mode (with different languages integration)
- Visual UI to enable/disable tools and MCPs at runtime
- Advanced security model (user roles, sandboxing, explicit confirmations)
- Optional HTTP/WebSocket transport in addition to stdio
- Monitoring tools
- Tools heirarchy mechanics
- Tools orchestration
- Add Skills and prompts (just a prompt with no tool execution)

---

## 🤝 Contribute & Collaborate

If you're interested in contributing, testing new detection methods, or exploring related projects, feel free to contribute and reach me out — collaboration is welcome!

📩 Contact: marvinromson@gmail.com
and [LinkedIn](https://www.linkedin.com/in/roman-smirnov-09165b127/)

---

## ❤️ Support

If you find this project helpful or use it in your research, you can support further development:  
📩 Contact: marvinromson@gmail.com

