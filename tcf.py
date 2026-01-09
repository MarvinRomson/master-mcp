"""
Ideally when doing CodeExec this file should not be in tools repo, but in case you put it there - the naming is supposed to confuse agent to avoid it reading it.
"""
import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

from master_mcp_client import MCPClient

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
TOOLS_JSON = SCHEMA_DIR / "tools.json"
BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "code_tools"
TOOL_SERVER_CONFIG = GENERATED_DIR / "tool_server_config.json"



def load_tools_from_schema() -> List[Dict[str, Any]]:
    """Load the current tools list from ``schemas/tools.json``.

    We always read the "modified" list so this works with whatever the
    schema editor last saved.
    """

    if not TOOLS_JSON.exists():
        return []

    with TOOLS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tools = data.get("modified") or []
    if not isinstance(tools, list):
        return []
    return tools


def _split_for_skills(tool_name: str) -> Tuple[str, str]:
    """Split tool name into (server_folder, function_folder) for skills layout.

    As requested, this uses the *last* ``'-'`` as the split point. The
    first part is treated as the server folder name; tools from the same
    server end up grouped underneath it.
    """

    idx = tool_name.rfind("-")
    if idx == -1:
        return "general", tool_name
    return tool_name[:idx], tool_name[idx + 1 :]


def _safe_identifier(name: str) -> str:
    """Convert a tool name into a Python/TypeScript-safe identifier."""

    ident = name.replace("-", "_").replace(" ", "_")
    # Very simple sanitization; callers can further adapt if needed
    return ident


def _is_port_open(host: str, port: int) -> bool:
    """Return True if a TCP port is open on the given host."""

    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class CallRequest(BaseModel):
    tool: str
    kwargs: Dict[str, Any] = {}
    has_output_schema: bool = False


tool_server_app = FastAPI()


@tool_server_app.post("/call_tool")
async def call_tool(req: CallRequest) -> Any:
    """Call a tool via the Master MCP server and return JSON-safe output."""

    client = MCPClient()
    await client.connect_to_server("python", ["master_mcp_client.py"])
    try:
        result = await client(req.tool, req.kwargs)
    finally:
        await client.cleanup()

    text: str | None = None
    try:
        if getattr(result, "content", None):
            first = result.content[0]
            text = getattr(first, "text", None)
    except Exception:
        text = None

    if req.has_output_schema and text is not None:
        try:
            return json.loads(text)
        except Exception:
            # Fall back to raw text wrapped as JSON.
            return {"text": text}

    if text is not None:
        return {"text": text}

    # Last-resort: stringified result object.
    try:
        return json.loads(str(result))
    except Exception:
        return {"result": str(result)}


def _find_free_port(start: int = 23050, end: int = 23150) -> int:
    """Find a free localhost TCP port in the given range."""

    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port available in range {start}-{end}")


def _ensure_tool_server() -> int:
    """Ensure the FastAPI tool server is running and return its port.

    If a config file exists and the port is open, reuse it. Otherwise,
    launch a new uvicorn process for `available_code_tools:tool_server_app`
    on a free port and write the config file.
    """

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse existing running server if possible
    if TOOL_SERVER_CONFIG.exists():
        try:
            data = json.loads(TOOL_SERVER_CONFIG.read_text(encoding="utf-8"))
            port = int(data.get("port"))
            host = data.get("host") or "127.0.0.1"
            if _is_port_open(host, port):
                return port
        except Exception:
            pass

    # Start a new server on a free port
    port = _find_free_port()
    TOOL_SERVER_CONFIG.write_text(
        json.dumps({"host": "127.0.0.1", "port": port}, indent=2),
        encoding="utf-8",
    )

    # Fire-and-forget uvicorn process
    subprocess.Popen(
        [
            "uvicorn",
            "available_code_tools:tool_server_app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    #time.sleep(0.25)

    return port


def _cleanup_generated_artifacts(mode: str, lang: str) -> None:
    """Remove previously generated wrappers for the selected language(s).

    This only touches the folders/files that this script itself creates:

    - Single-file wrappers under ``code_tools/``
      - ``code_tools/python_tools.py``
      - ``code_tools/typescript_tools.ts``
    - Skills-style folders under ``skills/``
      - ``skills/python/``
      - ``skills/typescript/``
    """

    mode = (mode or "single").lower()
    lang = (lang or "both").lower()

    python_enabled = lang in ("python", "both")
    ts_enabled = lang in ("typescript", "ts", "both")

    generated_dir = BASE_DIR / "code_tools"
    skills_python_dir = BASE_DIR / "skills" / "python"
    skills_ts_dir = BASE_DIR / "skills" / "typescript"

    # Language-specific single-file modules
    if python_enabled:
        py_single = generated_dir / "python_tools.py"
        if py_single.exists():
            py_single.unlink()
    if ts_enabled:
        ts_single = generated_dir / "typescript_tools.ts"
        if ts_single.exists():
            ts_single.unlink()

    # Skills folders (per-language trees)
    if python_enabled and skills_python_dir.exists():
        shutil.rmtree(skills_python_dir)
    if ts_enabled and skills_ts_dir.exists():
        shutil.rmtree(skills_ts_dir)


def _python_header() -> str:
    """Common Python header (imports + helpers).

    Python wrappers call a local FastAPI "tool server" over HTTP. The
    server hosts a single `/call_tool` endpoint that connects to the
    Master MCP server via ``MCPClient`` and returns JSON-serialisable
    results. The server is launched automatically by ``run_codeexec``
    and its port is stored in ``code_tools/tool_server_config.json``.
    """

    return '''from __future__ import annotations

import json
from typing import Any, Dict
from pathlib import Path

import httpx


_CONFIG_PATH = Path(__file__).resolve().parent / "tool_server_config.json"


def _load_server_config() -> tuple[str, int]:
    """Load the FastAPI tool server host and port from the config file.

    Raises a RuntimeError with a helpful message if the server has not
    been started yet (for example if `run_codeexec` has never been
    called in this session).
    """

    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        host = str(data.get("host") or "127.0.0.1")
        port = int(data.get("port"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Tool server config not found. Call `run_codeexec` at least once "
            "so the shared tool server can be started."
        ) from exc
    return host, port


async def _call_tool(tool_full_name: str, kwargs: Dict[str, Any], has_output_schema: bool) -> Any:
    """Call the shared FastAPI tool server and return its JSON result."""

    host, port = _load_server_config()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"http://{host}:{port}/call_tool",
            json={
                "tool": tool_full_name,
                "kwargs": kwargs,
                "has_output_schema": has_output_schema,
            },
        )
    resp.raise_for_status()
    return resp.json()
'''


def _ts_header() -> str:
    """Common TypeScript header (tool server client + helpers).

    The generated TypeScript wrappers call the same FastAPI tool server
    as the Python wrappers. The server host and port are loaded from
    ``tool_server_config.json`` which is written by ``run_codeexec`` in
    the same directory as the generated TypeScript file.
    """

    return """import fs from 'fs/promises';
import path from 'path';

type ToolServerConfig = { host: string; port: number };

async function loadToolServerConfig(): Promise<ToolServerConfig> {
  const configPath = path.resolve(__dirname, 'tool_server_config.json');
  const raw = await fs.readFile(configPath, 'utf8');
  const data = JSON.parse(raw);
  return {
    host: typeof data.host === 'string' && data.host.length > 0 ? data.host : '127.0.0.1',
    port: Number(data.port),
  };
}

async function callToolWithParsing(
  toolFullName: string,
  hasOutputSchema: boolean,
  params: any,
): Promise<any> {
  const { host, port } = await loadToolServerConfig();

  const res = await fetch(`http://${host}:${port}/call_tool`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tool: toolFullName,
      kwargs: params,
      has_output_schema: hasOutputSchema,
    }),
  });

  if (!res.ok) {
    throw new Error(`Tool server returned ${res.status}: ${await res.text()}`);
  }

  const result = await res.json();

  if (hasOutputSchema && typeof result === 'string') {
    try {
      return JSON.parse(result);
    } catch {
      return result;
    }
  }

  return result;
}

"""


def _python_function(tool: Dict[str, Any]) -> str:
    """Generate a single Python async wrapper function for a tool.

    The function name mirrors the tool name from tools.json with
    hyphens replaced by underscores. It connects to the Master MCP
    server, calls the tool by its original name and returns either
    structured JSON (when an output schema is present) or raw text.
    """

    full_name = tool.get("name", "")
    func_name = _safe_identifier(full_name)
    description = tool.get("description") or "(no description provided)"
    input_schema = tool.get("inputSchema") or {}
    props = input_schema.get("properties", {}) or {}
    required = set(input_schema.get("required", []) or [])
    has_output = tool.get("outputSchema") is not None

    lines: List[str] = []
    lines.append(f"async def {func_name}(kwargs: Dict[str, Any]) -> Any:")
    lines.append('    """')
    lines.append(f"    {description}")
    lines.append("")
    lines.append(f"    MCP tool name (as in tools.json): {full_name}")
    lines.append("")
    if props:
        lines.append("    Parameters (keys in `kwargs`):")
        for pname, spec in props.items():
            pdesc = spec.get("description") or ""
            ptype = spec.get("type", "any")
            flag = "required" if pname in required else "optional"
            lines.append(f"      - {pname} ({ptype}, {flag}) - {pdesc}")
    lines.append("")
    lines.append("    Returns")
    lines.append("    -------")
    lines.append("    Parsed JSON if the tool defines an output schema and returns JSON text;")
    lines.append("    otherwise the raw text response or underlying MCP result object.")
    lines.append('    """')
    lines.append(f"    # Each function starts with a connection to the Master MCP server,")
    lines.append(f"    # followed by execution of the tool and structured output parsing.")
    lines.append(
        f"    return await _call_tool('{full_name}', kwargs, {str(has_output).lower()})"
    )
    lines.append("")
    lines.append("")
    return "\n".join(lines)


def _ts_function(tool: Dict[str, Any]) -> str:
    """Generate a single TypeScript wrapper function for a tool.

    The function delegates actual MCP connectivity to a caller-supplied
    ``CallToolFn``. This keeps the generated code independent from any
    particular MCP JS/TS client implementation.
    """

    full_name = tool.get("name", "")
    func_name = _safe_identifier(full_name)
    description = tool.get("description") or "(no description provided)"
    input_schema = tool.get("inputSchema") or {}
    props = input_schema.get("properties", {}) or {}
    required = set(input_schema.get("required", []) or [])
    has_output = tool.get("outputSchema") is not None

    lines: List[str] = []
    lines.append(f"export async function {func_name}(callTool: CallToolFn, params: any): Promise<any> {{")
    lines.append("  /**")
    lines.append(f"   * {description}")
    lines.append("   *")
    lines.append(f"   * MCP tool: `{full_name}`")
    if props:
        lines.append("   *")
        lines.append("   * Parameters (keys in `params`):")
        for pname, spec in props.items():
            pdesc = spec.get("description") or ""
            ptype = spec.get("type", "any")
            flag = "required" if pname in required else "optional"
            lines.append(f"   *   - {pname} ({ptype}, {flag}) - {pdesc}")
    lines.append("   *")
    if has_output:
        lines.append("   * Returns structured JSON when the tool defines an output schema;")
        lines.append("   * otherwise returns the raw text or result from `callTool`." )
    else:
        lines.append("   * Returns the raw text or result from `callTool`." )
    lines.append("   */")
    lines.append(f"  const hasOutputSchema = {str(has_output).lower()};")
    lines.append(f"  return callToolWithParsing(callTool, '{full_name}', hasOutputSchema, params);")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def generate_python_single(tools: List[Dict[str, Any]]) -> None:
    """Generate a single Python module with wrappers for all tools.

    All wrappers connect to the Master MCP server and call tools by
    their full names (as in tools.json), with function names equal to
    the tool name but with ``'-'`` replaced by ``'_'``.
    """

    header = _python_header()

    body_parts: List[str] = [header]
    for tool in tools:
        full_name = tool.get("name", "")
        if not full_name:
            continue
        body_parts.append(_python_function(tool))

    out_dir = BASE_DIR / "code_tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "python_tools.py"
    out_file.write_text("\n".join(body_parts), encoding="utf-8")


def generate_typescript_single(tools: List[Dict[str, Any]]) -> None:
    """Generate a single TypeScript module with wrappers for all tools."""

    header = _ts_header()
    body_parts: List[str] = [header]
    for tool in tools:
        full_name = tool.get("name", "")
        if not full_name:
            continue
        body_parts.append(_ts_function(tool))

    out_dir = BASE_DIR / "code_tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "typescript_tools.ts"
    out_file.write_text("\n".join(body_parts), encoding="utf-8")


def generate_python_skills(tools: List[Dict[str, Any]]) -> None:
    """Generate a skills-style folder hierarchy with one Python module per tool.

    Layout:

    ``skills/python/<server_folder>/<function_folder>/__init__.py``
    """

    header = _python_header()

    base = BASE_DIR / "skills" / "python"
    base.mkdir(parents=True, exist_ok=True)

    for tool in tools:
        full_name = tool.get("name", "")
        if not full_name:
            continue
        server_folder, func_folder = _split_for_skills(full_name)
        target_dir = base / server_folder / func_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        # Each skills module is self-contained and includes the helper header
        content = header + "\n" + _python_function(tool)
        (target_dir / "__init__.py").write_text(content, encoding="utf-8")


def generate_typescript_skills(tools: List[Dict[str, Any]]) -> None:
    """Generate a skills-style folder hierarchy with one TS module per tool.

    Layout:

    ``skills/typescript/<server_folder>/<function_folder>/index.ts``
    """

    header = _ts_header()
    base = BASE_DIR / "skills" / "typescript"
    base.mkdir(parents=True, exist_ok=True)

    for tool in tools:
        full_name = tool.get("name", "")
        if not full_name:
            continue
        server_folder, func_folder = _split_for_skills(full_name)
        target_dir = base / server_folder / func_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        content = header + "\n" + _ts_function(tool)
        (target_dir / "index.ts").write_text(content, encoding="utf-8")


def generate_agents_markdown(
    tools: List[Dict[str, Any]],
    mode: str,
    instructions_path: Path,
    lang: str,
) -> None:
    """Generate AGENTS.md (or a custom-named instructions file).

    This file explains where the generated Python and TypeScript wrappers
    live and how to use them from an agent.
    """

    lines: List[str] = []
    lines.append("# Code Execution Tools")
    lines.append("")
    lines.append(
        "This file is explains that there are additional tools that you can use as function to insert into code to get things done." \
        "They can be very useful, check which are available if you think you are missing any tool to complete the task. "
    )
    lines.append("")

    lines.append("## Check provided folder and scripts.")
    lines.append("")

    lang = (lang or "both").lower()
    python_enabled = lang in ("python", "both")
    ts_enabled = lang in ("typescript", "ts", "both")

    if mode == "skills":
        if python_enabled:
            lines.append("Python wrappers (one module per tool):")
            lines.append("- `skills/python/<server>/<function>/__init__.py`")
            lines.append("")
        if ts_enabled:
            lines.append("TypeScript wrappers (one module per tool):")
            lines.append("- `skills/typescript/<server>/<function>/index.ts`")
    else:
        if python_enabled:
            lines.append("Python wrappers (all tools in a single module):")
            lines.append("- `code_tools/python_tools.py`")
            lines.append("")
        if ts_enabled:
            lines.append("TypeScript wrappers (all tools in a single module):")
            lines.append("- `code_tools/typescript_tools.ts`")
    lines.append("")

    lines.append("## How to call the wrappers")
    lines.append("")

    if python_enabled:
        lines.append("### Python")
        lines.append("")
        lines.append("Each generated Python function:")
        lines.append("- Is declared as `async def`.")
        lines.append("- Starts by connecting to the Master MCP server.")
        lines.append("- Calls the tool with a single `kwargs: Dict[str, Any]` argument.")
        lines.append(
            "- If the tool declares an `outputSchema`, attempts to parse the first text "
            "response as JSON and returns the resulting structure; otherwise returns "
            "the raw text or underlying MCP result."
        )
        lines.append("")

        lines.append("Example (single-mode module):")
        lines.append("")
        lines.append("```python")
        lines.append("from code_tools.python_tools import chart_generate_area_chart")
        lines.append("")
        lines.append("result = await chart_generate_area_chart({\"data\": [...], \"stack\": False})")
        lines.append("```")
        lines.append("")

    if ts_enabled:
        lines.append("### TypeScript")
        lines.append("")
        lines.append("TypeScript wrappers expect a caller-supplied `CallToolFn`: ")
        lines.append("```ts")
        lines.append("export type CallToolFn = (toolName: string, params: any) => Promise<any>;")
        lines.append("```")
        lines.append("")
        lines.append(
            "You are responsible for implementing `CallToolFn` so that it "
            "connects to the desired MCP server and calls the tool by name. "
            "The generated functions then delegate to this callback and apply "
            "the same JSON parsing logic as in the Python wrappers."
        )
        lines.append("")

    lines.append("## Explore the additional available tools here if you need any tools to solve the task.")
    lines.append("")
    for tool in tools:
        name = tool.get("name", "<unnamed>")
        description = tool.get("description") or "(no description provided)"
        #input_schema = tool.get("inputSchema") or {}
        #props = input_schema.get("properties", {}) or {}
        #required = set(input_schema.get("required", []) or [])

        func_name = _safe_identifier(name)
        lines.append(f"### `{name}` → function `{func_name}`")
        lines.append("")
        lines.append(description)
        lines.append("")

        # if props:
        #     lines.append("**Parameters (keys in `params`):**")
        #     lines.append("")
        #     for pname, spec in props.items():
        #         p_type = spec.get("type", "any")
        #         p_desc = spec.get("description") or ""
        #         is_req = "(required)" if pname in required else "(optional)"
        #         lines.append(f"- `{pname}` ({p_type}) {is_req} - {p_desc}")
        #     lines.append("")

    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    instructions_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_codeexec(mode: str, name: str, lang: str = "both") -> None:
    """Entry point for CodeExec generation.

    Parameters
    ----------
    mode:
        Either "single" (generate consolidated Python/TS modules) or
        "skills" (generate a skills folder hierarchy).
    name:
        Path to the markdown instructions file to generate (default:
        ``AGENTS.md``).
    """

    mode = (mode or "single").lower()
    lang = (lang or "both").lower()
    python_enabled = lang in ("python", "both")
    ts_enabled = lang in ("typescript", "ts", "both")
    instructions_path = (Path(name or "AGENTS.md").resolve())

    # Always clean up any previously generated wrappers so the new run
    # starts from a known state.
    _cleanup_generated_artifacts(mode, lang)

    # Ensure the shared tool server is running and its port persisted.
    _ensure_tool_server()

    tools = load_tools_from_schema()
    if not tools:
        # Nothing to generate
        return

    if mode == "skills":
        if python_enabled:
            generate_python_skills(tools)
        if ts_enabled:
            generate_typescript_skills(tools)
    else:
        if python_enabled:
            generate_python_single(tools)
        if ts_enabled:
            generate_typescript_single(tools)

    generate_agents_markdown(tools, mode, instructions_path, lang)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CodeExec instructions from MCP tools schema.")
    parser.add_argument("--mode", choices=["single", "skills"], default="single", help="Generation mode: single file or skills folder.")
    parser.add_argument("--name", default="AGENTS.md", help="Instructions filename (for single) or base folder (for skills).")
    parser.add_argument("--lang", choices=["python", "typescript", "both"], default="both", help="Which language wrappers to generate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_codeexec(args.mode, args.name, args.lang)


if __name__ == "__main__":
    main()
