"""
Based on the original modelcontextprotocol python-sdk, low level server and client implementations:

https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#server
"""

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
import asyncio
import logging
from typing import Optional
from contextlib import AsyncExitStack, asynccontextmanager

from typing import Any, List

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

import os, json, hashlib
from pathlib import Path
import time
from pathlib import Path
from schema_editor import _ensure_editor

BASE_DIR = Path(__file__).resolve().parent

# Set up logging
LOG_FILE = Path("master_mcp_client.log")


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("master_mcp_client")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        logger.propagate = False
    return logger


logger = _configure_logger()
logger.info("master_mcp_client module imported; logging to %s", LOG_FILE)

file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(file_dir)


def _read_ui_settings(schema_dir: str) -> dict:
    settings_path = os.path.join(schema_dir, "ui_settings.json")
    if not os.path.exists(settings_path):
        return {"expose_tools_search": False, "default_tool_names": [], "codeexec_enabled": False}
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"expose_tools_search": False, "default_tool_names": []}
        return {
            "expose_tools_search": bool(data.get("expose_tools_search", False)),
            "default_tool_names": list(data.get("default_tool_names", []) or []),
            "codeexec_enabled": bool(data.get("codeexec_enabled", False)),
        }
    except Exception:
        return {"expose_tools_search": False, "default_tool_names": [], "codeexec_enabled": False}


def _load_tools_json(schema_dir: str, filename: str) -> list[Tool]:
    path = os.path.join(schema_dir, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        parsed = json.load(f)
    tools = parsed.get("modified") if isinstance(parsed, dict) else None
    if not isinstance(tools, list):
        return []
    return [Tool.model_validate(x) for x in tools]


class MasterMCP:
    # initialize all the clients to external MCPs, but manages the tool names and description by itself
    def __init__(self, clients, tools, schema_dir=os.path.join(file_dir, "schemas")):
        self.clients = clients
        self.tools = tools
        self.tools_list = self._hide_tools_params(
            [v["tool"] for _, v in self.tools.items()]
        )

        self.schema_dir = schema_dir
        os.makedirs(schema_dir, exist_ok=True)

        # original hash
        self.original_hash = self._schema_hash([x.model_dump(exclude_none=False) for x in self.tools_list])

        # try load override
        self._load_modified()

        # save file
        self._save_schema(self.tools_list)

        _ensure_editor(8001)

    def _save_schema(self, schema):
        # `tools.json` is the default discovered tool catalog.
        with open(os.path.join(self.schema_dir, "tools.json"), "w") as f:
            json.dump(
                {"modified": [x.model_dump(exclude_none=False) for x in schema], "modified_hash": self._schema_hash([x.model_dump(exclude_none=False) for x in schema])},
                f,
                indent=2
            )

        # `search_tools.json` is the curated subset used by Search MCP.
        # We create it on first run as a copy of the default catalog so
        # users can later trim/edit it without affecting the default.
        search_path = os.path.join(self.schema_dir, "search_tools.json")
        if not os.path.exists(search_path):
            with open(search_path, "w") as f:
                json.dump(
                    {
                        "modified": [x.model_dump(exclude_none=False) for x in schema],
                        "modified_hash": self._schema_hash(
                            [x.model_dump(exclude_none=False) for x in schema]
                        ),
                    },
                    f,
                    indent=2,
                )

    def _schema_hash(self, schema):
        return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()

    def _load_modified(self):
        path = os.path.join(self.schema_dir, "tools.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        modified = data.get("modified")
        mod_hash = data.get("modified_hash")

        try:
            self.settings = _read_ui_settings(os.path.join(file_dir, "schemas"))
        except:
            self.settings = None

        default_tools_names = set(self.settings.get("default_tool_names") or [])
        
        if modified and mod_hash == self._schema_hash(modified):
            self.tools_list = [Tool.model_validate(json_str) for json_str in  modified]

        self.tools_list = [t for t in self.tools_list if t.name in default_tools_names]

        if self.settings and self.settings.get("expose_tools_search"):
            self.tools_list += self.clients['search'].tools

    @staticmethod
    def _hide_tools_params(tools_list):
        for tool in tools_list:
            tool.inputSchema["properties"] = {
                k: v
                for k, v in tool.inputSchema["properties"].items()
                if not (k[-1] == "_" and "default" in v)
            }
        return tools_list

    @classmethod
    async def create(cls, configs):
        # Always enable the Search MCP client internally so it is immediately
        # available to Master-MCP.
        # Exposure (whether it shows up in schemas/tools.json and list_tools)
        # is controlled separately via schema editor settings.
        configs = dict(configs)
        if "search" not in configs:
            configs["search"] = {
                "command": "python",
                "args": ["search_mcp_server.py"],
            }
        clients = {key: await cls.activate_client(cfg) for key, cfg in configs.items()}
        # avoid same names in the clients tools
        # for each MCP we also receive its name given by the user

        # we can redefine tools
        tools = {}
        for k, v in clients.items():
            logger.info(f"MasterMCP: Processing client '{k}' with {len(v.tools)} tools")
            for tool in v.tools:
                internal_tool_name = tool.name
                original_name = tool.name
                tool.name = k + "-" + tool.name
                logger.info(f"MasterMCP: Tool '{original_name}' -> '{tool.name}'")
                logger.info(
                    f"MasterMCP: Tool '{tool.name}' - inputSchema: {tool.inputSchema}"
                )
                logger.info(
                    f"MasterMCP: Tool '{tool.name}' - outputSchema: {tool.outputSchema}"
                )
                tools[tool.name] = {
                    "tool": tool,
                    "client": v,
                    "internal_tool_name": internal_tool_name,
                }
        
        logger.info(
            f"MasterMCP: Created {len(tools)} total tools: {list(tools.keys())}"
        )

        # Filter out search tools from the exposed schema unless enabled.
        # try:
        #     settings = _read_ui_settings(os.path.join(file_dir, "schemas"))
        #     expose_search = bool(settings.get("expose_tools_search", False))
        # except Exception:
        #     expose_search = False

        # if not expose_search:
        #     tools = {name: meta for name, meta in tools.items() if not name.startswith("search-")}

        return cls(clients, tools)

    @staticmethod
    async def activate_client(configuration):
        client = MCPClient()
        await client.connect_to_server(configuration["command"], configuration["args"])

        return client

    async def list_tools(self) -> list[types.Tool]:
        """List available tools with structured output schemas."""
        # Load tool schemas from disk (tools.json) and then apply "default tools"
        # filtering if configured via schema editor.
        self._load_modified()

        #settings = _read_ui_settings(self.schema_dir)
        return self.tools_list
        # default_names = set(settings.get("default_tool_names") or [])
        # if not default_names:
        #     # No defaults configured -> preserve previous behaviour
        #     return self.tools_list

        # # Use derived file if present, otherwise filter in-memory.
        # default_tools = _load_tools_json(self.schema_dir, "default_tools.json")
        # if default_tools:
        #     return default_tools

        # return [t for t in self.tools_list if t.name in default_names]

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        """Handle tool calls with structured output."""

        logger.info(f"MasterMCP: Calling tool '{name}' with arguments: {arguments}")
        original_name = self.tools[name]["internal_tool_name"]
        logger.info(f"MasterMCP: Internal tool name: '{original_name}'")
        tool_exec = await self.tools[name]["client"](original_name, arguments)
        #mcp_tool_latency_ms.record((time.time() - start_t) * 1000)
        logger.info(f"MasterMCP: Tool execution result: {tool_exec}")
        logger.info(f"MasterMCP: Tool execution content: {tool_exec.content}")
        logger.info(
            f"MasterMCP: Tool execution structured content: {tool_exec.structuredContent}"
        )
        logger.info(f"MasterMCP: Tool execution is error: {tool_exec.isError}")
        res = [types.TextContent(type="text", text=tool_exec.content[0].text)]
        logger.info(f"MasterMCP: Returning result: {res}")
        return {"result": tool_exec.content[0].text}

    async def cleanup_clients(self):
        for _, v in self.clients.items():
            await v.cleanup()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = None

    async def connect_to_server(self, command: str, args: list):
        server_params = StdioServerParameters(command=command, args=args, env=None)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        response = await self.session.list_tools()
        self.tools = response.tools

    async def __call__(self, tool_name, tool_args):
        result = await self.session.call_tool(tool_name, tool_args)
        logger.info(f"MCPClient: Tool result: {result} of type {type(result)}")
        return result

    async def cleanup(self):
        """Clean up resources.

        The underlying MCP Python SDK uses AnyIO task groups and cancel scopes
        in its stdio transport. Under some shutdown sequences (for example
        when the server process is cancelled while we're tearing down the
        stdio client), AnyIO can raise::

            RuntimeError: Attempted to exit a cancel scope that isn't the
            current tasks's current cancel scope

        from inside the async generator used by ``stdio_client``.

        This is a shutdown-only issue – by the time it happens, the
        client/server are already going away – but if we let it propagate it
        bubbles up through ``AsyncExitStack.aclose()`` and looks like a hard
        failure to callers.

        To keep caller code simple (``await client.cleanup()`` should not
        crash), we defensively swallow these specific shutdown errors while
        still allowing any other exceptions to surface.
        """
        try:
            await self.exit_stack.aclose()
        except BaseException as exc:  # pragma: no cover - defensive shutdown path
            # AnyIO-based cancellation uses a custom CancelledError subclass
            # which derives directly from BaseException, not Exception, so we
            # need to catch BaseException here.
            try:  # best-effort import; if this fails we still fall back to msg check
                import anyio  # type: ignore

                cancelled_type = anyio.get_cancelled_exc_class()
                is_cancel = isinstance(exc, cancelled_type)
            except Exception:  # pragma: no cover
                cancelled_type = None
                is_cancel = False

            msg = str(exc)
            looks_like_cancel_scope_bug = "Attempted to exit a cancel scope" in msg or "cancel scope" in msg

            if is_cancel or looks_like_cancel_scope_bug:
                logger.warning(
                    "Ignoring cancellation-related error during MCPClient cleanup: %r",
                    exc,
                )
                return

            # Re-raise unexpected cleanup errors so they aren't silently hidden
            raise

def load_mcp_configs():
    with open(os.path.join(BASE_DIR, "mcps.json"), "r") as file:
        data = json.loads(file.read())

    # Search MCP is always activated internally by Master-MCP (see MasterMCP.create).
    # The schema editor controls *exposure* (whether search tools are included in
    # schemas/tools.json and list_tools()), not connectivity.

    # Propagate CodeExec mode to child MCPs (e.g. Search MCP formatting).
    # This is set by the schema editor when the user enables CodeExec.
    try:
        settings = _read_ui_settings(os.path.join(file_dir, "schemas"))
        os.environ["MASTER_MCP_CODEEXEC"] = "1" if settings.get("codeexec_enabled") else ""
    except Exception:
        pass

    return data

@asynccontextmanager
async def main(_server):
    global master
    configs = load_mcp_configs()
    master = await MasterMCP.create(configs)
    tools = await master.list_tools()
    # @_server.list_tools()
    # async def list_tools():
    #     # load it from file
    #     logger.info("Calling to list the tools")
    #     return await master.list_tools()

    # @_server.call_tool()
    # async def call_tool(name, arguments):
    #     tool_exec = await master.call_tool(name, arguments)
    #     return tool_exec

    try:
        yield "connected"
    finally:
        await master.cleanup_clients()


server = Server("master-mcp-server", lifespan=main)

# workaround with global master - essential to make it working with mcp-proxy
master = None
@server.list_tools()
async def list_tools():
    # load it from file
    logger.info("Calling to list the tools")
    list = await master.list_tools()
    logger.info("Called to list the tools")
    return list

@server.call_tool()
async def call_tool(name, arguments):
    tool_exec = await master.call_tool(name, arguments)
    return tool_exec


async def server_run():
    """Run the server."""
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="Master-MCP-server",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except BaseException as exc:  # pragma: no cover - defensive shutdown path
        # Mirror the defensive logic in MCPClient.cleanup(): during shutdown,
        # AnyIO may raise cancellation-related errors (including a custom
        # CancelledError subclass and/or the
        # "Attempted to exit a cancel scope" RuntimeError). These indicate a
        # normal teardown path, not an application error, so we downgrade them
        # to a warning instead of crashing the process.

        try:  # best-effort import; if this fails we still fall back to msg check
            import anyio  # type: ignore

            cancelled_type = anyio.get_cancelled_exc_class()
            is_cancel = isinstance(exc, cancelled_type)
        except Exception:  # pragma: no cover
            cancelled_type = None
            is_cancel = False

        msg = str(exc)
        looks_like_cancel_scope_bug = "Attempted to exit a cancel scope" in msg or "cancel scope" in msg

        if is_cancel or looks_like_cancel_scope_bug:
            logger.warning(
                "Ignoring cancellation-related error while shutting down stdio server: %r",
                exc,
            )
            return

        # Re-raise unexpected errors so they aren't silently hidden.
        raise

# use python master_mcp_client.py to run the stdio MCP server
if __name__ == "__main__":
    asyncio.run(server_run())
