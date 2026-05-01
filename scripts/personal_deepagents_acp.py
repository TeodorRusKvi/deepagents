"""Launch a personal Deep Agents ACP server over stdio.

This script is intended to be launched by ACP-compatible editors such as the
VS Code ACP extension. It keeps the entrypoint small and configurable so you
can point it at a custom workspace, model, and prompt.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
from acp.schema import McpServerStdio, HttpMcpServer, SseMcpServer

if TYPE_CHECKING:
    from deepagents_acp.server import AgentSessionContext

load_dotenv("/Users/teodorrustadkvisberg/projects/current/deepagents/.env")

DEFAULT_MODEL = "openai:gpt-4o-mini"
BASE_SYSTEM_PROMPT = """You are a helpful coding assistant.

Behavior guidelines:
- Prefer concise, direct answers.
- Keep code changes minimal and focused.
- Preserve existing public APIs unless the user explicitly asks for a breaking change.
- Use repo-local instructions as higher priority than generic defaults.
- Use `read_file`, `ls`, and other filesystem tools to inspect files including any attachments saved in the `attachments/` directory.
"""


def _read_instruction_file(path: Path) -> str:
    """Read an instruction file if it exists.

    Args:
        path: Absolute or workspace-relative file path.

    Returns:
        The file contents, or an empty string if the file does not exist.
    """

    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(workspace_root: Path, *, mode: str | None = None) -> str:
    """Build the final system prompt for the agent.

    Args:
        workspace_root: Root directory for the active workspace.
        mode: Optional ACP session mode.

    Returns:
        A merged system prompt containing the base prompt and repo-local
        instructions from `AGENTS.md` and `GEMINI.md` when present.
    """

    agents_md = _read_instruction_file(workspace_root / "AGENTS.md")
    gemini_md = ""
    if not agents_md:
        gemini_md = _read_instruction_file(workspace_root / "GEMINI.md")

    parts: list[str] = [BASE_SYSTEM_PROMPT.strip()]
    parts.append(
        "Session context:\n"
        f"- workspace_root: {workspace_root.as_posix()}\n"
        f"- mode: {mode or 'auto'}",
    )
    if agents_md:
        parts.append("Workspace instructions from AGENTS.md:\n\n" + agents_md)
    elif gemini_md:
        parts.append("Supplemental instructions from GEMINI.md:\n\n" + gemini_md)

    return "\n\n".join(parts).strip()


def build_agent_config() -> dict[str, Any]:
    """Build the Deep Agents configuration for this personal server.

    Returns:
        A dictionary containing the agent configuration passed to
        `create_deep_agent`.
    """

    workspace_root = Path(
        os.environ.get("WORKSPACE_ROOT", Path.cwd().as_posix()),
    ).expanduser()
    mode = os.environ.get("DEEPAGENTS_MODE")
    system_prompt = os.environ.get("DEEPAGENTS_SYSTEM_PROMPT")
    if system_prompt is None:
        system_prompt = build_system_prompt(workspace_root, mode=mode)

    return {
        "model": os.environ.get("DEEPAGENTS_MODEL", DEFAULT_MODEL),
        "system_prompt": system_prompt,
        "workspace_root": workspace_root,
    }




async def build_agent(
    context: "AgentSessionContext",
) -> CompiledStateGraph:
    """Build a Deep Agents graph for the current ACP session."""

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_mcp_adapters.client import MultiServerMCPClient

        config = build_agent_config()
        workspace_root = cast(Path, config["workspace_root"])
        model = context.model or config["model"]
        
        print(f"DEBUG: Building agent for workspace: {workspace_root}, model: {model}", file=sys.stderr)
        
        system_prompt = build_system_prompt(
            workspace_root,
            mode=getattr(context, "mode", "auto"),
        )
        
        # Build tools
        tools = []
        
        # Load MCP tools if servers are provided
        if context.mcp_servers:
            print(f"DEBUG: Loading tools from {len(context.mcp_servers)} MCP servers", file=sys.stderr)
            server_configs = {}
            for i, server in enumerate(context.mcp_servers):
                server_id = getattr(server, "name", f"server_{i}")
                if isinstance(server, McpServerStdio):
                    server_configs[server_id] = {
                        "transport": "stdio",
                        "command": server.command,
                        "args": server.args,
                        "env": {e.name: e.value for e in server.env} if server.env else None,
                    }
                elif isinstance(server, HttpMcpServer):
                    server_configs[server_id] = {
                        "transport": "streamable_http",
                        "url": server.url,
                        "headers": server.headers,
                    }
                elif isinstance(server, SseMcpServer):
                    server_configs[server_id] = {
                        "transport": "sse",
                        "url": server.url,
                        "headers": server.headers,
                    }
            
            if server_configs:
                try:
                    print(f"DEBUG: Initializing MultiServerMCPClient with configs: {list(server_configs.keys())}", file=sys.stderr)
                    mcp_client = MultiServerMCPClient(server_configs)
                    mcp_tools = await mcp_client.get_tools()
                    print(f"DEBUG: Loaded {len(mcp_tools)} tools from MCP servers", file=sys.stderr)
                    tools.extend(mcp_tools)
                except Exception as mcp_err:
                    print(f"DEBUG: Failed to load MCP tools: {mcp_err}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    # If it's an ExceptionGroup (Python 3.11+), print sub-exceptions
                    if hasattr(mcp_err, "exceptions"):
                        for i, sub_err in enumerate(mcp_err.exceptions):
                            print(f"DEBUG: Sub-exception {i}: {sub_err}", file=sys.stderr)
                            if hasattr(sub_err, "__notes__"):
                                print(f"DEBUG: Notes: {sub_err.__notes__}", file=sys.stderr)

        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=MemorySaver(),
            backend=FilesystemBackend(root_dir=workspace_root, virtual_mode=True),
            debug=True,
        )
        print("DEBUG: Agent built successfully", file=sys.stderr)
        return agent
    except Exception as e:
        print(f"FAILED TO BUILD AGENT: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


async def main() -> None:
    """Start the ACP server."""

    from acp import run_agent
    try:
        from deepagents_acp.server import AgentServerACP
    except ImportError:
        # Fallback for environment where deepagents_acp is not in path correctly
        print("CRITICAL: deepagents_acp.server not found in PYTHONPATH", file=sys.stderr)
        raise

    server = AgentServerACP(build_agent)
    await run_agent(server)


if __name__ == "__main__":
    asyncio.run(main())
