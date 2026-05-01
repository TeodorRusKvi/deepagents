from __future__ import annotations

from pathlib import Path

from deepagents_acp.server import AgentServerACP


def _server(tmp_path: Path) -> AgentServerACP:
    server = AgentServerACP.__new__(AgentServerACP)
    server._cwd = tmp_path.as_posix()
    server._session_cwds = {"session-1": tmp_path.as_posix()}
    server._session_modes = {"session-1": "auto"}
    server._modes = None
    return server


def test_about_command_reports_context(tmp_path) -> None:
    server = _server(tmp_path)

    output = server._handle_command("session-1", "/about")

    assert output is not None
    assert "Deep Agents ACP server" in output
    assert f"cwd: {tmp_path.as_posix()}" in output
    assert "mode: auto" in output


def test_memory_command_reports_workspace(tmp_path) -> None:
    server = _server(tmp_path)
    (tmp_path / "AGENTS.md").write_text("rules", encoding="utf-8")

    output = server._handle_command("session-1", "/memory")

    assert output is not None
    assert f"workspace_root: {tmp_path.as_posix()}" in output
    assert "AGENTS.md:" in output


def test_init_command_creates_agents_md(tmp_path) -> None:
    server = _server(tmp_path)

    output = server._handle_command("session-1", "/init")

    assert output is not None
    assert (tmp_path / "AGENTS.md").exists()
    assert "Created" in output
