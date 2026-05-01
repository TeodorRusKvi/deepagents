from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[4] / "scripts" / "personal_deepagents_acp.py"
    spec = importlib.util.spec_from_file_location("personal_deepagents_acp", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_agent_config_uses_defaults(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("WORKSPACE_ROOT", "/tmp/empty-workspace")
    monkeypatch.delenv("DEEPAGENTS_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("DEEPAGENTS_MODEL", raising=False)

    config = module.build_agent_config()

    assert config["model"] == module.DEFAULT_MODEL
    assert config["system_prompt"].startswith("You are a helpful coding assistant.")
    assert config["workspace_root"] == Path("/tmp/empty-workspace")


def test_build_agent_config_honors_environment(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("WORKSPACE_ROOT", "/tmp/custom-workspace")
    monkeypatch.setenv("DEEPAGENTS_SYSTEM_PROMPT", "Stay focused.")
    monkeypatch.setenv("DEEPAGENTS_MODEL", "openai:gpt-5.1")

    config = module.build_agent_config()

    assert config["model"] == "openai:gpt-5.1"
    assert config["system_prompt"] == "Stay focused."
    assert config["workspace_root"] == Path("/tmp/custom-workspace")


def test_build_system_prompt_prefers_agents_over_gemini(tmp_path) -> None:
    module = _load_module()
    (tmp_path / "AGENTS.md").write_text("Agent rules", encoding="utf-8")
    (tmp_path / "GEMINI.md").write_text("Gemini notes", encoding="utf-8")

    prompt = module.build_system_prompt(tmp_path, mode="manual")

    assert "Behavior guidelines:" in prompt
    assert "Session context:" in prompt
    assert f"- workspace_root: {tmp_path.as_posix()}" in prompt
    assert "- mode: manual" in prompt
    assert "Workspace instructions from AGENTS.md:" in prompt
    assert "Agent rules" in prompt
    assert "GEMINI.md" not in prompt


def test_build_system_prompt_uses_gemini_when_agents_missing(tmp_path) -> None:
    module = _load_module()
    (tmp_path / "GEMINI.md").write_text("Gemini notes", encoding="utf-8")

    prompt = module.build_system_prompt(tmp_path)

    assert "Session context:" in prompt
    assert "Workspace instructions from AGENTS.md:" not in prompt
    assert "Supplemental instructions from GEMINI.md:" in prompt
    assert "Gemini notes" in prompt


def test_build_system_prompt_uses_base_prompt_when_files_missing(tmp_path) -> None:
    module = _load_module()

    prompt = module.build_system_prompt(tmp_path)

    assert prompt.startswith("You are a helpful coding assistant.")
    assert f"- workspace_root: {tmp_path.as_posix()}" in prompt
    assert "AGENTS.md" not in prompt
    assert "GEMINI.md" not in prompt


def test_build_attachment_tools_lists_saved_files(tmp_path) -> None:
    module = _load_module()
    saved = tmp_path / "attachments" / "session-1"
    saved.mkdir(parents=True)
    (saved / "note.txt").write_text("hello", encoding="utf-8")

    tools = module.build_attachment_tools(tmp_path)
    result = tools[0].invoke({})

    assert "attachments/session-1/note.txt" in result


def test_read_saved_attachment_rejects_outside_workspace(tmp_path) -> None:
    module = _load_module()
    tools = module.build_attachment_tools(tmp_path)

    try:
        tools[1].invoke({"file_path": "../secret.txt"})
    except ValueError as error:
        assert "attachments directory" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_read_saved_attachment_meta_reports_size(tmp_path) -> None:
    module = _load_module()
    saved = tmp_path / "attachments" / "session-1"
    saved.mkdir(parents=True)
    (saved / "report.pdf").write_bytes(b"pdf-bytes")

    tools = module.build_attachment_tools(tmp_path)
    result = tools[1].invoke({"file_path": "attachments/session-1/report.pdf"})

    assert "size_bytes=9" in result
    assert "suffix=.pdf" in result


def test_read_saved_attachment_bytes_returns_base64(tmp_path) -> None:
    module = _load_module()
    saved = tmp_path / "attachments" / "session-1"
    saved.mkdir(parents=True)
    (saved / "image.png").write_bytes(b"image-bytes")

    tools = module.build_attachment_tools(tmp_path)
    result = tools[2].invoke({"file_path": "attachments/session-1/image.png"})

    assert result == "aW1hZ2UtYnl0ZXM="


def test_build_agent_uses_session_model_when_available(monkeypatch, tmp_path) -> None:
    module = _load_module()
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("DEEPAGENTS_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("DEEPAGENTS_MODEL", raising=False)
    (tmp_path / "AGENTS.md").write_text("Agent rules", encoding="utf-8")

    captured: dict[str, object] = {}

    fake_deepagents = types.SimpleNamespace(
        create_deep_agent=lambda **kwargs: captured.update(kwargs) or "agent",
    )
    fake_memory = types.SimpleNamespace(MemorySaver=lambda: "memory-saver")
    monkeypatch.setitem(sys.modules, "deepagents", fake_deepagents)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.memory", fake_memory)

    context = type("Context", (), {"model": "openai:gpt-5.1"})()
    result = module.build_agent(context)

    assert result == "agent"
    assert captured["model"] == "openai:gpt-5.1"


def test_build_agent_falls_back_to_default_model(monkeypatch, tmp_path) -> None:
    module = _load_module()
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("DEEPAGENTS_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("DEEPAGENTS_MODEL", raising=False)

    captured: dict[str, object] = {}

    fake_deepagents = types.SimpleNamespace(
        create_deep_agent=lambda **kwargs: captured.update(kwargs) or "agent",
    )
    fake_memory = types.SimpleNamespace(MemorySaver=lambda: "memory-saver")
    monkeypatch.setitem(sys.modules, "deepagents", fake_deepagents)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.memory", fake_memory)

    context = type("Context", (), {"model": None})()
    result = module.build_agent(context)

    assert result == "agent"
    assert captured["model"] == module.DEFAULT_MODEL
