# VS Code ACP Webview Prototype

This folder contains a minimal VS Code webview extension prototype for a custom
Deep Agents UI.

Goals:
- expose workspace-aware command buttons
- show current workspace and mode
- send prompts and attachments to a local ACP bridge
- keep the Python ACP server reusable

This is intentionally small and local. It is not packaged for publishing.

## Flow

```txt
VS Code webview
  -> local bridge endpoint
    -> personal_deepagents_acp.py
      -> Deep Agents
```

## Planned UI actions

- `/about`
- `/memory`
- `/init`
- `/restore`
- file upload
- screenshot preview
- image annotation

