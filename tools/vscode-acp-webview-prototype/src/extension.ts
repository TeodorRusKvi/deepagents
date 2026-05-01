import * as vscode from "vscode";

export function activate(context: vscode.ExtensionContext) {
  const disposable = vscode.commands.registerCommand("acpPrototype.openPanel", () => {
    const panel = vscode.window.createWebviewPanel(
      "acpPrototype",
      "ACP Prototype",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
      },
    );

    panel.webview.html = getHtml(panel.webview);
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}

function getHtml(webview: vscode.Webview): string {
  const nonce = String(Date.now());
  const csp = [
    "default-src 'none'",
    `img-src ${webview.cspSource} https: data:`,
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `script-src 'nonce-${nonce}'`,
  ].join("; ");

  return `<!doctype html>
  <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta http-equiv="Content-Security-Policy" content="${csp}" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>ACP Prototype</title>
      <style>
        body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; }
        .row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        button { padding: 8px 12px; border: 1px solid #666; border-radius: 8px; background: #1f1f1f; color: white; }
        textarea { width: 100%; min-height: 140px; margin-top: 8px; }
        .box { border: 1px solid #333; border-radius: 10px; padding: 12px; margin-top: 12px; }
      </style>
    </head>
    <body>
      <h2>ACP Prototype</h2>
      <div class="box">
        <div class="row">
          <button data-command="/about">/about</button>
          <button data-command="/memory">/memory</button>
          <button data-command="/init">/init</button>
          <button data-command="/restore">/restore</button>
        </div>
        <div class="row">
          <button id="send">Send Prompt</button>
        </div>
        <label for="prompt">Prompt</label>
        <textarea id="prompt" placeholder="Type a prompt or command..."></textarea>
      </div>
      <div class="box">
        <strong>Output</strong>
        <pre id="output"></pre>
      </div>
      <script nonce="${nonce}">
        const output = document.getElementById("output");
        const prompt = document.getElementById("prompt");
        const show = (text) => { output.textContent = text; };

        document.querySelectorAll("button[data-command]").forEach((button) => {
          button.addEventListener("click", () => {
            prompt.value = button.getAttribute("data-command") || "";
            show("Ready to send: " + prompt.value);
          });
        });

        document.getElementById("send").addEventListener("click", () => {
          show("Would send to ACP bridge:\\n" + prompt.value);
        });
      </script>
    </body>
  </html>`;
}

