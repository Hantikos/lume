"""Builders that turn Lume source code into shippable artefacts.

* :func:`build_model_bundle` exports a trained model as JSON + a Python loader.
* :func:`build_app_bundle` packages a model into a runnable mini-application
  (stdlib-only HTTP server + HTML UI) that anybody can double-click.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile

from .errors import LumeError
from .interpreter import Interpreter
from .nn import Model
from .parser import parse

APP_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__APP_TITLE__</title>
<style>
  :root { color-scheme: dark; --bg:#0b1020; --panel:#141a2e; --line:#243055;
          --text:#e8ecf8; --muted:#8ea0c8; --accent:#7c5cff; --accent2:#22d3ee; }
  * { box-sizing: border-box; }
  body { margin:0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: radial-gradient(1200px 600px at 10% -10%, #1d2547 0%, var(--bg) 60%);
         color: var(--text); min-height:100vh; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 32px 20px 60px; }
  header { display:flex; align-items:center; gap:14px; margin-bottom:26px; }
  .logo { width:44px;height:44px;border-radius:12px; display:grid;place-items:center;
          background:linear-gradient(135deg,var(--accent),var(--accent2)); font-weight:800; color:#fff;}
  h1 { font-size:22px; margin:0; }
  .sub { color:var(--muted); font-size:13px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:20px;
          box-shadow: 0 18px 40px rgba(0,0,0,.35); margin-bottom:18px; }
  label { display:block; font-size:12px; text-transform:uppercase; letter-spacing:.08em;
          color:var(--muted); margin-bottom:8px; }
  textarea, input { width:100%; background:#0e1424; color:var(--text); border:1px solid var(--line);
          border-radius:10px; padding:12px; font: 14px/1.5 ui-monospace, Menlo, Consolas, monospace; }
  button { margin-top:12px; border:0; border-radius:10px; padding:11px 20px; cursor:pointer;
          background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff;
          font-weight:600; font-size:14px; }
  button:hover { filter:brightness(1.1); }
  pre { background:#0b1020; border:1px solid var(--line); border-radius:12px; padding:14px;
        overflow:auto; font:13px/1.6 ui-monospace, Menlo, Consolas, monospace; color:#cfe0ff; }
  .pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px;
          background:rgba(124,92,255,.18); border:1px solid rgba(124,92,255,.4); color:#c9bcff; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  @media (max-width:760px){ .grid { grid-template-columns:1fr; } }
  a { color:var(--accent2); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">L</div>
    <div>
      <h1>__APP_TITLE__</h1>
      <div class="sub">Built with Lume &middot; model <span class="pill">__MODEL_NAME__</span></div>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <label>Input features (comma separated)</label>
      <input id="features" value="__SAMPLE_INPUT__" />
      <button id="run">Predict</button>
    </div>
    <div class="card">
      <label>Result</label>
      <pre id="out">Waiting for input…</pre>
    </div>
  </div>

  <div class="card">
    <label>Model</label>
    <pre id="info">__MODEL_INFO__</pre>
  </div>
</div>
<script>
const runBtn = document.getElementById('run');
runBtn.addEventListener('click', async () => {
  const out = document.getElementById('out');
  out.textContent = 'Predicting…';
  const features = document.getElementById('features').value
      .split(',').map(v => parseFloat(v.trim())).filter(v => !Number.isNaN(v));
  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({features})
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = 'Error: ' + err;
  }
});
</script>
</body>
</html>
"""

APP_PY = '''"""__APP_TITLE__ - a standalone Lume application.

Run it with:   python app.py
Then open:     http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.json")


def load_model():
    with open(MODEL_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def predict(data, features):
    """Run the exported model over a flat list of numbers."""
    layers = data.get("layers", [])
    values = [float(v) for v in features]
    for layer in layers:
        kind = layer.get("type")
        if kind != "Dense":
            continue
        weights = layer.get("W", [])          # in_dim rows x units columns
        bias = layer.get("b", [])
        if not weights:
            continue
        units = len(bias) if bias else len(weights[0])
        activation = layer.get("activation", "none")
        out = []
        for j in range(units):
            total = bias[j] if j < len(bias) else 0.0
            for i, row in enumerate(weights):
                if j < len(row) and i < len(values):
                    total += row[j] * values[i]
            out.append(total)
        values = _activate(out, activation)
    if len(values) == 1:
        return {"prediction": values[0], "class": 1 if values[0] > 0.5 else 0}
    best = max(range(len(values)), key=lambda i: values[i])
    return {"prediction": values[best], "class": best, "probabilities": values}


def _activate(values, kind):
    import math
    kind = (kind or "none").lower()
    if kind == "relu":
        return [v if v > 0 else 0.0 for v in values]
    if kind == "sigmoid":
        return [1 / (1 + math.exp(-v)) for v in values]
    if kind == "tanh":
        return [math.tanh(v) for v in values]
    if kind == "softmax":
        top = max(values)
        exps = [math.exp(v - top) for v in values]
        total = sum(exps)
        return [e / total for e in exps]
    return values


HTML = open(os.path.join(HERE, "index.html"), "r", encoding="utf-8").read()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="text/html; charset=utf-8"):
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML)
        elif self.path == "/model":
            self._send(200, json.dumps(load_model(), indent=2), "application/json")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or "{}")
        except json.JSONDecodeError:
            self._send(400, '{"error": "bad json"}', "application/json")
            return
        features = payload.get("features", [])
        try:
            result = predict(load_model(), features)
        except Exception as exc:  # pragma: no cover - runtime safety
            result = {"error": str(exc)}
        self._send(200, json.dumps(result, indent=2), "application/json")

    def log_message(self, *args):
        pass


def main():
    port = 8000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    if "--predict" in sys.argv:
        features = sys.argv[sys.argv.index("--predict") + 1].split(",")
        print(json.dumps(predict(load_model(), features), indent=2))
        return
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"__APP_TITLE__ is running on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\nbye")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
'''

README_MD = """# {title}

A model built with [Lume](https://github.com/{repo}) and packaged as a
standalone application.

## Run

```
python app.py            # then open http://127.0.0.1:8000
python app.py --predict "0.3,0.7"
```

## What is inside

| file | purpose |
| --- | --- |
| `app.py` | tiny stdlib server + inference (no third-party packages) |
| `index.html` | the user interface |
| `model.json` | weights and architecture |

## Model

```
{summary}
```
"""

MODEL_LOADER_PY = '''"""Load a Lume model exported to JSON."""

import json
import os


def load(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def predict(model, features):
    values = [float(v) for v in features]
    for layer in model.get("layers", []):
        if layer.get("type") != "Dense":
            continue
        weights = layer.get("W", [])          # in_dim rows x units columns
        bias = layer.get("b", [])
        if not weights:
            continue
        units = len(bias) if bias else len(weights[0])
        out = []
        for j in range(units):
            total = bias[j] if j < len(bias) else 0.0
            for i, row in enumerate(weights):
                if j < len(row) and i < len(values):
                    total += row[j] * values[i]
            out.append(total)
        values = out
    return values
'''


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _run_program(source: str, filename: str = "<build>") -> Interpreter:
    program = parse(source, filename)
    interp = Interpreter(write=lambda text: print(text))
    interp.visit(program, interp.global_scope)
    return interp


def _select_model(interp: Interpreter, preferred: str | None = None) -> Model:
    if not interp.models:
        raise LumeError("no model found in this file - define one with `model Name { ... }`")
    if preferred:
        for model in interp.models:
            if model.name == preferred:
                return model
    return interp.models[-1]


def build_model_bundle(source: str, name: str | None = None, outdir: str = "build",
                       repo: str = "lume-lang/lume") -> dict:
    """Train/execute the source and export the model as a JSON + Python bundle."""
    interp = _run_program(source)
    model = _select_model(interp, name)
    folder = os.path.join(outdir, model.name)
    os.makedirs(folder, exist_ok=True)
    model_path = os.path.join(folder, "model.json")
    model.save(model_path)
    _write(os.path.join(folder, "model.py"), MODEL_LOADER_PY)
    _write(os.path.join(folder, "README.md"), README_MD.format(
        title=f"{model.name} (Lume model)",
        repo=repo,
        summary=_model_summary(model),
    ))
    _write(os.path.join(folder, "source.lume"), source)
    _write(os.path.join(folder, "requirements.txt"), "# no third-party packages required\n")
    return {
        "ok": True,
        "kind": "model",
        "name": model.name,
        "path": os.path.abspath(folder),
        "files": ["model.json", "model.py", "README.md", "source.lume"],
        "metrics": model.metrics,
    }


def build_app_bundle(source: str, name: str | None = None, outdir: str = "build",
                     repo: str = "lume-lang/lume") -> dict:
    """Package the model from ``source`` into a runnable mini-application."""
    interp = _run_program(source)
    model = _select_model(interp, name)
    app_name = f"{model.name} App"
    folder = os.path.join(outdir, app_name.replace(" ", ""))
    os.makedirs(folder, exist_ok=True)
    model.save(os.path.join(folder, "model.json"))
    sample = _sample_input(model)
    info = json.dumps(model.to_dict(), indent=2, ensure_ascii=False)
    _write(os.path.join(folder, "app.py"), APP_PY.replace("__APP_TITLE__", app_name))
    _write(os.path.join(folder, "index.html"), APP_HTML
           .replace("__APP_TITLE__", app_name)
           .replace("__MODEL_NAME__", model.name)
           .replace("__SAMPLE_INPUT__", sample)
           .replace("__MODEL_INFO__", info.replace("`", "'")))
    _write(os.path.join(folder, "README.md"), README_MD.format(
        title=app_name,
        repo=repo,
        summary=_model_summary(model),
    ))
    _write(os.path.join(folder, "run.sh"), "#!/usr/bin/env bash\npython3 app.py\n")
    _write(os.path.join(folder, "run.bat"), "@echo off\r\npython app.py\r\n")
    _write(os.path.join(folder, "source.lume"), source)
    _write(os.path.join(folder, "requirements.txt"), "# no third-party packages required\n")
    archive = shutil.make_archive(os.path.join(outdir, app_name.replace(" ", "")), "zip", folder)
    return {
        "ok": True,
        "kind": "app",
        "name": model.name,
        "path": os.path.abspath(folder),
        "archive": os.path.abspath(archive),
        "files": sorted(os.listdir(folder)),
    }


def _model_summary(model: Model) -> str:
    lines = [f"name: {model.name}"]
    for index, layer in enumerate(model.layers):
        detail = ""
        if hasattr(layer, "units"):
            detail = f"{layer.units} units"
        elif hasattr(layer, "rate"):
            detail = f"rate {layer.rate}"
        elif hasattr(layer, "dim"):
            detail = f"{layer.vocab_size} x {layer.dim}"
        lines.append(f"  layer {index + 1}: {layer.type_name} {detail}".rstrip())
    if model.metrics:
        lines.append(f"accuracy: {model.metrics.get('accuracy', 0):.3f}")
    return "\n".join(lines)


def _sample_input(model: Model) -> str:
    for layer in model.layers:
        if hasattr(layer, "in_dim") and layer.in_dim:
            return ",".join("0.0" for _ in range(layer.in_dim))
    return "1.0,0.0"
