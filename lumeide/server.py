"""Lume Studio - a tiny desktop-like IDE for the Lume language.

The whole studio is built on the Python standard library: a local HTTP server
serves a single page application that talks to the Lume interpreter, to the
build pipeline and to the keyless Pollinations AI assistant.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lume.errors import LumeError
from lume.interpreter import Interpreter
from lume.parser import parse

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")

POLLINATIONS_TEXT = "https://text.pollinations.ai"
POLLINATIONS_IMAGE = "https://image.pollinations.ai"
DEFAULT_TEXT_MODEL = "openai-fast"

TEXT_MODELS = [
    {"id": "openai-fast", "name": "GPT-OSS 20B (fast, default)"},
    {"id": "openai", "name": "GPT-OSS 20B"},
    {"id": "mistral", "name": "Mistral"},
    {"id": "llama", "name": "Llama"},
    {"id": "qwen-coder", "name": "Qwen Coder"},
]

SYSTEM_PROMPT = (
    "You are Lume Assistant, embedded in Lume Studio - an IDE for the Lume "
    "programming language (a tiny language for building AI models).\n"
    "Lume syntax essentials:\n"
    "  fn name(a, b) { return a + b }        # functions, defaults: fn f(a, b = 2)\n"
    "  let x = 1   const y = 2               # variables\n"
    "  data set = dataset.xor()             # datasets\n"
    "  model Net { layer Dense(8, \"relu\")   # a neural network\n"
    "             layer Dense(1, \"sigmoid\")\n"
    "             train(data = set, epochs = 50, lr = 0.1, optimizer = \"adam\",\n"
    "                   loss = \"binary_crossentropy\")\n"
    "             eval(data = set) }\n"
    "  print \"text\"   f\"text {value}\"     # printing and interpolation\n"
    "  for x in list { }  while cond { }  if cond { } else { }\n"
    "  layers: Dense, Activation, Dropout, Flatten, Embedding\n"
    "  losses: mse, mae, binary_crossentropy, crossentropy\n"
    "  optimizers: sgd, momentum, rmsprop, adam\n"
    "  datasets: dataset.xor(), dataset.linear(), dataset.blobs(), dataset.spiral(), dataset.csv(path)\n"
    "  builtins: print, len, sum, min, max, abs, range, sqrt, tensor, zeros, ones, randn, softmax\n"
    "Answer with short, practical Lume code. Use fenced code blocks with the lume tag."
)


def _http_json(url: str, payload: dict | None = None, timeout: int = 120, raw: bool = False):
    data = None
    headers = {"User-Agent": "LumeStudio/0.1", "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if raw:
        return body
    text = body.decode("utf-8", "replace").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


class LumeHandler(BaseHTTPRequestHandler):
    server_version = "LumeStudio/0.1"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------
    def log_message(self, *args):  # keep the console clean
        pass

    def _send(self, code: int, body: bytes, content_type="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload, code=200):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _file(self, relative: str) -> str:
        root = os.path.abspath(self.server.workspace)
        target = os.path.abspath(os.path.join(root, relative))
        if not target.startswith(root):
            raise LumeError("path outside of the workspace is not allowed")
        return target

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/examples":
                return self._json(self._examples())
            if path.startswith("/api/examples/"):
                name = path[len("/api/examples/"):]
                return self._json(self._example(name))
            if path == "/api/tree":
                return self._json(self._tree())
            if path == "/api/file":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                relative = (query.get("path") or [""])[0]
                return self._json(self._read_file(relative))
            if path == "/api/models":
                return self._json({"models": self._text_models()})
            if path == "/api/health":
                return self._json({"ok": True, "workspace": self.server.workspace})
            self._json({"error": "not found"}, 404)
        except LumeError as exc:
            self._json({"error": exc.message}, 400)
        except Exception as exc:  # pragma: no cover - safety net
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        payload = self._read_json()
        try:
            if path == "/api/run":
                return self._json(self._run(payload))
            if path == "/api/check":
                return self._json(self._check(payload))
            if path == "/api/chat":
                return self._json(self._chat(payload))
            if path == "/api/image":
                return self._json(self._image(payload))
            if path == "/api/save":
                return self._json(self._save(payload))
            if path == "/api/build/model":
                return self._json(self._build_model(payload))
            if path == "/api/build/app":
                return self._json(self._build_app(payload))
            if path == "/api/build/exe":
                return self._json(self._build_exe(payload))
            self._json({"error": "not found"}, 404)
        except LumeError as exc:
            self._json({"ok": False, "error": exc.message}, 400)
        except Exception as exc:  # pragma: no cover - safety net
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- static -----------------------------------------------------------
    def _static(self, name: str):
        target = os.path.normpath(os.path.join(STATIC_DIR, name))
        if not target.startswith(STATIC_DIR) or not os.path.isfile(target):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        with open(target, "rb") as handle:
            self._send(200, handle.read(), ctype)

    # -- language ---------------------------------------------------------
    def _run(self, payload):
        code = payload.get("code", "")
        filename = payload.get("filename", "main.lume")
        output: list[str] = []
        started = time.time()
        try:
            program = parse(code, filename)
        except LumeError as exc:
            return {"ok": False, "error": exc.render(), "line": exc.line, "col": exc.col,
                    "output": output}
        interp = Interpreter(write=lambda text: output.append(text))
        try:
            interp.visit(program, interp.global_scope)
        except LumeError as exc:
            return {"ok": False, "error": exc.render(), "line": exc.line, "col": exc.col,
                    "output": output}
        except RecursionError:
            return {"ok": False, "error": "RecursionError: the program is too deeply nested",
                    "output": output}
        models = [{"name": m.name, "layers": m.layer_names(), "metrics": m.metrics}
                  for m in interp.models]
        return {"ok": True, "output": output, "models": models,
                "ms": round((time.time() - started) * 1000, 1)}

    def _check(self, payload):
        code = payload.get("code", "")
        filename = payload.get("filename", "main.lume")
        try:
            parse(code, filename)
        except LumeError as exc:
            return {"ok": False, "error": exc.render(), "line": exc.line, "col": exc.col}
        return {"ok": True}

    # -- assistant --------------------------------------------------------
    def _text_models(self):
        try:
            data = _http_json(f"{POLLINATIONS_TEXT}/models", timeout=8)
            models = []
            for item in data if isinstance(data, list) else []:
                models.append({"id": item.get("name", ""), "name": item.get("description", "")})
            if models:
                return models
        except Exception:
            pass
        return TEXT_MODELS

    def _chat(self, payload):
        message = (payload.get("message") or "").strip()
        if not message:
            return {"ok": False, "error": "empty message"}
        code = payload.get("code") or ""
        history = payload.get("history") or []
        model = payload.get("model") or DEFAULT_TEXT_MODEL
        system = SYSTEM_PROMPT
        if code.strip():
            system += "\n\nThe user is editing this Lume file right now:\n```lume\n" + code[:6000] + "\n```"
        messages = [{"role": "system", "content": system}]
        for item in history[-6:]:
            role = item.get("role")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        try:
            answer = _http_json(f"{POLLINATIONS_TEXT}/openai",
                                {"model": model, "messages": messages, "temperature": 0.4})
        except urllib.error.HTTPError as exc:
            return {"ok": False, "error": f"assistant HTTP {exc.code}"}
        except Exception as exc:
            return {"ok": False, "error": f"assistant is unreachable: {exc}"}
        text = ""
        if isinstance(answer, dict):
            choices = answer.get("choices") or []
            if choices:
                text = (choices[0].get("message") or {}).get("content", "")
            elif answer.get("text"):
                text = answer["text"]
        return {"ok": True, "reply": text.strip(), "model": model}

    def _image(self, payload):
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return {"ok": False, "error": "empty prompt"}
        seed = payload.get("seed") or int(time.time() * 1000) % 100000
        width = int(payload.get("width", 640))
        height = int(payload.get("height", 400))
        query = urllib.parse.urlencode({
            "width": width, "height": height, "seed": seed, "nologo": "true",
            "model": payload.get("model", "sana"),
        })
        url = f"{POLLINATIONS_IMAGE}/prompt/{urllib.parse.quote(prompt)}?{query}"
        return {"ok": True, "url": url}

    # -- workspace --------------------------------------------------------
    def _tree(self):
        root = os.path.abspath(self.server.workspace)
        entries = []
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d != "__pycache__"]
            rel_base = os.path.relpath(base, root)
            for name in sorted(files):
                if name.startswith("."):
                    continue
                rel = name if rel_base == "." else os.path.join(rel_base, name)
                entries.append({"path": rel.replace(os.sep, "/"),
                                "size": os.path.getsize(os.path.join(base, name))})
        return {"workspace": root, "files": entries}

    def _save(self, payload):
        relative = (payload.get("path") or "").strip()
        content = payload.get("content", "")
        if not relative:
            return {"ok": False, "error": "no file name given"}
        target = self._file(relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content)
        return {"ok": True, "path": relative, "bytes": len(content.encode("utf-8"))}

    def _read_file(self, relative: str):
        if not relative:
            raise LumeError("no file name given")
        target = self._file(relative)
        if not os.path.isfile(target):
            raise LumeError(f"file '{relative}' not found")
        with open(target, "r", encoding="utf-8") as handle:
            return {"path": relative, "content": handle.read()}

    def _examples(self):
        if not os.path.isdir(EXAMPLES_DIR):
            return {"files": []}
        files = []
        for name in sorted(os.listdir(EXAMPLES_DIR)):
            if name.endswith(".lume"):
                files.append({"name": name, "title": _title_of(name)})
        return {"files": files}

    def _example(self, name):
        name = os.path.basename(name)
        target = os.path.join(EXAMPLES_DIR, name)
        if not os.path.isfile(target):
            raise LumeError(f"example '{name}' not found")
        with open(target, "r", encoding="utf-8") as handle:
            return {"name": name, "content": handle.read()}

    # -- building ---------------------------------------------------------
    def _build_model(self, payload):
        from ..builders import build_model_bundle
        code = payload.get("code", "")
        name = payload.get("name") or None
        outdir = self._file(payload.get("outdir") or "build")
        result = build_model_bundle(code, name=name, outdir=outdir)
        return {"ok": True, **result}

    def _build_app(self, payload):
        from ..builders import build_app_bundle
        code = payload.get("code", "")
        name = payload.get("name") or None
        outdir = self._file(payload.get("outdir") or "build")
        result = build_app_bundle(code, name=name, outdir=outdir)
        return {"ok": True, **result}

    def _build_exe(self, payload):
        """Build a single-file executable for the current project (PyInstaller)."""
        code = payload.get("code", "")
        name = (payload.get("name") or "LumeApp").strip()
        if not name.replace("_", "").replace("-", "").isalnum():
            return {"ok": False, "error": "the application name may only contain letters, digits, - and _"}
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            return {"ok": False, "error": "PyInstaller is not installed on this machine. "
                                          "Install it with:  pip install pyinstaller"}
        workdir = os.path.join(os.path.abspath(self.server.workspace), "build", name)
        os.makedirs(workdir, exist_ok=True)
        source = os.path.join(workdir, "main.lume")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(code)
        launcher = os.path.join(workdir, "launcher.py")
        with open(launcher, "w", encoding="utf-8") as handle:
            handle.write(_EXE_LAUNCHER)
        command = [sys.executable, "-m", "PyInstaller", "--onefile", "--noconfirm",
                   "--name", name, "--distpath", os.path.join(workdir, "dist"),
                   "--workpath", os.path.join(workdir, "build-tmp"),
                   "--specpath", workdir, launcher]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "the build took too long and was stopped"}
        if result.returncode != 0:
            tail = (result.stdout or "")[-1500:] + (result.stderr or "")[-1500:]
            return {"ok": False, "error": "PyInstaller failed:\n" + tail}
        binary = os.path.join(workdir, "dist", name + (".exe" if os.name == "nt" else ""))
        return {"ok": True, "path": binary,
                "note": "Start it with the same command line you would use for 'lume run'."}


def _title_of(name: str) -> str:
    stem = name[:-5] if name.endswith(".lume") else name
    return stem.replace("_", " ").replace("-", " ").title()


_EXE_LAUNCHER = '''"""Launcher used by Lume Studio when it builds a standalone executable."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    source = os.path.join(HERE, "main.lume")
    try:
        from lume.cli import main as lume_main
    except ImportError:
        sys.path.insert(0, HERE)
        sys.path.insert(0, os.path.dirname(HERE))
        from lume.cli import main as lume_main
    return lume_main(["run", source])


if __name__ == "__main__":
    sys.exit(main())
'''


class LumeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, workspace: str):
        super().__init__(address, LumeHandler)
        self.workspace = os.path.abspath(workspace)


def serve(workspace: str = ".", host: str = "127.0.0.1", port: int = 0):
    """Start the studio server and return it (the caller owns the thread)."""
    server = LumeServer((host, port), workspace)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
