"""Command line interface for Lume."""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .errors import LumeError
from .interpreter import Interpreter
from .parser import parse

BANNER = r"""
   _                      _
  | |    ___  _   _ _ __ | |_ _   _
  | |   / _ \| | | | '_ \| __| | | |
  | |__|  __/| |_| | | | | |_| |_| |
  |_____\___| \__, |_| |_|\__|\__, |
               |___/             |___/
  a tiny language for AI models
"""


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _run_source(source: str, filename: str = "<script>", quiet: bool = False) -> Interpreter:
    program = parse(source, filename)
    started = time.time()
    state = {"broken": False}

    def write(text: str):
        if quiet or state["broken"]:
            return
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except BrokenPipeError:      # e.g. `lume run file.lume | head`
            state["broken"] = True

    interp = Interpreter(write=write)
    interp.visit(program, interp.global_scope)
    if not quiet and not state["broken"]:
        elapsed = (time.time() - started) * 1000
        write(f"— done in {elapsed:.1f} ms\n")
    return interp


def cmd_run(args) -> int:
    source = _read(args.file)
    _run_source(source, args.file)
    return 0


def cmd_check(args) -> int:
    source = _read(args.file)
    try:
        parse(source, args.file)
    except LumeError as exc:
        print(f"✖ {exc}")
        return 1
    print(f"✔ {args.file}: syntax is valid")
    return 0


def cmd_repl(args) -> int:
    print(BANNER)
    print("Type an expression, or 'exit' to leave.\n")
    scope = None
    interp = Interpreter(write=lambda text: print(text))
    scope = interp.global_scope
    buffer = []
    while True:
        prompt = "... " if buffer else ">>> "
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not buffer and line.strip() in ("exit", "quit"):
            print("bye")
            return 0
        buffer.append(line)
        source = "\n".join(buffer)
        try:
            program = parse(source, "<repl>")
        except LumeError as exc:
            message = exc.message
            if "end of file" in message or "expected" in message and "got" not in message:
                continue  # keep reading a multi-line block
            print(f"✖ {message}")
            buffer = []
            continue
        buffer = []
        try:
            interp.visit(program, scope)
        except LumeError as exc:
            print(f"✖ {exc}")
        except Exception as exc:  # pragma: no cover - safety net
            print(f"✖ {type(exc).__name__}: {exc}")


STARTER = '''# {name} - built with Lume
# Run with:  lume run main.lume

print "Hello from {name}!"

# --- a tiny neural network ---------------------------------------------
data points = dataset.blobs(classes = 2, per_class = 30)

model Classifier {{
    layer Dense(16, "relu")
    layer Dense(8, "relu")
    layer Dense(2, "softmax")

    train(data = points, epochs = 30, lr = 0.05, optimizer = "adam", loss = "crossentropy")
    eval(data = points)
}}
'''


def cmd_new(args) -> int:
    root = args.name
    if os.path.exists(root) and os.listdir(root):
        print(f"✖ folder '{root}' already exists and is not empty")
        return 1
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    name = os.path.basename(os.path.abspath(root))
    with open(os.path.join(root, "main.lume"), "w", encoding="utf-8") as handle:
        handle.write(STARTER.format(name=name))
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(f"# {name}\n\nCreated with `lume new {name}`.\n\n"
                     f"```\nlume run main.lume\n```\n")
    print(f"✔ created project '{name}'")
    print(f"  cd {root} && lume run main.lume")
    return 0


def cmd_build(args) -> int:
    from .builders import build_app_bundle, build_model_bundle
    source = _read(args.file)
    outdir = args.out
    try:
        if args.target == "model":
            result = build_model_bundle(source, name=args.model, outdir=outdir)
            print(f"✔ model bundle ready: {result['path']}")
        else:
            result = build_app_bundle(source, name=args.model, outdir=outdir)
            print(f"✔ application bundle ready: {result['path']}")
            print(f"  archive: {result['archive']}")
    except LumeError as exc:
        print(f"✖ {exc}")
        return 1
    return 0


def cmd_examples(args) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(here, "examples")
    if not os.path.isdir(folder):
        print("no examples folder found")
        return 1
    for entry in sorted(os.listdir(folder)):
        if entry.endswith(".lume"):
            print(f"  {entry}")
    return 0


def cmd_version(args) -> int:
    print(f"Lume {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lume", description="Lume - a tiny language for AI models")
    parser.add_argument("--version", action="version", version=f"Lume {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a .lume file")
    run.add_argument("file")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="check the syntax of a .lume file")
    check.add_argument("file")
    check.set_defaults(func=cmd_check)

    repl = sub.add_parser("repl", help="interactive prompt")
    repl.set_defaults(func=cmd_repl)

    new = sub.add_parser("new", help="create a new Lume project")
    new.add_argument("name")
    new.set_defaults(func=cmd_new)

    build = sub.add_parser("build", help="build a model or a whole application")
    build.add_argument("target", choices=["model", "app"])
    build.add_argument("file")
    build.add_argument("-o", "--out", default="build", help="output folder (default: build)")
    build.add_argument("-m", "--model", default=None, help="which model to export")
    build.set_defaults(func=cmd_build)

    examples = sub.add_parser("examples", help="list bundled examples")
    examples.set_defaults(func=cmd_examples)

    version = sub.add_parser("version", help="show the version")
    version.set_defaults(func=cmd_version)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"✖ file not found: {exc.filename}")
        return 1
    except LumeError as exc:
        print(f"✖ {exc}")
        return 1
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
