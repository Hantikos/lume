"""Tests for the Lume language, the builders and the studio API.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lume.builders import build_app_bundle, build_model_bundle  # noqa: E402
from lume.errors import LumeError  # noqa: E402
from lume.interpreter import Interpreter  # noqa: E402
from lume.parser import parse  # noqa: E402


def run(source: str):
    """Execute Lume source and return (output lines, interpreter)."""
    lines: list[str] = []
    interp = Interpreter(write=lambda text: lines.append(text))
    interp.visit(parse(source, "<test>"), interp.global_scope)
    return lines, interp


class TestLexerAndParser(unittest.TestCase):
    def test_syntax_error_reports_line(self):
        with self.assertRaises(LumeError) as ctx:
            parse("let x = \nfn (", "<test>")
        self.assertIn("Line", ctx.exception.render())

    def test_unterminated_string(self):
        with self.assertRaises(LumeError):
            parse('print "hello', "<test>")

    def test_missing_brace(self):
        with self.assertRaises(LumeError):
            parse("fn a() { print 1 ", "<test>")

    def test_fstring(self):
        lines, _ = run('print f"a = {1 + 2}"')
        self.assertEqual("a = 3", lines[0].strip())

    def test_semicolons_are_optional(self):
        lines, _ = run('let a = 1; let b = 2; print a + b')
        self.assertEqual("3", lines[0].strip())


class TestLanguage(unittest.TestCase):
    def test_variables_and_types(self):
        lines, _ = run("""
let n = 7
let f = 1.5
let s = "text"
let b = true
let n2 = null
print type(n), type(f), type(s), type(b), type(n2)
""")
        self.assertEqual("int float str bool null", lines[0].strip())

    def test_functions_and_recursion(self):
        lines, _ = run("""
fn fact(n) {
    if n <= 1 {
        return 1
    }
    return n * fact(n - 1)
}
print fact(6)
""")
        self.assertEqual("720", lines[0].strip())

    def test_default_arguments(self):
        lines, _ = run("""
fn greet(name, greeting = "hi") {
    return greeting + " " + name
}
print greet("world")
print greet("world", "yo")
""")
        self.assertEqual("hi world", lines[0].strip())
        self.assertEqual("yo world", lines[1].strip())

    def test_lists_and_index_assignment(self):
        lines, _ = run("""
let xs = [1, 2, 3]
xs[1] = 20
print xs
print len(xs)
""")
        self.assertEqual("[1, 20, 3]", lines[0].strip())
        self.assertEqual("3", lines[1].strip())

    def test_dictionaries(self):
        lines, _ = run("""
let d = {"a": 1, "b": 2}
print d["a"]
d["c"] = 3
print keys(d)
""")
        self.assertEqual("1", lines[0].strip())
        self.assertEqual("['a', 'b', 'c']".replace("'", '"'), lines[1].strip())

    def test_loops(self):
        lines, _ = run("""
let total = 0
for i in range(1, 5) {
    total = total + i
}
print total

let j = 0
while j < 3 {
    j = j + 1
}
print j
""")
        self.assertEqual("10", lines[0].strip())
        self.assertEqual("3", lines[1].strip())

    def test_break_and_continue(self):
        lines, _ = run("""
for i in range(10) {
    if i == 5 { break }
    if i % 2 == 0 { continue }
    print i
}
""")
        self.assertEqual(["1", "3"], [line.strip() for line in lines])

    def test_string_plus_number(self):
        lines, _ = run('print "value: " + 42')
        self.assertEqual("value: 42", lines[0].strip())

    def test_tensors(self):
        lines, _ = run("""
let t = tensor([[1, 2], [3, 4]])
print t
print t.shape
print t @ tensor([[1], [1]])
""")
        self.assertEqual("[[1.0, 2.0], [3.0, 4.0]]", lines[0].strip())
        self.assertEqual("[2, 2]", lines[1].strip())
        self.assertEqual("[[3.0], [7.0]]", lines[2].strip())

    def test_import_module(self):
        lines, _ = run("""
import math
print math.sqrt(16)
""")
        self.assertEqual("4", lines[0].strip())

    def test_unknown_name_raises(self):
        with self.assertRaises(LumeError):
            run("print not_defined_anywhere")


class TestModels(unittest.TestCase):
    def test_xor_is_learned(self):
        lines, interp = run("""
data points = dataset.xor()

model Net {
    layer Dense(16, "relu")
    layer Dense(1, "sigmoid")
    train(data = points, epochs = 150, lr = 0.2, optimizer = "adam", loss = "binary_crossentropy")
}
print predict(Net, [[0, 0]])
print predict(Net, [[0, 1]])
print predict(Net, [[1, 0]])
print predict(Net, [[1, 1]])
""")
        self.assertEqual(["[0]", "[1]", "[1]", "[0]"], [line.strip() for line in lines[-4:]])
        self.assertTrue(interp.models)
        self.assertEqual(2, len(interp.models[0].layers))

    def test_metrics_are_reported(self):
        lines, _ = run("""
data points = dataset.xor()
model Net {
    layer Dense(16, "relu")
    layer Dense(1, "sigmoid")
    train(data = points, epochs = 60, lr = 0.2, optimizer = "adam", loss = "binary_crossentropy")
    eval(data = points)
}
""")
        text = "\n".join(lines)
        self.assertIn("accuracy", text)
        self.assertIn("confusion matrix", text)

    def test_model_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.json")
            lines, _ = run(f"""
data points = dataset.xor()
model Net {{
    layer Dense(4, "relu")
    layer Dense(1, "sigmoid")
    train(data = points, epochs = 10, lr = 0.2, optimizer = "adam", loss = "binary_crossentropy")
    save("{path}")
}}
""")
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual("Net", data["name"])
            self.assertEqual(2, len(data["layers"]))

    def test_regression_learns_line(self):
        lines, _ = run("""
data points = dataset.linear(count = 40, noise = 0.05)
model Line {
    layer Dense(16, "relu")
    layer Dense(1, "none")
    train(data = points, epochs = 150, lr = 0.02, optimizer = "adam", loss = "mse")
}
print forward(Line, [[1.0]])
""")
        value = float(lines[-1].strip().strip("[]"))
        self.assertAlmostEqual(5.0, value, delta=1.0)  # y = 3x + 2


class TestBuilders(unittest.TestCase):
    SOURCE = """
data points = dataset.xor()
model Net {
    layer Dense(16, "relu")
    layer Dense(1, "sigmoid")
    train(data = points, epochs = 30, lr = 0.2, optimizer = "adam", loss = "binary_crossentropy")
}
"""

    def test_build_model_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            result = build_model_bundle(self.SOURCE, outdir=folder)
            self.assertTrue(result["ok"])
            self.assertTrue(os.path.exists(os.path.join(result["path"], "model.json")))
            self.assertTrue(os.path.exists(os.path.join(result["path"], "model.py")))
            self.assertTrue(os.path.exists(os.path.join(result["path"], "README.md")))

    def test_build_app_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            result = build_app_bundle(self.SOURCE, outdir=folder)
            self.assertTrue(result["ok"])
            for name in ("app.py", "index.html", "model.json", "run.bat", "run.sh"):
                self.assertIn(name, result["files"])
            self.assertTrue(os.path.exists(result["archive"]))

    def test_build_without_model_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(LumeError):
                build_model_bundle("print 1", outdir=folder)


class TestStudio(unittest.TestCase):
    def test_api_roundtrip(self):
        import urllib.request

        from lumeide.server import serve

        with tempfile.TemporaryDirectory() as folder:
            server = serve(workspace=folder)
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=10) as response:
                    self.assertTrue(json.load(response)["ok"])
                with urllib.request.urlopen(base + "/", timeout=10) as response:
                    self.assertIn(b"Lume Studio", response.read())
                payload = json.dumps({"code": 'print "hi"'}).encode()
                request = urllib.request.Request(base + "/api/run", data=payload,
                                               headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.load(response)
                self.assertTrue(data["ok"])
                self.assertEqual("hi\n", data["output"][0])
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
