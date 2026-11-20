"""Built-in functions and tiny modules available to every Lume program."""

from __future__ import annotations

import json
import math
import os
import random

from .errors import LumeRuntimeError, LumeTypeError
from .nn import (Dataset, Dense, Model, Tensor, classification_metrics, softmax,
                 train_model)


class Builtin:
    """A Python function exposed to Lume code."""

    def __init__(self, fn, name: str, needs_interp: bool = False, doc: str = ""):
        self.fn = fn
        self.name = name
        self.needs_interp = needs_interp
        self.doc = doc

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __getattr__(self, item):
        # namespaced built-ins, e.g. dataset.xor or tensor.zeros
        members = NAMESPACES.get(self.name, {})
        if item in members:
            return members[item]
        raise AttributeError(f"builtin '{self.name}' has no member '{item}'")

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"<builtin {self.name}>"


#: name -> Builtin, and "namespace.member" -> Builtin
BUILTINS: dict = {}
NAMESPACES: dict = {}


class Module:
    """A tiny namespace object that can be imported."""

    def __init__(self, name: str, members: dict):
        self._name = name
        self._members = members

    def __getattr__(self, item):
        try:
            return self._members[item]
        except KeyError:
            raise LumeRuntimeError(f"module '{self._name}' has no member '{item}'") from None

    def __dir__(self):
        return sorted(self._members)

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"<module {self._name}>"


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LumeTypeError(f"expected a number, got {type(value).__name__}")
    return value


def _as_list(value):
    if isinstance(value, Tensor):
        return list(value.data)
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return list(value)
    raise LumeTypeError(f"expected a list, got {type(value).__name__}")


def _numbers(value):
    return [_num(v) for v in _as_list(value)]


# --------------------------------------------------------------------------
# conversion helpers
# --------------------------------------------------------------------------
def to_dataset(value, name: str = "dataset") -> Dataset:
    """Accept a Dataset, a dict with x/y or a list of [features, label] pairs."""
    if isinstance(value, Dataset):
        return value
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            x = value["x"]
            y = value["y"]
            if not isinstance(x, Tensor):
                x = Tensor.from_nested(x)
            return Dataset(x, y, name)
        raise LumeTypeError("dataset dict must contain 'x' and 'y'")
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)) and len(value[0]) == 2:
        xs = [row[0] for row in value]
        ys = [row[1] for row in value]
        return Dataset(Tensor.from_nested(xs), ys, name)
    raise LumeTypeError("expected a dataset value (use dataset(x, y))")


def _fmt_table(rows, header=None):
    widths = []
    rows = [list(map(str, row)) for row in rows]
    if header:
        rows = [list(map(str, header))] + rows
    for row in rows:
        for index, cell in enumerate(row):
            if index >= len(widths):
                widths.append(len(cell))
            else:
                widths[index] = max(widths[index], len(cell))
    lines = []
    for row in rows:
        lines.append("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return "\n".join(lines)


# --------------------------------------------------------------------------
# built-ins
# --------------------------------------------------------------------------
def _build_builtins() -> dict:
    b = {}

    def add(name, fn, needs_interp=False):
        builtin = Builtin(fn, name, needs_interp)
        b[name] = builtin
        if "." in name:
            namespace, member = name.split(".", 1)
            NAMESPACES.setdefault(namespace, {})[member] = builtin
        return builtin

    # -- output -----------------------------------------------------------
    def _print(interp, *args, sep=" "):
        interp.write(sep.join(_display(a) for a in args) + "\n")
    add("print", _print, True)

    def _println(interp, *args, sep=" "):
        interp.write(sep.join(_display(a) for a in args) + "\n")
    add("println", _println, True)

    # -- basics -----------------------------------------------------------
    add("type", lambda x: _type_name(x))
    add("len", lambda x: len(_as_list(x)) if not isinstance(x, (str, dict)) else len(x))
    add("str", lambda x: _display(x))
    add("int", lambda x: int(float(x)))
    add("float", lambda x: float(x))
    add("bool", lambda x: bool(x))
    add("list", lambda x=None: [] if x is None else _as_list(x))
    add("dict", lambda x=None: dict(x) if isinstance(x, dict) else {})
    add("range", lambda *args: list(range(*[int(a) for a in args])) if args else [])
    add("sum", lambda xs: sum(_numbers(xs)))
    add("min", lambda *args: min(args[0]) if len(args) == 1 else min(args))
    add("max", lambda *args: max(args[0]) if len(args) == 1 else max(args))
    add("abs", lambda x: abs(_num(x)))
    add("round", lambda x, n=0: round(_num(x), int(n)))
    add("floor", lambda x: math.floor(_num(x)))
    add("ceil", lambda x: math.ceil(_num(x)))
    add("sqrt", lambda x: math.sqrt(_num(x)))
    add("exp", lambda x: math.exp(_num(x)))
    add("log", lambda x: math.log(_num(x)))
    add("log2", lambda x: math.log2(_num(x)))
    add("sin", lambda x: math.sin(_num(x)))
    add("cos", lambda x: math.cos(_num(x)))
    add("tan", lambda x: math.tan(_num(x)))
    add("pow", lambda x, y: _num(x) ** _num(y))
    add("mean", lambda xs: sum(_numbers(xs)) / max(1, len(_as_list(xs))))
    add("median", lambda xs: _median(_numbers(xs)))
    add("argmax", lambda xs: max(range(len(xs)), key=lambda i: xs[i]) if len(xs) else -1)
    add("softmax", lambda xs: softmax(_numbers(xs)))
    add("sigmoid", lambda x: _sigmoid_scalar(_num(x)) if not isinstance(x, list) else
        [_sigmoid_scalar(v) for v in _numbers(x)])
    add("relu", lambda xs: [max(0.0, _num(v)) for v in _numbers(xs)])
    add("seed", lambda n=0: random.seed(int(n)))
    add("shuffle", lambda xs: random.sample(_as_list(xs), len(_as_list(xs))))
    add("sort", lambda xs: sorted(_as_list(xs)))
    add("unique", lambda xs: list(dict.fromkeys(_as_list(xs))))
    add("count", lambda xs, value: sum(1 for v in _as_list(xs) if v == value))
    add("keys", lambda d: list(d.keys()) if isinstance(d, dict) else [])
    add("values", lambda d: list(d.values()) if isinstance(d, dict) else [])
    add("upper", lambda s: str(s).upper())
    add("lower", lambda s: str(s).lower())
    add("strip", lambda s: str(s).strip())
    add("split", lambda s, sep=" ": str(s).split(sep))
    add("join", lambda xs, sep=", ": sep.join(str(v) for v in _as_list(xs)))
    add("replace", lambda s, a, b: str(s).replace(a, b))
    add("startswith", lambda s, p: str(s).startswith(p))
    add("endswith", lambda s, p: str(s).endswith(p))
    add("contains", lambda s, p: p in str(s))

    def _input(interp, prompt=""):
        interp.write(str(prompt))
        try:
            return input()
        except EOFError:
            return ""
    add("input", _input, True)

    # -- tensors ----------------------------------------------------------
    add("tensor", lambda values: Tensor.from_nested(values))
    add("zeros", lambda rows, cols=1: Tensor.zeros(int(rows), int(cols)))
    add("ones", lambda rows, cols=1: Tensor.ones(int(rows), int(cols)))
    add("randn", lambda rows, cols=1, seed=0: Tensor.randn(int(rows), int(cols), random.Random(int(seed))))
    add("eye", lambda n: Tensor(int(n), int(n), [1.0 if i == j else 0.0
                                                 for i in range(int(n)) for j in range(int(n))]))
    add("transpose", lambda t: _transpose(t))
    add("tensor.zeros", lambda rows, cols=1: Tensor.zeros(int(rows), int(cols)))
    add("tensor.ones", lambda rows, cols=1: Tensor.ones(int(rows), int(cols)))
    add("tensor.randn", lambda rows, cols=1, seed=0: Tensor.randn(int(rows), int(cols),
                                                                  random.Random(int(seed))))
    add("tensor.eye", lambda n: Tensor(int(n), int(n), [1.0 if i == j else 0.0
                                                        for i in range(int(n)) for j in range(int(n))]))

    # -- datasets ---------------------------------------------------------
    add("dataset", lambda x, y, name="dataset", group=1: Dataset(
        x if isinstance(x, Tensor) else Tensor.from_nested(x), y, name, group))

    def _xor():
        x = Tensor.from_nested([[0, 0], [0, 1], [1, 0], [1, 1]])
        return Dataset(x, [0, 1, 1, 0], "xor")

    def _linear(count=50, noise=0.0, seed=1):
        rng = random.Random(seed)
        rows, labels = [], []
        for _ in range(int(count)):
            x = rng.uniform(-1, 1)
            y = 3.0 * x + 2.0 + rng.uniform(-noise, noise)
            rows.append([x])
            labels.append(y)
        return Dataset(Tensor.from_nested(rows), labels, "linear")

    def _blobs(classes=3, per_class=40, seed=3):
        rng = random.Random(seed)
        rows, labels = [], []
        for c in range(int(classes)):
            cx, cy = rng.uniform(-3, 3), rng.uniform(-3, 3)
            for _ in range(int(per_class)):
                rows.append([cx + rng.gauss(0, 0.8), cy + rng.gauss(0, 0.8)])
                labels.append(c)
        return Dataset(Tensor.from_nested(rows), labels, "blobs")

    def _spiral(count=200, noise=0.2, seed=5):
        rng = random.Random(seed)
        rows, labels = [], []
        for c in range(2):
            for i in range(int(count) // 2):
                r = i / max(1, (count // 2))
                angle = r * 3.2 + c * math.pi
                rows.append([r * math.cos(angle) + rng.gauss(0, noise),
                             r * math.sin(angle) + rng.gauss(0, noise)])
                labels.append(c)
        return Dataset(Tensor.from_nested(rows), labels, "spiral")

    def _bars(count=40, size=8, noise=0.15, seed=11):
        """Tiny images with a vertical bar (class 0) or a horizontal bar (class 1)."""
        rng = random.Random(seed)
        size = int(size)
        rows, labels = [], []
        for index in range(int(count)):
            image = [0.0] * (size * size)
            vertical = index % 2 == 0
            column = rng.randrange(size)
            line = rng.randrange(size)
            for step in range(size):
                position = (line * size + step) if vertical else (step * size + column)
                image[position] = 1.0
            image = [max(0.0, min(1.0, value + rng.uniform(-noise, noise))) for value in image]
            rows.append(image)
            labels.append(0 if vertical else 1)
        return Dataset(Tensor.from_nested(rows), labels, "bars")

    add("dataset.xor", _xor)
    add("dataset.linear", _linear)
    add("dataset.blobs", _blobs)
    add("dataset.spiral", _spiral)
    add("dataset.bars", _bars)

    def _csv(path, delimiter=","):
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                parts = [p.strip() for p in raw.split(delimiter)]
                converted = []
                for part in parts:
                    try:
                        converted.append(float(part))
                    except ValueError:
                        converted.append(part)
                rows.append(converted)
        return rows

    add("read_csv", _csv)

    def _write_csv(path, rows):
        with open(path, "w", encoding="utf-8") as handle:
            for row in _as_list(rows):
                handle.write(",".join(str(v) for v in row) + "\n")
        return path
    add("write_csv", _write_csv)

    # -- model training ---------------------------------------------------
    def _train(interp, model, data, epochs=10, lr=0.01, optimizer="sgd", loss="mse",
               batch_size=16, verbose=True):
        if not isinstance(model, Model):
            raise LumeTypeError("train() expects a model as the first argument")
        dataset = to_dataset(data, "train-set")
        history = train_model(
            model, dataset,
            epochs=int(epochs), lr=float(lr), optimizer=str(optimizer), loss=str(loss),
            batch_size=int(batch_size), verbose=bool(verbose),
            emit=lambda text: interp.write(text + "\n"),
        )
        return history

    add("train", _train, True)

    def _evaluate(interp, model, data):
        if not isinstance(model, Model):
            raise LumeTypeError("evaluate() expects a model as the first argument")
        dataset = to_dataset(data, "test-set")
        result = model.evaluate(dataset)
        interp.write(_format_metrics(model, dataset, result) + "\n")
        return result

    add("evaluate", _evaluate, True)
    add("eval", _evaluate, True)

    def _predict(interp, model, x):
        if not isinstance(model, Model):
            raise LumeTypeError("predict() expects a model as the first argument")
        tensor = x if isinstance(x, Tensor) else Tensor.from_nested(x)
        return model.predict(tensor)

    add("predict", _predict, True)

    def _forward(interp, model, x):
        """Raw network output (useful for regression)."""
        if not isinstance(model, Model):
            raise LumeTypeError("forward() expects a model as the first argument")
        tensor = x if isinstance(x, Tensor) else Tensor.from_nested(x)
        out = model.forward(tensor)
        if out.cols == 1:
            return list(out.data)
        return out.to_list()

    add("forward", _forward, True)

    def _save(interp, model, path):
        if not isinstance(model, Model):
            raise LumeTypeError("save() can only be used inside a model block")
        return model.save(str(path))

    add("save", _save, True)

    def _load(interp, model, path):
        if not isinstance(model, Model):
            raise LumeTypeError("load() can only be used inside a model block")
        loaded = Model.load(str(path))
        model.layers = list(loaded.layers)
        model.metrics = dict(loaded.metrics)
        return model

    add("load", _load, True)

    def _predict_proba(model, x):
        tensor = x if isinstance(x, Tensor) else Tensor.from_nested(x)
        return model.predict_proba(tensor)

    add("predict_proba", _predict_proba)

    def _save_model(model, path):
        if not isinstance(model, Model):
            raise LumeTypeError("save_model() expects a model")
        return model.save(str(path))

    add("save_model", _save_model)

    def _load_model(path):
        return Model.load(str(path))

    add("load_model", _load_model)

    def _confusion(preds, truth):
        return classification_metrics(preds, truth)["confusion_matrix"]

    add("confusion_matrix", _confusion)

    return b


def _transpose(tensor: Tensor) -> Tensor:
    from .nn import transpose
    return transpose(tensor)


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, Tensor):
        return "tensor"
    if isinstance(value, Model):
        return "model"
    if isinstance(value, Dataset):
        return "dataset"
    return type(value).__name__


def _display(value) -> str:
    from .interpreter import to_display
    return to_display(value)


def _format_metrics(model: Model, dataset: Dataset, result: dict) -> str:
    lines = [
        f"evaluation for {model.name}:",
        f"  samples     {result['total']}",
        f"  accuracy    {result['accuracy']:.3f}",
        f"  precision   {result.get('precision', 0):.3f}",
        f"  recall      {result.get('recall', 0):.3f}",
        f"  f1          {result.get('f1', 0):.3f}",
    ]
    matrix = result.get("confusion_matrix")
    if matrix:
        lines.append("  confusion matrix (rows = actual, cols = predicted):")
        lines.append(_fmt_table(matrix))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------
def _make_modules() -> dict:
    math_module = Module("math", {
        "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log2": math.log2,
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor,
        "ceil": math.ceil, "pow": pow, "fabs": math.fabs, "pi": math.pi, "tau": math.tau,
        "e": math.e,
    })
    json_module = Module("json", {
        "dumps": lambda obj: json.dumps(obj, ensure_ascii=False),
        "loads": lambda text: json.loads(text),
    })
    lume_module = Module("lume", {
        "tensor": Builtin(lambda values: Tensor.from_nested(values), "tensor"),
        "dataset": Builtin(lambda x, y, name="dataset": Dataset(
            x if isinstance(x, Tensor) else Tensor.from_nested(x), y, name), "dataset"),
        "train": Builtin(lambda model, data, **kw: train_model(
            model, to_dataset(data), **{k: v for k, v in kw.items()
                                        if k in {"epochs", "lr", "optimizer", "loss", "batch_size"}}),
            "train"),
        "predict": Builtin(lambda model, x: model.predict(
            x if isinstance(x, Tensor) else Tensor.from_nested(x)), "predict"),
    })
    return {"math": math_module, "json": json_module, "lume": lume_module}


MODULES = _make_modules()
MODULE_NAMES = sorted(MODULES)


def get_module(name: str):
    return MODULES.get(name)


_BUILTIN_CACHE: dict | None = None


def install_builtins(scope, interp):
    global _BUILTIN_CACHE
    if _BUILTIN_CACHE is None:
        _BUILTIN_CACHE = _build_builtins()
    for name, builtin in _BUILTIN_CACHE.items():
        if "." not in name:
            scope.define(name, builtin)
