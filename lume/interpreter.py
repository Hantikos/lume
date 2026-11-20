"""Tree-walking interpreter for Lume."""

from __future__ import annotations

import math

from . import ast_nodes as A
from .errors import LumeError, LumeRuntimeError, LumeTypeError
from .nn import Dataset, Model, Tensor

#: built-ins that work on the model that is currently being defined
MODEL_AWARE_BUILTINS = {"train", "evaluate", "eval", "predict", "forward", "save", "load"}


class Scope:
    """Lexical scope with a parent link."""

    def __init__(self, parent: "Scope | None" = None):
        self.vars: dict = {}
        self.parent = parent

    def get(self, name: str):
        scope = self
        while scope is not None:
            if name in scope.vars:
                return scope.vars[name]
            scope = scope.parent
        raise LumeRuntimeError(f"unknown name '{name}'")

    def has(self, name: str) -> bool:
        scope = self
        while scope is not None:
            if name in scope.vars:
                return True
            scope = scope.parent
        return False

    def set(self, name: str, value):
        scope = self
        while scope is not None:
            if name in scope.vars:
                scope.vars[name] = value
                return
            scope = scope.parent
        self.vars[name] = value

    def define(self, name: str, value):
        self.vars[name] = value


class LumeFunction:
    def __init__(self, name, params, defaults, body, closure: Scope, interp=None):
        self.name = name
        self.params = params
        self.defaults = defaults
        self.body = body
        self.closure = closure
        self.interp = interp

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"<function {self.name}>"


class ReturnSignal(Exception):
    def __init__(self, value=None):
        self.value = value

class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


def truthy(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) > 0
    return True


def _quote(value) -> str:
    """Show strings with quotes so lists stay readable."""
    if isinstance(value, str):
        return f'"{value}"'
    return to_display(value)


def to_display(value) -> str:
    if isinstance(value, Tensor):
        rows = value.rows
        preview = value.to_list() if rows <= 6 else value.to_list()[:6]
        text = "[" + ", ".join(str(row) for row in preview) + "]"
        if rows > 6:
            text += f" ... ({rows}x{value.cols})"
        return text
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(_quote(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_quote(k)}: {_quote(v)}" for k, v in value.items()) + "}"
    if isinstance(value, Model):
        return f"<model {value.name}: {len(value.layers)} layers>"
    if isinstance(value, Dataset):
        return f"<dataset {value.name}: {len(value)} samples>"
    return str(value)


class Interpreter:
    def __init__(self, write=None, globals_: dict | None = None):
        self.write = write or (lambda text: None)
        self.global_scope = Scope()
        self.models: list[Model] = []
        self.current_model: Model | None = None
        self.output: list[str] = []
        self.call_depth = 0
        from .stdlib import install_builtins
        install_builtins(self.global_scope, self)
        if globals_:
            for key, value in globals_.items():
                self.global_scope.define(key, value)

    # -- helpers ----------------------------------------------------------
    def log(self, text: str):
        self.output.append(text)
        self.write(text + "\n")

    def visit(self, node, scope: Scope):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise LumeRuntimeError(f"cannot execute {type(node).__name__}")
        return method(node, scope)

    def execute_block(self, block: A.Block, scope: Scope):
        result = None
        for stmt in block.body:
            result = self.visit(stmt, scope)
        return result

    # -- statements -------------------------------------------------------
    def visit_Program(self, node, scope):
        result = None
        for stmt in node.body:
            result = self.visit(stmt, scope)
        return result

    def visit_ExprStmt(self, node, scope):
        return self.visit(node.expr, scope)

    def visit_LetStmt(self, node, scope):
        value = self.visit(node.value, scope)
        if node.const and scope.has(node.name) and node.name in scope.vars:
            raise LumeRuntimeError(f"'{node.name}' is a constant and cannot be reassigned")
        scope.define(node.name, value)
        return value

    def visit_DataStmt(self, node, scope):
        value = self.visit(node.value, scope)
        scope.define(node.name, value)
        return value

    def visit_Assign(self, node, scope):
        value = self.visit(node.value, scope)
        target = node.target
        if isinstance(target, A.Name):
            scope.set(target.name, value)
            return value
        if isinstance(target, A.Index):
            container = self.visit(target.obj, scope)
            index = self.visit(target.index, scope)
            if isinstance(container, list):
                idx = int(index)
                if idx < 0:
                    idx += len(container)
                if idx < 0 or idx >= len(container):
                    raise LumeRuntimeError(f"index {idx} is out of range (length {len(container)})")
                container[idx] = value
                return value
            if isinstance(container, dict):
                container[index] = value
                return value
            if isinstance(container, Tensor):
                idx = int(index)
                if idx < 0:
                    idx += container.rows
                if idx < 0 or idx >= container.rows:
                    raise LumeRuntimeError(f"index {idx} is out of range for a tensor with "
                                           f"{container.rows} rows")
                if container.cols == 1:
                    container.set(idx, 0, float(value))
                    return value
                row = list(value) if isinstance(value, (list, tuple)) else [value]
                if len(row) != container.cols:
                    raise LumeRuntimeError(f"expected {container.cols} values, got {len(row)}")
                for j, cell in enumerate(row):
                    container.set(idx, j, float(cell))
                return value
            raise LumeTypeError(f"cannot assign into {type(container).__name__}")
        raise LumeRuntimeError("invalid assignment target")

    def visit_IfStmt(self, node, scope):
        if truthy(self.visit(node.test, scope)):
            return self.execute_block(node.body, Scope(scope))
        if node.orelse is not None:
            return self.execute_block(node.orelse, Scope(scope))
        return None

    def visit_ForStmt(self, node, scope):
        iterable = self.visit(node.iterable, scope)
        if isinstance(iterable, Tensor):
            items = iterable.to_list()
        elif isinstance(iterable, dict):
            items = list(iterable.keys())
        elif isinstance(iterable, (list, tuple, range)):
            items = list(iterable)
        elif isinstance(iterable, str):
            items = list(iterable)
        else:
            raise LumeTypeError(f"cannot loop over {type(iterable).__name__}")
        for item in items:
            inner = Scope(scope)
            inner.define(node.var, item)
            try:
                self.execute_block(node.body, inner)
            except ContinueSignal:
                continue
            except BreakSignal:
                break
        return None

    def visit_WhileStmt(self, node, scope):
        guard = 0
        while truthy(self.visit(node.test, scope)):
            guard += 1
            if guard > 10_000_000:
                raise LumeRuntimeError("while loop did not terminate")
            inner = Scope(scope)
            try:
                self.execute_block(node.body, inner)
            except ContinueSignal:
                continue
            except BreakSignal:
                break
        return None

    def visit_ReturnStmt(self, node, scope):
        value = self.visit(node.value, scope) if node.value is not None else None
        raise ReturnSignal(value)

    def visit_BreakStmt(self, node, scope):
        raise BreakSignal()

    def visit_ContinueStmt(self, node, scope):
        raise ContinueSignal()

    def visit_FuncDef(self, node, scope):
        fn = LumeFunction(node.name, node.params, node.defaults, node.body, scope, self)
        scope.define(node.name, fn)
        return fn

    def visit_ImportStmt(self, node, scope):
        from .stdlib import get_module
        module = get_module(node.module)
        if module is None:
            raise LumeRuntimeError(f"module '{node.module}' not found "
                                   f"(available: {', '.join(sorted(get_module_names()))})")
        scope.define(node.alias or node.module, module)
        return module

    def visit_FromImportStmt(self, node, scope):
        from .stdlib import get_module
        module = get_module(node.module)
        if module is None:
            raise LumeRuntimeError(f"module '{node.module}' not found")
        for name, alias in node.names:
            if not hasattr(module, name):
                raise LumeRuntimeError(f"module '{node.module}' has no member '{name}'")
            scope.define(alias or name, getattr(module, name))
        return module

    def visit_ModelDef(self, node, scope):
        model = Model(node.name)
        self.models.append(model)
        inner = Scope(scope)
        previous = self.current_model
        self.current_model = model
        try:
            for stmt in node.body.body:
                self.visit(stmt, inner)
        finally:
            self.current_model = previous
        scope.define(node.name, model)
        self.log(f"model {model.name} ready ({len(model.layers)} layers)")
        return model

    def visit_LayerDecl(self, node, scope):
        if self.current_model is None:
            raise LumeRuntimeError("'layer' can only be used inside a model block")
        call = node.call
        func = call.func
        if not isinstance(func, A.Name):
            raise LumeRuntimeError("layer name must be a plain identifier, e.g. Dense")
        args = [self.visit(a, scope) for a in call.args]
        kwargs = {name: self.visit(value, scope) for name, value in call.kwargs}
        layer = self.make_layer(func.name, args, kwargs)
        self.current_model.add_layer(layer)
        return layer

    def make_layer(self, name: str, args, kwargs):
        from .nn import LAYER_TYPES
        key = name.lower()
        if key not in LAYER_TYPES:
            raise LumeRuntimeError(
                f"unknown layer '{name}'. Available layers: {', '.join(sorted(LAYER_TYPES))}")
        cls = LAYER_TYPES[key]
        if cls.__name__ == "Dense":
            units = args[0] if args else kwargs.get("units", kwargs.get("outputs", 1))
            activation = args[1] if len(args) > 1 else kwargs.get("activation", "none")
            return cls(int(units), str(activation))
        if cls.__name__ == "Activation":
            kind = args[0] if args else kwargs.get("activation", kwargs.get("kind", "relu"))
            return cls(str(kind))
        if cls.__name__ == "Dropout":
            rate = args[0] if args else kwargs.get("rate", 0.5)
            return cls(float(rate))
        if cls.__name__ == "Embedding":
            vocab = args[0] if args else kwargs.get("vocab_size", 1)
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 8)
            return cls(int(vocab), int(dim))
        if cls.__name__ == "Conv2d":
            filters = args[0] if args else kwargs.get("filters", 1)
            kernel = args[1] if len(args) > 1 else kwargs.get("kernel", 3)
            height = args[2] if len(args) > 2 else kwargs.get("height", 0)
            width = args[3] if len(args) > 3 else kwargs.get("width", 0)
            channels = args[4] if len(args) > 4 else kwargs.get("channels", 1)
            activation = args[5] if len(args) > 5 else kwargs.get("activation", "relu")
            stride = args[6] if len(args) > 6 else kwargs.get("stride", 1)
            return cls(int(filters), int(kernel), int(height), int(width), int(channels),
                       str(activation), int(stride))
        if cls.__name__ == "MaxPool":
            size = args[0] if args else kwargs.get("size", 2)
            stride = args[1] if len(args) > 1 else kwargs.get("stride", 2)
            return cls(int(size), int(stride))
        if cls.__name__ == "LSTM":
            units = args[0] if args else kwargs.get("units", 8)
            input_size = args[1] if len(args) > 1 else kwargs.get("input_size", 8)
            seq_len = args[2] if len(args) > 2 else kwargs.get("seq_len", 1)
            return cls(int(units), int(input_size), int(seq_len))
        if cls.__name__ == "MeanPool":
            seq_len = args[0] if args else kwargs.get("seq_len", 1)
            return cls(int(seq_len))
        return cls()

    # -- expressions ------------------------------------------------------
    def visit_NumberLit(self, node, scope):
        return node.value

    def visit_StringLit(self, node, scope):
        return node.value

    def visit_FStringLit(self, node, scope):
        out = []
        for part in node.parts:
            if isinstance(part, str):
                out.append(part)
            else:
                out.append(to_display(self.visit(part, scope)))
        return "".join(out)

    def visit_BoolLit(self, node, scope):
        return node.value

    def visit_NullLit(self, node, scope):
        return None

    def visit_ListLit(self, node, scope):
        return [self.visit(item, scope) for item in node.items]

    def visit_DictLit(self, node, scope):
        result = {}
        for key_node, value_node in node.pairs:
            key = self.visit(key_node, scope)
            if isinstance(key, (list, dict)):
                key = to_display(key)
            result[key] = self.visit(value_node, scope)
        return result

    def visit_Name(self, node, scope):
        try:
            return scope.get(node.name)
        except LumeRuntimeError as exc:
            raise LumeRuntimeError(f"unknown name '{node.name}'", node.line, node.col) from exc

    def visit_Unary(self, node, scope):
        value = self.visit(node.operand, scope)
        if node.op == "not":
            return not truthy(value)
        if node.op == "-":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LumeTypeError(f"cannot negate {type(value).__name__}")
            return -value
        if node.op == "+":
            return value
        raise LumeRuntimeError(f"unknown operator {node.op}")

    def visit_Logical(self, node, scope):
        left = self.visit(node.left, scope)
        if node.op == "and":
            return self.visit(node.right, scope) if truthy(left) else left
        return self.visit(node.right, scope) if not truthy(left) else left

    def visit_Compare(self, node, scope):
        left = self.visit(node.left, scope)
        right = self.visit(node.right, scope)
        op = node.op
        try:
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
        except TypeError:
            raise LumeTypeError(f"cannot compare {type(left).__name__} with {type(right).__name__}")
        raise LumeRuntimeError(f"unknown comparison {op}")

    def visit_Binary(self, node, scope):
        if node.op == "@":
            left = self.visit(node.left, scope)
            right = self.visit(node.right, scope)
            if not isinstance(left, Tensor) or not isinstance(right, Tensor):
                raise LumeTypeError("'@' (matmul) works on tensors only")
            from .nn import matmul
            return matmul(left, right)
        left = self.visit(node.left, scope)
        right = self.visit(node.right, scope)
        op = node.op
        if op == "+":
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, str) and isinstance(right, (int, float, bool)):
                return left + to_display(right)
            if isinstance(right, str) and isinstance(left, (int, float, bool)):
                return to_display(left) + right
            if isinstance(left, list) and isinstance(right, list):
                return left + right
            if isinstance(left, list):
                return left + [right]
            if isinstance(right, list):
                return [left] + right
            _numeric_check(left, right, "+")
            return left + right
        if op == "-":
            _numeric_check(left, right, "-")
            return left - right
        if op == "*":
            if isinstance(left, str) and isinstance(right, int):
                return left * right
            if isinstance(right, str) and isinstance(left, int):
                return right * left
            if isinstance(left, list) and isinstance(right, int):
                return left * right
            _numeric_check(left, right, "*")
            return left * right
        if op == "/":
            _numeric_check(left, right, "/")
            if right == 0:
                raise LumeRuntimeError("division by zero")
            return left / right
        if op == "%":
            _numeric_check(left, right, "%")
            if right == 0:
                raise LumeRuntimeError("division by zero")
            return left % right
        if op == "**":
            _numeric_check(left, right, "**")
            try:
                return left ** right
            except OverflowError:
                raise LumeRuntimeError("numeric overflow in '**'")
        raise LumeRuntimeError(f"unknown operator {op}")

    def visit_Index(self, node, scope):
        obj = self.visit(node.obj, scope)
        index = self.visit(node.index, scope)
        if isinstance(obj, Tensor):
            idx = int(index)
            if idx < 0:
                idx += obj.rows
            if idx < 0 or idx >= obj.rows:
                raise LumeRuntimeError(f"index {idx} is out of range for a tensor with {obj.rows} rows")
            row = obj.row(idx)
            if obj.cols == 1:
                return row[0]
            return list(row)
        if isinstance(obj, list):
            idx = int(index)
            if idx < 0:
                idx += len(obj)
            if idx < 0 or idx >= len(obj):
                raise LumeRuntimeError(f"index {idx} is out of range (length {len(obj)})")
            return obj[idx]
        if isinstance(obj, dict):
            if index not in obj:
                raise LumeRuntimeError(f"key {index!r} not found")
            return obj[index]
        if isinstance(obj, str):
            idx = int(index)
            if idx < 0:
                idx += len(obj)
            if idx < 0 or idx >= len(obj):
                raise LumeRuntimeError(f"index {idx} is out of range (length {len(obj)})")
            return obj[idx]
        raise LumeTypeError(f"cannot index {type(obj).__name__}")

    def visit_Attribute(self, node, scope):
        obj = self.visit(node.obj, scope)
        attr = node.attr
        if isinstance(obj, Model):
            if attr == "layers":
                return obj.layer_names()
            if attr == "metrics":
                return obj.metrics
            if attr == "name":
                return obj.name
            if attr == "history":
                return obj.history
            raise LumeRuntimeError(f"model has no attribute '{attr}'")
        if isinstance(obj, Dataset):
            if attr == "x":
                return obj.x
            if attr == "y":
                return obj.y
            if attr == "size":
                return len(obj)
            raise LumeRuntimeError(f"dataset has no attribute '{attr}'")
        if isinstance(obj, Tensor):
            if attr == "shape":
                return list(obj.shape)
            if attr == "rows":
                return obj.rows
            if attr == "cols":
                return obj.cols
            if attr == "data":
                return list(obj.data)
            raise LumeRuntimeError(f"tensor has no attribute '{attr}'")
        if isinstance(obj, dict):
            if attr in obj:
                return obj[attr]
            raise LumeRuntimeError(f"key '{attr}' not found")
        from .stdlib import Builtin, Module
        if isinstance(obj, Builtin):
            return getattr(obj, attr)
        if isinstance(obj, Module):
            return getattr(obj, attr)
        raise LumeTypeError(f"cannot read attribute '{attr}' of {type(obj).__name__}")

    def visit_Call(self, node, scope):
        func = self.visit(node.func, scope)
        args = [self.visit(a, scope) for a in node.args]
        kwargs = {name: self.visit(value, scope) for name, value in node.kwargs}
        return self.call(func, args, kwargs, node)

    def call(self, func, args, kwargs, node=None):
        from .stdlib import Builtin
        if isinstance(func, Builtin):
            if func.needs_interp:
                if (func.name in MODEL_AWARE_BUILTINS
                        and self.current_model is not None
                        and "model" not in kwargs
                        and not (args and isinstance(args[0], Model))):
                    args = [self.current_model] + list(args)
                return func.fn(self, *args, **kwargs)
            return func.fn(*args, **kwargs)
        if isinstance(func, LumeFunction):
            self.call_depth += 1
            if self.call_depth > 500:
                self.call_depth -= 1
                raise LumeRuntimeError("maximum call depth exceeded (recursion too deep)")
            inner = Scope(func.closure)
            for index, param in enumerate(func.params):
                if index < len(args):
                    inner.define(param, args[index])
                elif param in kwargs:
                    inner.define(param, kwargs[param])
                else:
                    default = func.defaults[index] if index < len(func.defaults) else None
                    if default is None:
                        self.call_depth -= 1
                        raise LumeRuntimeError(f"missing argument '{param}' for function '{func.name}'")
                    inner.define(param, self.visit(default, func.closure))
            try:
                self.execute_block(func.body, inner)
                return None
            except ReturnSignal as signal:
                return signal.value
            finally:
                self.call_depth -= 1
        if callable(func):
            return func(*args, **kwargs)
        raise LumeTypeError(f"{to_display(func)} is not callable")


def _numeric_check(left, right, op):
    if isinstance(left, bool) or isinstance(right, bool) or \
            not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise LumeTypeError(f"cannot apply '{op}' to {type(left).__name__} and {type(right).__name__}")


def get_module_names():
    from .stdlib import MODULE_NAMES
    return MODULE_NAMES


def run_program(program: A.Program, write=None, globals_: dict | None = None) -> Interpreter:
    interp = Interpreter(write=write, globals_=globals_)
    interp.visit(program, interp.global_scope)
    return interp
