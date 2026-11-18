"""AST node definitions for Lume."""

from __future__ import annotations


class Node:
    """Base class for all AST nodes."""

    def __init__(self, line: int = 0, col: int = 0):
        self.line = line
        self.col = col

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        fields = ", ".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
            if k not in ("line", "col")
        )
        return f"{type(self).__name__}({fields})"


# -- literals ---------------------------------------------------------------
class NumberLit(Node):
    def __init__(self, value, line=0, col=0):
        super().__init__(line, col)
        self.value = value


class StringLit(Node):
    def __init__(self, value, line=0, col=0):
        super().__init__(line, col)
        self.value = value


class FStringLit(Node):
    """Interpolated string: parts is a list of str | Node."""

    def __init__(self, parts, line=0, col=0):
        super().__init__(line, col)
        self.parts = parts


class BoolLit(Node):
    def __init__(self, value, line=0, col=0):
        super().__init__(line, col)
        self.value = value


class NullLit(Node):
    pass


class ListLit(Node):
    def __init__(self, items, line=0, col=0):
        super().__init__(line, col)
        self.items = items


class DictLit(Node):
    def __init__(self, pairs, line=0, col=0):
        super().__init__(line, col)
        self.pairs = pairs


# -- expressions ------------------------------------------------------------
class Name(Node):
    def __init__(self, name, line=0, col=0):
        super().__init__(line, col)
        self.name = name


class Unary(Node):
    def __init__(self, op, operand, line=0, col=0):
        super().__init__(line, col)
        self.op = op
        self.operand = operand


class Binary(Node):
    def __init__(self, op, left, right, line=0, col=0):
        super().__init__(line, col)
        self.op = op
        self.left = left
        self.right = right


class Logical(Node):
    def __init__(self, op, left, right, line=0, col=0):
        super().__init__(line, col)
        self.op = op
        self.left = left
        self.right = right


class Compare(Node):
    def __init__(self, op, left, right, line=0, col=0):
        super().__init__(line, col)
        self.op = op
        self.left = left
        self.right = right


class Call(Node):
    def __init__(self, func, args, kwargs, line=0, col=0):
        super().__init__(line, col)
        self.func = func
        self.args = args
        self.kwargs = kwargs  # list of (name, node)


class Index(Node):
    def __init__(self, obj, index, line=0, col=0):
        super().__init__(line, col)
        self.obj = obj
        self.index = index


class Attribute(Node):
    def __init__(self, obj, attr, line=0, col=0):
        super().__init__(line, col)
        self.obj = obj
        self.attr = attr


# -- statements -------------------------------------------------------------
class Block(Node):
    def __init__(self, body, line=0, col=0):
        super().__init__(line, col)
        self.body = body


class ExprStmt(Node):
    def __init__(self, expr, line=0, col=0):
        super().__init__(line, col)
        self.expr = expr


class Assign(Node):
    def __init__(self, target, value, line=0, col=0):
        super().__init__(line, col)
        self.target = target  # Name
        self.value = value


class LetStmt(Node):
    def __init__(self, name, value, const=False, type_hint=None, line=0, col=0):
        super().__init__(line, col)
        self.name = name
        self.value = value
        self.const = const
        self.type_hint = type_hint


class DataStmt(Node):
    def __init__(self, name, value, line=0, col=0):
        super().__init__(line, col)
        self.name = name
        self.value = value


class IfStmt(Node):
    def __init__(self, test, body, orelse=None, line=0, col=0):
        super().__init__(line, col)
        self.test = test
        self.body = body
        self.orelse = orelse


class ForStmt(Node):
    def __init__(self, var, iterable, body, line=0, col=0):
        super().__init__(line, col)
        self.var = var
        self.iterable = iterable
        self.body = body


class WhileStmt(Node):
    def __init__(self, test, body, line=0, col=0):
        super().__init__(line, col)
        self.test = test
        self.body = body


class ReturnStmt(Node):
    def __init__(self, value=None, line=0, col=0):
        super().__init__(line, col)
        self.value = value


class BreakStmt(Node):
    pass


class ContinueStmt(Node):
    pass


class FuncDef(Node):
    def __init__(self, name, params, defaults, body, line=0, col=0):
        super().__init__(line, col)
        self.name = name
        self.params = params          # list[str]
        self.defaults = defaults        # list[node | None]
        self.body = body


class ModelDef(Node):
    def __init__(self, name, body, line=0, col=0):
        super().__init__(line, col)
        self.name = name
        self.body = body


class LayerDecl(Node):
    def __init__(self, call, line=0, col=0):
        super().__init__(line, col)
        self.call = call


class ImportStmt(Node):
    def __init__(self, module, alias=None, line=0, col=0):
        super().__init__(line, col)
        self.module = module
        self.alias = alias


class FromImportStmt(Node):
    def __init__(self, module, names, line=0, col=0):
        super().__init__(line, col)
        self.module = module
        self.names = names  # list[(name, alias|None)]


class Program(Node):
    def __init__(self, body, line=0, col=0):
        super().__init__(line, col)
        self.body = body
