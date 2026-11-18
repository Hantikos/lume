"""Errors raised by the Lume toolchain."""

from __future__ import annotations


class LumeError(Exception):
    """Base class for every error produced by Lume."""

    def __init__(self, message: str, line: int = 0, col: int = 0, source: str = ""):
        self.message = message
        self.line = line
        self.col = col
        self.source = source
        super().__init__(self.render())

    def render(self) -> str:
        if self.line:
            return f"Line {self.line}:{self.col}  {self.message}"
        return self.message


class LumeSyntaxError(LumeError):
    """Raised while tokenizing or parsing."""


class LumeRuntimeError(LumeError):
    """Raised while executing a program."""


class LumeTypeError(LumeRuntimeError):
    """Raised when a value of the wrong type is used."""


class LumeNameError(LumeRuntimeError):
    """Raised when an unknown name is referenced."""


def location(node, filename: str = "") -> str:
    line = getattr(node, "line", 0) or 0
    col = getattr(node, "col", 0) or 0
    if filename:
        return f"{filename}:{line}:{col}"
    return f"{line}:{col}"
