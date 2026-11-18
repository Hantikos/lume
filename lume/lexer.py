"""Tokenizer for the Lume language.

Lume uses curly-brace blocks (``{ ... }``) and optional semicolons, so the
lexer only needs to know where a statement ends (newlines) and where brackets
are open (inside brackets newlines are ignored).
"""

from __future__ import annotations

from .errors import LumeSyntaxError

KEYWORDS = {
    "fn", "let", "const", "if", "else", "for", "in", "while", "return",
    "break", "continue", "import", "from", "as", "true", "false", "null",
    "and", "or", "not",
}

OPERATORS = [
    "**", "==", "!=", "<=", ">=", "->", "+=", "-=", "*=", "/=",
    "+", "-", "*", "/", "%", "=", "<", ">", "(", ")", "{", "}", "[", "]",
    ",", ":", ".", "@", "!", ";",
]


class Token:
    __slots__ = ("kind", "value", "line", "col")

    def __init__(self, kind: str, value, line: int, col: int):
        self.kind = kind
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Token({self.kind}, {self.value!r}, {self.line}:{self.col})"


class Lexer:
    def __init__(self, source: str, filename: str = "<script>"):
        self.src = source
        self.filename = filename
        self.tokens: list[Token] = []
        self.depth = 0  # bracket nesting depth

    # -- helpers ---------------------------------------------------------
    def error(self, message: str, line: int, col: int):
        raise LumeSyntaxError(message, line, col, self.filename)

    def tokenize(self) -> list[Token]:
        lines = self.src.splitlines()
        for number, raw in enumerate(lines, start=1):
            self._tokenize_line(raw, number)
            if self.depth == 0:
                self.tokens.append(Token("NEWLINE", "\n", number, len(raw) + 1))
        self.tokens.append(Token("EOF", None, (len(lines) or 1) + 1, 1))
        return self.tokens

    # -- main tokenizer --------------------------------------------------
    def _tokenize_line(self, raw: str, number: int):
        expanded = raw.expandtabs(4)
        text = expanded.lstrip(" ")
        col = len(expanded) - len(text) + 1
        i = 0
        length = len(text)
        while i < length:
            ch = text[i]
            if ch in " \t":
                i += 1
                continue
            if ch == "#":
                break  # comment to end of line
            if ch in "\"'":
                value, i = self._read_string(text, i, number, col + i)
                self.tokens.append(Token("STRING", value, number, col + i))
                continue
            if ch.isdigit() or (ch == "." and i + 1 < length and text[i + 1].isdigit()):
                value, i = self._read_number(text, i, number)
                self.tokens.append(Token("NUMBER", value, number, col + i))
                continue
            if ch.isalpha() or ch == "_":
                start = i
                while i < length and (text[i].isalnum() or text[i] == "_"):
                    i += 1
                word = text[start:i]
                if word == "f" and i < length and text[i] in "\"'":
                    value, i = self._read_string(text, i, number, col + i)
                    self.tokens.append(Token("FSTRING", value, number, col + i))
                    continue
                kind = "KEYWORD" if word in KEYWORDS else "ID"
                self.tokens.append(Token(kind, word, number, col + start))
                continue
            matched = None
            for op in OPERATORS:
                if text.startswith(op, i):
                    matched = op
                    break
            if matched is None:
                self.error(f"unexpected character {ch!r}", number, col + i)
            if matched in "([":
                self.depth += 1
            elif matched in ")]":
                self.depth = max(0, self.depth - 1)
            self.tokens.append(Token("OP", matched, number, col + i))
            i += len(matched)

    def _read_string(self, text: str, i: int, number: int, col: int):
        quote = text[i]
        i += 1
        out = []
        escapes = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0"}
        while True:
            if i >= len(text):
                self.error("unterminated string literal", number, col)
            ch = text[i]
            if ch == "\\":
                i += 1
                if i >= len(text):
                    self.error("unterminated escape sequence", number, col)
                esc = text[i]
                if esc == "u":
                    hexa = text[i + 1:i + 5]
                    try:
                        out.append(chr(int(hexa, 16)))
                    except ValueError:
                        self.error("invalid \\u escape", number, col)
                    i += 5
                    continue
                out.append(escapes.get(esc, esc))
                i += 1
                continue
            if ch == quote:
                i += 1
                return "".join(out), i
            out.append(ch)
            i += 1

    def _read_number(self, text: str, i: int, number: int):
        start = i
        length = len(text)
        while i < length and (text[i].isdigit() or text[i] == "_"):
            i += 1
        is_float = False
        if i < length and text[i] == "." and i + 1 < length and text[i + 1].isdigit():
            is_float = True
            i += 1
            while i < length and (text[i].isdigit() or text[i] == "_"):
                i += 1
        if i < length and text[i] in "eE":
            j = i + 1
            if j < length and text[j] in "+-":
                j += 1
            if j < length and text[j].isdigit():
                is_float = True
                i = j
                while i < length and text[i].isdigit():
                    i += 1
        raw = text[start:i].replace("_", "")
        try:
            value = float(raw) if is_float else int(raw)
        except ValueError:
            self.error(f"invalid number {raw!r}", number, start + 1)
        return value, i


def tokenize(source: str, filename: str = "<script>") -> list[Token]:
    return Lexer(source, filename).tokenize()
