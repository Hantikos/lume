"""Recursive-descent parser for Lume."""

from __future__ import annotations

from . import ast_nodes as A
from .errors import LumeSyntaxError
from .lexer import Token, tokenize

COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">="}


class Parser:
    def __init__(self, source: str, filename: str = "<script>"):
        self.filename = filename
        self.tokens = tokenize(source, filename)
        self.pos = 0

    # -- token helpers ----------------------------------------------------
    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        index = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def advance(self) -> Token:
        token = self.tokens[self.pos]
        if token.kind != "EOF":
            self.pos += 1
        return token

    def check(self, kind: str, value=None) -> bool:
        token = self.current
        if token.kind != kind:
            return False
        return value is None or token.value == value

    def match(self, kind: str, value=None) -> Token | None:
        if self.check(kind, value):
            return self.advance()
        return None

    def expect(self, kind: str, value=None, what: str = "") -> Token:
        if self.check(kind, value):
            return self.advance()
        expected = what or (value if isinstance(value, str) else kind)
        token = self.current
        got = token.value if token.kind != "EOF" else "end of file"
        self.error(f"expected {expected}, got {got!r}", token)

    def error(self, message: str, token: Token | None = None):
        token = token or self.current
        raise LumeSyntaxError(message, token.line, token.col, self.filename)

    def skip_newlines(self):
        while self.check("NEWLINE"):
            self.advance()

    def at_end_of_statement(self) -> bool:
        return (self.check("NEWLINE") or self.check("EOF") or self.check("OP", "}")
                or self.check("OP", ";"))

    def peek_past_newlines(self) -> Token:
        index = self.pos
        while index < len(self.tokens) and self.tokens[index].kind == "NEWLINE":
            index += 1
        return self.tokens[min(index, len(self.tokens) - 1)]

    # -- program ----------------------------------------------------------
    def parse(self) -> A.Program:
        body = []
        self.skip_newlines()
        while not self.check("EOF"):
            body.append(self.parse_statement())
            self.finish_statement()
            self.skip_newlines()
        return A.Program(body)

    def finish_statement(self):
        """Consume the separator that ends a statement."""
        if self.check("NEWLINE"):
            self.advance()
        elif self.check("OP", ";"):
            self.advance()
        elif self.check("EOF") or self.check("OP", "}"):
            return
        else:
            token = self.current
            got = token.value if token.kind != "EOF" else "end of file"
            self.error(f"expected a new line after the statement, got {got!r}", token)

    # -- statements -------------------------------------------------------
    def parse_statement(self) -> A.Node:
        token = self.current
        line, col = token.line, token.col

        if self.check("KEYWORD"):
            handler = getattr(self, f"_stmt_{token.value}", None)
            if handler is not None:
                return handler(line, col)

        # contextual keywords
        if token.kind == "ID":
            if token.value == "model" and self.peek().kind == "ID":
                return self._stmt_model(line, col)
            if token.value == "layer" and self.peek().kind in ("ID", "KEYWORD"):
                return self._stmt_layer(line, col)
            if token.value == "data" and self.peek().kind == "ID" and self.peek(2).kind == "OP" \
                    and self.peek(2).value == "=":
                return self._stmt_data(line, col)

        expr = self.parse_expression()
        if self.check("OP", "="):
            self.advance()
            value = self.parse_expression()
            if not isinstance(expr, (A.Name, A.Index)):
                self.error("only a variable or a list item can be assigned to", token)
            return A.Assign(expr, value, line, col)
        return A.ExprStmt(expr, line, col)

    def _stmt_fn(self, line, col):
        self.advance()  # fn
        name = self.expect("ID", what="function name").value
        self.expect("OP", "(")
        params, defaults = self._parse_params()
        self.expect("OP", ")")
        body = self.parse_block()
        return A.FuncDef(name, params, defaults, body, line, col)

    def _parse_params(self):
        params, defaults = [], []
        while not self.check("OP", ")"):
            pname = self.expect("ID", what="parameter name").value
            if self.match("OP", ":"):
                self.parse_expression()  # optional type hint (documentation only)
            default = None
            if self.match("OP", "="):
                default = self.parse_expression()
            params.append(pname)
            defaults.append(default)
            if not self.match("OP", ","):
                break
        return params, defaults

    def _stmt_model(self, line, col):
        self.advance()  # model
        name = self.expect("ID", what="model name").value
        body = self.parse_block()
        return A.ModelDef(name, body, line, col)

    def _stmt_layer(self, line, col):
        self.advance()  # layer
        call = self.parse_expression()
        if not isinstance(call, A.Call):
            self.error("'layer' must be followed by a layer call, e.g. layer Dense(64)")
        return A.LayerDecl(call, line, col)

    def _stmt_data(self, line, col):
        self.advance()  # data
        name = self.expect("ID", what="dataset name").value
        self.expect("OP", "=")
        value = self.parse_expression()
        return A.DataStmt(name, value, line, col)

    def _stmt_let(self, line, col):
        self.advance()  # let / const
        const = self.tokens[self.pos - 1].value == "const"
        name = self.expect("ID", what="variable name").value
        type_hint = None
        if self.match("OP", ":"):
            type_hint = self.parse_expression()
        self.expect("OP", "=")
        value = self.parse_expression()
        return A.LetStmt(name, value, const, type_hint, line, col)

    _stmt_const = _stmt_let

    def _stmt_if(self, line, col):
        self.advance()  # if
        test = self.parse_expression()
        body = self.parse_block()
        orelse = None
        if self.peek_past_newlines().kind == "KEYWORD" and self.peek_past_newlines().value == "else":
            self.skip_newlines()
            self.advance()  # else
            if self.check("KEYWORD", "if"):
                orelse = A.Block([self._stmt_if(self.current.line, self.current.col)], self.current.line)
            else:
                orelse = self.parse_block()
        return A.IfStmt(test, body, orelse, line, col)

    def _stmt_else(self, line, col):  # `else` without `if` -> error
        self.error("'else' without a matching 'if'")

    def _stmt_for(self, line, col):
        self.advance()  # for
        var = self.expect("ID", what="loop variable").value
        if not self.check("KEYWORD", "in"):
            self.error("expected 'in' in a for loop")
        self.advance()
        iterable = self.parse_expression()
        body = self.parse_block()
        return A.ForStmt(var, iterable, body, line, col)

    def _stmt_while(self, line, col):
        self.advance()  # while
        test = self.parse_expression()
        body = self.parse_block()
        return A.WhileStmt(test, body, line, col)

    def _stmt_return(self, line, col):
        self.advance()
        value = None
        if not self.at_end_of_statement():
            value = self.parse_expression()
        return A.ReturnStmt(value, line, col)

    def _stmt_break(self, line, col):
        self.advance()
        return A.BreakStmt(line, col)

    def _stmt_continue(self, line, col):
        self.advance()
        return A.ContinueStmt(line, col)

    def _stmt_import(self, line, col):
        self.advance()
        module = self.expect("ID", what="module name").value
        alias = None
        if self.match("KEYWORD", "as"):
            alias = self.expect("ID", what="alias").value
        return A.ImportStmt(module, alias, line, col)

    def _stmt_from(self, line, col):
        self.advance()
        module = self.expect("ID", what="module name").value
        if not self.check("KEYWORD", "import"):
            self.error("expected 'import'")
        self.advance()
        names = []
        while True:
            name = self.expect("ID", what="imported name").value
            alias = None
            if self.match("KEYWORD", "as"):
                alias = self.expect("ID", what="alias").value
            names.append((name, alias))
            if not self.match("OP", ","):
                break
        return A.FromImportStmt(module, names, line, col)

    def _stmt_true(self, line, col):
        self.advance()
        return A.ExprStmt(A.BoolLit(True, line, col), line, col)

    def _stmt_false(self, line, col):
        self.advance()
        return A.ExprStmt(A.BoolLit(False, line, col), line, col)

    def _stmt_null(self, line, col):
        self.advance()
        return A.ExprStmt(A.NullLit(line, col), line, col)

    def _stmt_and(self, line, col):
        return self._expression_statement(line, col)

    _stmt_or = _stmt_and
    _stmt_not = _stmt_and

    def _expression_statement(self, line, col):
        expr = self.parse_expression()
        return A.ExprStmt(expr, line, col)

    # -- blocks -----------------------------------------------------------
    def parse_block(self) -> A.Block:
        start = self.current
        self.expect("OP", "{", what="'{'")
        body = []
        while True:
            self.skip_newlines()
            if self.check("OP", "}"):
                self.advance()
                break
            if self.check("EOF"):
                self.error("unexpected end of file, '}' expected")
            body.append(self.parse_statement())
            if self.check("NEWLINE"):
                self.advance()
            elif self.check("OP", ";"):
                self.advance()
            elif not self.check("OP", "}") and not self.check("EOF"):
                token = self.current
                got = token.value if token.kind != "EOF" else "end of file"
                self.error(f"expected a new line after the statement, got {got!r}", token)
        return A.Block(body, start.line, start.col)

    # -- expressions ------------------------------------------------------
    def parse_expression(self):
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.check("KEYWORD", "or"):
            line, col = self.current.line, self.current.col
            self.advance()
            right = self.parse_and()
            left = A.Logical("or", left, right, line, col)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.check("KEYWORD", "and"):
            line, col = self.current.line, self.current.col
            self.advance()
            right = self.parse_not()
            left = A.Logical("and", left, right, line, col)
        return left

    def parse_not(self):
        if self.check("KEYWORD", "not"):
            line, col = self.current.line, self.current.col
            self.advance()
            return A.Unary("not", self.parse_not(), line, col)
        return self.parse_compare()

    def parse_compare(self):
        left = self.parse_additive()
        while self.current.kind == "OP" and self.current.value in COMPARE_OPS:
            token = self.advance()
            right = self.parse_additive()
            left = A.Compare(token.value, left, right, token.line, token.col)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current.kind == "OP" and self.current.value in ("+", "-"):
            token = self.advance()
            right = self.parse_multiplicative()
            left = A.Binary(token.value, left, right, token.line, token.col)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current.kind == "OP" and self.current.value in ("*", "/", "%", "@"):
            token = self.advance()
            right = self.parse_unary()
            left = A.Binary(token.value, left, right, token.line, token.col)
        return left

    def parse_unary(self):
        if self.current.kind == "OP" and self.current.value in ("-", "+"):
            token = self.advance()
            return A.Unary(token.value, self.parse_unary(), token.line, token.col)
        return self.parse_power()

    def parse_power(self):
        base = self.parse_postfix()
        if self.check("OP", "**"):
            token = self.advance()
            exponent = self.parse_unary()
            return A.Binary("**", base, exponent, token.line, token.col)
        return base

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.check("OP", "("):
                token = self.advance()
                args, kwargs = self.parse_call_args()
                self.expect("OP", ")")
                node = A.Call(node, args, kwargs, token.line, token.col)
            elif self.check("OP", "["):
                token = self.advance()
                index = self.parse_expression()
                self.expect("OP", "]")
                node = A.Index(node, index, token.line, token.col)
            elif self.check("OP", "."):
                token = self.advance()
                attr = self.expect("ID", what="attribute name").value
                node = A.Attribute(node, attr, token.line, token.col)
            else:
                break
        return node

    def parse_call_args(self):
        args, kwargs = [], []
        self.skip_newlines()
        while not self.check("OP", ")"):
            if self.current.kind == "ID" and self.peek().kind == "OP" and self.peek().value in ("=", ":"):
                name = self.advance().value
                self.advance()
                kwargs.append((name, self.parse_expression()))
            else:
                args.append(self.parse_expression())
            self.skip_newlines()
            if not self.match("OP", ","):
                break
            self.skip_newlines()
        self.skip_newlines()
        return args, kwargs

    def parse_primary(self):
        token = self.current
        if token.kind == "NUMBER":
            self.advance()
            return A.NumberLit(token.value, token.line, token.col)
        if token.kind == "STRING":
            self.advance()
            return A.StringLit(token.value, token.line, token.col)
        if token.kind == "FSTRING":
            self.advance()
            return A.FStringLit(self.parse_fstring(token.value, token.line, token.col),
                                token.line, token.col)
        if token.kind == "KEYWORD":
            if token.value == "true":
                self.advance()
                return A.BoolLit(True, token.line, token.col)
            if token.value == "false":
                self.advance()
                return A.BoolLit(False, token.line, token.col)
            if token.value == "null":
                self.advance()
                return A.NullLit(token.line, token.col)
            self.error(f"unexpected keyword {token.value!r}", token)
        if token.kind == "ID":
            self.advance()
            # friendly sugar:  print "hello"  ==  print("hello")
            #                    print x + 1   ==  print(x + 1)
            if self.current.kind in ("STRING", "FSTRING", "NUMBER", "ID"):
                arguments = [self.parse_expression()]
                while self.match("OP", ","):
                    arguments.append(self.parse_expression())
                return A.Call(A.Name(token.value, token.line, token.col), arguments, [],
                              token.line, token.col)
            return A.Name(token.value, token.line, token.col)
        if self.check("OP", "("):
            self.advance()
            expr = self.parse_expression()
            self.expect("OP", ")")
            return expr
        if self.check("OP", "["):
            self.advance()
            items = []
            self.skip_newlines()
            while not self.check("OP", "]"):
                items.append(self.parse_expression())
                self.skip_newlines()
                if not self.match("OP", ","):
                    break
                self.skip_newlines()
            self.expect("OP", "]")
            return A.ListLit(items, token.line, token.col)
        if self.check("OP", "{"):
            self.advance()
            pairs = []
            self.skip_newlines()
            while not self.check("OP", "}"):
                key = self.parse_expression()
                self.expect("OP", ":")
                value = self.parse_expression()
                pairs.append((key, value))
                self.skip_newlines()
                if not self.match("OP", ","):
                    break
                self.skip_newlines()
            self.expect("OP", "}")
            return A.DictLit(pairs, token.line, token.col)
        self.error(f"unexpected token {token.value!r}", token)

    def parse_fstring(self, raw: str, line: int, col: int):
        parts = []
        buf = ""
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "{" and i + 1 < len(raw) and raw[i + 1] == "{":
                buf += "{"
                i += 2
                continue
            if ch == "}" and i + 1 < len(raw) and raw[i + 1] == "}":
                buf += "}"
                i += 2
                continue
            if ch == "{":
                depth = 1
                j = i + 1
                while j < len(raw) and depth:
                    if raw[j] == "{":
                        depth += 1
                    elif raw[j] == "}":
                        depth -= 1
                    j += 1
                if depth:
                    self.error("unterminated '{' in interpolated string", Token("STRING", raw, line, col))
                expression = raw[i + 1:j - 1]
                if buf:
                    parts.append(buf)
                    buf = ""
                parts.append(self.parse_expression_source(expression, line, col))
                i = j
                continue
            if ch == "}":
                self.error("unmatched '}' in interpolated string", Token("STRING", raw, line, col))
            buf += ch
            i += 1
        if buf:
            parts.append(buf)
        return parts

    def parse_expression_source(self, source: str, line: int, col: int):
        sub = Parser(source, self.filename)
        expr = sub.parse_expression()
        sub.skip_newlines()
        if not sub.check("EOF"):
            sub.error("invalid expression inside interpolated string")
        expr.line, expr.col = line, col
        return expr


def parse(source: str, filename: str = "<script>") -> A.Program:
    return Parser(source, filename).parse()
