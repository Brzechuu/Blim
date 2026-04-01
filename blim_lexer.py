import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    PACKAGE = auto()
    USE = auto()
    AS = auto()
    ASM = auto()
    STRUCT = auto()
    FUN = auto()
    RETURN = auto()
    WHILE = auto()
    BREAK = auto()
    CONTINUE = auto()
    IF = auto()
    ELSE = auto()
    TYPE = auto()

    HASH_DIRECTIVE = auto()

    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()

    ASSIGN = auto()  # =
    EQUAL = auto()  # ==
    NOT_EQUAL = auto()  # !=
    LESS = auto()  # <
    MORE = auto()  # >
    LESS_EQUAL = auto()  # <=
    MORE_EQUAL = auto()  # >=
    PLUS = auto()  # +
    MINUS = auto()  # -
    STAR = auto()  # *
    BANG_AMPERSAND = auto()  # !&
    BANG_PIPE = auto()  # !|
    BANG_CARET = auto()  # !^
    AMPERSAND = auto()  # &
    PIPE = auto()  # |
    CARET = auto()  # ^
    BANG = auto()  # !
    LEFT_SHIFT = auto()  # <<
    RIGHT_SHIFT = auto()  # >>
    LEFT_ROTATE = auto()  # <<<
    RIGHT_ROTATE = auto()  # >>>
    RIGHT_ARITHMETIC_SHIFT = auto()  # >=>
    ARROW = auto()  # ->
    PERCENT = auto()  # %

    COLON = auto()  # :
    COMMA = auto()  # ,
    DOT = auto()  # .
    LEFT_BRACKET = auto()  # (
    RIGHT_BRACKET = auto()  # )
    LEFT_BRACE = auto()  # {
    RIGHT_BRACE = auto()  # }
    LEFT_SQUARE_BRACKET = auto()  # [
    RIGHT_SQUARE_BRACKET = auto()  # ]

    NEW_LINE = auto()
    EOF = auto()
    ILLEGAL = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __str__(self):
        return f"[{self.line}:{self.column}] {self.type.name}: {self.value}"


class Lexer:
    def __init__(self, code: str):
        self.code = code

        self.keywords = {
            "package": TokenType.PACKAGE,
            "use": TokenType.USE,
            "as": TokenType.AS,
            "asm": TokenType.ASM,
            "struct": TokenType.STRUCT,
            "fun": TokenType.FUN,
            "return": TokenType.RETURN,
            "while": TokenType.WHILE,
            "break": TokenType.BREAK,
            "continue": TokenType.CONTINUE,
            "if": TokenType.IF,
            "else": TokenType.ELSE,
            "u16": TokenType.TYPE,
            "i16": TokenType.TYPE,
            "u8": TokenType.TYPE,
            # "i8": TokenType.TYPE,
        }

        rules = [
            ("COMMENT", r"//.*"),
            ("WHITESPACE", r"[ \t\r]+"),
            ("NEW_LINE", r"\n"),
            ("HASH_DIRECTIVE", r"#[a-zA-Z_]+"),
            ("NUMBER", r"0[xX][0-9a-fA-F]+|0[bB][01]+|\d+"),
            ("STRING", r'"[^"\\]*(\\.[^"\\]*)*"'),
            ("IDENTIFIER", r"[a-zA-Z_][a-zA-Z0-9_]*"),
            ("LEFT_ROTATE", r"<<<"),
            ("RIGHT_ROTATE", r">>>"),
            ("RIGHT_ARITHMETIC_SHIFT", r">=>"),
            ("EQUAL", r"=="),
            ("NOT_EQUAL", r"!="),
            ("LESS_EQUAL", r"<="),
            ("MORE_EQUAL", r">="),
            ("LEFT_SHIFT", r"<<"),
            ("RIGHT_SHIFT", r">>"),
            ("ARROW", r"->"),
            ("ASSIGN", r"="),
            ("LESS", r"<"),
            ("MORE", r">"),
            ("PLUS", r"\+"),
            ("MINUS", r"-"),
            ("STAR", r"\*"),
            ("BANG_AMPERSAND", r"!&"),
            ("BANG_PIPE", r"!\|"),
            ("BANG_CARET", r"!\^"),
            ("AMPERSAND", r"&"),
            ("PIPE", r"\|"),
            ("CARET", r"\^"),
            ("BANG", r"!"),
            ("PERCENT", r"%"),
            ("COLON", r":"),
            ("COMMA", r","),
            ("DOT", r"\."),
            ("LEFT_BRACKET", r"\("),
            ("RIGHT_BRACKET", r"\)"),
            ("LEFT_BRACE", r"\{"),
            ("RIGHT_BRACE", r"\}"),
            ("LEFT_SQUARE_BRACKET", r"\["),
            ("RIGHT_SQUARE_BRACKET", r"\]"),
            ("ILLEGAL", r"."),
        ]

        self.regex = re.compile(
            "|".join(f"(?P<{name}>{pattern})" for name, pattern in rules)
        )

    def tokenize(self):
        pos = 0
        line = 1
        column = 1
        code_len = len(self.code)

        while pos < code_len:
            match = self.regex.match(self.code, pos)

            if not match:
                yield Token(TokenType.ILLEGAL, self.code[pos], line, column)
                pos += 1
                column += 1
                continue

            kind = match.lastgroup
            value = match.group(0)
            start_line = line
            start_column = column

            if kind == "WHITESPACE":
                pos += len(value)
                column += len(value)
                continue

            if kind == "COMMENT":
                pos += len(value)
                column += len(value)
                continue

            if kind == "NEW_LINE":
                yield Token(TokenType.NEW_LINE, value, start_line, start_column)
                pos += 1
                line += 1
                column = 1
                continue

            pos += len(value)
            column += len(value)

            if kind == "IDENTIFIER":
                yield Token(
                    self.keywords.get(value, TokenType.IDENTIFIER),
                    value,
                    start_line,
                    start_column,
                )
                continue

            if kind == "NUMBER":
                yield Token(
                    TokenType.NUMBER, str(int(value, 0)), start_line, start_column
                )
                continue

            if kind == "STRING":
                yield Token(TokenType.STRING, value, start_line, start_column)
                continue

            if kind is None:
                yield Token(TokenType.ILLEGAL, value, start_line, start_column)
                continue

            yield Token(TokenType[kind], value, start_line, start_column)

        yield Token(TokenType.EOF, "", line, column)
