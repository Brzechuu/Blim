from dataclasses import dataclass
from pathlib import Path

from blim_lexer import Token


class Node:
    pass


class Expression(Node):
    pass


class Statement(Node):
    pass


@dataclass
class FileAst(Node):
    path: Path
    package: str


class Parser:
    def __init__(self, tokens: list[Token], path: Path):
        self.tokens = tokens
        self.pos = 0
        self.path = path
