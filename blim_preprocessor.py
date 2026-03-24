from pathlib import Path

from blim_lexer import Token, TokenType
from blim_reporter import Reporter


class Preprocessor:
    def __init__(
        self,
        tokens: list[Token],
        path: Path,
        project_path: Path,
        reporter: Reporter,
    ):
        self.tokens = tokens
        self.path = path
        self.project_path = project_path
        self.r = reporter
        self.pos = 0

    def get_token(self, offset: int = 0) -> Token:
        index = self.pos + offset
        if index < 0:
            index = 0
        elif index >= len(self.tokens):
            index = len(self.tokens) - 1
        return self.tokens[index]

    def match(self, expected_type: TokenType) -> bool:
        self.skip_newlines()
        if self.get_token().type == expected_type:
            self.pos += 1
            return True
        return False

    def expect(self, expected_type: TokenType) -> Token:
        self.skip_newlines()
        token = self.get_token()
        if token.type == expected_type:
            self.pos += 1
            return token

        self.r.error(
            f"Expected {expected_type.name}, but got {token.type.name} ('{token.value}') in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
        )
        raise SystemExit(1)

    def skip_newlines(self):
        while self.get_token().type == TokenType.NEW_LINE:
            self.pos += 1

    def process(self) -> list[Token]:
        defines: dict[str, str] = {}
        result: list[Token] = []

        while self.pos < len(self.tokens):
            token = self.get_token()

            if token.type == TokenType.HASH_DEF:
                self.pos += 1
                name_token = self.expect(TokenType.IDENTIFIER)
                name = name_token.value

                if name in defines:
                    self.r.error(
                        f"Redefinition of #def '{name}' in {self.path.relative_to(self.project_path)}:{name_token.line}:{name_token.column}"
                    )
                    raise SystemExit(1)

                value_token = self.expect(TokenType.NUMBER)
                defines[name] = value_token.value
            else:
                if token.type == TokenType.IDENTIFIER and token.value in defines:
                    result.append(
                        Token(
                            TokenType.NUMBER,
                            defines[token.value],
                            token.line,
                            token.column,
                        )
                    )
                else:
                    result.append(token)

                self.pos += 1

        return result
