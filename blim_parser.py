import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from blim_lexer import Token, TokenType


class IntType(Enum):
    U8 = auto()
    U16 = auto()
    # I8 = auto()
    I16 = auto()


@dataclass
class Node:
    line: int
    column: int


@dataclass
class Expression(Node):
    pass


@dataclass
class Statement(Node):
    pass


@dataclass
class Type(Node):
    base_type: IntType | str
    pointer_depth: int = 0
    array_size: Expression | None = None


@dataclass
class Number(Expression):
    value: int


@dataclass
class Name(Expression):
    value: str


@dataclass
class Operation1(Expression):
    op: str
    value: Expression


@dataclass
class Operation2(Expression):
    op: str
    left: Expression
    right: Expression


@dataclass
class Call(Expression):
    value: Expression
    args: list[Expression] = field(default_factory=list)


@dataclass
class MemberAccess(Expression):
    value: Expression
    member: str


@dataclass
class Index(Expression):
    value: Expression
    index: Expression


@dataclass
class ArrayValue(Expression):
    values: list[Expression] = field(default_factory=list)


@dataclass
class StructValueField(Node):
    name: str
    value: Expression


@dataclass
class StructValue(Expression):
    name: str
    fields: list[StructValueField] = field(default_factory=list)


@dataclass
class Block(Statement):
    statements: list[Statement] = field(default_factory=list)


@dataclass
class Variable(Statement):
    name: str
    type: Type
    value: Expression | None = None


@dataclass
class Assign(Statement):
    targets: list[Expression] = field(default_factory=list)
    value: Expression | None = None


@dataclass
class Return(Statement):
    values: list[Expression] = field(default_factory=list)


@dataclass
class If(Statement):
    condition: Expression
    then_block: Statement
    else_block: Statement | None = None


@dataclass
class While(Statement):
    condition: Expression
    body: Statement


@dataclass
class Break(Statement):
    pass


@dataclass
class Continue(Statement):
    pass


@dataclass
class ExprStatement(Statement):
    value: Expression


@dataclass
class Asm(Statement):
    lines: list[str] = field(default_factory=list)


@dataclass
class Use(Node):
    package: str
    alias: str | None = None


@dataclass
class Define(Node):
    name: str
    value: Expression


@dataclass
class Field(Node):
    name: str
    type: Type


@dataclass
class Struct(Node):
    name: str
    fields: list[Field] = field(default_factory=list)


@dataclass
class Param(Node):
    name: str
    type: Type


@dataclass
class Result(Node):
    name: str
    type: Type


@dataclass
class Function(Node):
    name: str
    params: list[Param] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    body: Block | None = None


@dataclass
class GlobalVariable(Node):
    name: str
    type: Type
    value: Expression | None = None


@dataclass
class FileAst(Node):
    path: Path
    package: str
    imports: list[Use] = field(default_factory=list)
    defines: list[Define] = field(default_factory=list)
    structures: list[Struct] = field(default_factory=list)
    global_variables: list[GlobalVariable] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)


class Parser:
    def __init__(self, tokens: list[Token], path: Path, project_path: Path):
        self.tokens = tokens
        self.pos = 0
        self.path = path
        self.project_path = project_path
        self.current_function_results: list[Result] = []

        self.PRECEDENCE = {
            TokenType.PIPE: 1,
            TokenType.BANG_PIPE: 1,
            TokenType.CARET: 2,
            TokenType.BANG_CARET: 2,
            TokenType.AMPERSAND: 3,
            TokenType.BANG_AMPERSAND: 3,
            TokenType.EQUAL: 4,
            TokenType.NOT_EQUAL: 4,
            TokenType.LESS: 5,
            TokenType.MORE: 5,
            TokenType.LESS_EQUAL: 5,
            TokenType.MORE_EQUAL: 5,
            TokenType.LEFT_SHIFT: 6,
            TokenType.RIGHT_SHIFT: 6,
            TokenType.PLUS: 7,
            TokenType.MINUS: 7,
            TokenType.STAR: 8,
            TokenType.PERCENT: 8,
        }

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

        print(
            f"Error: Expected {expected_type.name}, but got {token.type.name} ('{token.value}') in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
        )
        sys.exit(1)

    def skip_newlines(self):
        while self.get_token().type == TokenType.NEW_LINE:
            self.pos += 1

    def parse_expression(self, precedence: int = 0) -> Expression:
        self.skip_newlines()
        token = self.get_token()
        self.pos += 1

        left: Expression

        if token.type == TokenType.NUMBER:
            left = Number(
                line=token.line, column=token.column, value=int(token.value, 0)
            )

        elif token.type == TokenType.IDENTIFIER:
            left = Name(line=token.line, column=token.column, value=token.value)

        elif token.type == TokenType.LEFT_BRACKET:
            left = self.parse_expression(0)
            self.expect(TokenType.RIGHT_BRACKET)

        elif token.type in [
            TokenType.MINUS,
            TokenType.BANG,
            TokenType.AMPERSAND,
            TokenType.STAR,
        ]:
            operator = token.value
            operand = self.parse_expression(10)
            left = Operation1(
                line=token.line, column=token.column, op=operator, value=operand
            )

        else:
            print(
                f"Error: Expected expression, but got {token.type.name} ('{token.value}') in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
            )
            sys.exit(1)

        self.skip_newlines()

        while precedence < self.PRECEDENCE.get(
            self.get_token().type, 0
        ) or self.get_token().type in [
            TokenType.DOT,
            TokenType.LEFT_SQUARE_BRACKET,
            TokenType.LEFT_BRACKET,
        ]:
            infix_token = self.get_token()

            if infix_token.type == TokenType.LEFT_BRACKET:
                self.pos += 1
                args = []
                self.skip_newlines()
                if self.get_token().type != TokenType.RIGHT_BRACKET:
                    args.append(self.parse_expression(0))
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expression(0))
                self.expect(TokenType.RIGHT_BRACKET)
                left = Call(line=left.line, column=left.column, value=left, args=args)

            elif infix_token.type == TokenType.DOT:
                self.pos += 1
                member = self.expect(TokenType.IDENTIFIER)
                left = MemberAccess(
                    line=left.line, column=left.column, value=left, member=member.value
                )

            elif infix_token.type == TokenType.LEFT_SQUARE_BRACKET:
                self.pos += 1
                index_expr = self.parse_expression(0)
                self.expect(TokenType.RIGHT_SQUARE_BRACKET)
                left = Index(
                    line=left.line, column=left.column, value=left, index=index_expr
                )

            else:
                self.pos += 1
                operator_str = infix_token.value
                current_prec = self.PRECEDENCE.get(infix_token.type, 0)
                right = self.parse_expression(current_prec)
                left = Operation2(
                    line=left.line,
                    column=left.column,
                    op=operator_str,
                    left=left,
                    right=right,
                )

            self.skip_newlines()

        return left

    def parse_block(self) -> Block:
        token = self.expect(TokenType.LEFT_BRACE)
        statements = []
        self.skip_newlines()
        while not self.match(TokenType.RIGHT_BRACE):
            if self.get_token().type == TokenType.EOF:
                print(
                    f"Error: Unclosed '{{' in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
                )
                sys.exit(1)
            statements.append(self.parse_statement())
            self.skip_newlines()
        return Block(line=token.line, column=token.column, statements=statements)

    def parse_statement(self) -> Statement:
        self.skip_newlines()
        token = self.get_token()

        if self.match(TokenType.RETURN):
            next_t = self.get_token()
            values: list[Expression] = []

            if next_t.type in (
                TokenType.NEW_LINE,
                TokenType.RIGHT_BRACE,
                TokenType.EOF,
            ):
                for res in self.current_function_results:
                    values.append(
                        Name(line=token.line, column=token.column, value=res.name)
                    )
            else:
                while True:
                    values.append(self.parse_expression())
                    if not self.match(TokenType.COMMA):
                        break
            return Return(line=token.line, column=token.column, values=values)

        if self.match(TokenType.WHILE):
            self.expect(TokenType.LEFT_BRACKET)
            cond = self.parse_expression()
            self.expect(TokenType.RIGHT_BRACKET)
            self.skip_newlines()
            if self.get_token().type == TokenType.LEFT_BRACE:
                body = self.parse_block()
            else:
                body = self.parse_statement()
            return While(
                line=token.line, column=token.column, condition=cond, body=body
            )

        if self.match(TokenType.IF):
            self.expect(TokenType.LEFT_BRACKET)
            cond = self.parse_expression()
            self.expect(TokenType.RIGHT_BRACKET)
            self.skip_newlines()
            if self.get_token().type == TokenType.LEFT_BRACE:
                then_block = self.parse_block()
            else:
                then_block = self.parse_statement()
            if self.match(TokenType.ELSE):
                self.skip_newlines()
                if self.get_token().type == TokenType.LEFT_BRACE:
                    else_block = self.parse_block()
                else:
                    else_block = self.parse_statement()
            else:
                else_block = None
            return If(
                line=token.line,
                column=token.column,
                condition=cond,
                then_block=then_block,
                else_block=else_block,
            )

        if self.match(TokenType.BREAK):
            return Break(line=token.line, column=token.column)

        if self.match(TokenType.CONTINUE):
            return Continue(line=token.line, column=token.column)

        if self.match(TokenType.HASH_ASM):
            self.expect(TokenType.LEFT_BRACE)
            lines = []
            while not self.match(TokenType.RIGHT_BRACE):
                if self.get_token().type == TokenType.EOF:
                    print(
                        f"Error: Unclosed #asm block in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
                    )
                    sys.exit(1)

                asm_line = []
                while self.get_token().type not in [
                    TokenType.NEW_LINE,
                    TokenType.EOF,
                    TokenType.RIGHT_BRACE,
                ]:
                    asm_line.append(self.get_token().value)
                    self.pos += 1
                if asm_line:
                    lines.append(" ".join(asm_line))
                self.skip_newlines()
            return Asm(line=token.line, column=token.column, lines=lines)

        if token.type == TokenType.IDENTIFIER:
            next_token = self.get_token(1)

            if next_token.type == TokenType.COLON:
                self.pos += 2
                var_type = self.parse_type()
                value = None
                if self.match(TokenType.ASSIGN):
                    value = self.parse_expression()
                return Variable(
                    line=token.line,
                    column=token.column,
                    name=token.value,
                    type=var_type,
                    value=value,
                )

            saved_pos = self.pos
            targets: list[Expression] = []

            while self.get_token().type == TokenType.IDENTIFIER:
                ident = self.get_token()
                targets.append(
                    Name(line=ident.line, column=ident.column, value=ident.value)
                )
                self.pos += 1
                if not self.match(TokenType.COMMA):
                    break

            if self.match(TokenType.ASSIGN):
                value = self.parse_expression()
                return Assign(
                    line=token.line, column=token.column, targets=targets, value=value
                )

            self.pos = saved_pos

        expr = self.parse_expression()

        if self.match(TokenType.ASSIGN):
            val_expr = self.parse_expression()
            return Assign(
                line=token.line, column=token.column, targets=[expr], value=val_expr
            )

        return ExprStatement(line=token.line, column=token.column, value=expr)

    def parse_type(self) -> Type:
        token = self.get_token()
        pointer_depth = 0
        while self.match(TokenType.STAR):
            pointer_depth += 1

        token = self.get_token()
        base_type = None

        if token.type == TokenType.TYPE:
            self.pos += 1
            base_type = IntType[token.value.upper()]
        elif token.type == TokenType.IDENTIFIER:
            self.pos += 1
            base_type = token.value
        else:
            print(
                f"Error: Expected type, but got {token.type.name} ('{token.value}') in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
            )
            sys.exit(1)

        array_size = None
        if self.match(TokenType.LEFT_SQUARE_BRACKET):
            array_size = self.parse_expression()
            self.expect(TokenType.RIGHT_SQUARE_BRACKET)

        return Type(
            line=token.line,
            column=token.column,
            base_type=base_type,
            pointer_depth=pointer_depth,
            array_size=array_size,
        )

    def parse(self) -> FileAst:
        self.skip_newlines()
        start_token = self.get_token()

        if self.match(TokenType.PACKAGE):
            package_name = self.expect(TokenType.IDENTIFIER).value
            self.skip_newlines()
        else:
            package_name = "main"

        ast = FileAst(
            line=start_token.line,
            column=start_token.column,
            path=self.path,
            package=package_name,
        )

        while not self.get_token().type == TokenType.EOF:
            self.skip_newlines()
            if self.get_token().type == TokenType.EOF:
                break

            token = self.get_token()

            if token.type == TokenType.USE:
                ast.imports.append(self.parse_use())
            elif token.type == TokenType.HASH_DEF:
                ast.defines.append(self.parse_define())
            elif token.type == TokenType.IDENTIFIER:
                self.pos += 1
                self.expect(TokenType.COLON)

                next_tok = self.get_token()
                if self.match(TokenType.STRUCT):
                    ast.structures.append(self.parse_struct(token))
                elif self.match(TokenType.FUN):
                    ast.functions.append(self.parse_function(token))
                elif next_tok.type in [
                    TokenType.TYPE,
                    TokenType.IDENTIFIER,
                    TokenType.STAR,
                ]:
                    ast.global_variables.append(self.parse_global_var(token))
                else:
                    print(
                        f"Error: Expected 'struct', 'fun' or type, but got {token.type.name} ('{token.value}') in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
                    )
                    sys.exit(1)
            else:
                print(
                    f"Error: Unexpected token {token.type.name} ('{token.value}') in {self.path.relative_to(self.project_path)}:{token.line}:{token.column}"
                )
                sys.exit(1)

        return ast

    def parse_use(self) -> Use:
        line_tok = self.expect(TokenType.USE)
        pkg = self.expect(TokenType.IDENTIFIER).value
        alias = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENTIFIER).value
        return Use(line=line_tok.line, column=line_tok.column, package=pkg, alias=alias)

    def parse_define(self) -> Define:
        line_tok = self.expect(TokenType.HASH_DEF)
        name = self.expect(TokenType.IDENTIFIER).value
        val = self.parse_expression()
        return Define(line=line_tok.line, column=line_tok.column, name=name, value=val)

    def parse_struct(self, name_token: Token) -> Struct:
        self.expect(TokenType.LEFT_BRACE)
        fields = []
        self.skip_newlines()
        while not self.match(TokenType.RIGHT_BRACE):
            f_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)

            f_type = self.parse_type()
            fields.append(
                Field(
                    line=name_token.line,
                    column=name_token.column,
                    name=f_name,
                    type=f_type,
                )
            )
            self.skip_newlines()

        return Struct(
            line=name_token.line,
            column=name_token.column,
            name=name_token.value,
            fields=fields,
        )

    def parse_global_var(self, name_token: Token) -> GlobalVariable:
        g_type = self.parse_type()

        initial_value = None
        if self.match(TokenType.ASSIGN):
            initial_value = self.parse_expression()

        return GlobalVariable(
            line=name_token.line,
            column=name_token.column,
            name=name_token.value,
            type=g_type,
            value=initial_value,
        )

    def parse_function(self, name_token: Token) -> Function:
        self.expect(TokenType.LEFT_BRACKET)
        params = []

        while not self.match(TokenType.RIGHT_BRACKET):
            self.skip_newlines()
            param_token = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.COLON)
            param_type = self.parse_type()

            params.append(
                Param(
                    line=param_token.line,
                    column=param_token.column,
                    name=param_token.value,
                    type=param_type,
                )
            )

            if not self.match(TokenType.COMMA):
                self.expect(TokenType.RIGHT_BRACKET)
                break

        results = []
        if self.match(TokenType.ARROW):
            while True:
                self.skip_newlines()
                res_token = self.expect(TokenType.IDENTIFIER)
                self.expect(TokenType.COLON)
                res_type = self.parse_type()

                results.append(
                    Result(
                        line=res_token.line,
                        column=res_token.column,
                        name=res_token.value,
                        type=res_type,
                    )
                )

                if not self.match(TokenType.COMMA):
                    break

        previous_results = self.current_function_results
        self.current_function_results = results

        self.skip_newlines()
        body = self.parse_block()

        self.current_function_results = previous_results

        return Function(
            line=name_token.line,
            column=name_token.column,
            name=name_token.value,
            params=params,
            results=results,
            body=body,
        )
