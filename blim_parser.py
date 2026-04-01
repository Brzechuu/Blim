from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from blim_lexer import Token, TokenType
from blim_reporter import Reporter


class IntType(Enum):
    U8 = auto()
    U16 = auto()
    # I8 = auto()
    I16 = auto()


class MemberAccessType(Enum):
    UNKNOWN = auto()
    PACKAGE = auto()
    FIELD = auto()


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
class StringValue(Expression):
    value: str


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
    type: MemberAccessType = MemberAccessType.UNKNOWN


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
    pass


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
    body: Block
    params: list[Param] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)


@dataclass
class GlobalVariable(Node):
    name: str
    type: Type
    value: Expression | None = None


@dataclass
class Define(Node):
    name: str
    value: str


@dataclass
class InterruptVector(Node):
    func_name: str
    vector_number: int


@dataclass
class FileAst(Node):
    path: Path
    package: str
    imports: list[Use] = field(default_factory=list)
    structures: list[Struct] = field(default_factory=list)
    global_variables: list[GlobalVariable] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    defines: list[Define] = field(default_factory=list)
    interrupt_vectors: list[InterruptVector] = field(default_factory=list)


class Parser:
    def __init__(
        self, tokens: list[Token], path: Path, project_path: Path, reporter: Reporter
    ):
        self.tokens = tokens
        self.pos = 0
        self.path = path
        self.project_path = project_path
        self.r = reporter

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
            TokenType.LEFT_ROTATE: 6,
            TokenType.RIGHT_ROTATE: 6,
            TokenType.RIGHT_ARITHMETIC_SHIFT: 6,
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

        self.r.error(
            f"Expected {expected_type.name}, but got {token.type.name} ('{token.value}')",
            self.path,
            token.line,
            token.column,
        )
        raise SystemExit(1)

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

        elif token.type == TokenType.STRING:
            left = StringValue(line=token.line, column=token.column, value=token.value)

        elif token.type == TokenType.LEFT_BRACE:
            values = []
            self.skip_newlines()
            if self.get_token().type != TokenType.RIGHT_BRACE:
                values.append(self.parse_expression(0))
                self.skip_newlines()
                while self.match(TokenType.COMMA):
                    self.skip_newlines()
                    values.append(self.parse_expression(0))
                    self.skip_newlines()
            self.expect(TokenType.RIGHT_BRACE)
            left = ArrayValue(line=token.line, column=token.column, values=values)

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
            self.r.error(
                f"Expected EXPRESSION, but got {token.type.name} ('{token.value}')",
                self.path,
                token.line,
                token.column,
            )
            raise SystemExit(1)

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

        return left

    def parse_block(self) -> Block:
        token = self.expect(TokenType.LEFT_BRACE)
        statements = []
        self.skip_newlines()
        while not self.match(TokenType.RIGHT_BRACE):
            if self.get_token().type == TokenType.EOF:
                self.r.error(
                    "Unclosed '{'",
                    self.path,
                    token.line,
                    token.column,
                )
            statements.append(self.parse_statement())
            self.skip_newlines()
        return Block(line=token.line, column=token.column, statements=statements)

    def parse_statement(self) -> Statement:
        self.skip_newlines()
        token = self.get_token()

        if self.match(TokenType.RETURN):
            return Return(line=token.line, column=token.column)

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

        if self.match(TokenType.ASM):
            self.expect(TokenType.LEFT_BRACE)
            lines = []
            while not self.match(TokenType.RIGHT_BRACE):
                if self.get_token().type == TokenType.EOF:
                    self.r.error(
                        "Unclosed asm block",
                        self.path,
                        token.line,
                        token.column,
                    )

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

            if self.match(TokenType.DOT):
                member_token = self.expect(TokenType.IDENTIFIER)
                base_type = f"{base_type}.{member_token.value}"
        else:
            self.r.error(
                f"Expected TYPE, but got {token.type.name} ('{token.value}')",
                self.path,
                token.line,
                token.column,
            )
            raise SystemExit(1)

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

            elif token.type == TokenType.HASH_DIRECTIVE:
                self.pos += 1
                directive_name = token.value[1:]

                if directive_name == "def":
                    name_tok = self.expect(TokenType.IDENTIFIER)
                    val_tok = self.expect(TokenType.NUMBER)

                    ast.defines.append(
                        Define(
                            line=token.line,
                            column=token.column,
                            name=name_tok.value,
                            value=val_tok.value,
                        )
                    )

                elif directive_name == "vec":
                    func_name_tok = self.expect(TokenType.IDENTIFIER)
                    vec_num_tok = self.expect(TokenType.NUMBER)

                    ast.interrupt_vectors.append(
                        InterruptVector(
                            line=token.line,
                            column=token.column,
                            func_name=func_name_tok.value,
                            vector_number=int(vec_num_tok.value, 0),
                        )
                    )

                else:
                    self.r.error(
                        f"Unknown directive '#{directive_name}'",
                        self.path,
                        token.line,
                        token.column,
                    )

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
                    self.r.error(
                        f"Expected 'struct', 'fun' or type, but got {token.type.name} ('{token.value}')",
                        self.path,
                        token.line,
                        token.column,
                    )
            else:
                self.r.error(
                    f"Unexpected token {token.type.name} ('{token.value}')",
                    self.path,
                    token.line,
                    token.column,
                )
                self.pos += 1

        return ast

    def parse_use(self) -> Use:
        line_tok = self.expect(TokenType.USE)
        pkg = self.expect(TokenType.IDENTIFIER).value
        alias = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENTIFIER).value
        return Use(line=line_tok.line, column=line_tok.column, package=pkg, alias=alias)

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

        self.skip_newlines()
        body = self.parse_block()

        return Function(
            line=name_token.line,
            column=name_token.column,
            name=name_token.value,
            params=params,
            results=results,
            body=body,
        )
