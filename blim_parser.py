from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path


class IntType(Enum):
    U8 = auto()
    U16 = auto()
    I8 = auto()
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
    then_block: Block
    else_block: Block | None = None


@dataclass
class While(Statement):
    condition: Expression
    body: Block


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
