from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto

from blim_parser import (
    ArrayValue,
    Asm,
    Assign,
    Block,
    Break,
    Call,
    Continue,
    Expression,
    ExprStatement,
    FileAst,
    Function,
    If,
    Index,
    IntType,
    MemberAccess,
    Name,
    Number,
    Operation1,
    Operation2,
    Return,
    Statement,
    StructValue,
    Variable,
    While,
)
from blim_reporter import Reporter


class Register(IntEnum):
    R0 = 0
    A0 = 1
    A1 = 2
    A2 = 3
    A3 = 4
    B0 = 5
    B1 = 6
    B2 = 7
    B3 = 8
    G0 = 9
    G1 = 10
    G2 = 11
    G3 = 12
    FL = 13
    SP = 14
    PC = 15


class RegisterState(Enum):
    FREE = auto()
    ALLOCATED = auto()
    RESERVED = auto()


class RegisterType(Enum):
    A = auto()
    B = auto()
    G = auto()
    SPECIAL = auto()


class SymbolType(Enum):
    STRUCT = auto()
    ARRAY = auto()


class SymbolRange(Enum):
    GLOBAL = auto()
    LOCAL = auto()
    PARAM = auto()
    RESULT = auto()


@dataclass
class Symbol:
    name: str
    type: SymbolType | IntType
    range: SymbolRange
    size: int = 1
    offset: int = 0
    label: str | None = None
    fields: list["Symbol"] = field(default_factory=list)


ARGUMENT_REGISTERS = (
    Register.A0,
    Register.B0,
    Register.A1,
    Register.B1,
    Register.A2,
    Register.B2,
    Register.A3,
    Register.B3,
)

RESULT_REGISTERS = (
    Register.A0,
    Register.A1,
    Register.A2,
    Register.A3,
)


class RegisterAllocator:
    def __init__(self, reporter: Reporter):
        self.reg_states: dict[Register, RegisterState] = {
            Register.R0: RegisterState.RESERVED,  # Always zero
            Register.A0: RegisterState.FREE,
            Register.A1: RegisterState.FREE,
            Register.A2: RegisterState.FREE,
            Register.A3: RegisterState.FREE,
            Register.B0: RegisterState.FREE,
            Register.B1: RegisterState.FREE,
            Register.B2: RegisterState.FREE,
            Register.B3: RegisterState.FREE,
            Register.G0: RegisterState.FREE,
            Register.G1: RegisterState.FREE,
            Register.G2: RegisterState.FREE,
            Register.G3: RegisterState.RESERVED,  # Frame pointer
            Register.FL: RegisterState.RESERVED,  # Flag register
            Register.SP: RegisterState.RESERVED,  # Stack pointer
            Register.PC: RegisterState.RESERVED,  # Program counter
        }
        self.r = reporter

    def reg_type(self, register: Register) -> RegisterType:
        if register in (Register.A0, Register.A1, Register.A2, Register.A3):
            return RegisterType.A
        if register in (Register.B0, Register.B1, Register.B2, Register.B3):
            return RegisterType.B
        if register in (Register.G0, Register.G1, Register.G2, Register.G3):
            return RegisterType.G
        return RegisterType.SPECIAL

    def reg_state(self, register: Register) -> RegisterState:
        return self.reg_states[register]

    def reg_name(self, register: Register) -> str:
        return register.name.lower()

    def reg_alloc(self, reg_type: RegisterType) -> Register:
        for register, state in self.reg_states.items():
            if self.reg_type(register) == reg_type and state == RegisterState.FREE:
                self.reg_states[register] = RegisterState.ALLOCATED
                return register
        self.r.error(f"No free {reg_type.name.lower()} registers.")
        raise SystemExit(1)

    def reg_free(self, register: Register):
        if register == Register.R0:
            return
        if self.reg_state(register) == RegisterState.ALLOCATED:
            self.reg_states[register] = RegisterState.FREE
            return
        self.r.error(f"Cannot free {self.reg_name(register)}.")
        raise SystemExit(1)

    def reg_reserve(self, register: Register):
        if self.reg_state(register) == RegisterState.FREE:
            self.reg_states[register] = RegisterState.RESERVED
            return
        self.r.error(f"Cannot reserve {self.reg_name(register)}.")
        raise SystemExit(1)

    def reg_unreserve(self, register: Register):
        if self.reg_state(register) == RegisterState.RESERVED:
            self.reg_states[register] = RegisterState.FREE
            return
        self.r.error(f"Cannot unreserve {self.reg_name(register)}.")
        raise SystemExit(1)


class CodeGenerator:
    def __init__(self, project_ast: dict[str, list[FileAst]], reporter: Reporter):
        self.project_ast = project_ast
        self.lines: list[str] = []
        self.r = reporter
        self.allocator = RegisterAllocator(reporter)
        self.id_counter = 0
        self.loop_stack: list[int] = []

    def emit(self, text: str = ""):
        self.lines.append(text)

    def statement_id(self) -> int:
        self.id_counter += 1
        return self.id_counter

    # ---------------

    def gen_function(self, package: str, function: Function):
        self.id_counter = 0
        self.emit(f"{package}__{function.name}:")

        self.gen_statement(function.body)

        if not self.lines or self.lines[-1] != "\tret":
            self.emit("\tret")
        self.emit()

    def gen_statement(self, statement: Statement):
        if isinstance(statement, Block):
            for s in statement.statements:
                self.gen_statement(s)

        elif isinstance(statement, Variable):
            pass

        elif isinstance(statement, Assign):
            pass

        elif isinstance(statement, Return):
            self.emit("\tret")

        elif isinstance(statement, If):
            id = self.statement_id()

            if statement.else_block:
                self.gen_condition(
                    statement.condition, false_jump_label=f".if_else_{id}"
                )
                self.gen_statement(statement.then_block)
                self.emit(f"\tjmp .if_end_{id}")
                self.emit(f".if_else_{id}:")
                self.gen_statement(statement.else_block)
                self.emit(f".if_end_{id}:")
            else:
                self.gen_condition(
                    statement.condition, false_jump_label=f".if_end_{id}"
                )
                self.gen_statement(statement.then_block)
                self.emit(f".if_end_{id}:")

        elif isinstance(statement, While):
            id = self.statement_id()
            self.loop_stack.append(id)
            self.emit(f".while_start_{id}:")
            self.gen_condition(statement.condition, false_jump_label=f".while_end_{id}")
            self.gen_statement(statement.body)
            self.emit(f"\tjmp .while_start_{id}")
            self.emit(f".while_end_{id}:")

            self.loop_stack.pop()

        elif isinstance(statement, Break):
            if not self.loop_stack:
                self.r.error("Cannot 'break' outside of a loop.")
                raise SystemExit(1)
            id = self.loop_stack[-1]
            self.emit(f"\tjmp .while_end_{id}")

        elif isinstance(statement, Continue):
            if not self.loop_stack:
                self.r.error("Cannot 'continue' outside of a loop.")
                raise SystemExit(1)
            id = self.loop_stack[-1]
            self.emit(f"\tjmp .while_start_{id}")

        elif isinstance(statement, ExprStatement):
            pass

        elif isinstance(statement, Asm):
            for line in statement.lines:
                self.emit(f"\t{line}")

    def gen_condition(self, expression: Expression, false_jump_label: str):
        if isinstance(expression, Operation2):
            if expression.op in (
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
            ):
                if expression.op == "<" or expression.op == "<=":
                    left_reg = self.allocator.reg_alloc(RegisterType.B)  # TODO
                    right_reg = self.allocator.reg_alloc(RegisterType.A)  # TODO
                else:
                    left_reg = self.allocator.reg_alloc(RegisterType.A)  # TODO
                    right_reg = self.allocator.reg_alloc(RegisterType.B)  # TODO

                left_name = self.allocator.reg_name(left_reg)
                right_name = self.allocator.reg_name(right_reg)

                if expression.op == "==":
                    self.emit(f"\tsub {left_name}, {right_name}, r0")
                    self.emit(f"\tjmp ne, {false_jump_label}")
                elif expression.op == "!=":
                    self.emit(f"\tsub {left_name}, {right_name}, r0")
                    self.emit(f"\tjmp zr, {false_jump_label}")
                elif expression.op == "<":
                    self.emit(f"\tsub {right_name}, {left_name}, r0")
                    self.emit(f"\tjmp zv, {false_jump_label}")
                elif expression.op == "<=":
                    self.emit(f"\tsub {right_name}, {left_name}, r0")
                    self.emit(f"\tjmp nv, {false_jump_label}")
                elif expression.op == ">":
                    self.emit(f"\tsub {left_name}, {right_name}, r0")
                    self.emit(f"\tjmp zv, {false_jump_label}")
                elif expression.op == ">=":
                    self.emit(f"\tsub {left_name}, {right_name}, r0")
                    self.emit(f"\tjmp nv, {false_jump_label}")

                self.allocator.reg_free(left_reg)
                self.allocator.reg_free(right_reg)
            else:
                self.r.error(f"Operation '{expression.op}' is not supported.")
                raise SystemExit(1)
        elif isinstance(expression, Operation1):
            self.r.error(f"Operation '{expression.op}' is not supported (yet).")  # TODO
            raise SystemExit(1)

    def gen_expression(
        self,
        expression: Expression,
        target_register_type: RegisterType = RegisterType.A,
    ) -> Register:
        if isinstance(expression, Number):
            reg = self.allocator.reg_alloc(target_register_type)
            self.emit(f"\tmov {expression.value}, {self.allocator.reg_name(reg)}")
            return reg

        elif isinstance(expression, Name):
            pass

        elif isinstance(expression, Operation1):
            pass

        elif isinstance(expression, Operation2):
            target: Register
            basic_ops = {
                "+": "add",
                "-": "sub",
                "&": "and",
                "|": "or",
                "^": "xor",
                "!&": "nand",
                "!|": "nor",
                "!^": "xnor",
            }
            shifts = {
                "<<": "sll",
                ">>": "srl",
                "<<<": "rol",
                ">>>": "ror",
                ">=>": "sra",
            }
            if expression.op in basic_ops:
                # TODO: optimization
                left_reg = self.gen_expression(expression.left, RegisterType.A)
                right_reg = self.gen_expression(expression.right, RegisterType.B)
                if target_register_type == RegisterType.A:
                    target = left_reg
                    self.allocator.reg_free(right_reg)
                elif target_register_type == RegisterType.B:
                    target = right_reg
                    self.allocator.reg_free(left_reg)
                else:
                    target = self.allocator.reg_alloc(target_register_type)
                    self.allocator.reg_free(left_reg)
                    self.allocator.reg_free(right_reg)
                self.emit(
                    f"\t{basic_ops[expression.op]} {self.allocator.reg_name(left_reg)}, {self.allocator.reg_name(right_reg)}, {self.allocator.reg_name(target)}"
                )
                return target
            elif expression.op in shifts:
                if not isinstance(expression.right, Number):
                    self.r.error("Dynamic shifts is not supported (yet).")  # TODO
                    raise SystemExit(1)
                if expression.right.value >= 16:
                    self.r.error("You can't shift by 16 bits or more.")
                    raise SystemExit(1)
                if expression.right.value == 0:
                    reg = self.gen_expression(expression.left, target_register_type)
                    return reg
                if target_register_type in (RegisterType.A, RegisterType.B):
                    reg = self.gen_expression(expression.left, target_register_type)
                    target = reg
                else:
                    reg = self.gen_expression(expression.left, RegisterType.A)
                    target = self.allocator.reg_alloc(target_register_type)
                if expression.op == "<<<" or expression.op == ">>>":
                    for _ in range(1, expression.right.value):
                        self.emit(
                            f"\t{shifts[expression.op]} {self.allocator.reg_name(reg)}, {self.allocator.reg_name(reg)}"
                        )
                    self.emit(
                        f"\t{shifts[expression.op]} {self.allocator.reg_name(reg)}, {self.allocator.reg_name(target)}"
                    )
                elif expression.right.value == 8:
                    self.emit(
                        f"\t{shifts[expression.op]}8 {self.allocator.reg_name(reg)}, {self.allocator.reg_name(target)}"
                    )
                elif expression.right.value > 8:
                    self.emit(
                        f"\t{shifts[expression.op]}8 {self.allocator.reg_name(reg)}, {self.allocator.reg_name(reg)}"
                    )
                    for _ in range(1, expression.right.value - 8):
                        self.emit(
                            f"\t{shifts[expression.op]} {self.allocator.reg_name(reg)}, {self.allocator.reg_name(reg)}"
                        )
                    self.emit(
                        f"\t{shifts[expression.op]} {self.allocator.reg_name(reg)}, {self.allocator.reg_name(target)}"
                    )
                else:
                    for _ in range(1, expression.right.value):
                        self.emit(
                            f"\t{shifts[expression.op]} {self.allocator.reg_name(reg)}, {self.allocator.reg_name(reg)}"
                        )
                    self.emit(
                        f"\t{shifts[expression.op]} {self.allocator.reg_name(reg)}, {self.allocator.reg_name(target)}"
                    )
                if target != reg:
                    self.allocator.reg_free(reg)
                return target

        elif isinstance(expression, Call):
            pass

        elif isinstance(expression, MemberAccess):
            pass

        elif isinstance(expression, Index):
            pass

        elif isinstance(expression, ArrayValue):
            pass

        elif isinstance(expression, StructValue):
            pass

        return Register.R0

    # ---------------

    def generate_asm_code(self) -> str:
        self.emit("; Generated by Blim for MANIAC 1.0")
        self.emit()

        interrupt_map: dict[int, str] = {}
        for files_ast in self.project_ast.values():
            for file_ast in files_ast:
                for vec in file_ast.interrupt_vectors:
                    interrupt_map[vec.vector_number] = (
                        f"{file_ast.package}__{vec.func_name}"
                    )

        if 0 not in interrupt_map:
            self.r.error("Interrupt 0 is unhandled.")
            raise SystemExit(1)

        self.emit("; ------ INTERRUPT VECTOR TABLE ------")
        for i in range(max(15, max(interrupt_map.keys())) + 1):
            if i == 16:
                self.emit("")
                self.emit("; ------- SYSCALL VECTOR TABLE -------")
            if i in interrupt_map:
                self.emit(f"\t#d16 {interrupt_map[i]} ; Vector {i}")
            else:
                self.emit(f"\t#d16 unhandled ; Vector {i}")
        self.emit()

        self.emit("unhandled:")
        self.emit("\tret")
        self.emit()

        self.emit("; --------------- CODE ---------------")
        for files_ast in self.project_ast.values():
            for file_ast in files_ast:
                for function in file_ast.functions:
                    self.gen_function(file_ast.package, function)

        return "\n".join(self.lines)
