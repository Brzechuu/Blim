from enum import IntEnum

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
    A0 = R1 = 1
    A1 = R2 = 2
    A2 = R3 = 3
    A3 = R4 = 4
    B0 = R5 = 5
    B1 = R6 = 6
    B2 = R7 = 7
    B3 = R8 = 8
    G0 = R9 = 9
    G1 = R10 = 10
    G2 = R11 = 11
    G3 = R12 = 12
    FL = R13 = 13
    SP = R14 = 14
    PC = R15 = 15


class RegisterState(IntEnum):
    FREE = 0
    ALLOCATED = 1
    RESERVED = 2


class RegisterType(IntEnum):
    A = 0
    B = 1
    G = 2
    SPECIAL = 3


class RegisterAllocator:
    def __init__(self, reporter: Reporter):
        self.reg_states: dict[Register, RegisterState] = {
            Register.R0: RegisterState.RESERVED,
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
            Register.G3: RegisterState.FREE,
            Register.FL: RegisterState.RESERVED,
            Register.SP: RegisterState.RESERVED,
            Register.PC: RegisterState.RESERVED,
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

    def emit_instruction(self, instruction: str = ""):
        self.lines.append(f"\t{instruction}")

    def emit_label(self, label: str):
        self.lines.append(f"{label}:")

    def emit_raw(self, text: str = ""):
        self.lines.append(text)

    # ---------------

    def gen_function(self, package: str, function: Function):
        self.emit_label(f"{package}__{function.name}")
        self.gen_statement(function.body)
        if self.lines[-1] != "\tret":
            self.emit_instruction("ret")
        self.emit_raw()

    def gen_statement(self, statement: Statement):
        if isinstance(statement, Block):
            for s in statement.statements:
                self.gen_statement(s)

        elif isinstance(statement, Variable):
            pass

        elif isinstance(statement, Assign):
            pass

        elif isinstance(statement, Return):
            self.emit_instruction("ret")

        elif isinstance(statement, If):
            pass

        elif isinstance(statement, While):
            pass

        elif isinstance(statement, Break):
            pass

        elif isinstance(statement, Continue):
            pass

        elif isinstance(statement, ExprStatement):
            pass

        elif isinstance(statement, Asm):
            for line in statement.lines:
                self.emit_instruction(line)

    def gen_expression(self, expression: Expression) -> Register:
        if isinstance(expression, Number):
            pass

        elif isinstance(expression, Name):
            pass

        elif isinstance(expression, Operation1):
            pass

        elif isinstance(expression, Operation2):
            pass

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
        self.emit_raw("; Generated by Blim for MANIAC 1.0")
        self.emit_raw()
        self.emit_label("START")
        self.emit_instruction("jmp main__main")
        self.emit_raw()

        for files_ast in self.project_ast.values():
            for file_ast in files_ast:
                for function in file_ast.functions:
                    self.gen_function(file_ast.package, function)

        return "\n".join(self.lines)
