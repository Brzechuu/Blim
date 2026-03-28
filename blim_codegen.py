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
    MemberAccessType,
    Name,
    Number,
    Operation1,
    Operation2,
    Return,
    Statement,
    StructValue,
    Type,
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


class SymbolRange(Enum):
    GLOBAL = auto()
    LOCAL = auto()
    PARAM = auto()
    RESULT = auto()


@dataclass
class Symbol:
    name: str
    type: Type
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
        self.loop_stack: list[tuple[int, int]] = []

        self.scopes: list[dict[str, Symbol]] = []
        self.current_package: str = ""
        self.current_stack_offset: int = 0

    def emit(self, text: str = ""):
        self.lines.append(text)

    def statement_id(self) -> int:
        self.id_counter += 1
        return self.id_counter

    def is_array(self, var_type: Type) -> bool:
        return var_type.array_size is not None

    def is_struct(self, var_type: Type) -> bool:
        return isinstance(var_type.base_type, str) and var_type.pointer_depth == 0

    def is_pointer(self, var_type: Type) -> bool:
        return var_type.pointer_depth > 0

    def adjust_sp(self, delta: int):
        if delta == 0:
            return

        tmp_a = self.allocator.reg_alloc(RegisterType.A)
        tmp_b = self.allocator.reg_alloc(RegisterType.B)

        self.emit(f"\tmov sp, {self.allocator.reg_name(tmp_a)}")
        self.emit(f"\tmov {abs(delta)}, {self.allocator.reg_name(tmp_b)}")

        if delta > 0:
            self.emit(
                f"\tadd {self.allocator.reg_name(tmp_a)}, {self.allocator.reg_name(tmp_b)}, sp"
            )
        else:
            self.emit(
                f"\tsub {self.allocator.reg_name(tmp_a)}, {self.allocator.reg_name(tmp_b)}, sp"
            )

        self.allocator.reg_free(tmp_a)
        self.allocator.reg_free(tmp_b)

    def unwind_stack_to(self, target_offset: int):
        leak_size = target_offset - self.current_stack_offset
        if leak_size > 0:
            self.adjust_sp(leak_size)

    def resolve_type_size(self, var_type: Type, current_package: str) -> int:
        base_size = 1

        if isinstance(var_type.base_type, IntType):
            base_size = 1
        elif isinstance(var_type.base_type, str):
            struct_name = var_type.base_type
            target_package = current_package
            if "." in struct_name:
                target_package, struct_name = struct_name.split(".", 1)

            found_struct = False
            if target_package in self.project_ast:
                for file_ast in self.project_ast[target_package]:
                    for struct_def in file_ast.structures:
                        if struct_def.name == struct_name:
                            base_size = sum(
                                self.resolve_type_size(f.type, target_package)
                                for f in struct_def.fields
                            )
                            found_struct = True
                            break
                    if found_struct:
                        break
            if not found_struct:
                self.r.error(f"Unknown type: {var_type.base_type}")
                raise SystemExit(1)

        if var_type.array_size:
            if isinstance(var_type.array_size, Number):
                return base_size * var_type.array_size.value
            else:
                self.r.error("Array size must be a constant number.")
                raise SystemExit(1)

        return base_size

    def get_struct_field_offset(
        self, struct_name: str, field_name: str, current_package: str
    ) -> int:
        target_package = current_package
        if "." in struct_name:
            target_package, struct_name = struct_name.split(".", 1)

        if target_package in self.project_ast:
            for file_ast in self.project_ast[target_package]:
                for struct_def in file_ast.structures:
                    if struct_def.name == struct_name:
                        current_offset = 0
                        for field in struct_def.fields:
                            if field.name == field_name:
                                return current_offset
                            current_offset += self.resolve_type_size(
                                field.type, target_package
                            )

                        self.r.error(
                            f"Field '{field_name}' not found in struct '{struct_name}'"
                        )
                        raise SystemExit(1)

        self.r.error(f"Unknown struct type: {struct_name}")
        raise SystemExit(1)

    def get_struct_field_type(
        self, struct_name: str, field_name: str, current_package: str
    ) -> Type:
        target_package = current_package
        if "." in struct_name:
            target_package, struct_name = struct_name.split(".", 1)

        if target_package in self.project_ast:
            for file_ast in self.project_ast[target_package]:
                for struct_def in file_ast.structures:
                    if struct_def.name == struct_name:
                        for field in struct_def.fields:
                            if field.name == field_name:
                                return field.type

                        self.r.error(
                            f"Field '{field_name}' not found in struct '{struct_name}'"
                        )
                        raise SystemExit(1)

        self.r.error(f"Unknown struct type: {struct_name}")
        raise SystemExit(1)

    def lookup_variable(self, name: str) -> Symbol:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        self.r.error(f"Undefined variable '{name}'")
        raise SystemExit(1)

    def get_expression_type(self, expression: Expression) -> Type:
        if isinstance(expression, Number):
            return Type(
                line=expression.line, column=expression.column, base_type=IntType.I16
            )

        if isinstance(expression, Name):
            return self.lookup_variable(expression.value).type

        if (
            isinstance(expression, MemberAccess)
            and expression.type == MemberAccessType.FIELD
        ):
            if not isinstance(expression.value, Name):
                self.r.error("Complex member access bases not supported (yet).")  # TODO
                raise SystemExit(1)

            base_symbol = self.lookup_variable(expression.value.value)
            base_type = base_symbol.type

            if not self.is_struct(base_type):
                self.r.error(f"Variable '{expression.value.value}' is not a struct")
                raise SystemExit(1)

            struct_name = base_type.base_type
            if not isinstance(struct_name, str):
                self.r.error("Invalid struct type encountered.")
                raise SystemExit(1)

            return self.get_struct_field_type(
                struct_name, expression.member, self.current_package
            )

        if isinstance(expression, (Operation1, Operation2)):
            inner_expr = getattr(expression, "value", getattr(expression, "left", None))
            if inner_expr:
                return self.get_expression_type(inner_expr)

        self.r.error("Unsupported expression for type resolution.")
        raise SystemExit(1)

    def get_memory_offset(self, expression: Expression) -> int:
        if isinstance(expression, Name):
            return self.lookup_variable(expression.value).offset
        elif (
            isinstance(expression, MemberAccess)
            and expression.type == MemberAccessType.FIELD
        ):
            base_offset = self.get_memory_offset(expression.value)

            if isinstance(expression.value, Name):
                base_name = expression.value.value
                symbol = self.lookup_variable(base_name)
                if self.is_struct(symbol.type):
                    struct_name = symbol.type.base_type
                    if isinstance(struct_name, str):
                        field_offset = self.get_struct_field_offset(
                            struct_name, expression.member, self.current_package
                        )
                        return base_offset + field_offset
                    self.r.error(f"Variable '{base_name}' is not a struct")
                    raise SystemExit(1)
                else:
                    self.r.error(f"Variable '{base_name}' is not a struct")
                    raise SystemExit(1)
            else:
                self.r.error("Complex member access bases not supported (yet).")  # TODO
                raise SystemExit(1)
        else:
            self.r.error("Invalid left value")
            raise SystemExit(1)

    # ---------------

    def gen_function(self, package: str, function: Function):
        self.id_counter = 0
        self.scopes.clear()
        self.current_stack_offset = 0
        self.current_package = package

        self.emit(f"{package}__{function.name}:")
        self.emit("\tpush g3")
        self.emit("\tmov sp, g3")

        self.gen_statement(function.body)

        self.emit(" .return:")
        self.emit("\tmov g3, sp")
        self.emit("\tpop g3")
        self.emit("\tret")
        self.emit()

    def gen_statement(self, statement: Statement):
        if isinstance(statement, Block):
            self.scopes.append({})

            block_local_size = 0
            start_offset = self.current_stack_offset

            for s in statement.statements:
                if isinstance(s, Variable):
                    if s.name in self.scopes[-1]:
                        self.r.error(
                            f"Duplicate declaration of '{s.name}' in current scope."
                        )
                        raise SystemExit(1)

                    size = self.resolve_type_size(s.type, self.current_package)
                    start_offset -= size

                    self.scopes[-1][s.name] = Symbol(
                        name=s.name,
                        type=s.type,
                        range=SymbolRange.LOCAL,
                        size=size,
                        offset=start_offset,
                    )
                    block_local_size += size

            self.current_stack_offset -= block_local_size
            self.adjust_sp(-block_local_size)

            for s in statement.statements:
                self.gen_statement(s)

            self.scopes.pop()
            self.adjust_sp(block_local_size)
            self.current_stack_offset += block_local_size

        elif isinstance(statement, Variable):
            if statement.value:
                symbol = self.scopes[-1][statement.name]
                if self.is_array(symbol.type):
                    self.r.error(f"Variable '{statement.name}' cannot be an array.")
                    raise SystemExit(1)
                if self.is_struct(symbol.type):
                    self.r.error(
                        f"Variable '{statement.name}' cannot be a struct value."
                    )
                    raise SystemExit(1)

                value_type = self.get_expression_type(statement.value)
                if self.is_array(value_type):
                    self.r.error("Initializer cannot be an array.")
                    raise SystemExit(1)
                if self.is_struct(value_type):
                    self.r.error("Initializer cannot be a struct value.")
                    raise SystemExit(1)

                value_reg = self.gen_expression(statement.value, RegisterType.B)
                base_reg = self.allocator.reg_alloc(RegisterType.A)
                offset_reg = self.allocator.reg_alloc(RegisterType.B)

                self.emit(f"\tmov g3, {self.allocator.reg_name(base_reg)}")

                if symbol.offset == 0:
                    self.emit(f"\tmov r0, {self.allocator.reg_name(offset_reg)}")
                else:
                    self.emit(
                        f"\tmov {symbol.offset}, {self.allocator.reg_name(offset_reg)}"
                    )

                self.emit(
                    f"\tadd {self.allocator.reg_name(base_reg)}, {self.allocator.reg_name(offset_reg)}, {self.allocator.reg_name(base_reg)}"
                )
                self.emit(
                    f"\tstore {self.allocator.reg_name(value_reg)}, [{self.allocator.reg_name(base_reg)}]"
                )

                self.allocator.reg_free(value_reg)
                self.allocator.reg_free(base_reg)
                self.allocator.reg_free(offset_reg)

        elif isinstance(statement, Assign):
            if statement.value:
                target_expr = statement.targets[0]
                if isinstance(target_expr, (Name, MemberAccess)):
                    target_type = self.get_expression_type(target_expr)
                    if self.is_array(target_type):
                        self.r.error("Assignment target cannot be an array.")
                        raise SystemExit(1)
                    if self.is_struct(target_type):
                        self.r.error("Assignment target cannot be a struct value.")
                        raise SystemExit(1)

                    value_type = self.get_expression_type(statement.value)
                    if self.is_array(value_type):
                        self.r.error("Assigned value cannot be an array.")
                        raise SystemExit(1)
                    if self.is_struct(value_type):
                        self.r.error("Assigned value cannot be a struct value.")
                        raise SystemExit(1)

                    value_reg = self.gen_expression(statement.value, RegisterType.B)
                    final_offset = self.get_memory_offset(target_expr)

                    base_reg = self.allocator.reg_alloc(RegisterType.A)
                    offset_reg = self.allocator.reg_alloc(RegisterType.B)
                    self.emit(f"\tmov g3, {self.allocator.reg_name(base_reg)}")
                    if final_offset == 0:
                        self.emit(f"\tmov r0, {self.allocator.reg_name(offset_reg)}")
                    else:
                        self.emit(
                            f"\tmov {final_offset}, {self.allocator.reg_name(offset_reg)}"
                        )
                    self.emit(
                        f"\tadd {self.allocator.reg_name(base_reg)}, {self.allocator.reg_name(offset_reg)}, {self.allocator.reg_name(base_reg)}"
                    )
                    self.emit(
                        f"\tstore {self.allocator.reg_name(value_reg)}, [{self.allocator.reg_name(base_reg)}]"
                    )
                    self.allocator.reg_free(base_reg)
                    self.allocator.reg_free(offset_reg)
                    self.allocator.reg_free(value_reg)
                else:
                    self.r.error(
                        "Assignments to complex structures are not supported (yet)."
                    )
                    raise SystemExit(1)

        elif isinstance(statement, Return):
            self.emit("\tjmp .return")

        elif isinstance(statement, If):
            id = self.statement_id()
            self.emit(f" .if_start_{id}:")
            if statement.else_block:
                self.gen_condition(
                    statement.condition, false_jump_label=f".if_else_{id}"
                )
                self.gen_statement(statement.then_block)
                self.emit(f"\tjmp .if_end_{id}")
                self.emit(f" .if_else_{id}:")
                self.gen_statement(statement.else_block)
                self.emit(f" .if_end_{id}:")
            else:
                self.gen_condition(
                    statement.condition, false_jump_label=f".if_end_{id}"
                )
                self.gen_statement(statement.then_block)
                self.emit(f" .if_end_{id}:")

        elif isinstance(statement, While):
            id = self.statement_id()
            self.loop_stack.append((id, self.current_stack_offset))
            self.emit(f" .while_start_{id}:")
            self.gen_condition(statement.condition, false_jump_label=f".while_end_{id}")
            self.gen_statement(statement.body)
            self.emit(f"\tjmp .while_start_{id}")
            self.emit(f" .while_end_{id}:")

            self.loop_stack.pop()

        elif isinstance(statement, Break):
            if not self.loop_stack:
                self.r.error("Cannot 'break' outside of a loop.")
                raise SystemExit(1)
            id, loop_start_offset = self.loop_stack[-1]
            self.unwind_stack_to(loop_start_offset)
            self.emit(f"\tjmp .while_end_{id}")

        elif isinstance(statement, Continue):
            if not self.loop_stack:
                self.r.error("Cannot 'continue' outside of a loop.")
                raise SystemExit(1)
            id, loop_start_offset = self.loop_stack[-1]
            self.unwind_stack_to(loop_start_offset)
            self.emit(f"\tjmp .while_start_{id}")

        elif isinstance(statement, ExprStatement):
            result_reg = self.gen_expression(statement.value)
            self.allocator.reg_free(result_reg)

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
                    left_reg = self.gen_expression(expression.left, RegisterType.B)
                    right_reg = self.gen_expression(expression.right, RegisterType.A)
                else:
                    left_reg = self.gen_expression(expression.left, RegisterType.A)
                    right_reg = self.gen_expression(expression.right, RegisterType.B)

                if expression.op == "==":
                    self.emit(
                        f"\tsub {self.allocator.reg_name(left_reg)}, {self.allocator.reg_name(right_reg)}, r0"
                    )
                    self.emit(f"\tjmp ne, {false_jump_label}")
                elif expression.op == "!=":
                    self.emit(
                        f"\tsub {self.allocator.reg_name(left_reg)}, {self.allocator.reg_name(right_reg)}, r0"
                    )
                    self.emit(f"\tjmp zr, {false_jump_label}")
                elif expression.op == "<":
                    self.emit(
                        f"\tsub {self.allocator.reg_name(right_reg)}, {self.allocator.reg_name(left_reg)}, r0"
                    )
                    self.emit(f"\tjmp zv, {false_jump_label}")
                elif expression.op == "<=":
                    self.emit(
                        f"\tsub {self.allocator.reg_name(right_reg)}, {self.allocator.reg_name(left_reg)}, r0"
                    )
                    self.emit(f"\tjmp nv, {false_jump_label}")
                elif expression.op == ">":
                    self.emit(
                        f"\tsub {self.allocator.reg_name(left_reg)}, {self.allocator.reg_name(right_reg)}, r0"
                    )
                    self.emit(f"\tjmp zv, {false_jump_label}")
                elif expression.op == ">=":
                    self.emit(
                        f"\tsub {self.allocator.reg_name(left_reg)}, {self.allocator.reg_name(right_reg)}, r0"
                    )
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
            if expression.value == 0:
                self.emit(f"\tmov r0, {self.allocator.reg_name(reg)}")
            else:
                self.emit(f"\tmov {expression.value}, {self.allocator.reg_name(reg)}")
            return reg

        elif isinstance(expression, (Name, MemberAccess)):
            if (
                isinstance(expression, MemberAccess)
                and expression.type == MemberAccessType.PACKAGE
            ):
                self.r.error("Package member access is not supported (yet).")  # TODO
                raise SystemExit(1)

            expr_type = self.get_expression_type(expression)
            if self.is_array(expr_type):
                self.r.error("Expression cannot be an array.")
                raise SystemExit(1)
            if self.is_struct(expr_type):
                self.r.error("Expression cannot be a struct value.")
                raise SystemExit(1)

            final_offset = self.get_memory_offset(expression)

            base_reg = self.allocator.reg_alloc(RegisterType.A)
            offset_reg = self.allocator.reg_alloc(RegisterType.B)

            self.emit(f"\tmov g3, {self.allocator.reg_name(base_reg)}")

            if final_offset == 0:
                self.emit(f"\tmov r0, {self.allocator.reg_name(offset_reg)}")
            else:
                self.emit(
                    f"\tmov {final_offset}, {self.allocator.reg_name(offset_reg)}"
                )

            self.emit(
                f"\tadd {self.allocator.reg_name(base_reg)}, {self.allocator.reg_name(offset_reg)}, {self.allocator.reg_name(base_reg)}"
            )

            target = self.allocator.reg_alloc(target_register_type)
            self.emit(
                f"\tload [{self.allocator.reg_name(base_reg)}], {self.allocator.reg_name(target)}"
            )

            self.allocator.reg_free(base_reg)
            self.allocator.reg_free(offset_reg)

            return target

        elif isinstance(expression, Operation1):
            if expression.op == "!":
                if target_register_type in (RegisterType.A, RegisterType.B):
                    reg = self.gen_expression(expression.value, target_register_type)
                else:
                    reg = self.gen_expression(expression.value, RegisterType.A)

                self.emit(
                    f"\tnot {self.allocator.reg_name(reg)}, {self.allocator.reg_name(reg)}"
                )
                return reg

            elif expression.op == "-":
                value_type = self.get_expression_type(expression.value)
                if self.is_array(value_type):
                    self.r.error("Operand '-' cannot be an array.")
                    raise SystemExit(1)
                if self.is_struct(value_type):
                    self.r.error("Operand '-' cannot be a struct value.")
                    raise SystemExit(1)

                value_reg = self.gen_expression(expression.value, RegisterType.B)
                zero_reg = self.allocator.reg_alloc(RegisterType.A)

                self.emit(f"\tmov r0, {self.allocator.reg_name(zero_reg)}")
                self.emit(
                    f"\tsub {self.allocator.reg_name(zero_reg)}, {self.allocator.reg_name(value_reg)}, {self.allocator.reg_name(zero_reg)}"
                )

                self.allocator.reg_free(value_reg)
                return zero_reg

            else:
                self.r.error(f"Operation '{expression.op}' is not supported.")
                raise SystemExit(1)

        elif isinstance(expression, Operation2):
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
                left_type = self.get_expression_type(expression.left)
                right_type = self.get_expression_type(expression.right)
                if self.is_array(left_type):
                    self.r.error(
                        f"Left operand of '{expression.op}' cannot be an array."
                    )
                    raise SystemExit(1)
                if self.is_struct(left_type):
                    self.r.error(
                        f"Left operand of '{expression.op}' cannot be a struct value."
                    )
                    raise SystemExit(1)
                if self.is_array(right_type):
                    self.r.error(
                        f"Right operand of '{expression.op}' cannot be an array."
                    )
                    raise SystemExit(1)
                if self.is_struct(right_type):
                    self.r.error(
                        f"Right operand of '{expression.op}' cannot be a struct value."
                    )
                    raise SystemExit(1)

                left_reg = self.gen_expression(expression.left, RegisterType.A)
                right_reg = self.gen_expression(expression.right, RegisterType.B)
                if target_register_type == RegisterType.A:
                    target = left_reg
                    to_free = [right_reg]
                elif target_register_type == RegisterType.B:
                    target = right_reg
                    to_free = [left_reg]
                else:
                    target = self.allocator.reg_alloc(target_register_type)
                    to_free = [left_reg, right_reg]

                self.emit(
                    f"\t{basic_ops[expression.op]} {self.allocator.reg_name(left_reg)}, {self.allocator.reg_name(right_reg)}, {self.allocator.reg_name(target)}"
                )
                for reg in to_free:
                    if reg != target:
                        self.allocator.reg_free(reg)

                return target
            elif expression.op in shifts:
                left_type = self.get_expression_type(expression.left)
                if self.is_array(left_type):
                    self.r.error(
                        f"Left operand of '{expression.op}' cannot be an array."
                    )
                    raise SystemExit(1)
                if self.is_struct(left_type):
                    self.r.error(
                        f"Left operand of '{expression.op}' cannot be a struct value."
                    )
                    raise SystemExit(1)

                if not isinstance(expression.right, Number):
                    self.r.error("Dynamic shifts is not supported (yet).")  # TODO
                    raise SystemExit(1)
                if expression.right.value >= 16:
                    self.r.error("You can't shift by 16 bits or more.")
                    raise SystemExit(1)
                if expression.right.value == 0:
                    return self.gen_expression(expression.left, target_register_type)

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

            else:
                self.r.error(f"Operation '{expression.op}' is not supported.")
                raise SystemExit(1)

        elif isinstance(expression, Call):
            self.r.error("Call is not supported (yet).")  # TODO
            raise SystemExit(1)

        elif isinstance(expression, Index):
            self.r.error("Index is not supported (yet).")  # TODO
            raise SystemExit(1)

        elif isinstance(expression, ArrayValue):
            self.r.error("ArrayValue is not supported (yet).")  # TODO
            raise SystemExit(1)

        elif isinstance(expression, StructValue):
            self.r.error("StructValue is not supported (yet).")  # TODO
            raise SystemExit(1)

        self.r.error("Unsupported expression.")
        raise SystemExit(1)

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
                self.emit(f"#d16 {interrupt_map[i]} ; Vector {i}")
            else:
                self.emit(f"#d16 unhandled ; Vector {i}")
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
