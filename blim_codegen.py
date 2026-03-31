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


@dataclass
class MemAddress:
    register: Register | None = None
    label: str | None = None


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
    register: Register | None = None


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
        self.locked_regs: set[Register] = set()
        self.r = reporter

    def reg_lock(self, register: Register):
        if register in self.locked_regs:
            return
        if self.reg_state(register) != RegisterState.ALLOCATED:
            self.r.error(f"Cannot lock {self.reg_name(register)} – not allocated.")
            raise SystemExit(1)
        self.locked_regs.add(register)

    def reg_unlock(self, register: Register):
        if register not in self.locked_regs:
            self.r.error(f"Cannot unlock {self.reg_name(register)} – not locked.")
            raise SystemExit(1)
        self.locked_regs.remove(register)

    def reg_unlock_and_reset_states(self):
        for reg in self.reg_states:
            if reg in (Register.G3, Register.FL, Register.SP, Register.PC, Register.R0):
                self.reg_states[reg] = RegisterState.RESERVED
            else:
                self.reg_states[reg] = RegisterState.FREE
        self.locked_regs.clear()

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

    def reg_alloc_specific(self, register: Register):
        if self.reg_state(register) == RegisterState.FREE:
            self.reg_states[register] = RegisterState.ALLOCATED
            return
        self.r.error(
            f"Cannot allocate specific register {self.reg_name(register)} – it is already {self.reg_state(register).name}."
        )
        raise SystemExit(1)

    def reg_free(self, register: Register):
        if register == Register.R0:
            return
        if register in self.locked_regs:
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
        self.package_globals: dict[str, dict[str, Symbol]] = {}

    def emit(self, text: str = ""):
        self.lines.append(text)

    def statement_id(self) -> int:
        self.id_counter += 1
        return self.id_counter

    def type_name(self, var_type: Type) -> str:
        if isinstance(var_type.base_type, str):
            base = var_type.base_type
        else:
            base = var_type.base_type.name.lower()

        if var_type.pointer_depth > 0:
            base = "*" * var_type.pointer_depth + base

        if var_type.array_size is not None:
            if isinstance(var_type.array_size, Number):
                base += f"[{var_type.array_size.value}]"
            else:
                base += "[]"

        return base

    def is_array(self, var_type: Type) -> bool:
        return var_type.array_size is not None

    def is_struct(self, var_type: Type) -> bool:
        return isinstance(var_type.base_type, str) and var_type.pointer_depth == 0

    def is_pointer(self, var_type: Type) -> bool:
        return var_type.pointer_depth > 0

    def same_array_size(self, a: Type, b: Type) -> bool:
        if a.array_size is None and b.array_size is None:
            return True
        if a.array_size is None or b.array_size is None:
            return False
        if isinstance(a.array_size, Number) and isinstance(b.array_size, Number):
            return a.array_size.value == b.array_size.value
        return False

    def same_type(self, a: Type, b: Type) -> bool:
        return (
            a.base_type == b.base_type
            and a.pointer_depth == b.pointer_depth
            and self.same_array_size(a, b)
        )

    def is_integer(self, var_type: Type) -> bool:
        return (
            isinstance(var_type.base_type, IntType)
            and var_type.pointer_depth == 0
            and var_type.array_size is None
        )

    def registers_for_type(self, reg_type: RegisterType) -> tuple[Register, ...]:
        if reg_type == RegisterType.A:
            return (Register.A0, Register.A1, Register.A2, Register.A3)
        if reg_type == RegisterType.B:
            return (Register.B0, Register.B1, Register.B2, Register.B3)
        if reg_type == RegisterType.G:
            return (Register.G0, Register.G1, Register.G2, Register.G3)
        return (Register.R0, Register.FL, Register.SP, Register.PC)

    def alloc_temp_register(
        self,
        preferred_types: tuple[RegisterType, ...],
        avoid: set[Register] | None = None,
    ) -> Register:
        if avoid is None:
            avoid = set()

        for reg_type in preferred_types:
            for register in self.registers_for_type(reg_type):
                if register in avoid:
                    continue
                if self.allocator.reg_state(register) == RegisterState.FREE:
                    self.allocator.reg_alloc_specific(register)
                    return register

        pref = ", ".join(rt.name.lower() for rt in preferred_types)
        self.r.error(f"No free temporary register available ({pref}).")
        raise SystemExit(1)

    def find_free_register_in_state(
        self,
        reg_states: dict[Register, RegisterState],
        reg_type: RegisterType,
    ) -> Register:
        for register in self.registers_for_type(reg_type):
            if reg_states[register] == RegisterState.FREE:
                return register
        self.r.error(
            f"No free {reg_type.name.lower()} register available for call result."
        )
        raise SystemExit(1)

    def save_allocator_state(
        self,
    ) -> tuple[dict[Register, RegisterState], set[Register]]:
        return dict(self.allocator.reg_states), set(self.allocator.locked_regs)

    def restore_allocator_state(
        self,
        reg_states: dict[Register, RegisterState],
        locked_regs: set[Register],
    ):
        self.allocator.reg_states = dict(reg_states)
        self.allocator.locked_regs = set(locked_regs)

    def lookup_function(self, expression: Expression) -> tuple[str, Function]:
        if isinstance(expression, Name):
            func_name = expression.value
            for file_ast in self.project_ast.get(self.current_package, []):
                for function in file_ast.functions:
                    if function.name == func_name:
                        return self.current_package, function

            self.r.error(f"Undefined function '{func_name}'.")
            raise SystemExit(1)

        if (
            isinstance(expression, MemberAccess)
            and expression.type == MemberAccessType.PACKAGE
            and isinstance(expression.value, Name)
        ):
            target_package = expression.value.value
            func_name = expression.member

            for file_ast in self.project_ast.get(target_package, []):
                for function in file_ast.functions:
                    if function.name == func_name:
                        return target_package, function

            self.r.error(
                f"Undefined function '{func_name}' in package '{target_package}'."
            )
            raise SystemExit(1)

        self.r.error("Unsupported function call target.")
        raise SystemExit(1)

    def unify_int_types(
        self,
        left_expr: Expression,
        left_type: Type,
        right_expr: Expression,
        right_type: Type,
        context: str,
    ) -> Type:
        if not self.is_integer(left_type):
            self.r.error(
                f"Left operand of {context} must be an integer, got {self.type_name(left_type)}."
            )
            raise SystemExit(1)

        if not self.is_integer(right_type):
            self.r.error(
                f"Right operand of {context} must be an integer, got {self.type_name(right_type)}."
            )
            raise SystemExit(1)

        if left_type.base_type == right_type.base_type:
            return left_type

        if isinstance(left_expr, Number):
            return right_type

        if isinstance(right_expr, Number):
            return left_type

        self.r.error(
            f"Type mismatch in {context}: {self.type_name(left_type)} vs {self.type_name(right_type)}."
        )
        raise SystemExit(1)

    def ensure_assignable(
        self,
        target_type: Type,
        value_expr: Expression,
        value_type: Type,
        context: str,
    ):
        if self.is_array(target_type):
            self.r.error(f"{self.type_name(target_type)} target cannot be an array.")
            raise SystemExit(1)

        if self.is_struct(target_type):
            self.r.error(
                f"{self.type_name(target_type)} target cannot be a struct value."
            )
            raise SystemExit(1)

        if self.is_array(value_type):
            self.r.error(f"{self.type_name(value_type)} value cannot be an array.")
            raise SystemExit(1)

        if self.is_struct(value_type):
            self.r.error(
                f"{self.type_name(value_type)} value cannot be a struct value."
            )
            raise SystemExit(1)

        if self.same_type(target_type, value_type):
            return

        if self.is_integer(target_type) and self.is_integer(value_type):
            if isinstance(value_expr, Number):
                return

        self.r.error(
            f"Type mismatch in {context}: cannot assign {self.type_name(value_type)} to {self.type_name(target_type)}."
        )
        raise SystemExit(1)

    def push_register(self, register: Register):
        self.emit(f"\tpush {self.allocator.reg_name(register)}")

    def pop_register(self, register: Register):
        self.emit(f"\tpop {self.allocator.reg_name(register)}")

    def spill_registers_for_call(self) -> list[Register]:
        spilled: list[Register] = []
        for reg in (
            Register.A0,
            Register.A1,
            Register.A2,
            Register.A3,
            Register.B0,
            Register.B1,
            Register.B2,
            Register.B3,
            Register.G0,
            Register.G1,
            Register.G2,
        ):
            if self.allocator.reg_state(reg) == RegisterState.ALLOCATED:
                spilled.append(reg)

        for reg in spilled:
            self.push_register(reg)

        return spilled

    def restore_registers_after_call(self, spilled: list[Register]):
        for reg in reversed(spilled):
            self.pop_register(reg)

    def make_stack_address(
        self, offset: int, avoid: set[Register] | None = None
    ) -> Register:
        if avoid is None:
            avoid = set()

        base_reg = self.alloc_temp_register((RegisterType.A,), avoid=avoid)
        off_reg = self.alloc_temp_register((RegisterType.B,), avoid=avoid | {base_reg})

        self.emit(f"\tmov g3, {self.allocator.reg_name(base_reg)}")
        if offset == 0:
            self.emit(f"\tmov r0, {self.allocator.reg_name(off_reg)}")
        else:
            self.emit(f"\tmov {offset}, {self.allocator.reg_name(off_reg)}")

        self.emit(
            f"\tadd {self.allocator.reg_name(base_reg)}, {self.allocator.reg_name(off_reg)}, {self.allocator.reg_name(base_reg)}"
        )
        self.allocator.reg_free(off_reg)
        return base_reg

    def make_stack_address_flexible(
        self,
        offset: int,
        avoid: set[Register] | None = None,
    ) -> Register:
        if avoid is None:
            avoid = set()

        base_reg = self.alloc_temp_register((RegisterType.A,), avoid=avoid)
        off_reg = self.alloc_temp_register((RegisterType.B,), avoid=avoid | {base_reg})

        self.emit(f"\tmov g3, {self.allocator.reg_name(base_reg)}")
        if offset == 0:
            self.emit(f"\tmov r0, {self.allocator.reg_name(off_reg)}")
        else:
            self.emit(f"\tmov {offset}, {self.allocator.reg_name(off_reg)}")

        self.emit(
            f"\tadd {self.allocator.reg_name(base_reg)}, "
            f"{self.allocator.reg_name(off_reg)}, "
            f"{self.allocator.reg_name(base_reg)}"
        )
        self.allocator.reg_free(off_reg)

        return base_reg

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

        if isinstance(expression, Operation1):
            operand_type = self.get_expression_type(expression.value)

            if expression.op in ("!", "-"):
                if not self.is_integer(operand_type):
                    self.r.error(
                        f"Operand '{expression.op}' must be an integer, got {self.type_name(operand_type)}."
                    )
                    raise SystemExit(1)
                return operand_type

            self.r.error(f"Operation '{expression.op}' is not supported.")
            raise SystemExit(1)

        if isinstance(expression, Operation2):
            left_type = self.get_expression_type(expression.left)
            right_type = self.get_expression_type(expression.right)

            if expression.op in ("+", "-", "&", "|", "^", "!&", "!|", "!^"):
                return self.unify_int_types(
                    expression.left,
                    left_type,
                    expression.right,
                    right_type,
                    f"'{expression.op}'",
                )

            if expression.op in ("<<", ">>", "<<<", ">>>", ">=>"):
                if not self.is_integer(left_type):
                    self.r.error(
                        f"Left operand of '{expression.op}' must be an integer, got {self.type_name(left_type)}."
                    )
                    raise SystemExit(1)

                if not isinstance(expression.right, Number):
                    self.r.error("Dynamic shifts is not supported (yet).")  # TODO
                    raise SystemExit(1)

                return left_type

            if expression.op in ("==", "!=", "<", "<=", ">", ">="):
                self.unify_int_types(
                    expression.left,
                    left_type,
                    expression.right,
                    right_type,
                    f"'{expression.op}'",
                )
                return Type(
                    line=expression.line, column=expression.column, base_type=IntType.U8
                )

        if isinstance(expression, Call):
            _, function = self.lookup_function(expression.value)

            if len(function.results) == 0:
                self.r.error(f"Function '{function.name}' does not return a value.")
                raise SystemExit(1)

            if len(function.results) > 1:
                self.r.error(
                    f"Function '{function.name}' returns multiple values, which are not supported in expressions (yet)."
                )
                raise SystemExit(1)

            return function.results[0].type

        self.r.error("Unsupported expression for type resolution.")
        raise SystemExit(1)

    def gen_address(
        self, expression: Expression, avoid: set[Register] | None = None
    ) -> MemAddress:
        if avoid is None:
            avoid = set()

        if isinstance(expression, Name):
            symbol = self.lookup_variable(expression.value)

            if symbol.range == SymbolRange.GLOBAL:
                if not symbol.label:
                    self.r.error(f"Global symbol '{symbol.name}' has no label.")
                    raise SystemExit(1)
                return MemAddress(label=symbol.label)
            else:
                reg = self.make_stack_address_flexible(symbol.offset, avoid=avoid)
                return MemAddress(register=reg)

        elif (
            isinstance(expression, MemberAccess)
            and expression.type == MemberAccessType.FIELD
        ):
            base_addr = self.gen_address(expression.value, avoid=avoid)

            if not isinstance(expression.value, Name):
                self.r.error("Complex member access bases not supported (yet).")  # TODO
                raise SystemExit(1)

            base_name = expression.value.value
            symbol = self.lookup_variable(base_name)

            if not self.is_struct(symbol.type):
                self.r.error(f"Variable '{base_name}' is not a struct")
                raise SystemExit(1)

            struct_name = symbol.type.base_type
            if not isinstance(struct_name, str):
                self.r.error("Invalid struct type encountered.")
                raise SystemExit(1)

            field_offset = self.get_struct_field_offset(
                struct_name, expression.member, self.current_package
            )

            if field_offset == 0:
                return base_addr

            if base_addr.label is not None:
                return MemAddress(label=f"({base_addr.label} + {field_offset})")

            if base_addr.register is None:
                self.r.error("Base address must have a register if it has no label.")
                raise SystemExit(1)
            base_addr_reg = base_addr.register
            avoid_with_base = set(avoid)
            avoid_with_base.add(base_addr_reg)

            offset_reg = self.alloc_temp_register(
                (RegisterType.B,), avoid=avoid_with_base
            )

            target_reg = base_addr_reg
            if self.allocator.reg_type(base_addr_reg) != RegisterType.A:
                target_reg = self.alloc_temp_register(
                    (RegisterType.A,), avoid=avoid_with_base
                )
                self.emit(
                    f"\tmov {self.allocator.reg_name(base_addr_reg)}, {self.allocator.reg_name(target_reg)}"
                )
                self.allocator.reg_free(base_addr_reg)

            self.emit(f"\tmov {field_offset}, {self.allocator.reg_name(offset_reg)}")
            self.emit(
                f"\tadd {self.allocator.reg_name(target_reg)}, {self.allocator.reg_name(offset_reg)}, {self.allocator.reg_name(target_reg)}"
            )

            self.allocator.reg_free(offset_reg)
            return MemAddress(register=target_reg)

        self.r.error("Invalid left value")
        raise SystemExit(1)

    # ---------------

    def emit_call(
        self,
        expression: Call,
        target_package: str,
        function: Function,
    ) -> tuple[dict[Register, RegisterState], set[Register], list[Register]]:
        if len(expression.args) != len(function.params):
            self.r.error(
                f"Function '{function.name}' expects {len(function.params)} arguments, got {len(expression.args)}."
            )
            raise SystemExit(1)

        if len(function.params) > len(ARGUMENT_REGISTERS):
            self.r.error(f"Function '{function.name}' has too many arguments.")
            raise SystemExit(1)

        saved_reg_states, saved_locked_regs = self.save_allocator_state()
        spilled = self.spill_registers_for_call()

        self.allocator.reg_unlock_and_reset_states()

        for i, arg in enumerate(expression.args):
            arg_type = self.get_expression_type(arg)
            param_type = function.params[i].type

            self.ensure_assignable(
                target_type=param_type,
                value_expr=arg,
                value_type=arg_type,
                context=f"argument {i + 1} of call to '{function.name}'",
            )

            dst_reg = ARGUMENT_REGISTERS[i]
            reg_type = self.allocator.reg_type(dst_reg)
            tmp_reg = self.gen_expression(arg, reg_type)

            if tmp_reg != dst_reg:
                if self.allocator.reg_state(dst_reg) == RegisterState.FREE:
                    self.allocator.reg_alloc_specific(dst_reg)
                self.emit(
                    f"\tmov {self.allocator.reg_name(tmp_reg)}, {self.allocator.reg_name(dst_reg)}"
                )
                self.allocator.reg_free(tmp_reg)

            self.allocator.reg_lock(dst_reg)

        self.emit(f"\tcall _fun__{target_package}__{function.name}")

        return saved_reg_states, saved_locked_regs, spilled

    def gen_function(self, package: str, function: Function):
        self.id_counter = 0
        self.scopes.clear()
        self.current_stack_offset = 0
        self.current_package = package
        self.allocator.reg_unlock_and_reset_states()

        if len(function.params) > len(ARGUMENT_REGISTERS):
            self.r.error(f"Function '{function.name}' has too many parameters.")
            raise SystemExit(1)

        if len(function.results) > len(RESULT_REGISTERS):
            self.r.error(f"Function '{function.name}' has too many results.")
            raise SystemExit(1)

        self.emit(f"_fun__{package}__{function.name}:")
        self.emit("\tpush g3")
        self.emit("\tmov sp, g3")

        function_scope: dict[str, Symbol] = {}
        function_scope.update(self.package_globals.get(package, {}))
        self.scopes.append(function_scope)

        frame_size = 0
        start_offset = self.current_stack_offset

        for param in function.params:
            size = self.resolve_type_size(param.type, self.current_package)
            if size != 1 or self.is_array(param.type) or self.is_struct(param.type):
                self.r.error(
                    f"Function parameter '{param.name}' must fit in one register."
                )
                raise SystemExit(1)

            start_offset -= size
            function_scope[param.name] = Symbol(
                name=param.name,
                type=param.type,
                range=SymbolRange.PARAM,
                size=size,
                offset=start_offset,
            )
            frame_size += size

        for result in function.results:
            size = self.resolve_type_size(result.type, self.current_package)
            if size != 1 or self.is_array(result.type) or self.is_struct(result.type):
                self.r.error(
                    f"Function result '{result.name}' must fit in one register."
                )
                raise SystemExit(1)

            start_offset -= size
            function_scope[result.name] = Symbol(
                name=result.name,
                type=result.type,
                range=SymbolRange.RESULT,
                size=size,
                offset=start_offset,
            )
            frame_size += size

        self.current_stack_offset -= frame_size
        self.adjust_sp(-frame_size)

        for i, param in enumerate(function.params):
            src_reg = ARGUMENT_REGISTERS[i]
            param_symbol = function_scope[param.name]

            addr_reg = self.make_stack_address_flexible(
                param_symbol.offset, avoid={src_reg}
            )
            self.emit(
                f"\tstore {self.allocator.reg_name(src_reg)}, [{self.allocator.reg_name(addr_reg)}]"
            )
            self.allocator.reg_free(addr_reg)

        self.gen_statement(function.body)

        self.emit(" .return:")

        for i, result in enumerate(function.results):
            dst_reg = RESULT_REGISTERS[i]
            if self.allocator.reg_state(dst_reg) == RegisterState.FREE:
                self.allocator.reg_alloc_specific(dst_reg)

            result_symbol = function_scope[result.name]

            addr_reg = self.make_stack_address_flexible(
                result_symbol.offset, avoid={dst_reg}
            )
            self.emit(
                f"\tload [{self.allocator.reg_name(addr_reg)}], {self.allocator.reg_name(dst_reg)}"
            )
            self.allocator.reg_free(addr_reg)

        self.scopes.pop()
        self.emit("\tmov g3, sp")
        self.emit("\tpop g3")
        self.emit("\tret")
        self.emit()

        self.allocator.reg_unlock_and_reset_states()

    def gen_multi_result_assign(
        self,
        targets: list[Expression],
        call: Call,
    ):
        target_package, function = self.lookup_function(call.value)

        if len(function.results) == 0:
            self.r.error(f"Function '{function.name}' does not return any values.")
            raise SystemExit(1)

        if len(function.results) == 1:
            self.r.error(
                f"Function '{function.name}' returns a single value; use a normal assignment instead."
            )
            raise SystemExit(1)

        if len(targets) != len(function.results):
            self.r.error(
                f"Function '{function.name}' returns {len(function.results)} values, but assignment has {len(targets)} targets."
            )
            raise SystemExit(1)

        for i, target_expr in enumerate(targets):
            if not isinstance(target_expr, (Name, MemberAccess)):
                self.r.error(
                    "Assignments to complex structures are not supported (yet)."  # TODO
                )
                raise SystemExit(1)

            target_type = self.get_expression_type(target_expr)
            result_type = function.results[i].type
            self.ensure_assignable(
                target_type=target_type,
                value_expr=call,
                value_type=result_type,
                context=f"assignment target {i} of call to '{function.name}'",
            )

        saved_reg_states, saved_locked_regs, spilled = self.emit_call(
            call, target_package, function
        )

        live_result_regs: set[Register] = set(RESULT_REGISTERS[: len(function.results)])

        for i, target_expr in enumerate(targets):
            result_reg = RESULT_REGISTERS[i]
            addr = self.gen_address(target_expr, avoid=live_result_regs)
            if addr.label is not None:
                self.emit(
                    f"\tstore {self.allocator.reg_name(result_reg)}, {addr.label}"
                )
            else:
                assert addr.register is not None
                self.emit(
                    f"\tstore {self.allocator.reg_name(result_reg)}, [{self.allocator.reg_name(addr.register)}]"
                )
                self.allocator.reg_free(addr.register)

        self.restore_registers_after_call(spilled)
        self.restore_allocator_state(saved_reg_states, saved_locked_regs)

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
                value_type = self.get_expression_type(statement.value)

                self.ensure_assignable(
                    target_type=symbol.type,
                    value_expr=statement.value,
                    value_type=value_type,
                    context=f"initializer of '{statement.name}'",
                )

                value_reg = self.gen_expression(statement.value, RegisterType.B)
                addr = self.gen_address(Name(0, 0, statement.name), avoid={value_reg})
                if addr.label is not None:
                    self.emit(
                        f"\tstore {self.allocator.reg_name(value_reg)}, {addr.label}"
                    )
                else:
                    assert addr.register is not None
                    self.emit(
                        f"\tstore {self.allocator.reg_name(value_reg)}, [{self.allocator.reg_name(addr.register)}]"
                    )
                    self.allocator.reg_free(addr.register)
                self.allocator.reg_free(value_reg)

        elif isinstance(statement, Assign):
            if statement.value is None:
                return

            if len(statement.targets) > 1:
                if not isinstance(statement.value, Call):
                    self.r.error(
                        "Multiple assignment requires a function call on the right side."
                    )
                    raise SystemExit(1)
                self.gen_multi_result_assign(statement.targets, statement.value)
                return

            target_expr = statement.targets[0]
            if not isinstance(target_expr, (Name, MemberAccess)):
                self.r.error(
                    "Assignments to complex structures are not supported (yet)."  # TODO
                )
                raise SystemExit(1)

            if isinstance(statement.value, Call):
                _, function = self.lookup_function(statement.value.value)
                if len(function.results) > 1:
                    self.r.error(
                        f"Function '{function.name}' returns multiple values; use the same number of assignment targets."
                    )
                    raise SystemExit(1)

            target_type = self.get_expression_type(target_expr)
            value_type = self.get_expression_type(statement.value)

            self.ensure_assignable(
                target_type=target_type,
                value_expr=statement.value,
                value_type=value_type,
                context="assignment",
            )

            value_reg = self.gen_expression(statement.value, RegisterType.B)
            addr = self.gen_address(target_expr, avoid={value_reg})
            if addr.label is not None:
                self.emit(f"\tstore {self.allocator.reg_name(value_reg)}, {addr.label}")
            elif addr.register is not None:
                self.emit(
                    f"\tstore {self.allocator.reg_name(value_reg)}, [{self.allocator.reg_name(addr.register)}]"
                )
                self.allocator.reg_free(addr.register)
            else:
                self.r.error("Invalid memory address.")
                raise SystemExit(1)
            self.allocator.reg_free(value_reg)

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
        if not isinstance(expression, Operation2):
            self.r.error("Condition must be a comparison operation.")
            raise SystemExit(1)

        if expression.op not in ("==", "!=", "<", "<=", ">", ">="):
            self.r.error(f"Operation '{expression.op}' is not supported in conditions.")
            raise SystemExit(1)

        left_type = self.get_expression_type(expression.left)
        right_type = self.get_expression_type(expression.right)

        cmp_type = self.unify_int_types(
            expression.left,
            left_type,
            expression.right,
            right_type,
            f"condition '{expression.op}'",
        )

        if expression.op in ("<", "<="):
            left_reg = self.gen_expression(expression.left, RegisterType.B)
            right_reg = self.gen_expression(expression.right, RegisterType.A)
            self.emit(
                f"\tsub {self.allocator.reg_name(right_reg)}, {self.allocator.reg_name(left_reg)}, r0"
            )
        else:
            left_reg = self.gen_expression(expression.left, RegisterType.A)
            right_reg = self.gen_expression(expression.right, RegisterType.B)
            self.emit(
                f"\tsub {self.allocator.reg_name(left_reg)}, {self.allocator.reg_name(right_reg)}, r0"
            )

        if expression.op == "==":
            self.emit(f"\tjmp nz, {false_jump_label}")

        elif expression.op == "!=":
            self.emit(f"\tjmp zr, {false_jump_label}")

        else:
            base_type = cmp_type.base_type
            if not isinstance(base_type, IntType):
                self.r.error("Comparison type must be an integer type.")
                raise SystemExit(1)

            if base_type == IntType.I16:
                false_jumps = {
                    "<": "zv",
                    "<=": "nv",
                    ">": "zv",
                    ">=": "nv",
                }
            else:
                false_jumps = {
                    "<": "zc",
                    "<=": "cr",
                    ">": "zc",
                    ">=": "cr",
                }

            self.emit(f"\tjmp {false_jumps[expression.op]}, {false_jump_label}")

        self.allocator.reg_free(left_reg)
        self.allocator.reg_free(right_reg)

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
                self.r.error(f"{self.type_name(expr_type)} cannot be an array.")
                raise SystemExit(1)
            if self.is_struct(expr_type):
                self.r.error(f"{self.type_name(expr_type)} cannot be a struct value.")
                raise SystemExit(1)

            addr = self.gen_address(expression)
            target = self.allocator.reg_alloc(target_register_type)
            if addr.label is not None:
                self.emit(f"\tload [{addr.label}], {self.allocator.reg_name(target)}")
            else:
                assert addr.register is not None
                self.emit(
                    f"\tload [{self.allocator.reg_name(addr.register)}], {self.allocator.reg_name(target)}"
                )
                self.allocator.reg_free(addr.register)
            return target

        elif isinstance(expression, Operation1):
            if expression.op == "!":
                value_type = self.get_expression_type(expression.value)
                if not self.is_integer(value_type):
                    self.r.error(
                        f"Operand '!' must be an integer, got {self.type_name(value_type)}."
                    )
                    raise SystemExit(1)

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
                if not self.is_integer(value_type):
                    self.r.error(
                        f"Operand '-' must be an integer, got {self.type_name(value_type)}."
                    )
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
                self.unify_int_types(
                    expression.left,
                    self.get_expression_type(expression.left),
                    expression.right,
                    self.get_expression_type(expression.right),
                    f"'{expression.op}'",
                )

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
                if not self.is_integer(left_type):
                    self.r.error(
                        f"Left operand of '{expression.op}' must be an integer, got {self.type_name(left_type)}."
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
            target_package, function = self.lookup_function(expression.value)

            if len(function.results) > 1:
                self.r.error(
                    f"Function '{function.name}' returns multiple values, which are not supported in expressions (yet)."
                )  # TODO
                raise SystemExit(1)

            saved_reg_states, saved_locked_regs = self.save_allocator_state()
            result_target: Register | None = None

            if len(function.results) == 1:
                result_target = self.find_free_register_in_state(
                    saved_reg_states, target_register_type
                )

            _, _, spilled = self.emit_call(expression, target_package, function)

            if result_target is not None:
                abi_result_reg = RESULT_REGISTERS[0]
                if result_target != abi_result_reg:
                    self.emit(
                        f"\tmov {self.allocator.reg_name(abi_result_reg)}, {self.allocator.reg_name(result_target)}"
                    )

            self.restore_registers_after_call(spilled)
            self.restore_allocator_state(saved_reg_states, saved_locked_regs)

            if result_target is None:
                return Register.R0

            self.allocator.reg_states[result_target] = RegisterState.ALLOCATED
            self.allocator.locked_regs.discard(result_target)
            return result_target

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
                        f"_fun__{file_ast.package}__{vec.func_name}"
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

        self.package_globals = {}
        self.emit("; -------------- GLOBAL --------------")
        for package, files_ast in self.project_ast.items():
            globals_map: dict[str, Symbol] = {}
            self.package_globals[package] = globals_map
            for file_ast in files_ast:
                for global_var in file_ast.global_variables:
                    if global_var.name in globals_map:
                        self.r.error(
                            f"Duplicate global variable '{global_var.name}' in package '{package}'."
                        )
                        raise SystemExit(1)

                    size = self.resolve_type_size(global_var.type, package)
                    label = f"_global__{package}__{global_var.name}"

                    globals_map[global_var.name] = Symbol(
                        name=global_var.name,
                        type=global_var.type,
                        range=SymbolRange.GLOBAL,
                        size=size,
                        label=label,
                    )

                    self.emit(f"{label}:")
                    if global_var.value and isinstance(global_var.value, Number):
                        self.emit(f"\t#d16 {global_var.value.value}")
                        for _ in range(1, size):
                            self.emit("\t#d16 0")
                    elif global_var.value:
                        self.r.error(
                            f"Global initializer for '{global_var.name}' must be a constant number."
                        )
                        raise SystemExit(1)
                    else:
                        for _ in range(size):
                            self.emit("\t#d16 0")
        self.emit()

        self.emit("; --------------- CODE ---------------")
        for files_ast in self.project_ast.values():
            for file_ast in files_ast:
                for function in file_ast.functions:
                    self.gen_function(file_ast.package, function)

        return "\n".join(self.lines)
