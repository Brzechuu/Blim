from blim_parser import (
    ArrayValue,
    Assign,
    Block,
    Call,
    ExprStatement,
    Expression,
    FileAst,
    If,
    Index,
    MemberAccess,
    MemberAccessType,
    Name,
    Number,
    Operation1,
    Operation2,
    StringValue,
    StructValue,
    Type,
    Variable,
    While,
)
from blim_reporter import Reporter


class Preprocessor:
    def __init__(self, project_ast: dict[str, list[FileAst]], reporter: Reporter):
        self.project_ast = project_ast
        self.r = reporter
        self.all_defines: dict[str, dict[str, int]] = {}
        self.package_aliases: dict[str, str] = {}

    def preprocess(self) -> None:
        self.all_defines = self.build_defines_map()

        for package, files_ast in self.project_ast.items():
            package_defines = self.all_defines.get(package, {})
            for file_ast in files_ast:
                self.process_file(file_ast, package_defines)

    def build_defines_map(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for package, files_ast in self.project_ast.items():
            package_defines: dict[str, int] = {}
            for file_ast in files_ast:
                for define in file_ast.defines:
                    package_defines[define.name] = int(define.value, 0)
            result[package] = package_defines
        return result

    def process_file(self, file_ast: FileAst, defines: dict[str, int]) -> None:
        self.package_aliases = {}
        for use in file_ast.imports:
            if use.alias:
                self.package_aliases[use.alias] = use.package
            else:
                self.package_aliases[use.package] = use.package

        for struct in file_ast.structures:
            for field in struct.fields:
                self.process_type(field.type, defines)

        for global_var in file_ast.global_variables:
            self.process_type(global_var.type, defines)
            if global_var.value is not None:
                global_var.value = self.fold_expression(global_var.value, defines)

        for function in file_ast.functions:
            for param in function.params:
                self.process_type(param.type, defines)
            for result in function.results:
                self.process_type(result.type, defines)
            self.process_block(function.body, defines)

    def process_type(self, var_type: Type, defines: dict[str, int]) -> None:
        if var_type.array_size is not None:
            var_type.array_size = self.fold_expression(var_type.array_size, defines)

    def process_block(self, block: Block, defines: dict[str, int]) -> None:
        for statement in block.statements:
            self.process_statement(statement, defines)

    def process_statement(self, statement, defines: dict[str, int]) -> None:
        if isinstance(statement, Block):
            self.process_block(statement, defines)

        elif isinstance(statement, Variable):
            self.process_type(statement.type, defines)
            if statement.value is not None:
                statement.value = self.fold_expression(statement.value, defines)

        elif isinstance(statement, Assign):
            for target in statement.targets:
                self.fold_target(target, defines)
            if statement.value is not None:
                statement.value = self.fold_expression(statement.value, defines)

        elif isinstance(statement, If):
            statement.condition = self.fold_expression(statement.condition, defines)
            self.process_statement(statement.then_block, defines)
            if statement.else_block is not None:
                self.process_statement(statement.else_block, defines)

        elif isinstance(statement, While):
            statement.condition = self.fold_expression(statement.condition, defines)
            self.process_statement(statement.body, defines)

        elif isinstance(statement, ExprStatement):
            statement.value = self.fold_expression(statement.value, defines)

    def fold_target(self, target, defines: dict[str, int]) -> None:
        if isinstance(target, Index):
            target.value = self.fold_expression(target.value, defines)
            target.index = self.fold_expression(target.index, defines)
        elif isinstance(target, MemberAccess):
            self.fold_target(target.value, defines)

    def fold_expression(self, expression, defines: dict[str, int]) -> Expression:
        if isinstance(expression, (Number, StringValue)):
            return expression

        if isinstance(expression, Name):
            if expression.value in defines:
                return Number(
                    line=expression.line,
                    column=expression.column,
                    value=defines[expression.value],
                )
            return expression

        if isinstance(expression, Operation1):
            if expression.op in ("&", "*", "++", "--"):
                expression.value = self.fold_expression(expression.value, defines)
                return expression
            folded = self.fold_expression(expression.value, defines)
            if isinstance(folded, Number):
                if expression.op == "-":
                    return Number(
                        folded.line, folded.column, (-folded.value) & 0xFFFF
                    )
                if expression.op == "!":
                    return Number(
                        folded.line, folded.column, (~folded.value) & 0xFFFF
                    )
            expression.value = folded
            return expression

        if isinstance(expression, Operation2):
            left = self.fold_expression(expression.left, defines)
            right = self.fold_expression(expression.right, defines)
            if isinstance(left, Number) and isinstance(right, Number):
                folded = self.fold_binary(
                    expression.op, left.value, right.value
                )
                if folded is not None:
                    return Number(
                        line=expression.line,
                        column=expression.column,
                        value=folded,
                    )
            expression.left = left
            expression.right = right
            return expression

        if isinstance(expression, Call):
            for i, arg in enumerate(expression.args):
                expression.args[i] = self.fold_expression(arg, defines)
            return expression

        if isinstance(expression, MemberAccess):
            if isinstance(expression.value, Name):
                prefix = expression.value.value
                is_package = (
                    expression.type == MemberAccessType.PACKAGE
                    or prefix in self.package_aliases
                )
                if is_package:
                    target_package = self.package_aliases.get(prefix, prefix)
                    package_defines = self.all_defines.get(target_package, {})
                    if expression.member in package_defines:
                        return Number(
                            line=expression.line,
                            column=expression.column,
                            value=package_defines[expression.member],
                        )
                    return expression
            expression.value = self.fold_expression(expression.value, defines)
            return expression

        if isinstance(expression, Index):
            expression.value = self.fold_expression(expression.value, defines)
            expression.index = self.fold_expression(expression.index, defines)
            return expression

        if isinstance(expression, ArrayValue):
            for i, value in enumerate(expression.values):
                expression.values[i] = self.fold_expression(value, defines)
            return expression

        if isinstance(expression, StructValue):
            for field in expression.fields:
                field.value = self.fold_expression(field.value, defines)
            return expression

        return expression

    def fold_binary(self, op: str, left: int, right: int) -> int | None:
        if op == "+":
            return (left + right) & 0xFFFF
        if op == "-":
            return (left - right) & 0xFFFF
        if op == "*":
            return (left * right) & 0xFFFF
        if op == "&":
            return left & right
        if op == "|":
            return left | right
        if op == "^":
            return left ^ right
        if op == "!&":
            return (~(left & right)) & 0xFFFF
        if op == "!|":
            return (~(left | right)) & 0xFFFF
        if op == "!^":
            return (~(left ^ right)) & 0xFFFF
        if op == "<<":
            return (left << right) & 0xFFFF if right >= 0 else None
        if op == ">>":
            return (left >> right) & 0xFFFF if right >= 0 else None
        if op == "<<<":
            r = right % 16
            return ((left << r) | (left >> (16 - r))) & 0xFFFF
        if op == ">>>":
            r = right % 16
            return ((left >> r) | (left << (16 - r))) & 0xFFFF
        if op == ">=>":
            if right < 0:
                return None
            sign = left & 0x8000
            shifted = left >> right
            if sign:
                shifted |= (0xFFFF << (16 - right)) & 0xFFFF
            return shifted & 0xFFFF
        return None