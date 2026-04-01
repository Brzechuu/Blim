from dataclasses import dataclass

from blim_parser import (
    ArrayValue,
    Asm,
    Assign,
    Block,
    Break,
    Call,
    Continue,
    Define,
    Expression,
    ExprStatement,
    FileAst,
    Function,
    GlobalVariable,
    If,
    Index,
    MemberAccess,
    MemberAccessType,
    Name,
    Number,
    Operation1,
    Operation2,
    Return,
    Statement,
    Struct,
    StructValue,
    Variable,
    While,
)
from blim_reporter import Reporter


@dataclass
class PackageEnv:
    variables: dict[str, GlobalVariable]
    functions: dict[str, Function]
    structs: dict[str, Struct]
    defines: dict[str, Define]


class SemanticAnalyzer:
    def __init__(self, project_ast: dict[str, list[FileAst]], reporter: Reporter):
        self.project_ast = project_ast
        self.r = reporter
        self.packages_env: dict[str, PackageEnv] = {}

    def check_duplication(self, package_name: str, files_ast: list[FileAst]):
        package_symbols: dict[str, str] = {}

        for file_ast in files_ast:
            packages: set[str] = set()
            packages_alias: set[str] = set()

            for use in file_ast.imports:
                if use.alias:
                    name = use.alias
                else:
                    name = use.package

                if name in packages_alias:
                    if use.package in packages and not use.alias:
                        self.r.error(
                            f"Package '{use.package}' is imported multiple times",
                            file_ast.path,
                            use.line,
                            use.column,
                        )
                    else:
                        self.r.error(
                            f"Package alias '{name}' is already in use",
                            file_ast.path,
                            use.line,
                            use.column,
                        )

                packages_alias.add(name)
                packages.add(use.package)

            for global_var in file_ast.global_variables:
                if global_var.name in package_symbols:
                    self.r.error(
                        f"Duplicate name '{global_var.name}' in package '{package_name}'. Already declared as {package_symbols[global_var.name]}",
                        file_ast.path,
                        global_var.line,
                        global_var.column,
                    )
                if global_var.name in packages_alias and isinstance(
                    global_var.type.base_type, str
                ):
                    self.r.error(
                        f"Global variable '{global_var.name}' conflicts with imported package or its alias",
                        file_ast.path,
                        global_var.line,
                        global_var.column,
                    )
                package_symbols[global_var.name] = "global variable"

            for define in file_ast.defines:
                if define.name in package_symbols:
                    self.r.error(
                        f"Duplicate name '{define.name}' in package '{package_name}'. Already declared as {package_symbols[define.name]}",
                        file_ast.path,
                        define.line,
                        define.column,
                    )
                package_symbols[define.name] = "define"

            for function in file_ast.functions:
                if function.name in package_symbols:
                    self.r.error(
                        f"Duplicate name '{function.name}' in package '{package_name}'. Already declared as {package_symbols[function.name]}",
                        file_ast.path,
                        function.line,
                        function.column,
                    )
                package_symbols[function.name] = "function"

            for struct in file_ast.structures:
                if struct.name in package_symbols:
                    self.r.error(
                        f"Duplicate name '{struct.name}' in package '{package_name}'. Already declared as {package_symbols[struct.name]}",
                        file_ast.path,
                        struct.line,
                        struct.column,
                    )
                package_symbols[struct.name] = "struct"

    def build_environments(self):
        for pkg_name, files_ast in self.project_ast.items():
            env = PackageEnv(variables={}, functions={}, structs={}, defines={})

            for file_ast in files_ast:
                for func in file_ast.functions:
                    env.functions[func.name] = func
                for gvar in file_ast.global_variables:
                    env.variables[gvar.name] = gvar
                for struct in file_ast.structures:
                    env.structs[struct.name] = struct
                for define in file_ast.defines:
                    env.defines[define.name] = define

            self.packages_env[pkg_name] = env

    def analyze_package(self, package_name: str, files_ast: list[FileAst]):
        env = self.packages_env[package_name]

        for file_ast in files_ast:
            for vec in file_ast.interrupt_vectors:
                if vec.func_name not in env.functions:
                    self.r.error(
                        f"Interrupt vector '{vec.vector_number}' points to an undefined function '{vec.func_name}' in package '{package_name}'",
                        file_ast.path,
                        vec.line,
                        vec.column,
                    )
                else:
                    func = env.functions[vec.func_name]
                    if len(func.params) > 0 and vec.vector_number < 16:
                        self.r.error(
                            f"Interrupt handler function '{vec.func_name}' cannot take parameters",
                            file_ast.path,
                            vec.line,
                            vec.column,
                        )
                    if len(func.results) > 0 and vec.vector_number < 16:
                        self.r.error(
                            f"Interrupt handler function '{vec.func_name}' cannot return results",
                            file_ast.path,
                            vec.line,
                            vec.column,
                        )

            for struct in file_ast.structures:
                for field in struct.fields:
                    if field.name in env.defines:
                        self.r.error(
                            f"Field '{field.name}' in struct '{struct.name}' cannot use the same name as a define",
                            file_ast.path,
                            field.line,
                            field.column,
                        )

            packages: dict[str, str] = {}
            for use in file_ast.imports:
                if use.alias:
                    packages[use.alias] = use.package
                else:
                    packages[use.package] = use.package

            scopes: list[dict[str, object]] = [dict(env.variables)]

            for function in file_ast.functions:
                self.analyze_function(function, scopes, env, packages, file_ast)

    def analyze_function(
        self,
        function: Function,
        scopes: list[dict[str, object]],
        env: PackageEnv,
        packages: dict[str, str],
        file_ast: FileAst,
    ):
        fun_scope: dict[str, object] = {}

        for param in function.params:
            if param.name in env.defines:
                self.r.error(
                    f"Parameter '{param.name}' cannot use the same name as a define",
                    file_ast.path,
                    param.line,
                    param.column,
                )
            fun_scope[param.name] = param

        for result in function.results:
            if result.name in env.defines:
                self.r.error(
                    f"Result '{result.name}' cannot use the same name as a define",
                    file_ast.path,
                    result.line,
                    result.column,
                )
            fun_scope[result.name] = result

        scopes.append(fun_scope)

        for statement in function.body.statements:
            self.analyze_statement(statement, scopes, env, packages, file_ast)

        scopes.pop()

    def analyze_statement(
        self,
        statement: Statement,
        scopes: list[dict[str, object]],
        env: PackageEnv,
        packages: dict[str, str],
        file_ast: FileAst,
    ):
        if isinstance(statement, Block):
            scopes.append({})
            for stmt in statement.statements:
                self.analyze_statement(stmt, scopes, env, packages, file_ast)
            scopes.pop()

        elif isinstance(statement, Variable):
            if statement.name in env.defines:
                self.r.error(
                    f"Local variable '{statement.name}' cannot use the same name as a define",
                    file_ast.path,
                    statement.line,
                    statement.column,
                )

            if statement.value:
                self.analyze_expression(
                    statement.value, scopes, env, packages, file_ast
                )
            scopes[-1][statement.name] = statement

        elif isinstance(statement, Assign):
            for target in statement.targets:
                if isinstance(target, Name) and target.value in env.defines:
                    self.r.error(
                        f"Cannot assign to define '{target.value}'",
                        file_ast.path,
                        target.line,
                        target.column,
                    )

                self.analyze_expression(target, scopes, env, packages, file_ast)
            if statement.value:
                self.analyze_expression(
                    statement.value, scopes, env, packages, file_ast
                )

        elif isinstance(statement, Return):
            pass

        elif isinstance(statement, If):
            self.analyze_expression(
                statement.condition, scopes, env, packages, file_ast
            )
            self.analyze_statement(
                statement.then_block, scopes, env, packages, file_ast
            )
            if statement.else_block:
                self.analyze_statement(
                    statement.else_block, scopes, env, packages, file_ast
                )

        elif isinstance(statement, While):
            self.analyze_expression(
                statement.condition, scopes, env, packages, file_ast
            )
            self.analyze_statement(statement.body, scopes, env, packages, file_ast)

        elif isinstance(statement, Break):
            pass

        elif isinstance(statement, Continue):
            pass

        elif isinstance(statement, ExprStatement):
            self.analyze_expression(statement.value, scopes, env, packages, file_ast)

        elif isinstance(statement, Asm):
            pass

    def analyze_expression(
        self,
        expression: Expression,
        scopes: list[dict[str, object]],
        env: PackageEnv,
        packages: dict[str, str],
        file_ast: FileAst,
    ):
        if isinstance(expression, Number):
            pass

        elif isinstance(expression, Name):
            name = expression.value
            is_variable = any(name in scope for scope in reversed(scopes))

            if is_variable:
                pass
            elif name in env.defines:
                pass
            elif name in packages:
                pass
            else:
                self.r.error(
                    f"Use of undeclared identifier '{name}'",
                    file_ast.path,
                    expression.line,
                    expression.column,
                )

        elif isinstance(expression, Operation1):
            self.analyze_expression(expression.value, scopes, env, packages, file_ast)

        elif isinstance(expression, Operation2):
            self.analyze_expression(expression.left, scopes, env, packages, file_ast)
            self.analyze_expression(expression.right, scopes, env, packages, file_ast)

        elif isinstance(expression, Call):
            if isinstance(expression.value, Name):
                name = expression.value.value

                if name not in env.functions:
                    self.r.error(
                        f"Use of undeclared function '{name}'",
                        file_ast.path,
                        expression.line,
                        expression.column,
                    )
            elif isinstance(expression.value, MemberAccess):
                package_call = False

                if isinstance(expression.value.value, Name):
                    pkg_alias = expression.value.value.value
                    if pkg_alias in packages:
                        package_call = True
                        expression.value.type = MemberAccessType.PACKAGE
                        func_name = expression.value.member

                        target_env = self.packages_env.get(packages[pkg_alias])
                        if target_env is None:
                            self.r.error(
                                f"Package '{packages[pkg_alias]}' not found",
                                file_ast.path,
                                expression.line,
                                expression.column,
                            )
                        elif func_name not in target_env.functions:
                            self.r.error(
                                f"Function '{func_name}' not found in package '{pkg_alias}'",
                                file_ast.path,
                                expression.line,
                                expression.column,
                            )

                if not package_call:
                    self.analyze_expression(
                        expression.value, scopes, env, packages, file_ast
                    )

            else:
                self.analyze_expression(
                    expression.value, scopes, env, packages, file_ast
                )

            for arg in expression.args:
                self.analyze_expression(arg, scopes, env, packages, file_ast)

        elif isinstance(expression, MemberAccess):
            if isinstance(expression.value, Name):
                ident = expression.value.value

                is_variable = any(ident in scope for scope in reversed(scopes))

                if is_variable:
                    expression.type = MemberAccessType.FIELD
                elif ident in packages:
                    expression.type = MemberAccessType.PACKAGE

                    target_env = self.packages_env.get(packages[ident])
                    member_name = expression.member

                    if target_env is None:
                        self.r.error(
                            f"Package '{packages[ident]}' not found",
                            file_ast.path,
                            expression.line,
                            expression.column,
                        )
                    elif (
                        member_name not in target_env.variables
                        and member_name not in target_env.defines
                    ):
                        self.r.error(
                            f"Variable or define '{member_name}' not found in package '{ident}'",
                            file_ast.path,
                            expression.line,
                            expression.column,
                        )
                elif ident in env.defines:
                    self.r.error(
                        f"Cannot use dot notation on define '{ident}'",
                        file_ast.path,
                        expression.line,
                        expression.column,
                    )
                else:
                    self.r.error(
                        f"Unknown identifier '{ident}' before '.'",
                        file_ast.path,
                        expression.line,
                        expression.column,
                    )
            else:
                self.analyze_expression(
                    expression.value, scopes, env, packages, file_ast
                )
                expression.type = MemberAccessType.FIELD

        elif isinstance(expression, Index):
            self.analyze_expression(expression.value, scopes, env, packages, file_ast)
            self.analyze_expression(expression.index, scopes, env, packages, file_ast)

        elif isinstance(expression, ArrayValue):
            for val in expression.values:
                self.analyze_expression(val, scopes, env, packages, file_ast)

        elif isinstance(expression, StructValue):
            for field in expression.fields:
                self.analyze_expression(field.value, scopes, env, packages, file_ast)

    def analyze(self):
        global_vectors: set[int] = set()
        for files_ast in self.project_ast.values():
            for file_ast in files_ast:
                for vec in file_ast.interrupt_vectors:
                    if vec.vector_number in global_vectors:
                        self.r.error(
                            f"Duplicate interrupt vector '{vec.vector_number}'",
                            file_ast.path,
                            vec.line,
                            vec.column,
                        )
                    global_vectors.add(vec.vector_number)

        for package_name, files_ast in self.project_ast.items():
            self.check_duplication(package_name, files_ast)

        self.build_environments()

        for package_name, files_ast in self.project_ast.items():
            self.analyze_package(package_name, files_ast)
