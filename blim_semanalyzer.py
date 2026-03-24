from dataclasses import dataclass

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


class SemanticAnalyzer:
    def __init__(self, project_ast: dict[str, list[FileAst]], reporter: Reporter):
        self.project_ast = project_ast
        self.r = reporter
        self.packages_env: dict[str, PackageEnv] = {}

    def check_duplication(self, package_name: str, files_ast: list[FileAst]):
        global_vars: set[str] = set()
        structs: set[str] = set()
        functions: set[str] = set()
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
                            f"Package '{use.package}' is imported multiple times at {file_ast.path.name}:{use.line}:{use.column}"
                        )
                    else:
                        self.r.error(
                            f"Package alias '{name}' is already in use at {file_ast.path.name}:{use.line}:{use.column}"
                        )
                packages_alias.add(name)
                packages.add(use.package)

            for global_var in file_ast.global_variables:
                if global_var.name in global_vars:
                    self.r.error(
                        f"Duplicate global variable '{global_var.name}' in package '{package_name}' at {file_ast.path.name}:{global_var.line}:{global_var.column}"
                    )
                if global_var.name in packages_alias:
                    if isinstance(global_var.type.base_type, str):
                        self.r.error(
                            f"Global variable '{global_var.name}' conflicts with used package or its alias at {file_ast.path.name}:{global_var.line}:{global_var.column}"
                        )
                global_vars.add(global_var.name)

            for function in file_ast.functions:
                if function.name in functions:
                    self.r.error(
                        f"Duplicate function '{function.name}' in package '{package_name}' at {file_ast.path.name}:{function.line}:{function.column}"
                    )
                functions.add(function.name)

            for struct in file_ast.structures:
                if struct.name in structs:
                    self.r.error(
                        f"Duplicate struct '{struct.name}' in package '{package_name}' at {file_ast.path.name}:{struct.line}:{struct.column}"
                    )
                structs.add(struct.name)

    def build_environments(self):
        for pkg_name, files_ast in self.project_ast.items():
            env = PackageEnv(variables={}, functions={}, structs={})

            for file_ast in files_ast:
                for func in file_ast.functions:
                    env.functions[func.name] = func
                for gvar in file_ast.global_variables:
                    env.variables[gvar.name] = gvar
                for struct in file_ast.structures:
                    env.structs[struct.name] = struct

            self.packages_env[pkg_name] = env

    def analyze_package(self, package_name: str, files_ast: list[FileAst]):
        env = self.packages_env[package_name]

        for file_ast in files_ast:
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
            fun_scope[param.name] = param

        for result in function.results:
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
            if statement.value:
                self.analyze_expression(
                    statement.value, scopes, env, packages, file_ast
                )
            scopes[-1][statement.name] = statement

        elif isinstance(statement, Assign):
            for target in statement.targets:
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
            elif name in packages:
                pass
            else:
                self.r.error(
                    f"Use of undeclared identifier '{name}' at {file_ast.path.name}:{expression.line}:{expression.column}"
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
                        f"Use of undeclared function '{name}' at {file_ast.path.name}:{expression.line}:{expression.column}"
                    )
            elif isinstance(expression.value, MemberAccess):
                self.analyze_expression(
                    expression.value, scopes, env, packages, file_ast
                )

                if expression.value.type == MemberAccessType.PACKAGE:
                    if isinstance(expression.value.value, Name):
                        pkg_alias = expression.value.value.value
                        func_name = expression.value.member

                        name = packages.get(pkg_alias)
                        if name is not None:
                            target_env = self.packages_env.get(name)

                            if not target_env or func_name not in target_env.functions:
                                self.r.error(
                                    f"Function '{func_name}' not found in package '{pkg_alias}' at {file_ast.path.name}:{expression.line}:{expression.column}"
                                )
            else:
                self.analyze_expression(
                    expression.value, scopes, env, packages, file_ast
                )

            for arg in expression.args:
                self.analyze_expression(arg, scopes, env, packages, file_ast)

        elif isinstance(expression, MemberAccess):
            self.analyze_expression(expression.value, scopes, env, packages, file_ast)

            if isinstance(expression.value, Name):
                ident = expression.value.value

                is_variable = any(ident in scope for scope in reversed(scopes))

                if is_variable:
                    expression.type = MemberAccessType.FIELD
                elif ident in packages:
                    expression.type = MemberAccessType.PACKAGE
                else:
                    self.r.error(
                        f"Unknown identifier '{ident}' before '.' at {file_ast.path.name}:{expression.line}:{expression.column}"
                    )
            else:
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
        for package_name, files_ast in self.project_ast.items():
            self.check_duplication(package_name, files_ast)

        self.build_environments()

        for package_name, files_ast in self.project_ast.items():
            self.analyze_package(package_name, files_ast)
