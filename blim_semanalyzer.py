from blim_parser import FileAst
from blim_reporter import Reporter


class SemanticAnalyzer:
    def __init__(self, project_ast: dict[str, list[FileAst]], reporter: Reporter):
        self.project_ast = project_ast
        self.r = reporter

    def check_duplication(self):
        for package_name, files_ast in self.project_ast.items():
            globals: set[str] = set()
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
                    if global_var.name in globals:
                        self.r.error(
                            f"Duplicate global variable or define '{global_var.name}' in package '{package_name}' at {file_ast.path.name}:{global_var.line}:{global_var.column}"
                        )
                    if global_var.name in packages_alias:
                        if isinstance(global_var.type.base_type, str):
                            self.r.error(
                                f"Global variable '{global_var.name}' conflicts with used package or its alias at {file_ast.path.name}:{global_var.line}:{global_var.column}"
                            )
                    globals.add(global_var.name)

                for define in file_ast.defines:
                    if define.name in globals:
                        self.r.error(
                            f"Duplicate global variable or define '{define.name}' in package '{package_name}' at {file_ast.path.name}:{define.line}:{define.column}"
                        )
                    globals.add(define.name)

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

    def analyze(self):
        self.check_duplication()
