from blim_parser import FileAst
from blim_reporter import Reporter


class SemanticAnalyzer:
    def __init__(self, project_ast: dict[str, list[FileAst]], reporter: Reporter):
        self.project_ast = project_ast
        self.r = reporter

    def analyze(self):
        globals: set[str] = set()
        functions: set[str] = set()
        structs: set[str] = set()
        for package_name, files_ast in self.project_ast.items():
            for file_ast in files_ast:
                for global_var in file_ast.global_variables:
                    if f"{package_name}__{global_var.name}" in globals:
                        self.r.error(
                            f"Duplicate global variable '{global_var.name}' in package '{package_name}' at {file_ast.path.name}:{global_var.line}:{global_var.column}"
                        )
                    else:
                        globals.add(f"{package_name}__{global_var.name}")

                for function in file_ast.functions:
                    if f"{package_name}__{function.name}" in functions:
                        self.r.error(
                            f"Duplicate function '{function.name}' in package '{package_name}' at {file_ast.path.name}:{function.line}:{function.column}"
                        )
                    else:
                        functions.add(f"{package_name}__{function.name}")

                for struct in file_ast.structures:
                    if f"{package_name}__{struct.name}" in structs:
                        self.r.error(
                            f"Duplicate struct '{struct.name}' in package '{package_name}' at {file_ast.path.name}:{struct.line}:{struct.column}"
                        )
                    else:
                        structs.add(f"{package_name}__{struct.name}")
