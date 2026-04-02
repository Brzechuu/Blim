#!/bin/python
import argparse
from pathlib import Path

from blim_codegen import CodeGenerator
from blim_lexer import Lexer, TokenType
from blim_parser import FileAst, Parser
from blim_reporter import Reporter
from blim_semanalyzer import SemanticAnalyzer


def find_src_files(directory: Path):
    try:
        for item in directory.iterdir():
            if item.is_dir():
                if not (item.name.startswith(".") or item.name.startswith("_")):
                    yield from find_src_files(item)
            elif item.is_file() and item.suffix == ".blim":
                yield item
    except PermissionError:
        pass


def main():
    r = Reporter()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", type=str, help="Project path")
    arg_parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug info", dest="debug"
    )
    # arg_parser.add_argument("-f", "--format", required=True, help="Format", dest="format")
    args = arg_parser.parse_args()

    project_path = Path(args.path).resolve()

    if not project_path.exists():
        r.error(f"Path '{project_path}' does not exist.")
        raise SystemExit(1)

    if not project_path.is_dir():
        r.error(f"Path '{project_path}' is not a directory.")
        raise SystemExit(1)

    print("Blim compiler 0.2")
    print(f"Building project: {project_path.name}")

    src_files = list(find_src_files(project_path))
    ast: dict[str, list[FileAst]] = {}

    if not src_files:
        r.error("Blim files not found.")
        raise SystemExit(1)

    for path in src_files:
        try:
            with open(path, "r", encoding="utf-8") as file:
                code = file.read()
        except Exception as e:
            r.warn(f"{e}")
            continue

        tokens = list(Lexer(code).tokenize())

        for token in tokens:
            if token.type == TokenType.ILLEGAL:
                r.error(
                    f"Illegal token '{token.value}'",
                    path,
                    token.line,
                    token.column,
                )

        if r.error_counter:
            raise SystemExit(1)

        file_ast = Parser(tokens, path, project_path, r).parse()

        if file_ast.package not in ast:
            ast[file_ast.package] = []
        ast[file_ast.package].append(file_ast)

    SemanticAnalyzer(ast, r).analyze()

    if r.error_counter:
        raise SystemExit(1)

    if args.debug:
        for package, files_ast in ast.items():
            print(f"Package: {package}:")
            for file_ast in files_ast:
                print(f"* File: {file_ast.path.relative_to(project_path)}:")
                if file_ast.imports:
                    print("  + Imports:")
                    for item in file_ast.imports:
                        print(f"    - {item.package}")
                if file_ast.global_variables:
                    print("  + Global variables:")
                    for item in file_ast.global_variables:
                        print(f"    - {item.name}")
                if file_ast.structures:
                    print("  + Structures:")
                    for item in file_ast.structures:
                        print(f"    - {item.name}")
                if file_ast.functions:
                    print("  + Functions:")
                    for item in file_ast.functions:
                        print(f"    - {item.name}")

    asm_code = CodeGenerator(ast, r).generate_asm_code()

    if r.error_counter:
        raise SystemExit(1)

    output_file = project_path / f"{project_path.name}.asm"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(asm_code)
        print(f"File saved to '{output_file}'")
    except Exception as e:
        r.error(f"{e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
