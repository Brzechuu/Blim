import argparse
import sys
from pathlib import Path

from blim_lexer import Lexer, TokenType

# from blim_parser import Parser


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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", type=str, help="Project path")
    arg_parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug info", dest="debug"
    )
    # arg_parser.add_argument("-f", "--format", required=True, help="Format", dest="format")
    args = arg_parser.parse_args()

    project_path = Path(args.path).resolve()

    if not project_path.exists():
        print(f"Error: Path '{project_path}' does not exist.")
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Error: Path '{project_path}' is not a directory.")
        sys.exit(1)

    print("Blim compiler 0.2")
    print(f"Building project: {project_path.name}")

    src_files = list(find_src_files(project_path))

    if not src_files:
        print("Error: Blim files not found.")
        sys.exit(1)

    if args.debug:
        print("[DEBUG] =============== LEXER, PARSER ===============")

    for path in src_files:
        try:
            with open(path, "r", encoding="utf-8") as file:
                code = file.read()
        except Exception as e:
            print("Error:", e)
            continue

        tokens = list(Lexer(code).tokenize())

        if args.debug:
            print(f"[DEBUG] --- File: {path.relative_to(project_path)} ---")
            for token in tokens:
                print(f"[DEBUG] {token}")

        for token in tokens:
            if token.type == TokenType.ILLEGAL:
                print(
                    f"Error: Illegal token '{token.value}' in {path.relative_to(project_path)}:{token.line}:{token.column}"
                )
                sys.exit(1)

        # ast = Parser(tokens, path).parse()


if __name__ == "__main__":
    main()
