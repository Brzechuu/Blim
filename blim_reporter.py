from pathlib import Path


class Reporter:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __init__(self):
        self.error_counter = 0
        self.warning_counter = 0

    def error(
        self,
        message: str,
        file: str | Path | None = None,
        line: int | None = None,
        column: int | None = None,
    ):
        self.error_counter += 1
        if file and line and column:
            print(
                f"{self.RED}Error:{self.RESET} {message}\n{self.RED} └─At:{self.RESET} {file}:{line}:{column}"
            )
        elif file and line:
            print(
                f"{self.RED}Error:{self.RESET} {message}\n{self.RED} └─At:{self.RESET} {file}:{line}"
            )
        elif file:
            print(
                f"{self.RED}Error:{self.RESET} {message}\n{self.RED} └─At:{self.RESET} {file}"
            )
        else:
            print(f"{self.RED}Error:{self.RESET} {message}")

    def warn(
        self,
        message: str,
        file: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ):
        self.warning_counter += 1
        if file and line and column:
            print(
                f"{self.YELLOW}Warning:{self.RESET} {message}\n{self.YELLOW} └─At:{self.RESET} {file}:{line}:{column}"
            )
        elif file and line:
            print(
                f"{self.YELLOW}Warning:{self.RESET} {message}\n{self.YELLOW} └─At:{self.RESET} {file}:{line}"
            )
        elif file:
            print(
                f"{self.YELLOW}Warning:{self.RESET} {message}\n{self.YELLOW} └─At:{self.RESET} {file}"
            )
        else:
            print(f"{self.YELLOW}Warning:{self.RESET} {message}")
