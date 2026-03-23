class Reporter:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __init__(self):
        self.error_counter = 0
        self.warning_counter = 0

    def error(self, message: str):
        self.error_counter += 1
        print(f"{self.RED}Error:{self.RESET} {message}")

    def warn(self, message: str):
        self.warning_counter += 1
        print(f"{self.YELLOW}Warning:{self.RESET} {message}")
