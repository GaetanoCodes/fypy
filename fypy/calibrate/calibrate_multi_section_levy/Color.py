class Color:
    def __init__(self):
        self._dict = {
            "PURPLE": "\033[95m",
            "CYAN": "\033[96m",
            "DARKCYAN": "\033[36m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "YELLOW": "\033[93m",
            "RED": "\033[91m",
            "BOLD": "\033[1m",
            "UNDERLINE": "\033[4m",
            "END": "\033[0m",
        }

    def write(
        self, text: str, color: str = "END", style: list[str] = [], indent: int = 0
    ):
        style_code = "".join([self._dict[st] for st in style])
        print(self._dict[color] + style_code + indent * "\t" + text + self._dict["END"])


COLOR = Color()
