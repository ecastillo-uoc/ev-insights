"""Shared console utilities for colored terminal output.

The ``Colors`` class provides ANSI escape codes for direct use inside
``logging`` f-strings (the codes travel through any handler that writes
to a real terminal).  The ``rich`` :class:`Console` instance and its
helpers are available for richer interactive output.
"""

from rich.console import Console

console = Console()


class Colors:
    """ANSI color codes (valid SGR sequences).

    https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
    """
    # reset normal
    NORMAL = "\033[0m"

    # Text styles
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Standard foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    LIGHT_GRAY = "\033[37m"

    # Bold + color
    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_PURPLE = "\033[1;35m"


def log_success(message: str) -> None:
    console.print(f"[green]{message}[/green]")


def log_warning(message: str) -> None:
    console.print(f"[yellow]{message}[/yellow]")


def log_error(message: str) -> None:
    console.print(f"[bold red]{message}[/bold red]")


def log_info(message: str, style: str = "blue") -> None:
    console.print(f"[{style}]{message}[/{style}]")
