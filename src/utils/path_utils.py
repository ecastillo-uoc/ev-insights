"""Cross-platform path normalization utilities.

Config files may contain Windows-style backslash paths (e.g. 'output\\logs')
because the project was originally developed on Windows. On POSIX systems
these backslashes are treated as literal characters rather than separators.

PureWindowsPath(...).as_posix() converts any backslashes to forward slashes,
making paths work correctly on both Windows and Linux/macOS.
"""
from pathlib import Path, PureWindowsPath


def normalize_path(path_str: str) -> Path:
    """Convert a potentially Windows-backslash path string to a POSIX-safe Path."""
    return Path(PureWindowsPath(path_str).as_posix())
