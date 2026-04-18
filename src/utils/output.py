"""Shared output utilities for saving results to file."""
import json
from pathlib import Path


def save_results_to_json(results: dict, output_dir: Path, name: str):
    """Save results dict to a JSON file.

    Uses json.dump() for valid, machine-readable JSON output
    (replaces the previous pprint-based approach).
    """
    filename = output_dir / f"{name}.json"
    with filename.open('w') as file:
        json.dump(results, file, indent=2, default=str)
