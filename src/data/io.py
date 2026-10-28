import csv
import json
from pathlib import Path
from typing import Any


def save_dataset_file(
    filepath: Path, nodes: list[str], data: list[list[int]]
) -> None:
    """
    Saves time-series data in the transposed tab-delimited format required by BNFinder.
    Rows represent variables (genes); columns represent time points (1:1, 1:2, ...).
    """
    time_steps = len(data)
    headers = ["ExpTitle"] + [f"1:{t + 1}" for t in range(time_steps)]

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)

        for n_idx, node_name in enumerate(nodes):
            row = [node_name] + [str(data[t][n_idx]) for t in range(time_steps)]
            writer.writerow(row)


def extract_edges_from_structure(
    structure: dict[str, dict[str, Any]],
) -> set[tuple[str, str]]:
    """Extracts ground-truth directed edges (parent, child) from a structure dict."""
    edges: set[tuple[str, str]] = set()
    for child, info in structure.items():
        for parent in info["parents"]:
            edges.add((parent, child))
    return edges


def load_ground_truth_edges(json_file: Path) -> set[tuple[str, str]]:
    """Loads ground-truth directed edges from a serialized JSON network file."""
    with json_file.open("r", encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = json.load(f)
    return extract_edges_from_structure(data)