import ast
from pathlib import Path
from typing import Any


def evaluate_network(
    true_edges: set[tuple[str, str]], inferred_file: Path
) -> tuple[float, float, float, int]:
    """
    Compares the inferred network against ground truth.
    Returns (Precision, Recall, F1-Score, Structural Hamming Distance).
    """
    inferred_edges: set[tuple[str, str]] = set()

    if inferred_file.exists():
        try:
            content = inferred_file.read_text(encoding="utf-8").strip()
            if content:
                data: dict[str, dict[str, Any]] = ast.literal_eval(content)
                for child, props in data.items():
                    parents: list[str] = props.get("pars", [])
                    for p in parents:
                        inferred_edges.add((p, child))
        except Exception as exc:
            print(f" [Parse Error] {exc}")

    tp = len(true_edges.intersection(inferred_edges))
    fp = len(inferred_edges - true_edges)
    fn = len(true_edges - inferred_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    hamming = fp + fn

    return precision, recall, f1, hamming