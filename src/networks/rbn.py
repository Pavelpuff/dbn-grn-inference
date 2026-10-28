import json
import random
from pathlib import Path
from typing import Any


class RandomBooleanNetwork:
    """Generates a Random Boolean Network (RBN) with random topology and truth tables."""

    def __init__(self, n_nodes: int, max_parents: int = 3) -> None:
        self.n_nodes = n_nodes
        self.max_parents = max_parents
        self.nodes: list[str] = [f"x{i}" for i in range(n_nodes)]
        self.structure: dict[str, dict[str, Any]] = {}

    def generate(self) -> None:
        """Generates random parent sets and truth tables for each node."""
        for node in self.nodes:
            k = random.randint(1, self.max_parents)
            parents = random.sample(self.nodes, k)
            table_size = 2**k
            truth_table = [random.randint(0, 1) for _ in range(table_size)]

            self.structure[node] = {
                "parents": parents,
                "truth_table": truth_table,
            }

    def save_to_file(self, filepath: Path) -> None:
        """Serializes the ground-truth network topology to a JSON file."""
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.structure, f, indent=4)