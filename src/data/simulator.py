import json
import random
from pathlib import Path
from typing import Any


class BooleanNetworkSimulator:
    """Simulates discrete-time dynamics of a Boolean Network."""

    def __init__(self, structure: dict[str, dict[str, Any]]) -> None:
        self.structure = structure
        self.nodes = self._sort_nodes(list(structure.keys()))
        self.n_nodes = len(self.nodes)

    @staticmethod
    def _sort_nodes(nodes: list[str]) -> list[str]:
        """Sorts x0..xN numerically and biological gene names alphabetically."""
        if all(n.startswith("x") and n[1:].isdigit() for n in nodes):
            return sorted(nodes, key=lambda x: int(x[1:]))
        return sorted(nodes)

    @classmethod
    def from_json(cls, network_file: Path) -> "BooleanNetworkSimulator":
        """Instantiates the simulator from a saved JSON topology file."""
        with network_file.open("r", encoding="utf-8") as f:
            structure: dict[str, dict[str, Any]] = json.load(f)
        return cls(structure)

    def _get_next_val(self, node: str, current_state_dict: dict[str, int]) -> int:
        """Computes the next binary state of a single node from its parent states."""
        node_data = self.structure[node]
        parents: list[str] = node_data["parents"]
        table: list[int] = node_data["truth_table"]

        if not parents:
            return current_state_dict.get(node, 0)

        parent_vals = [str(current_state_dict[p]) for p in parents]
        idx = int("".join(parent_vals), 2)
        return table[idx]

    def simulate(
        self, steps: int, mode: str = "sync", sampling: int = 1
    ) -> list[list[int]]:
        """
        Runs the dynamical simulation.
        :param steps: Number of discrete time steps.
        :param mode: 'sync' (simultaneous updates) or 'async' (single random node update).
        :param sampling: Sub-sampling interval (records state every Nth step).
        """
        start_state = [random.randint(0, 1) for _ in range(self.n_nodes)]
        trajectory: list[list[int]] = [start_state]
        current = start_state

        for t in range(1, steps + 1):
            curr_dict = dict(zip(self.nodes, current, strict=True))

            if mode == "sync":
                next_state = [self._get_next_val(n, curr_dict) for n in self.nodes]
            else:
                next_state = list(current)
                idx = random.randint(0, self.n_nodes - 1)
                next_state[idx] = self._get_next_val(self.nodes[idx], curr_dict)

            current = next_state

            if t % sampling == 0:
                trajectory.append(current)

        return trajectory