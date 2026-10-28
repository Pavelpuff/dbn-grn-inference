import json
import re
from pathlib import Path
from typing import Any


class BNetModel:
    """Parses .bnet Boolean rule files into truth-table structures for simulation."""

    def __init__(self, bnet_file: Path) -> None:
        self.structure: dict[str, dict[str, Any]] = {}
        self.nodes: list[str] = []
        self.parse_bnet(bnet_file)

    def parse_bnet(self, filepath: Path) -> None:
        """Extracts variables, detects implicit input nodes, and compiles truth tables."""
        rules_raw: dict[str, str] = {}
        all_mentioned_vars: set[str] = set()

        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip()
                if not cleaned or cleaned.startswith("targets") or cleaned.startswith("#"):
                    continue
                if "," in cleaned:
                    target, rule = map(str.strip, cleaned.split(",", 1))
                    rules_raw[target] = rule
                    all_mentioned_vars.add(target)

        for rule in rules_raw.values():
            tokens = re.findall(r"[a-zA-Z0-9_]+", rule)
            for token in tokens:
                if token not in {"not", "and", "or", "true", "false", "0", "1"}:
                    all_mentioned_vars.add(token)

        self.nodes = sorted(all_mentioned_vars)

        for node in self.nodes:
            if node in rules_raw:
                expression = rules_raw[node]
            else:
                print(f"   -> Detected input node (adding self-loop): {node}")
                expression = node

            parents = [
                p
                for p in self.nodes
                if re.search(r"\b" + re.escape(p) + r"\b", expression)
            ]
            truth_table = self._generate_truth_table(expression, parents)

            self.structure[node] = {
                "parents": parents,
                "truth_table": truth_table,
                "rule_str": expression,
            }

    @staticmethod
    def _generate_truth_table(expression: str, parents: list[str]) -> list[int]:
        """Evaluates the boolean expression for all 2^k parent state combinations."""
        k = len(parents)
        table_size = 2**k
        table: list[int] = []

        py_expr = (
            expression.replace("!", " not ").replace("&", " and ").replace("|", " or ")
        )

        for i in range(table_size):
            bin_vals = format(i, f"0{k}b")
            context: dict[str, int] = {
                parent: int(bin_vals[idx]) for idx, parent in enumerate(parents)
            }
            try:
                res = eval(py_expr, {"__builtins__": {}}, context)
                table.append(1 if res else 0)
            except Exception as exc:
                print(f"Error evaluating rule for {parents}: {expression} -> {exc}")
                return [0] * table_size

        return table

    def save_to_json(self, filepath: Path) -> None:
        """Saves the parsed structure to a JSON artifact."""
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.structure, f, indent=4)