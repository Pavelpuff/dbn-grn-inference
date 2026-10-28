import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    BNFINDER_EXEC,
    DATASETS_DIR,
    NETWORK_SIZES,
    NETWORKS_DIR,
    RESULTS_DIR,
    SCORING_FUNCTIONS,
    SEED,
    ensure_directories,
)
from src.data.io import load_ground_truth_edges, save_dataset_file
from src.data.simulator import BooleanNetworkSimulator
from src.inference.bnfinder import run_bnfinder
from src.networks.rbn import RandomBooleanNetwork
from src.utils.metrics import evaluate_network


def main() -> None:
    random.seed(SEED)
    ensure_directories()
    print(f"=== PART I: SYNTHETIC RBN PIPELINE (Sizes: {NETWORK_SIZES}) ===")

    if not BNFINDER_EXEC.exists():
        print(f"CRITICAL ERROR: BNFinder executable not found at: {BNFINDER_EXEC}")
        print("Run `make setup-legacy` or copy your Python 2.7 environment to `.venv_legacy`.")
        return

    print("\n[Task 1] Generating Random Boolean Networks...")
    for size in NETWORK_SIZES:
        net = RandomBooleanNetwork(size)
        net.generate()
        net.save_to_file(NETWORKS_DIR / f"net_{size}.json")

    print("\n[Task 2] Simulating Time-Series Datasets...")
    dataset_map: list[dict[str, Any]] = []
    scenarios: list[tuple[str, int, str, int]] = [
        ("sync_base", 100, "sync", 1),
        ("async", 200, "async", 1),
        ("sampled", 150, "sync", 3),
        ("transient", 20, "sync", 1),
    ]

    for size in NETWORK_SIZES:
        net_file = NETWORKS_DIR / f"net_{size}.json"
        sim = BooleanNetworkSimulator.from_json(net_file)

        for mode_suffix, steps, mode, samp in scenarios:
            data = sim.simulate(steps, mode, samp)
            name = f"data_{size}_{mode_suffix}"
            path = DATASETS_DIR / f"{name}.txt"

            save_dataset_file(path, sim.nodes, data)
            dataset_map.append(
                {
                    "path": path,
                    "net_file": net_file,
                    "name": name,
                    "size": size,
                    "mode": mode_suffix,
                }
            )

    print("\n[Task 3] Running BNFinder2 Inference...")
    results_table: list[dict[str, Any]] = []

    for ds in dataset_map:
        print(f"Processing {ds['name']}...", end=" ", flush=True)
        true_edges = load_ground_truth_edges(ds["net_file"])

        for method in SCORING_FUNCTIONS:
            out_dir: Path = RESULTS_DIR / ds["name"] / method
            out_dir.mkdir(parents=True, exist_ok=True)

            success, _ = run_bnfinder(ds["path"], out_dir, method)
            if success:
                res_file = out_dir / "structure.cpd"
                prec, rec, f1, hamm = evaluate_network(true_edges, res_file)
                results_table.append(
                    {
                        "Dataset": ds["name"],
                        "Size": ds["size"],
                        "Mode": ds["mode"],
                        "Method": method,
                        "Precision": prec,
                        "Recall": rec,
                        "F1": f1,
                        "Hamming": hamm,
                    }
                )
        print("Done.")

    df = pd.DataFrame(results_table)
    csv_path = RESULTS_DIR / "part1_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[*] Saved Part I metrics to {csv_path}")

    print("\n" + "=" * 85)
    print(f"{'DATASET':<30} | {'METH':<5} | {'PREC':<5} | {'REC':<5} | {'F1':<5} | {'HAMM':<5}")
    print("-" * 85)
    for r in results_table:
        print(
            f"{r['Dataset']:<30} | {r['Method']:<5} | {r['Precision']:.2f}  | "
            f"{r['Recall']:.2f}  | {r['F1']:.2f}  | {r['Hamming']:<5}"
        )
    print("=" * 85)


if __name__ == "__main__":
    main()