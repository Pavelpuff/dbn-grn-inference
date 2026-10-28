import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    BNFINDER_EXEC,
    DATASETS_DIR,
    MODEL_BNET_FILE,
    MODEL_NAME,
    NETWORKS_DIR,
    RESULTS_DIR,
    SCORING_FUNCTIONS,
    SEED,
    ensure_directories,
)
from src.data.io import extract_edges_from_structure, save_dataset_file
from src.data.simulator import BooleanNetworkSimulator
from src.inference.bnfinder import run_bnfinder
from src.networks.bnet_parser import BNetModel
from src.utils.metrics import evaluate_network


def main() -> None:
    random.seed(SEED)
    ensure_directories()
    print(f"=== PART II: REAL NETWORK RECONSTRUCTION ({MODEL_NAME}) ===")

    if not BNFINDER_EXEC.exists():
        print(f"CRITICAL ERROR: BNFinder executable not found at: {BNFINDER_EXEC}")
        return

    if not MODEL_BNET_FILE.exists():
        print(f"Error: {MODEL_BNET_FILE} not found. Place 'model.bnet' in 'data/raw/'.")
        return

    print("\n[Step 1] Parsing .bnet biological model...")
    model = BNetModel(MODEL_BNET_FILE)
    json_path = NETWORKS_DIR / f"{MODEL_NAME}.json"
    model.save_to_json(json_path)
    print(f"   Nodes: {len(model.nodes)} | Saved structure to {json_path}")

    print("\n[Step 2] Simulating Multi-Replicate Trajectories...")
    sim = BooleanNetworkSimulator(model.structure)
    scenarios: list[tuple[str, str, int, str, int]] = [
        ("sync_long", "Sync Long", 100, "sync", 1),
        ("transient_bursts", "Transient", 20, "sync", 1),
        ("async_long", "Async Long", 200, "async", 1),
    ]

    dataset_map: list[dict[str, Any]] = []
    for suffix, label, steps, mode, samp in scenarios:
        all_data: list[list[int]] = []
        for _ in range(5):
            traj = sim.simulate(steps, mode, samp)
            all_data.extend(traj)

        name = f"{MODEL_NAME}_{suffix}"
        path = DATASETS_DIR / f"{name}.txt"
        save_dataset_file(path, sim.nodes, all_data)
        dataset_map.append({"path": path, "name": name, "scenario_label": label})

    print("\n[Step 3] Running BNFinder2 Inference...")
    results_table: list[dict[str, Any]] = []
    true_edges = extract_edges_from_structure(model.structure)
    print(f"   Ground Truth Edges: {len(true_edges)}")

    for ds in dataset_map:
        print(f"Processing {ds['name']}...", end=" ", flush=True)
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
                        "Scenario": ds["scenario_label"],
                        "Method": method,
                        "Precision": prec,
                        "Recall": rec,
                        "F1": f1,
                        "Hamming": hamm,
                    }
                )
        print("Done.")

    df = pd.DataFrame(results_table)
    csv_path = RESULTS_DIR / "part2_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[*] Saved Part II metrics to {csv_path}")

    print("\n" + "=" * 95)
    print(f"{'DATASET':<40} | {'METH':<5} | {'PREC':<5} | {'REC':<5} | {'F1':<5} | {'HAMM':<5}")
    print("-" * 95)
    for r in results_table:
        print(
            f"{r['Dataset']:<40} | {r['Method']:<5} | {r['Precision']:.2f}  | "
            f"{r['Recall']:.2f}  | {r['F1']:.2f}  | {r['Hamming']:<5}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()