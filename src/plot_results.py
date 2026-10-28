import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ASSETS_DIR, NETWORK_SIZES, RESULTS_DIR, ensure_directories

plt.style.use("seaborn-v0_8-whitegrid")

MODES = ["transient", "sync_base", "async", "sampled"]
COLORS = {
    "transient": "#d62728",
    "sync_base": "#1f77b4",
    "async": "gray",
    "sampled": "orange",
}
LABELS = {
    "transient": "Transient",
    "sync_base": "Sync Base",
    "async": "Async",
    "sampled": "Sampled",
}
MARKERS = {"transient": "o", "sync_base": "s", "async": "^", "sampled": "x"}


def plot_part1_metric(df_mdl: pd.DataFrame, metric_name: str, filename: str, fig_num: int) -> None:
    """Renders line plots for Part I synthetic network benchmarks (MDL scoring)."""
    plt.figure(figsize=(10, 6))
    for mode in MODES:
        subset = df_mdl[df_mdl["Mode"] == mode].sort_values("Size")
        if not subset.empty:
            plt.plot(
                subset["Size"],
                subset[metric_name],
                marker=MARKERS[mode],
                label=LABELS[mode],
                color=COLORS[mode],
                linewidth=2,
                markersize=8,
            )

    plt.title(f"Figure {fig_num}: {metric_name} vs Network Size (MDL)", fontsize=14)
    plt.xlabel("Network Size (Nodes)", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(NETWORK_SIZES)
    plt.ylim(-0.05, 1.1)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = ASSETS_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"   -> Saved {out_path}")


def plot_part2_figures(df2: pd.DataFrame) -> None:
    """Renders bar charts for Part II Mammalian Cell Cycle reconstruction."""
    scenarios = ["Sync Long", "Transient", "Async Long"]
    mdl_df = df2[df2["Method"] == "MDL"].set_index("Scenario").reindex(scenarios)
    bde_df = df2[df2["Method"] == "BDE"].set_index("Scenario").reindex(scenarios)

    x = np.arange(len(scenarios))

    # Figure 4: Detailed Metrics for MDL
    plt.figure(figsize=(10, 6))
    width = 0.25

    plt.bar(x - width, mdl_df["Precision"], width, label="Precision", color="#2ca02c", alpha=0.9)
    plt.bar(x, mdl_df["F1"], width, label="F1 Score", color="#1f77b4", alpha=0.9)
    plt.bar(x + width, mdl_df["Recall"], width, label="Recall", color="#aec7e8", alpha=0.9)

    plt.title("Figure 4: Mammalian Cell Cycle Metrics (MDL)", fontsize=14)
    plt.xticks(x, scenarios, fontsize=11)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.legend(loc="upper left", frameon=True)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for i, v in enumerate(mdl_df["Precision"]):
        plt.text(i - width, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
    for i, v in enumerate(mdl_df["F1"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_fig4 = ASSETS_DIR / "part2_metrics.png"
    plt.savefig(out_fig4, dpi=300)
    plt.close()
    print(f"   -> Saved {out_fig4}")

    # Figure 5: Method Comparison (MDL vs BDE)
    plt.figure(figsize=(8, 6))
    width_comp = 0.35
    plt.bar(x - width_comp / 2, mdl_df["F1"], width_comp, label="MDL", color="#1f77b4", alpha=0.9)
    plt.bar(x + width_comp / 2, bde_df["F1"], width_comp, label="BDE", color="#ff7f0e", alpha=0.9)

    plt.title("Figure 5: Scoring Method Comparison (F1 Score)", fontsize=14)
    plt.xticks(x, scenarios, fontsize=11)
    plt.ylim(0, 0.7)
    plt.ylabel("F1 Score")
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_fig5 = ASSETS_DIR / "part2_method_comparison.png"
    plt.savefig(out_fig5, dpi=300)
    plt.close()
    print(f"   -> Saved {out_fig5}")


def main() -> None:
    ensure_directories()
    part1_csv = RESULTS_DIR / "part1_metrics.csv"
    part2_csv = RESULTS_DIR / "part2_metrics.csv"

    if part1_csv.exists():
        df1 = pd.read_csv(part1_csv)
        df1_mdl = df1[df1["Method"] == "MDL"]
        plot_part1_metric(df1_mdl, "F1", "part1_f1.png", 1)
        plot_part1_metric(df1_mdl, "Precision", "part1_precision.png", 2)
        plot_part1_metric(df1_mdl, "Recall", "part1_recall.png", 3)
    else:
        print(f"Warning: {part1_csv} not found. Run `make synthetic` first.")

    if part2_csv.exists():
        df2 = pd.read_csv(part2_csv)
        plot_part2_figures(df2)
    else:
        print(f"Warning: {part2_csv} not found. Run `make biological` first.")


if __name__ == "__main__":
    main()