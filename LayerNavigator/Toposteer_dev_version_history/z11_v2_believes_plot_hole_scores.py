"""
HOLE Score Layer Visualizer — Euclidean Distance Only
======================================================
Plots every HOLE metric across layers for the euclidean distance metric
in a single figure, subplots organised by metric group.

Output
------
  hole_scores_euclidean_all.png
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR   = "/data/project/le-lab/Adaptive_Layer_Steering/LayerNavigator/1_LLama3_8B"
TASK       = "conscientiousness"
METHOD     = "md"
DIST       = "euclidean"
FOLDER     = f"Score_HOLE-standard-{DIST}"

OUTPUT_DIR = "./z11_v2_hole_score_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── All HOLE metrics to plot, in display order ─────────────────────────────────
# Grouped with section headers (title row) for visual separation

SECTIONS = {
    "Persistence — Mean": [
        "mean_persistence_H0", "mean_persistence_H1", "mean_persistence_H2",
    ],
    "Persistence — Max": [
        "max_persistence_H0", "max_persistence_H1", "max_persistence_H2",
    ],
    "Persistence — Total & Combined": [
        "total_persistence_H0", "total_persistence_H1", "total_persistence_H2",
        "total_persistence", "mean_all_persistence",
    ],
    "Feature Counts": [
        "count_H0", "count_H1", "count_H2",
    ],
    "Paradoxical Variants": [
        "robust_clusters_fragile_loops",
        "robust_clusters_fragile_voids",
        "persistence_inverseH1_inverseH2",
    ],
    "Entropy": [
        "entropy_H0", "entropy_H1",
        "inverse_h0_entropy", "inverse_h1_entropy",
    ],
    "Betti Curve AUC": [
        "betti_curve_auc_H0", "betti_curve_auc_H1",
    ],
    "Strong Features": [
        "strong_loops_count", "inverse_strong_loops_count",
        "strong_cluster_count", "inverse_strong_cluster_count",
    ],
    "Dominance & Inverse Means": [
        "portion_of_persistence_dominance_h0",
        "portion_of_persistence_dominance_h1",
        "avg_max_persistence_score",
        "inverse_mean_H0", "inverse_mean_H1",
    ],
}

# Flatten to ordered list for subplot indexing
ALL_METRICS = [m for metrics in SECTIONS.values() for m in metrics]

# Build a lookup: metric_key -> section title (for subplot titles)
METRIC_SECTION = {m: sec for sec, metrics in SECTIONS.items() for m in metrics}

# Color
LINE_COLOR = "#377eb8"   # blue for euclidean


# ── Load data ──────────────────────────────────────────────────────────────────

def load_euclidean() -> dict:
    """Returns dict[layer_int -> scores_dict]"""
    path = Path(BASE_DIR) / FOLDER / TASK / f"{TASK}+{METHOD}" / "all_layers.json"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found:\n  {path}")
    with open(path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}
    print(f"Loaded {len(data)} layers from: {path}")
    return data


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_all(data: dict) -> None:

    n_metrics = len(ALL_METRICS)
    n_cols    = 4
    n_rows    = (n_metrics + n_cols - 1) // n_cols

    layers = sorted(data.keys())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"HOLE Scores — Euclidean Distance  [{TASK}]",
        fontsize=14, fontweight="bold", y=1.005,
    )

    for idx, metric_key in enumerate(ALL_METRICS):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # Extract values
        values = []
        valid_layers = []
        for l in layers:
            v = data[l].get(metric_key)
            if v is not None:
                values.append(float(v))
                valid_layers.append(l)

        if values:
            ax.plot(
                valid_layers, values,
                color=LINE_COLOR, linewidth=1.8,
                marker="o", markersize=3, markeredgewidth=0,
                alpha=0.9,
            )
            # Light fill under line
            ax.fill_between(valid_layers, values,
                            alpha=0.08, color=LINE_COLOR)

            # Dashed vertical line at the layer with the highest score
            best_layer = valid_layers[values.index(max(values))]
            ax.axvline(
                x=best_layer,
                color="#333333", linestyle="--",
                linewidth=1.2, alpha=0.7, zorder=2,
            )
            # Label the best layer index just above the minimum value
            ax.text(
                best_layer, min(values),
                f" L{best_layer}",
                fontsize=6.5, color="#333333",
                va="bottom", ha="left",
            )
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")

        # Subplot title: metric name + section label in grey
        section = METRIC_SECTION.get(metric_key, "")
        ax.set_title(metric_key, fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel("Layer", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Light section background stripe (alternating groups)
        section_idx = list(SECTIONS.keys()).index(section)
        if section_idx % 2 == 0:
            ax.set_facecolor("#f7f7f7")

    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout(h_pad=2.5, w_pad=2.0)

    save_path = os.path.join(OUTPUT_DIR, "hole_scores_euclidean_all.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_euclidean()
    plot_all(data)