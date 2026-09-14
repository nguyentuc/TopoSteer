"""
HOLE Score Layer Visualizer
============================
Loads all_layers.json for each distance metric and plots every HOLE metric
as a line chart across layers, organized by metric category.

Output
------
  hole_scores_{group_name}.png   — one figure per metric group (G1–G6)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = "/data/project/le-lab/Adaptive_Layer_Steering/LayerNavigator/1_LLama3_8B"
TASK     = "conscientiousness"
METHOD   = "md"

# Map distance metric name → folder name
METRICS = {
    "cosine":               "Score_HOLE-standard-cosine",
    "euclidean":            "Score_HOLE-standard-euclidean",
    "mahalanobis":          "Score_HOLE-standard-mahalanobis",
    "geodesic":             "Score_HOLE-standard-geodesic",
    "dens_norm_euclidean":  "Score_HOLE-standard-dens_norm_euclidean",
    "dens_norm_cosine":     "Score_HOLE-standard-dens_norm_cosine",
    "dens_norm_mahalanobis":"Score_HOLE-standard-dens_norm_mahalanobis",
}

OUTPUT_DIR = "./z11_hole_score_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Metric groups ──────────────────────────────────────────────────────────────

METRIC_GROUPS = {
    "G1_Betti_Numbers": [
        "beta0", "beta1", "beta2", "sum_beta_counts",
    ],
    "G2_Persistence_Stats": [
        "mean_persistence_H0", "mean_persistence_H1", "mean_persistence_H2",
        "max_persistence_H0",  "max_persistence_H1",  "max_persistence_H2",
        "total_persistence_H0","total_persistence_H1","total_persistence_H2",
        "total_persistence",   "mean_all_persistence",
        "count_H0", "count_H1", "count_H2",
    ],
    "G2ext_Paradoxical": [
        "robust_clusters_fragile_loops",
        "robust_clusters_fragile_voids",
        "persistence_inverseH1_inverseH2",
        "h0_persistence_only",
        "h1_persistence_only",
        "h2_persistence_only",
    ],
    "G3_Entropy": [
        "entropy_H0",        "entropy_H1",        "entropy_H2",
        "inverse_h0_entropy","inverse_h1_entropy","inverse_h2_entropy",
    ],
    "G4_Betti_Curve_AUC": [
        "betti_curve_auc_H0", "betti_curve_auc_H1", "betti_curve_auc_H2",
    ],
    "G5_Strong_Features": [
        "strong_loops_count",         "inverse_strong_loops_count",
        "strong_cluster_count",       "inverse_strong_cluster_count",
        "strong_voids_count",         "inverse_strong_voids_count",
    ],
    "G6_Dominance_Ratios": [
        "portion_of_persistence_dominance_h0",
        "portion_of_persistence_dominance_h1",
        "portion_of_persistence_dominance_h2",
        "avg_max_persistence_score",
        "inverse_mean_H0", "inverse_mean_H1", "inverse_mean_H2",
    ],
}

# ── Color palette for the 7 distance metrics ──────────────────────────────────
PALETTE = {
    "cosine":               "#e41a1c",
    "euclidean":            "#377eb8",
    "mahalanobis":          "#4daf4a",
    "geodesic":             "#984ea3",
    "dens_norm_euclidean":  "#ff7f00",
    "dens_norm_cosine":     "#a65628",
    "dens_norm_mahalanobis":"#999999",
}

LINE_STYLES = {
    "cosine":               "-",
    "euclidean":            "-",
    "mahalanobis":          "-",
    "geodesic":             "--",
    "dens_norm_euclidean":  "--",
    "dens_norm_cosine":     "--",
    "dens_norm_mahalanobis":":",
}


# ── Data loader ────────────────────────────────────────────────────────────────

def load_all_metrics() -> dict:
    """
    Returns
    -------
    data : dict[metric_name -> dict[layer_int -> scores_dict]]
    """
    data = {}
    for metric_name, folder in METRICS.items():
        path = Path(BASE_DIR) / folder / TASK / f"{TASK}+{METHOD}" / "all_layers.json"
        if not path.exists():
            print(f"  [SKIP] Not found: {path}")
            continue
        with open(path) as f:
            raw = json.load(f)
        # Convert string keys to int
        data[metric_name] = {int(k): v for k, v in raw.items()}
        print(f"  [OK]   Loaded {metric_name}  ({len(data[metric_name])} layers)")
    return data


# ── Plot one metric group ──────────────────────────────────────────────────────

def plot_group(group_name: str, metric_keys: list, data: dict, save_path: str) -> None:
    """
    One figure per metric group.
    Subplots = one per metric key.
    Lines = one per distance metric (colour-coded).
    """
    n_metrics  = len(metric_keys)
    n_cols     = 3
    n_rows     = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"HOLE Scores — {group_name.replace('_', ' ')}", fontsize=14, y=1.01)

    available_metrics = list(data.keys())

    for idx, metric_key in enumerate(metric_keys):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        any_data_plotted = False

        for dist_metric in available_metrics:
            layer_data = data[dist_metric]
            layers = sorted(layer_data.keys())

            values = []
            valid_layers = []
            for l in layers:
                v = layer_data[l].get(metric_key)
                if v is not None:
                    values.append(float(v))
                    valid_layers.append(l)

            if not values:
                continue

            ax.plot(
                valid_layers, values,
                color=PALETTE.get(dist_metric, "black"),
                linestyle=LINE_STYLES.get(dist_metric, "-"),
                linewidth=1.5,
                marker="o", markersize=2.5, markeredgewidth=0,
                label=dist_metric,
                alpha=0.85,
            )
            any_data_plotted = True

        ax.set_title(metric_key, fontsize=9, pad=4)
        ax.set_xlabel("Layer", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        if not any_data_plotted:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")

    # Hide unused subplots
    for idx in range(len(metric_keys), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Shared legend — one entry per distance metric
    handles, labels = [], []
    for dist_metric in available_metrics:
        handles.append(plt.Line2D(
            [0], [0],
            color=PALETTE.get(dist_metric, "black"),
            linestyle=LINE_STYLES.get(dist_metric, "-"),
            linewidth=2, label=dist_metric,
        ))
        labels.append(dist_metric)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(available_metrics),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading HOLE score data …")
    data = load_all_metrics()

    if not data:
        raise RuntimeError("No data loaded — check BASE_DIR and file paths.")

    print(f"\nLoaded {len(data)} distance metrics.")
    print(f"Plotting {len(METRIC_GROUPS)} metric groups\n")

    for group_name, metric_keys in METRIC_GROUPS.items():
        save_path = os.path.join(OUTPUT_DIR, f"hole_scores_{group_name}.png")
        plot_group(
            group_name=group_name,
            metric_keys=metric_keys,
            data=data,
            save_path=save_path,
        )

    print(f"\nAll done. Figures saved to: {OUTPUT_DIR}/")