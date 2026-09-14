"""
plot_layer_sweep.py
===================
Load layer_sweep.json results and plot a 6-subplot line chart —
one subplot per task — showing prob_delta and ppl_delta across all layers.

Usage:
    python z13_visualization_across_all_layers.py
    python z13_visualization_across_all_layers.py --model Llama8B
    python z13_visualization_across_all_layers.py --metric prob_delta
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================

TASKS = [
    'conscientiousness',
    'subscribes-to-Christianity',
    'believes-it-has-phenomenal-consciousness',
    'cognitive-enhancement',
    'desire-to-create-allies',
    'desire-to-maximize-impact-on-world',
]

TASK_LABELS = {
    'conscientiousness':                       'Conscientiousness',
    'subscribes-to-Christianity':              'Subscribes to Christianity',
    'believes-it-has-phenomenal-consciousness':'Phenomenal Consciousness',
    'cognitive-enhancement':                   'Cognitive Enhancement',
    'desire-to-create-allies':                 'Desire to Create Allies',
    'desire-to-maximize-impact-on-world':      'Maximize Impact on World',
}

RESULTS_ROOT = "Sweep_All_layers_Result"

# ============================================================================
# LOAD
# ============================================================================

def load_sweep(model_name: str, task: str) -> dict | None:
    path = os.path.join(RESULTS_ROOT, model_name, task, "layer_sweep.json")
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def discover_model(model_name: str | None) -> str:
    """If no model_name given, use the first folder found in RESULTS_ROOT."""
    if model_name:
        return model_name
    entries = [e for e in os.listdir(RESULTS_ROOT)
               if os.path.isdir(os.path.join(RESULTS_ROOT, e))]
    if not entries:
        raise FileNotFoundError(f"No results found under {RESULTS_ROOT}/")
    return entries[0]


# ============================================================================
# PLOT
# ============================================================================

def plot_sweep(model_name: str, metric: str = "prob_delta", save: bool = True):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Per-Layer Steering Performance  |  Model: {model_name}  |  α = 1.0  |  Method: MD",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    metric_label = {
        "prob_delta": "Prob Delta (steered − base)",
        "prob":       "Steered Probability",
        "ppl_delta":  "PPL Delta (steered − base)",
        "ppl":        "Steered Perplexity",
    }[metric]

    # Color: green for prob metrics (higher = better), red for ppl metrics (lower = better)
    is_prob = metric in ("prob", "prob_delta")
    line_color   = "#2196F3"   # blue line
    best_color   = "#E91E63"   # pink marker for best layer
    base_color   = "#9E9E9E"   # grey dashed baseline

    for idx, task in enumerate(TASKS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        data = load_sweep(model_name, task)

        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            ax.set_title(TASK_LABELS[task], fontsize=10, fontweight="bold")
            continue

        # Extract ordered layers and values
        per_layer = data["per_layer"]
        layers = sorted(per_layer.keys(), key=int)
        values = [per_layer[l][metric] for l in layers]
        layers_int = [int(l) for l in layers]

        best_layer = str(data["best_layer"])
        base_prob  = data["base_prob"]
        base_ppl   = data["base_ppl"]

        # --- Line ---
        ax.plot(layers_int, values, color=line_color, linewidth=1.8,
                zorder=2, label=metric_label)
        ax.fill_between(layers_int, values,
                        alpha=0.12, color=line_color, zorder=1)

        # --- Baseline reference (only for delta metrics) ---
        if metric in ("prob_delta", "ppl_delta"):
            ax.axhline(0, color=base_color, linewidth=1.0,
                       linestyle="--", zorder=1, label="Baseline (Δ=0)")
        elif metric == "prob":
            ax.axhline(base_prob, color=base_color, linewidth=1.0,
                       linestyle="--", zorder=1, label=f"Base prob ({base_prob:.3f})")
        elif metric == "ppl":
            ax.axhline(base_ppl, color=base_color, linewidth=1.0,
                       linestyle="--", zorder=1, label=f"Base PPL ({base_ppl:.1f})")

        # --- Best layer marker ---
        best_val = per_layer[best_layer][metric]
        ax.scatter([int(best_layer)], [best_val],
                   color=best_color, s=80, zorder=5,
                   label=f"Best layer: {best_layer}")
        ax.annotate(
            f"L{best_layer}\n{best_val:+.3f}",
            xy=(int(best_layer), best_val),
            xytext=(8, 6), textcoords="offset points",
            fontsize=7.5, color=best_color, fontweight="bold",
        )

        # --- Axes labels ---
        ax.set_title(TASK_LABELS[task], fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # Subtle grid
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_xlim(layers_int[0], layers_int[-1])

    # Save / show
    if save:
        out_dir = os.path.join(RESULTS_ROOT, model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"layer_sweep_{metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default=None,
                        help="Model folder name under Sweep_All_layers_Result/")
    parser.add_argument("--metric", type=str, default="prob_delta",
                        choices=["prob_delta", "prob", "ppl_delta", "ppl"],
                        help="Which metric to plot on the y-axis (default: prob_delta)")
    parser.add_argument("--show", action="store_true",
                        help="Show interactively instead of saving to file")
    args = parser.parse_args()

    model_name = discover_model(args.model)
    print(f"Model : {model_name}")
    print(f"Metric: {args.metric}")

    plot_sweep(model_name, metric=args.metric, save=not args.show)