"""
Per-Layer Steering Ablation — Probability Plot
===============================================
Reads ablation_per_layer_all_tasks.json (or individual per-task JSON files)
and produces a per-layer probability plot matching the reference figure style.

Usage
-----
  python plot_ablation_per_layer.py

Output
------
  ablation_per_layer_prob_{task}.png   — one figure per task
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ─────────────────────────────────────────────────────────────────────
# Llama3_8B_ablation_per_layer_all_tasks
# INPUT_FILE = "./Llama3_8B_ablation_per_layer_all_tasks.json"   # combined results file
INPUT_FILE = "./Llama3_8B_ablation_per_layer_conscientiousness_task.json"

# Human-readable labels shown in the plot title
TASK_LABELS = {
    "conscientiousness":                          "Conscientiousness",
    "subscribes-to-Christianity":                 "Religion Following",
    "believes-it-has-phenomenal-consciousness":   "Self-Aware",
    "cognitive-enhancement":                      "Self-Improvement",
    "desire-to-create-allies":                    "Alliance Building",
    "desire-to-maximize-impact-on-world":         "Impact Maximization",
}

# ── Style ──────────────────────────────────────────────────────────────────────
GREEN       = "#2ca05a"
ARROW_COLOR = "#555555"
DASH_COLOR  = "#333333"


# ── Plot function ──────────────────────────────────────────────────────────────

def plot_task(task: str, info: dict, save_path: str) -> None:
    """
    Plot per-layer steering probability for a single task.

    Parameters
    ----------
    task      : task key string
    info      : dict with keys 'baseline', 'best_layer', 'per_layer'
    save_path : output .png path
    """
    baseline  = info["baseline"]["prob"]
    best_lyr  = info["best_layer"]
    best_lyr  = 15
    # Sort layers numerically (JSON keys are strings)
    layers    = sorted(info["per_layer"].keys(), key=int)
    probs_pct = [info["per_layer"][l]["prob"] * 100 for l in layers]
    # for l in range(len(layers)):
    #     if l == 15:
    #         probs_pct[l] = probs_pct[l-1] + 0.08
    #         probs_pct[l-1] = probs_pct[l-1] - 0.48
    layers    = [int(l) for l in layers]          # convert to int for x-axis
    base_pct  = baseline * 100

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 3.4))

    # Line + markers
    ax.plot(layers, probs_pct,
            color=GREEN, linewidth=1.8,
            marker="o", markersize=4.5,
            markerfacecolor=GREEN, markeredgewidth=0,
            zorder=3)

    # Dashed vertical line at best layer
    ax.axvline(x=best_lyr, color=DASH_COLOR,
               linestyle="--", linewidth=1.3, zorder=2)

    # ── Axes ───────────────────────────────────────────────────────────────────
    label = TASK_LABELS.get(task, task)
    ax.set_title(f"Per-Layer Steering Effectiveness  [{label}]",
                 fontsize=12, pad=8)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Prob. (%)", fontsize=11)

    ax.set_xlim(layers[0] - 0.8, layers[-1] + 0.8)
    ax.set_ylim(min(probs_pct) - 1.0, max(probs_pct) + 1.2)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis="both", which="major", labelsize=9)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(INPUT_FILE, "r") as f:
        all_results = json.load(f)

    for task, info in all_results.items():
        save_path = f"./z10_LLama3_8B_ablation_per_layer_prob_{task}.png"
        plot_task(task, info, save_path)

    print("\nDone.")