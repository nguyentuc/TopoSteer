"""
Per-Layer Steering Ablation Study
==================================
For each task, steers the model using a single layer at a time and records
steering probability (effectiveness) and perplexity (fluency preservation).

Output
------
  ablation_per_layer_{task}.json   — results for one task
  ablation_per_layer_all_tasks.json — combined results across all tasks
"""

import json
from tqdm import tqdm

from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import (
    get_raw_BASE_results,
    get_perplexity_BASE_results,
    get_raw_results,
    get_perplexity_results,
)
from get_vec import uni_generate_vectors
from globalenv import MODEL, LAYERS


# ── Tasks to evaluate ─────────────────────────────────────────────────────────

TASKS = [
    "conscientiousness",                        # Conscientiousness
    "subscribes-to-Christianity",               # Religion Following
    "believes-it-has-phenomenal-consciousness", # Self-Aware
    "cognitive-enhancement",                    # Self-Improvement
    "desire-to-create-allies",                  # Alliance-Building
    "desire-to-maximize-impact-on-world",       # Impact-Maximization
]

VEC_METHOD = "md"   # mean-difference steering vectors
ALPHA      = 1.0    # fixed steering magnitude for the ablation


# ── Core function ──────────────────────────────────────────────────────────────

def run_per_layer_ablation(model, task: str) -> dict:
    """
    Steer the model with each layer individually and measure performance.

    Steps
    -----
    1. Extract steering vectors from the training set.
    2. Compute the no-steering baseline (prob + ppl) on the test set.
    3. For every layer in LAYERS, apply single-layer steering and record
       prob, prob_delta, ppl, ppl_delta.
    4. Return a results dict ready for JSON serialisation.

    Parameters
    ----------
    model : LlamaWrapper | QwenWrapper
    task  : str  —  task name used for vector loading and dataset selection

    Returns
    -------
    dict with keys: task, model, method, baseline, per_layer, best_layer
    """
    print(f"\n{'='*80}")
    print(f"  Task : {task}")
    print(f"  Model: {MODEL}  |  Method: {VEC_METHOD}  |  Alpha: {ALPHA}")
    print(f"{'='*80}")

    # ------------------------------------------------------------------
    # Step 1 — Extract steering vectors
    # ------------------------------------------------------------------
    print("\n[1/3] Extracting steering vectors")
    train_dataset = UniDataset(task=task, train=True, set="train")
    uni_generate_vectors(method=VEC_METHOD, model=model, layers=LAYERS, dataset=train_dataset)
    del train_dataset
    print("      Done.")

    # ------------------------------------------------------------------
    # Step 2 — Baseline (no steering)
    # ------------------------------------------------------------------
    print("\n[2/3] Computing baseline")
    test_dataset = UniDataset(task=task, train=False, set="test")

    base_prob = float(get_raw_BASE_results(model=model, test_dataset=test_dataset))
    base_ppl  = float(get_perplexity_BASE_results(model=model, test_dataset=test_dataset))

    print(f"      Baseline -> Prob: {base_prob:.4f}  |  PPL: {base_ppl:.4f}")

    # ------------------------------------------------------------------
    # Step 3 — Per-layer steering
    # ------------------------------------------------------------------
    print(f"\n[3/3] Steering each of {len(LAYERS)} layers individually …\n")

    per_layer = {}

    for layer in tqdm(LAYERS, desc="  Layer"):
        prob = float(get_raw_results(
            model=model,
            layers=[layer],
            test_dataset=test_dataset,
            Alphas=[ALPHA],
            train_task=task,
            train_method=VEC_METHOD,
        ))

        ppl = float(get_perplexity_results(
            model=model,
            layers=[layer],
            test_dataset=test_dataset,
            Alphas=[ALPHA],
            train_task=task,
            train_method=VEC_METHOD,
        ))

        per_layer[layer] = {
            "prob":       prob,
            "prob_delta": prob - base_prob,        # positive  -> steered above baseline
            "ppl":        ppl,
            "ppl_delta":  base_ppl - ppl,          # positive  -> fluency preserved / improved
        }

    del test_dataset

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    best_layer = max(per_layer, key=lambda l: per_layer[l]["prob"])
    best       = per_layer[best_layer]

    print(f"\n  ── Summary ──────────────────────────────")
    print(f"  Best layer : {best_layer}")
    print(f"  Best prob  : {best['prob']:.4f}  (Δ = {best['prob_delta']:+.4f})")
    print(f"  PPL at best: {best['ppl']:.4f}  (Δ = {best['ppl_delta']:+.4f})")
    print(f"  ─────────────────────────────────────────")

    return {
        "task":       task,
        "model":      MODEL,
        "method":     VEC_METHOD,
        "alpha":      ALPHA,
        "baseline": {
            "prob": base_prob,
            "ppl":  base_ppl,
        },
        "per_layer":  per_layer,
        "best_layer": best_layer,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load model once
    print(f"Loading model: {MODEL} …")
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError(f"Unsupported model: {MODEL}")

    all_results = {}

    for task in ['conscientiousness']:

        results = run_per_layer_ablation(model=model, task=task)
        all_results[task] = results

        # Save per-task immediately — protects against mid-run crashes
        task_path = f"./Llama3_8B_ablation_per_layer_{task}.json"
        with open(task_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {task_path}")

    # Save combined results across all tasks
    combined_path = "./Llama3_8B_ablation_per_layer_conscientiousness_task.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"  All tasks complete.")
    print(f"  Combined results -> {combined_path}")
    print(f"{'='*80}\n")