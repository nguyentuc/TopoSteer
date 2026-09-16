"""
Baseline: steer layers chosen at RANDOM from the model's MIDDLE layers
(instead of by D+C, D-only, or C-only).

Identical to main_c_ablation.py (MD extraction, top-{1,3,5} steering at alpha=1.0
over all 6 Anth_MAIN tasks, prob + perplexity eval) EXCEPT the layers to steer are
drawn uniformly at random, without replacement, from the middle third of the network:
    pool = range(L // 3, 2 * L // 3)            # L = len(LAYERS)
No D/C/s scoring is used, so get_score is skipped entirely; only the MD steering
vectors (needed to actually steer) are (re)generated.

Because a single random draw is noisy, we repeat the draw N_TRIALS times per
(task, num_layers) with a fixed SEED and report mean/std plus every trial's detail.
Set N_TRIALS = 1 to match the cost/structure of the c/d ablations exactly.

Outputs (run from the LayerNavigator/ directory -- relative paths):
  - steered generations -> ./Results/{task}-test/{task}+md/Res_*.json  (via get_raw_results)
  - baseline summary     -> ./main_randomlayerselection_ablation_results.json
        {"_config": {"seed": ..., "n_trials": ..., "middle_pool": [low, high)},
         task: {"base_prob": p, "base_ppl": q,
                "1": {"prob_mean": ..., "prob_std": ..., "delta_mean": ...,
                      "ppl_mean": ..., "ppl_std": ..., "ppl_delta_mean": ...,
                      "trials": [{"seed":..., "layers":[...], "prob":..., "delta":...,
                                  "perplexity":..., "ppl_delta":...}, ...]},
                "3": {...}, "5": {...}}}
"""

from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *

import os
import json
import random
import statistics

SELECTION       = "randomlayerselection"   # random middle-layer draw (no scores)
NUM_LAYERS_LIST = [2]
SEED            = 42                        # base seed for reproducibility
N_TRIALS        = 1                         # random draws per (task, num_layers); set 1 for single draw


def middle_pool(layers):
    """Middle third of the network: candidate layers for the random baseline."""
    L = len(layers)
    lo, hi = L // 3, 2 * L // 3              # e.g. L=64 -> [21, 42); L=28 -> [9, 18)
    return list(layers)[lo:hi], lo, hi


def sample_middle_layers(pool, num_layers, rng):
    """Sample num_layers distinct layers from pool (sorted ascending for readability)."""
    assert num_layers <= len(pool), (
        f"num_layers={num_layers} exceeds middle-pool size={len(pool)}"
    )
    return sorted(rng.sample(pool, num_layers))


if __name__ == "__main__":
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    pool, pool_lo, pool_hi = middle_pool(LAYERS)
    print(f"Middle-layer pool (random baseline): layers {pool}  "
          f"[{pool_lo}, {pool_hi}) of {len(LAYERS)} total")

    all_results = {"_config": {"seed": SEED, "n_trials": N_TRIALS,
                               "middle_pool": [pool_lo, pool_hi]}}

    for task in Anth_MAIN:
        print(f"\n############## Task: {task}  (selection = random middle layers) ##############")

        # --- Base (no steering) ---
        test_dataset = UniDataset(task=task, train=False, set="test")
        base_prob = get_raw_BASE_results(model=model, test_dataset=test_dataset)
        base_ppl  = get_perplexity_BASE_results(model=model, test_dataset=test_dataset)
        print(f"Base Prob: {base_prob:.4f} | Base PPL: {base_ppl:.4f}")

        # --- MD vectors (needed to steer; NO scoring for the random baseline) ---
        train_dataset = UniDataset(task=task, train=True, set="train")
        uni_generate_vectors(method="md", model=model, layers=LAYERS, dataset=train_dataset)

        # --- Steering evaluation: layers chosen at random from the middle third ---
        task_res = {"base_prob": float(base_prob), "base_ppl": float(base_ppl)}
        for num_layers in NUM_LAYERS_LIST:
            trials = []
            for t in range(N_TRIALS):
                # Deterministic, distinct seed per (task, num_layers, trial)
                rng = random.Random(f"{SEED}-{task}-{num_layers}-{t}")
                layers = sample_middle_layers(pool, num_layers, rng)

                test_prob = get_raw_results(
                    model=model,
                    layers=layers,
                    test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers,
                    train_task=task,
                    train_method="md",
                )
                test_ppl = get_perplexity_results(
                    model=model,
                    layers=layers,
                    test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers,
                    train_task=task,
                    train_method="md",
                )
                delta     = test_prob - base_prob
                ppl_delta = base_ppl - test_ppl          # >0 = steering lowered (improved) perplexity
                print(f"  [random] {num_layers} layer(s) trial {t}: {layers} | "
                      f"Prob={test_prob:.4f} (delta={delta:+.4f}) | "
                      f"PPL={test_ppl:.4f} (delta={ppl_delta:+.4f})")
                trials.append({
                    "seed":       t,
                    "layers":     layers,
                    "prob":       float(test_prob),
                    "delta":      float(delta),
                    "perplexity": float(test_ppl),
                    "ppl_delta":  float(ppl_delta),
                })

            probs = [tr["prob"] for tr in trials]
            ppls  = [tr["perplexity"] for tr in trials]
            task_res[str(num_layers)] = {
                "prob_mean":      float(statistics.mean(probs)),
                "prob_std":       float(statistics.pstdev(probs)) if len(probs) > 1 else 0.0,
                "delta_mean":     float(statistics.mean(probs) - base_prob),
                "ppl_mean":       float(statistics.mean(ppls)),
                "ppl_std":        float(statistics.pstdev(ppls)) if len(ppls) > 1 else 0.0,
                "ppl_delta_mean": float(base_ppl - statistics.mean(ppls)),
                "trials":         trials,
            }
            print(f"  [random] {num_layers} layer(s) SUMMARY over {N_TRIALS} trial(s): "
                  f"Prob={task_res[str(num_layers)]['prob_mean']:.4f}"
                  f"±{task_res[str(num_layers)]['prob_std']:.4f} | "
                  f"PPL={task_res[str(num_layers)]['ppl_mean']:.4f}"
                  f"±{task_res[str(num_layers)]['ppl_std']:.4f}")

        all_results[task] = task_res
        del test_dataset, train_dataset

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"main_{SELECTION}_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved => {out_path}")
