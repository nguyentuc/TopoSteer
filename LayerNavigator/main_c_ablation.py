"""
Ablation: steering-layer selection by C-score ONLY (instead of D + C).

Identical to main.py (Llama-3-8B, MD extraction, D+C scoring, top-{1,3,5} steering at
alpha=1.0 over all 6 Anth_MAIN tasks) EXCEPT the layers to steer are chosen by the
C-score alone — UniStrategy(strategy="c") — rather than the combined s_score = D + C
(strategy="my" in main.py).

Scoring is unchanged: get_score still computes and saves d_score, c_score, s_score per
layer to Score-standard/{task}/{task}+md/L{l}.json; this script just RANKS by c_score.

Efficiency: MD extraction and D/C scoring are skipped if their artifacts already exist
(e.g. from a prior main.py run), so this only reruns the steering evaluation.

Outputs (run from the LayerNavigator/ directory — relative paths):
  - steered generations -> ./Results/{task}-test/{task}+md/Res_*.json  (via get_raw_results)
  - ablation summary    -> ./main_c_ablation_results.json
        {task: {"base_prob": p,
                "1": {"layers":[...], "scores":[...], "prob": p, "delta": d},
                "3": {...}, "5": {...}}}
"""

from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from strategy import UniStrategy

import os
import json

SELECTION       = "c"          # "c" = rank layers by C-score only
NUM_LAYERS_LIST = [2]


if __name__ == "__main__":
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    all_results = {}

    for task in Anth_MAIN:
        print(f"\n############## Task: {task}  (selection = {SELECTION}-score only) ##############")

        # --- Base (no steering) ---
        test_dataset = UniDataset(task=task, train=False, set="test")
        base_prob = get_raw_BASE_results(model=model, test_dataset=test_dataset)
        base_ppl  = get_perplexity_BASE_results(model=model, test_dataset=test_dataset)
        print(f"Base Prob: {base_prob:.4f} | Base PPL: {base_ppl:.4f}")

        # --- MD vectors + D/C/s scores (skip if already computed) ---
        train_dataset = UniDataset(task=task, train=True, set="train")

        uni_generate_vectors(method="md", model=model, layers=LAYERS, dataset=train_dataset)

        get_score(layers=LAYERS, dataset=train_dataset, vec_task=task,
                      vec_method="md", acts_pre="standard")

        # --- Steering evaluation: layers chosen by C-score only ---
        task_res = {"base_prob": float(base_prob), "base_ppl": float(base_ppl)}
        for num_layers in NUM_LAYERS_LIST:
            strategy = UniStrategy(task=task, strategy=SELECTION,
                                   num_layers=num_layers, method="md")
            test_prob = get_raw_results(
                model=model,
                layers=strategy.layers,
                test_dataset=test_dataset,
                Alphas=[1.0] * num_layers,
                train_task=task,
                train_method="md",
            )
            test_ppl = get_perplexity_results(
                model=model,
                layers=strategy.layers,
                test_dataset=test_dataset,
                Alphas=[1.0] * num_layers,
                train_task=task,
                train_method="md",
            )
            delta     = test_prob - base_prob
            ppl_delta = base_ppl - test_ppl          # >0 = steering lowered (improved) perplexity
            print(f"  [{SELECTION}] {num_layers} layer(s): {strategy.layers} | "
                  f"Prob={test_prob:.4f} (delta={delta:+.4f}) | "
                  f"PPL={test_ppl:.4f} (delta={ppl_delta:+.4f})")
            task_res[str(num_layers)] = {
                "layers":     strategy.layers,
                "scores":     strategy.scores,
                "prob":       float(test_prob),
                "delta":      float(delta),
                "perplexity": float(test_ppl),
                "ppl_delta":  float(ppl_delta),
            }

        all_results[task] = task_res
        del test_dataset, train_dataset

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"main_{SELECTION}_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved => {out_path}")
