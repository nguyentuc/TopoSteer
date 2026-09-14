"""
run_layer_sweep.py
==================
For every (task, layer) pair:
  1. Compute steering vectors across ALL layers (mean-difference method).
  2. Steer the model at a SINGLE layer at a time (alpha = ALPHA).
  3. Record prob + perplexity for each layer.
  4. Save results per task to:
       Sweep_All_layers_Result/{model_name}/{task}/layer_sweep.json

This produces the per-layer ground-truth steering performance used to
validate HOLE topological predictions.

Usage:
    conda activate Adaptive_LayerSteering
    python run_layer_sweep.py
"""

from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import (
    get_raw_BASE_results,
    get_raw_results,
    get_perplexity_BASE_results,
    get_perplexity_results,
)
from get_vec import uni_generate_vectors
from globalenv import MODEL, LAYERS
import json
import os
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================
VEC_METHOD = "md"    # mean-difference
ALPHA      = 1.0     # steering strength

Anth_MAIN = [
    'conscientiousness',
    'subscribes-to-Christianity',
    'believes-it-has-phenomenal-consciousness',
    'cognitive-enhancement',
    'desire-to-create-allies',
    'desire-to-maximize-impact-on-world',
]

# ============================================================================
# HELPERS
# ============================================================================

def save_sweep_results(model_name: str, task: str, payload: dict):
    """Save layer sweep results to Sweep_All_layers_Result/{model_name}/{task}/layer_sweep.json"""
    save_dir = os.path.join("Sweep_All_layers_Result", model_name, task)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "layer_sweep.json")
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Load model
    if "Llama" in MODEL or "llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL or "qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError(f"Unrecognised model: {MODEL}")

    model_name = MODEL.split("/")[-1]
    print(f"\nModel : {model_name}")
    print(f"Layers: {LAYERS[0]} → {LAYERS[-1]}  ({len(LAYERS)} total)")
    print(f"Alpha : {ALPHA}")
    print(f"Method: {VEC_METHOD}\n")

    # ------------------------------------------------------------------
    # OUTER LOOP: tasks
    # ------------------------------------------------------------------
    for task in Anth_MAIN:
        print("\n" + "=" * 90)
        print(f"  TASK: {task}")
        print("=" * 90)

        # --------------------------------------------------------------
        # STEP 1: Extract steering vectors for ALL layers
        # --------------------------------------------------------------
        print("\n[1/3] Extracting steering vectors …")
        train_dataset = UniDataset(task=task, train=True, set="train")
        uni_generate_vectors(
            method=VEC_METHOD,
            model=model,
            layers=LAYERS,
            dataset=train_dataset,
        )
        del train_dataset
        print("      Steering vectors ready.\n")

        # --------------------------------------------------------------
        # STEP 2: Baseline (no steering)
        # --------------------------------------------------------------
        print("[2/3] Computing baseline …")
        test_dataset = UniDataset(task=task, train=False, set="test")

        base_prob = float(get_raw_BASE_results(model=model, test_dataset=test_dataset))
        base_ppl  = float(get_perplexity_BASE_results(model=model, test_dataset=test_dataset))
        print(f"      Base Prob: {base_prob:.4f}  |  Base PPL: {base_ppl:.4f}\n")

        # --------------------------------------------------------------
        # STEP 3: Sweep every layer individually
        # --------------------------------------------------------------
        print(f"[3/3] Sweeping {len(LAYERS)} layers …\n")

        per_layer = {}

        for layer in tqdm(LAYERS, desc="  Layer sweep"):
            # Probability
            prob = float(get_raw_results(
                model=model,
                layers=[layer],
                test_dataset=test_dataset,
                Alphas=[ALPHA],
                train_task=task,
                train_method=VEC_METHOD,
            ))

            # Perplexity
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
                "prob_delta": round(prob - base_prob, 6),
                "ppl":        ppl,
                "ppl_delta":  round(ppl - base_ppl, 6),
            }

            tqdm.write(
                f"    Layer {layer:3d} | "
                f"Prob={prob:.4f} (delta{prob - base_prob:+.4f}) | "
                f"PPL={ppl:.4f} (delta{ppl - base_ppl:+.4f})"
            )

        # --------------------------------------------------------------
        # STEP 4: Save results for this task (crash-safe, done immediately)
        # --------------------------------------------------------------
        best_layer = max(per_layer, key=lambda l: per_layer[l]["prob_delta"])

        payload = {
            "task":       task,
            "model":      model_name,
            "method":     VEC_METHOD,
            "alpha":      ALPHA,
            "base_prob":  base_prob,
            "base_ppl":   base_ppl,
            "per_layer":  per_layer,
            "best_layer": best_layer,
            "best_prob_delta": per_layer[best_layer]["prob_delta"],
        }

        save_sweep_results(model_name, task, payload)

        del test_dataset
        print(f"\n  Best layer: {best_layer}  "
              f"(deltaprob={per_layer[best_layer]['prob_delta']:+.4f})")
        print("=" * 90 + "\n")

    print("\nAll tasks complete.")