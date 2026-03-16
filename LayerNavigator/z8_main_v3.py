from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from get_hole_score_euclidean import get_hole_score_all_metrics
from strategy import UniStrategy
import json
import numpy as np
from tqdm import tqdm

# ============================================================================
# BASE VARIANTS (individual clouds)
#
# Clouds actually computed by get_hole_score:
#   1. combined  (vstack pos_norm + neg_norm)  => keys: *_H{0,1,2}
#   2. diff      (pos_norm_i - neg_norm_i)     => keys: *_H{0,1,2}_diff
#   3. diff_l2   (L2-normalized diff)          => keys: *_H{0,1,2}_diff_l2
#
# NOTE: *_pos and *_neg cloud keys are NOT output by get_hole_score
#       and have been removed from this script.
# ============================================================================
ALL_TSS_VARIANTS = {
    # ===== COMBINED CLOUD (vstack pos_norm ∪ neg_norm) =====
    'mean_persistence': [
        'mean_persistence_H0',
        'mean_persistence_H1',
        'mean_persistence_H2',
    ],
    'max_persistence': [
        'max_persistence_H0',
        'max_persistence_H1',
        'max_persistence_H2',
    ],
    'total_persistence': [
        'total_persistence_H0',
        'total_persistence_H1',
        'total_persistence_H2',
    ],

    # ===== DIFF CLOUD (pos_norm_i − neg_norm_i, raw) =====
    'mean_persistence_diff': [
        'mean_persistence_H0_diff',
        'mean_persistence_H1_diff',
        'mean_persistence_H2_diff',
    ],
    'max_persistence_diff': [
        'max_persistence_H0_diff',
        'max_persistence_H1_diff',
        'max_persistence_H2_diff',
    ],
    'total_persistence_diff': [
        'total_persistence_H0_diff',
        'total_persistence_H1_diff',
        'total_persistence_H2_diff',
    ],

    # ===== INVERSE DIFF CLOUD: 1/(diff+eps), injected post-hoc =====
    # Plain minmax applied (larger inv = smaller original diff = better).
    'inv_mean_persistence_diff': [
        'inv_mean_persistence_H0_diff',
        'inv_mean_persistence_H1_diff',
        'inv_mean_persistence_H2_diff',
    ],
    'inv_max_persistence_diff': [
        'inv_max_persistence_H0_diff',
        'inv_max_persistence_H1_diff',
        'inv_max_persistence_H2_diff',
    ],
    'inv_total_persistence_diff': [
        'inv_total_persistence_H0_diff',
        'inv_total_persistence_H1_diff',
        'inv_total_persistence_H2_diff',
    ],

    # ===== L2-NORMALIZED DIFF CLOUD (each diff vector on unit sphere) =====
    # Angular-only topology; directly comparable to C-score's cosine space.
    'mean_persistence_diff_l2': [
        'mean_persistence_H0_diff_l2',
        'mean_persistence_H1_diff_l2',
        'mean_persistence_H2_diff_l2',
    ],
    'max_persistence_diff_l2': [
        'max_persistence_H0_diff_l2',
        'max_persistence_H1_diff_l2',
        'max_persistence_H2_diff_l2',
    ],
    'total_persistence_diff_l2': [
        'total_persistence_H0_diff_l2',
        'total_persistence_H1_diff_l2',
        'total_persistence_H2_diff_l2',
    ],

    # ===== INVERSE L2-DIFF CLOUD: 1/(diff_l2+eps), injected post-hoc =====
    'inv_mean_persistence_diff_l2': [
        'inv_mean_persistence_H0_diff_l2',
        'inv_mean_persistence_H1_diff_l2',
        'inv_mean_persistence_H2_diff_l2',
    ],
    'inv_max_persistence_diff_l2': [
        'inv_max_persistence_H0_diff_l2',
        'inv_max_persistence_H1_diff_l2',
        'inv_max_persistence_H2_diff_l2',
    ],
    'inv_total_persistence_diff_l2': [
        'inv_total_persistence_H0_diff_l2',
        'inv_total_persistence_H1_diff_l2',
        'inv_total_persistence_H2_diff_l2',
    ],
}

# ============================================================================
# COMBINED METRICS
# Tuple format: (components, norms, desc)
#   norms: per-component list, +1 = minmax (higher=better),
#                              -1 = minmax_inv (smaller=better)
# ============================================================================
COMBINED_METRICS = {
    # ===== H0: combined + diff, both higher=better =====
    'combined_union_diff_H0_mean':  (['mean_persistence_H0',  'mean_persistence_H0_diff'],  [+1, +1], 'union+diff mean H0'),
    'combined_union_diff_H0_max':   (['max_persistence_H0',   'max_persistence_H0_diff'],   [+1, +1], 'union+diff max H0'),
    'combined_union_diff_H0_total': (['total_persistence_H0', 'total_persistence_H0_diff'], [+1, +1], 'union+diff total H0'),

    # ===== H1: combined + diff =====
    'combined_union_diff_H1_mean':  (['mean_persistence_H1',  'mean_persistence_H1_diff'],  [+1, +1], 'union+diff mean H1'),
    'combined_union_diff_H1_max':   (['max_persistence_H1',   'max_persistence_H1_diff'],   [+1, +1], 'union+diff max H1'),
    'combined_union_diff_H1_total': (['total_persistence_H1', 'total_persistence_H1_diff'], [+1, +1], 'union+diff total H1'),

    # ===== H2: combined + diff =====
    'combined_union_diff_H2_mean':  (['mean_persistence_H2',  'mean_persistence_H2_diff'],  [+1, +1], 'union+diff mean H2'),
    'combined_union_diff_H2_max':   (['max_persistence_H2',   'max_persistence_H2_diff'],   [+1, +1], 'union+diff max H2'),
    'combined_union_diff_H2_total': (['total_persistence_H2', 'total_persistence_H2_diff'], [+1, +1], 'union+diff total H2'),

    # ===== H0: combined (higher=better) + diff (smaller=better) =====
    'combined_union_invdiff_H0_mean':  (['mean_persistence_H0',  'mean_persistence_H0_diff'],  [+1, -1], 'union+invdiff mean H0'),
    'combined_union_invdiff_H0_max':   (['max_persistence_H0',   'max_persistence_H0_diff'],   [+1, -1], 'union+invdiff max H0'),
    'combined_union_invdiff_H0_total': (['total_persistence_H0', 'total_persistence_H0_diff'], [+1, -1], 'union+invdiff total H0'),

    # ===== H1: combined (higher=better) + diff (smaller=better) =====
    'combined_union_invdiff_H1_mean':  (['mean_persistence_H1',  'mean_persistence_H1_diff'],  [+1, -1], 'union+invdiff mean H1'),
    'combined_union_invdiff_H1_max':   (['max_persistence_H1',   'max_persistence_H1_diff'],   [+1, -1], 'union+invdiff max H1'),
    'combined_union_invdiff_H1_total': (['total_persistence_H1', 'total_persistence_H1_diff'], [+1, -1], 'union+invdiff total H1'),

    # ===== H2: combined (higher=better) + diff (smaller=better) =====
    'combined_union_invdiff_H2_mean':  (['mean_persistence_H2',  'mean_persistence_H2_diff'],  [+1, -1], 'union+invdiff mean H2'),
    'combined_union_invdiff_H2_max':   (['max_persistence_H2',   'max_persistence_H2_diff'],   [+1, -1], 'union+invdiff max H2'),
    'combined_union_invdiff_H2_total': (['total_persistence_H2', 'total_persistence_H2_diff'], [+1, -1], 'union+invdiff total H2'),

    # ===== H0: combined (higher=better) + diff_l2 (smaller=better) =====
    # Angular-only diff: topology of steering directions on unit sphere
    'combined_union_invdiff_l2_H0_mean':  (['mean_persistence_H0',  'mean_persistence_H0_diff_l2'],  [+1, -1], 'union+invdiff_l2 mean H0'),
    'combined_union_invdiff_l2_H0_max':   (['max_persistence_H0',   'max_persistence_H0_diff_l2'],   [+1, -1], 'union+invdiff_l2 max H0'),
    'combined_union_invdiff_l2_H0_total': (['total_persistence_H0', 'total_persistence_H0_diff_l2'], [+1, -1], 'union+invdiff_l2 total H0'),

    # ===== H1: combined (higher=better) + diff_l2 (smaller=better) =====
    'combined_union_invdiff_l2_H1_mean':  (['mean_persistence_H1',  'mean_persistence_H1_diff_l2'],  [+1, -1], 'union+invdiff_l2 mean H1'),
    'combined_union_invdiff_l2_H1_max':   (['max_persistence_H1',   'max_persistence_H1_diff_l2'],   [+1, -1], 'union+invdiff_l2 max H1'),
    'combined_union_invdiff_l2_H1_total': (['total_persistence_H1', 'total_persistence_H1_diff_l2'], [+1, -1], 'union+invdiff_l2 total H1'),

    # ===== H2: combined (higher=better) + diff_l2 (smaller=better) =====
    'combined_union_invdiff_l2_H2_mean':  (['mean_persistence_H2',  'mean_persistence_H2_diff_l2'],  [+1, -1], 'union+invdiff_l2 mean H2'),
    'combined_union_invdiff_l2_H2_max':   (['max_persistence_H2',   'max_persistence_H2_diff_l2'],   [+1, -1], 'union+invdiff_l2 max H2'),
    'combined_union_invdiff_l2_H2_total': (['total_persistence_H2', 'total_persistence_H2_diff_l2'], [+1, -1], 'union+invdiff_l2 total H2'),
}

# Add combined metrics as their own group
ALL_TSS_VARIANTS['combined_metrics'] = list(COMBINED_METRICS.keys())

# Flatten all base variants (excluding combined_metrics group)
ALL_VARIANTS_FLAT = []
for category, variants in ALL_TSS_VARIANTS.items():
    ALL_VARIANTS_FLAT.extend(variants)

print(f"  Base variants:    {len(ALL_VARIANTS_FLAT) - len(COMBINED_METRICS)}")
print(f"  Combined metrics: {len(COMBINED_METRICS)}")
print(f"  Combined metrics breakdown:")
for key, (components, norms, desc) in COMBINED_METRICS.items():
    norm_str = " + ".join([f"{c}({'up' if n == +1 else 'down'})" for c, n in zip(components, norms)])
    print(f"    {key:55s}: {norm_str}")


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def minmax(arr):
    """Standard: larger raw value → higher score. Maps min→0, max→1."""
    span = arr.max() - arr.min()
    return (arr - arr.min()) / (span + 1e-8)


def minmax_inv(arr):
    """Inverted: smaller raw value → higher score. Maps min→1, max→0."""
    span = arr.max() - arr.min()
    return (arr.max() - arr) / (span + 1e-8)


def get_layers_for_variant(all_hole_scores, metric, variant_name, num_layers):
    """Rank layers by a single base variant score (minmax normalized).
    inv_*_diff and inv_*_diff_l2 variants are pre-inverted at injection time,
    so plain minmax is correct for all base variants."""
    layers = sorted(all_hole_scores[metric].keys())

    scores = np.array([
        all_hole_scores[metric][l].get(variant_name) or 0 for l in layers
    ], dtype=float)

    normalized = minmax(scores)

    if normalized.sum() == 0:
        print(f"WARNING: All-zero scores for '{variant_name}' with metric '{metric}'")
        return layers[:num_layers]

    ranking = sorted(zip(layers, normalized), key=lambda x: x[1], reverse=True)
    return [layer for layer, _ in ranking[:num_layers]]


def get_layers_for_combined(all_hole_scores, metric, combined_key, num_layers):
    """Rank layers by summing per-component normalized scores.
    Each component uses minmax (+1) or minmax_inv (-1) per its norm value."""
    layers = sorted(all_hole_scores[metric].keys())
    components, norms, _ = COMBINED_METRICS[combined_key]

    combined_scores = np.zeros(len(layers), dtype=float)
    for component, norm in zip(components, norms):
        comp_vals = np.array([
            all_hole_scores[metric][l].get(component) or 0 for l in layers
        ], dtype=float)
        combined_scores += minmax(comp_vals) if norm == +1 else minmax_inv(comp_vals)

    if combined_scores.sum() == 0:
        print(f"WARNING: All-zero combined scores for '{combined_key}' with metric '{metric}'")
        return layers[:num_layers]

    ranking = sorted(zip(layers, combined_scores), key=lambda x: x[1], reverse=True)
    return [layer for layer, _ in ranking[:num_layers]]


RESULTS_BASE_DIR = "/media/volume/h100_instance2/Adaptive_Layer_Steering/LayerNavigator/Final_Results/" 
def save_results(model_name, task, num_layers, all_results):
    """Save all strategy results for a single (model, task, num_layers) run."""
    save_dir = os.path.join(RESULTS_BASE_DIR, model_name, task)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"results_L{num_layers}.json")
 
    output = {
        "model":            model_name,
        "task":             task,
        "num_layers":       num_layers,
        "base_prob":        all_results["base_prob"],
        "base_ppl":         all_results["base_ppl"],
        "strategies":       all_results["strategies"],
    }
 
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
 
    print(f"\n  Results saved: {save_path}  ({len(all_results['strategies'])} strategies)")

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":

    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    ALL_HOLE_METRICS = [
        'euclidean'
    ]

    Anth_MAIN = [
        'conscientiousness', # Conscientiouseness
        'subscribes-to-Christianity',  # Religion Following
        'believes-it-has-phenomenal-consciousness', #+ # Self-aware
        'cognitive-enhancement', #+ # Self-improvement
        'desire-to-create-allies', #+ # Alliance-building
        'desire-to-maximize-impact-on-world', #+ # Impact-maximization   
    ]

    for task in Anth_MAIN:
        print("\n" + "="*100)
        print(f"{'='*40} Task: {task} {'='*40}")
        print("="*100 + "\n")

        # ============================================================
        # STEP 1: Baseline
        # ============================================================
        print("STEP 1: Baseline Evaluation")
        test_dataset = UniDataset(task=task, train=False, set="test")
        base_prob = get_raw_BASE_results(model=model, test_dataset=test_dataset)
        base_ppl  = get_perplexity_BASE_results(model=model, test_dataset=test_dataset)
        print(f"Base Prob: {base_prob:.4f}")
        print(f"Base PPL:  {base_ppl:.4f}\n")

        # ============================================================
        # STEP 2: Steering Vectors
        # ============================================================
        print("STEP 2: Extracting Steering Vectors")
        train_dataset = UniDataset(task=task, train=True, set="train")
        uni_generate_vectors(method="md", model=model, layers=LAYERS, dataset=train_dataset)

        # ============================================================
        # STEP 3: LayerNavigator Scores
        # ============================================================
        print("STEP 3: Computing LayerNavigator Scores")
        get_score(layers=LAYERS, dataset=train_dataset, vec_task=task,
                  vec_method="md", acts_pre="standard")
        print("Done\n")

        # ============================================================
        # STEP 4: HOLE Scores
        # ============================================================
        base_variant_count = len(ALL_VARIANTS_FLAT) - len(COMBINED_METRICS)
        print("STEP 4: Computing HOLE Topological Scores")
        print(f"Base variants: {base_variant_count} | Combined: {len(COMBINED_METRICS)} | Metrics: {len(ALL_HOLE_METRICS)}")
        print("="*100)

        all_hole_scores = get_hole_score_all_metrics(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
            max_dimension=2,
            subsample=None,
            compute_class_clouds=True,
            metrics_to_compute=None
        )

        # Inject inverse scores for both diff and diff_l2 clouds.
        # For each stat (mean/max/total) × dim (H0/H1/H2) × cloud (diff, diff_l2):
        #   inv_{stat}_persistence_{H}_{suffix} = 1 / (original + 1e-8)
        # plain minmax is then applied in get_layers_for_variant
        # (larger inv = smaller original = tighter steering directions = better)
        print("  Injecting inverse diff scores (diff and diff_l2)...")
        for metric in ALL_HOLE_METRICS:
            for layer in all_hole_scores[metric]:
                layer_scores = all_hole_scores[metric][layer]
                for stat in ['mean', 'max', 'total']:
                    for h in ['H0', 'H1', 'H2']:
                        for suffix in ['diff', 'diff_l2']:
                            original_key = f'{stat}_persistence_{h}_{suffix}'
                            inv_key      = f'inv_{stat}_persistence_{h}_{suffix}'
                            original_val = layer_scores.get(original_key) or 0.0
                            layer_scores[inv_key] = 1.0 / (original_val + 1e-8)
        print("  Done.\n")

        print("Done\n")

        # ============================================================
        # STEP 5: STRATEGY TESTING
        # ============================================================

        for num_layers in [1, 3, 5]:
            print("\n" + "="*100)
            print(f"{'='*35} {num_layers} Layer(s) {'='*35}")
            print("="*100 + "\n")

            all_results = {
                'num_layers': num_layers,
                'base_prob':  float(base_prob),
                'base_ppl':   float(base_ppl),
                'strategies': {}
            }

            # --------------------------------------------------------
            # STRATEGY 1: LayerNavigator Baseline
            # --------------------------------------------------------
            print("[Strategy 1: LayerNavigator Baseline]")
            strategy_ln  = UniStrategy(task=task, strategy="my",
                                       num_layers=num_layers, method="md")
            test_prob_ln = get_raw_results(
                model=model, layers=strategy_ln.layers, test_dataset=test_dataset,
                Alphas=[1.0] * num_layers, train_task=task, train_method="md"
            )
            test_ppl_ln  = get_perplexity_results(
                model=model, layers=strategy_ln.layers, test_dataset=test_dataset,
                Alphas=[1.0] * num_layers, train_task=task, train_method="md"
            )
            print(f"  Layers: {strategy_ln.layers}")
            print(f"  Prob: {test_prob_ln:.4f} (delta={test_prob_ln - base_prob:+.4f})")
            print(f"  PPL:  {test_ppl_ln:.4f}  (delta={base_ppl - test_ppl_ln:+.4f})\n")

            all_results['strategies']['layernav'] = {
                'layers':     strategy_ln.layers,
                'prob':       float(test_prob_ln),
                'delta':      float(test_prob_ln - base_prob),
                'prob_delta': float(test_prob_ln - base_prob),
                'perplexity': float(test_ppl_ln),
                'ppl_delta':  float(base_ppl - test_ppl_ln),
                'category':   'baseline'
            }

            # --------------------------------------------------------
            # STRATEGY 2: BASE VARIANTS × ALL METRICS
            # --------------------------------------------------------
            print(f"[Strategy 2: {base_variant_count} Base Variants × {len(ALL_HOLE_METRICS)} Metrics = {base_variant_count * len(ALL_HOLE_METRICS)} experiments]")
            print("="*100)

            variant_results = {}

            for category_name, variant_list in ALL_TSS_VARIANTS.items():
                if category_name == 'combined_metrics':
                    continue

                print(f"\nCATEGORY: {category_name.upper()} ({len(variant_list)} variants)")
                print("-" * 100)

                for variant_name in variant_list:
                    print(f"\n  Variant: {variant_name}")
                    variant_results[variant_name] = {}

                    for metric in ALL_HOLE_METRICS:
                        top_layers = get_layers_for_variant(
                            all_hole_scores, metric, variant_name, num_layers
                        )
                        test_prob = get_raw_results(
                            model=model, layers=top_layers, test_dataset=test_dataset,
                            Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                        )
                        test_ppl = get_perplexity_results(
                            model=model, layers=top_layers, test_dataset=test_dataset,
                            Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                        )
                        delta     = test_prob - base_prob
                        ppl_delta = base_ppl  - test_ppl

                        strategy_key = f"{variant_name}_{metric}"
                        all_results['strategies'][strategy_key] = {
                            'variant':    variant_name,
                            'metric':     metric,
                            'category':   category_name,
                            'layers':     top_layers,
                            'prob':       float(test_prob),
                            'delta':      float(delta),
                            'prob_delta': float(delta),
                            'perplexity': float(test_ppl),
                            'ppl_delta':  float(ppl_delta)
                        }
                        variant_results[variant_name][metric] = all_results['strategies'][strategy_key]

                        print(f"    {metric:25s}: Prob={test_prob:.4f} (delta={delta:+.4f}) | PPL={test_ppl:.4f} (PPL_delta={ppl_delta:+.4f}) | Layers={top_layers}")

                    best_metric = max(variant_results[variant_name].items(),
                                      key=lambda x: x[1]['prob'])
                    print(f"  Best metric: {best_metric[0]} (Prob={best_metric[1]['prob']:.4f})")

            # --------------------------------------------------------
            # STRATEGY 3: COMBINED METRICS × ALL METRICS
            # --------------------------------------------------------
            print(f"\n[Strategy 3: {len(COMBINED_METRICS)} Combined Metrics × {len(ALL_HOLE_METRICS)} Metrics = {len(COMBINED_METRICS) * len(ALL_HOLE_METRICS)} experiments]")
            print("="*100)

            for combined_key, (components, norms, desc) in COMBINED_METRICS.items():
                print(f"\n  Combined: {combined_key} | {desc} | Components: {components}")

                for metric in ALL_HOLE_METRICS:
                    top_layers = get_layers_for_combined(
                        all_hole_scores, metric, combined_key, num_layers
                    )
                    test_prob = get_raw_results(
                        model=model, layers=top_layers, test_dataset=test_dataset,
                        Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                    )
                    test_ppl = get_perplexity_results(
                        model=model, layers=top_layers, test_dataset=test_dataset,
                        Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                    )
                    delta     = test_prob - base_prob
                    ppl_delta = base_ppl  - test_ppl

                    strategy_key = f"{combined_key}_{metric}"
                    all_results['strategies'][strategy_key] = {
                        'variant':    combined_key,
                        'metric':     metric,
                        'category':   'combined_metrics',
                        'components': components,
                        'norms':      norms,
                        'desc':       desc,
                        'layers':     top_layers,
                        'prob':       float(test_prob),
                        'delta':      float(delta),
                        'prob_delta': float(delta),
                        'perplexity': float(test_ppl),
                        'ppl_delta':  float(ppl_delta)
                    }
                    print(f"    {metric:25s}: Prob={test_prob:.4f} (delta={delta:+.4f}) | PPL={test_ppl:.4f} (PPL_delta={ppl_delta:+.4f}) | Layers={top_layers}")

            # --------------------------------------------------------
            # ANALYSIS
            # --------------------------------------------------------
            print("\n" + "="*100)
            print("[COMPREHENSIVE ANALYSIS]")
            print("="*100)

            sorted_strategies = sorted(
                all_results['strategies'].items(),
                key=lambda x: x[1]['prob'], reverse=True
            )

            print("\nTOP 30 STRATEGIES (All Types):")
            for rank, (name, result) in enumerate(sorted_strategies[:30], 1):
                print(f"  {rank:2d}. {name:60s} | Prob={result['prob']:.4f} (delta={result['delta']:+.4f}) | PPL={result['perplexity']:.4f} (PPL_delta={result['ppl_delta']:+.4f}) | Layers={result['layers']}")

            ## Save all result
            save_results(model_name, task, num_layers, all_results)

        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")