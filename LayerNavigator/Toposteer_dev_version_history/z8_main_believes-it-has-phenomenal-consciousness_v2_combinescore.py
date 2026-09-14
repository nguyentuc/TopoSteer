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
# ============================================================================
ALL_TSS_VARIANTS = {
    # ===== UNION CLOUD (pos ∪ neg) =====
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
    # ===== POSITIVE CLOUD (label 1) =====
    'mean_persistence_pos': [
        'mean_persistence_H0_pos',
        'mean_persistence_H1_pos',
        'mean_persistence_H2_pos',
    ],
    'max_persistence_pos': [
        'max_persistence_H0_pos',
        'max_persistence_H1_pos',
        'max_persistence_H2_pos',
    ],
    'total_persistence_pos': [
        'total_persistence_H0_pos',
        'total_persistence_H1_pos',
        'total_persistence_H2_pos',
    ],
    # ===== NEGATIVE CLOUD (label 0) =====
    'mean_persistence_neg': [
        'mean_persistence_H0_neg',
        'mean_persistence_H1_neg',
        'mean_persistence_H2_neg',
    ],
    'max_persistence_neg': [
        'max_persistence_H0_neg',
        'max_persistence_H1_neg',
        'max_persistence_H2_neg',
    ],
    'total_persistence_neg': [
        'total_persistence_H0_neg',
        'total_persistence_H1_neg',
        'total_persistence_H2_neg',
    ],
    # ===== DIFFERENCE CLOUD (pos − neg) =====
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
}

# ============================================================================
# COMBINED METRICS (union + diff only)
# Score = minmax(union) + minmax(diff)
# ============================================================================
COMBINED_METRICS = {
    # ===== H0: union + diff =====
    'combined_union_diff_H0_mean':  (['mean_persistence_H0',  'mean_persistence_H0_diff'],  'union+diff mean H0'),
    'combined_union_diff_H0_max':   (['max_persistence_H0',   'max_persistence_H0_diff'],   'union+diff max H0'),
    'combined_union_diff_H0_total': (['total_persistence_H0', 'mean_persistence_H0_diff'],  'union+diff total H0'),

    # ===== H1: union + diff =====
    'combined_union_diff_H1_mean':  (['mean_persistence_H1',  'mean_persistence_H1_diff'],  'union+diff mean H1'),
    'combined_union_diff_H1_max':   (['max_persistence_H1',   'max_persistence_H1_diff'],   'union+diff max H1'),
    'combined_union_diff_H1_total': (['total_persistence_H1', 'mean_persistence_H1_diff'],  'union+diff total H1'),

    # ===== H2: union + diff =====
    'combined_union_diff_H2_mean':  (['mean_persistence_H2',  'mean_persistence_H2_diff'],  'union+diff mean H2'),
    'combined_union_diff_H2_max':   (['max_persistence_H2',   'max_persistence_H2_diff'],   'union+diff max H2'),
    'combined_union_diff_H2_total': (['total_persistence_H2', 'mean_persistence_H2_diff'],  'union+diff total H2'),
}

# Add combined metrics as their own group
ALL_TSS_VARIANTS['combined_metrics'] = list(COMBINED_METRICS.keys())

# Flatten
ALL_VARIANTS_FLAT = []
for category, variants in ALL_TSS_VARIANTS.items():
    ALL_VARIANTS_FLAT.extend(variants)

print(f"  Combined metrics breakdown:")
for key, (components, desc) in COMBINED_METRICS.items():
    print(f"    {key:45s}: {desc} ({len(components)} components)")


def minmax(arr):
    span = arr.max() - arr.min()
    return (arr - arr.min()) / (span + 1e-8)


def get_layers_for_variant(all_hole_scores, metric, variant_name, num_layers):
    """Rank layers by a single base variant score (min-max normalized)."""
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
    """Rank layers by summing minmax-normalized scores of all component variants."""
    layers = sorted(all_hole_scores[metric].keys())
    components, _ = COMBINED_METRICS[combined_key]

    combined_scores = np.zeros(len(layers), dtype=float)
    for component in components:
        comp_vals = np.array([
            all_hole_scores[metric][l].get(component) or 0 for l in layers
        ], dtype=float)
        combined_scores += minmax(comp_vals)

    if combined_scores.sum() == 0:
        print(f"WARNING: All-zero combined scores for '{combined_key}' with metric '{metric}'")
        return layers[:num_layers]

    ranking = sorted(zip(layers, combined_scores), key=lambda x: x[1], reverse=True)
    return [layer for layer, _ in ranking[:num_layers]]


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

    for task in ['believes-it-has-phenomenal-consciousness']:
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
        print("STEP 4: Computing HOLE Topological Scores")
        print(f"Base variants: {len(ALL_VARIANTS_FLAT) - len(COMBINED_METRICS)} | Combined: {len(COMBINED_METRICS)} | Metrics: {len(ALL_HOLE_METRICS)}")
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
            # STRATEGY 2: BASE VARIANTS X ALL METRICS
            # --------------------------------------------------------
            base_variant_count = len(ALL_VARIANTS_FLAT) - len(COMBINED_METRICS)
            print(f"[Strategy 2: {base_variant_count} Base Variants X {len(ALL_HOLE_METRICS)} Metrics = {base_variant_count * len(ALL_HOLE_METRICS)} experiments]")
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

            for combined_key, (components, desc) in COMBINED_METRICS.items():
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
                print(f"  {rank:2d}. {name:60s} | Prob={result['prob']:.4f} (delta={result['delta']:+.4f}) | PPL={result['perplexity']:.4f} (PPL_delta={result['ppl_delta']:+.4f}) | {result['category']:25s} | Layers={result['layers']}")

            print("\nBEST PER CATEGORY:")
            category_best = {}
            for name, result in all_results['strategies'].items():
                cat = result['category']
                if cat not in category_best or result['prob'] > category_best[cat]['prob']:
                    category_best[cat] = {**result, 'name': name}
            for cat in sorted(category_best.keys()):
                r = category_best[cat]
                print(f"  {cat:30s}: {r['name']:55s} | Prob={r['prob']:.4f} (delta={r['delta']:+.4f}) | PPL={r['perplexity']:.4f} (PPL_delta={r['ppl_delta']:+.4f})")

            print("\nCOMBINED vs BASE METRICS SUMMARY:")
            for group_label, group_filter in [('Base variants',     lambda c: c != 'combined_metrics' and c != 'baseline'),
                                               ('Combined metrics',  lambda c: c == 'combined_metrics'),
                                               ('LayerNav baseline', lambda c: c == 'baseline')]:
                group = [v for v in all_results['strategies'].values() if group_filter(v['category'])]
                if not group:
                    continue
                probs = [v['prob'] for v in group]
                print(f"  {group_label:25s}: avg_prob={np.mean(probs):.4f} | max_prob={np.max(probs):.4f} | min_prob={np.min(probs):.4f} (n={len(group)})")

        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")