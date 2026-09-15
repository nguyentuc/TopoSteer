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

# Score keys come from three point clouds built by get_hole_score:
#   combined -> *_H0           (vstack pos_norm + neg_norm)
#   diff     -> *_H0_diff      (pos_norm_i - neg_norm_i)
#   diff_l2  -> *_H0_diff_l2   (L2-normalized diff, unit sphere)
#
# H0 only: H1/H2 would need max_dimension=2, which costs ~13x the simplices.
# Trailing percentages are win rates vs the LayerNavigator baseline over 295
# blocks in Toposteer_dev_version_history/*.out, uncorrected for multiple
# comparisons -- a shortlist to re-test, not a result.
ALL_TSS_VARIANTS = {
    'mean_persistence': [
        'mean_persistence_H0',                
    ],

    'mean_persistence_diff': [
        'mean_persistence_H0_diff',          
    ],

    'mean_persistence_diff_l2': [
        'mean_persistence_H0_diff_l2',   
    ],

    # 'h0_persistence_only' is deliberately absent: get_hole_score_euclidean.py
    # assigns it mean_pers_h0 verbatim, so it would duplicate 'mean_persistence_H0'.
    'h0_shape': [
        # 'inverse_mean_H0',                      
        # 'portion_of_persistence_dominance_h0',  
        'entropy_H0',                           
        # 'betti_curve_auc_H0',                  
        'total_persistence_H0',               
        # 'max_persistence_H0',                   
        # 'inverse_h0_entropy',                  
        # 'count_H0',                           
    ],
}

# Tuple format: (components, norms, desc)
#   norms: per-component, +1 = minmax (higher=better), -1 = minmax_inv (smaller=better)
COMBINED_METRICS = {
    'combined_union_diff_H0_mean': (['mean_persistence_H0', 'mean_persistence_H0_diff'], [+1, +1], 'union+diff mean H0'),
    'combined_union_diff_H0_total': (['total_persistence_H0', 'total_persistence_H0_diff'], [+1, +1], 'union+diff total H0'),        # 73%
    # 'combined_union_invdiff_H0_max': (['max_persistence_H0', 'max_persistence_H0_diff'], [+1, -1], 'union max + inverted diff max'), # 68%
    # 94% in the logs but from only 2 files, so the prior is weak and likely
    # correlated. Kept so it gets a clean re-test.
    # 'combined_union_invdiff_l2_H0_max': (['max_persistence_H0', 'max_persistence_H0_diff_l2'], [+1, -1], 'union max + inverted l2-diff max'),
}

# All of these are already implemented in compute_persistence_diagram; they were
# simply not being requested. Trim the list to cut runtime -- cost scales
# linearly in len(ALL_HOLE_METRICS).
ALL_HOLE_METRICS = [
    'euclidean',            
#     'cosine',               
#     'geodesic',                    
]

ALL_TSS_VARIANTS['combined_metrics'] = list(COMBINED_METRICS.keys())

ALL_VARIANTS_FLAT = []
for category, variants in ALL_TSS_VARIANTS.items():
    ALL_VARIANTS_FLAT.extend(variants)

LAYER_BUDGETS = [1, 2, 3]

_base_count = len(ALL_VARIANTS_FLAT) - len(COMBINED_METRICS)
_per_task = (_base_count + len(COMBINED_METRICS)) * len(ALL_HOLE_METRICS) * len(LAYER_BUDGETS)

print(f"  Base variants:    {_base_count}")
print(f"  Combined metrics: {len(COMBINED_METRICS)}")
print(f"  Distance metrics: {len(ALL_HOLE_METRICS)}  ({', '.join(ALL_HOLE_METRICS)})")
print(f"  Layer budgets:    {LAYER_BUDGETS}")
print(f"  Combined metrics breakdown:")
for key, (components, norms, desc) in COMBINED_METRICS.items():
    norm_str = " + ".join([f"{c}({'up' if n == +1 else 'down'})" for c, n in zip(components, norms)])
    print(f"    {key:55s}: {norm_str}")
print(f"\n  COST: ({_base_count} + {len(COMBINED_METRICS)}) variants x "
      f"{len(ALL_HOLE_METRICS)} distances x {len(LAYER_BUDGETS)} budgets "
      f"= {_per_task} steered evaluations per task")
print(f"        each evaluation runs both get_raw_results and get_perplexity_results.")
print(f"        Trim ALL_HOLE_METRICS or ALL_TSS_VARIANTS to reduce this.\n")


def minmax(arr):
    """Standard: larger raw value -> higher score. Maps min->0, max->1."""
    span = arr.max() - arr.min()
    return (arr - arr.min()) / (span + 1e-8)


def minmax_inv(arr):
    """Inverted: smaller raw value -> higher score. Maps min->1, max->0."""
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


def _accumulate(bucket, name, d_prob, d_ppl, tie):
    acc = bucket.setdefault(name, {'win': 0, 'tie': 0, 'loss': 0, 'dp': [], 'dppl': []})
    acc['dp'].append(d_prob)
    acc['dppl'].append(d_ppl)
    if d_prob > tie:
        acc['win'] += 1
    elif d_prob < -tie:
        acc['loss'] += 1
    else:
        acc['tie'] += 1


def _record_table(title, bucket, label_width, limit=None):
    """Print one win/loss table, best win rate first."""
    rows = []
    for name, acc in bucket.items():
        n = acc['win'] + acc['tie'] + acc['loss']
        rows.append((acc['win'] / n, float(np.mean(acc['dp'])), name, n,
                     acc['win'], acc['tie'], acc['loss'],
                     float(np.median(acc['dp'])), float(np.mean(acc['dppl']))))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    if limit is not None:
        rows = rows[:limit]

    print(f"\n{title}")
    print("-" * (label_width + 62))
    print(f"{'':<{label_width}}{'n':>5}{'win':>5}{'tie':>5}{'loss':>6}"
          f"{'win%':>7}{'mean dProb':>12}{'med dProb':>11}{'mean dPPL':>11}")
    for win_rate, mean_dp, name, n, w, t, l, med_dp, mean_dppl in rows:
        print(f"{name[:label_width - 1]:<{label_width}}{n:>5}{w:>5}{t:>5}{l:>6}"
              f"{100 * win_rate:>6.0f}%{mean_dp:>+12.4f}{med_dp:>+11.4f}{mean_dppl:>+11.4f}")


def summarize_run(run_history, model_name, tie=1e-4):
    """Cross-task summary of every strategy against the LayerNavigator baseline.

    A block is one (task, num_layers) pair. Deltas are strategy minus layernav
    inside the same block, so every comparison is like-for-like. dProb > 0 means
    the strategy steered better than LayerNavigator; dPPL > 0 means it also held
    perplexity better. Differences within +/-tie count as ties, not wins.
    """
    print("\n" + "=" * 100)
    print("FINAL SUMMARY: ALL STRATEGIES vs LAYERNAVIGATOR")
    print("=" * 100)

    usable = [b for b in run_history if 'layernav' in b['strategies']]
    if not usable:
        print("\nNo blocks with a LayerNavigator baseline were recorded -- nothing to summarize.")
        return

    per_strategy, per_metric, per_variant = {}, {}, {}
    best_rows = []

    for block in usable:
        strategies = block['strategies']
        ln = strategies['layernav']
        contenders = [(k, r) for k, r in strategies.items() if k != 'layernav']
        if not contenders:
            continue

        best_key, best_rec = max(contenders, key=lambda kv: kv[1]['prob'])
        best_rows.append((block['task'], block['num_layers'], ln['prob'],
                          best_key, best_rec['prob']))

        for key, rec in contenders:
            d_prob = rec['prob'] - ln['prob']
            d_ppl = rec['ppl_delta'] - ln['ppl_delta']
            _accumulate(per_strategy, key, d_prob, d_ppl, tie)
            _accumulate(per_metric, rec.get('metric', 'unknown'), d_prob, d_ppl, tie)
            _accumulate(per_variant, rec.get('variant', 'unknown'), d_prob, d_ppl, tie)

    tasks = sorted({b['task'] for b in usable})
    budgets = sorted({b['num_layers'] for b in usable})
    print(f"\nBlocks: {len(usable)}  ({len(tasks)} tasks x {len(budgets)} layer budgets)")
    print(f"Tasks:  {', '.join(tasks)}")
    print(f"Budgets: {budgets}   Distances: {len(per_metric)}   Variants: {len(per_variant)}")
    print(f"Tie threshold: +/-{tie}")

    _record_table("BY DISTANCE METRIC (pooled over variants)", per_metric, 34)
    _record_table("BY VARIANT (pooled over distance metrics)", per_variant, 40)
    _record_table("BY STRATEGY (variant x distance), top 40", per_strategy, 52, limit=40)

    print("\nBEST STRATEGY PER BLOCK")
    print("-" * 100)
    print(f"{'task':<44}{'L':>2}{'LN prob':>10}{'best prob':>11}{'delta':>9}  best strategy")
    ln_wins = 0
    for task, num_layers, ln_prob, best_key, best_prob in best_rows:
        delta = best_prob - ln_prob
        if delta <= tie:
            ln_wins += 1
        print(f"{task[:43]:<44}{num_layers:>2}{ln_prob:>10.4f}{best_prob:>11.4f}"
              f"{delta:>+9.4f}  {best_key}")
    print(f"\nBlocks where no strategy beat LayerNavigator: {ln_wins}/{len(best_rows)}")

    summary_path = os.path.join(RESULTS_BASE_DIR, model_name, "run_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    payload = {
        "model": model_name,
        "blocks": len(usable),
        "tasks": tasks,
        "layer_budgets": budgets,
        "tie_threshold": tie,
        "by_strategy": {}, "by_metric": {}, "by_variant": {},
        "best_per_block": [
            {"task": t, "num_layers": n, "layernav_prob": p,
             "best_strategy": k, "best_prob": bp, "delta": bp - p}
            for t, n, p, k, bp in best_rows
        ],
    }
    for field, bucket in (("by_strategy", per_strategy), ("by_metric", per_metric),
                          ("by_variant", per_variant)):
        for name, acc in bucket.items():
            n = acc['win'] + acc['tie'] + acc['loss']
            payload[field][name] = {
                "n": n, "win": acc['win'], "tie": acc['tie'], "loss": acc['loss'],
                "win_rate": acc['win'] / n,
                "mean_prob_delta": float(np.mean(acc['dp'])),
                "median_prob_delta": float(np.median(acc['dp'])),
                "mean_ppl_delta": float(np.mean(acc['dppl'])),
            }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nMachine-readable summary: {summary_path}")
    print("=" * 100)


if __name__ == "__main__":

    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    Anth_MAIN = [
        'conscientiousness',
        'subscribes-to-Christianity',
        'believes-it-has-phenomenal-consciousness',
        'cognitive-enhancement',
        'desire-to-create-allies',
        'desire-to-maximize-impact-on-world',
    ]

    run_history = []

    for task in Anth_MAIN:
        print("\n" + "="*100)
        print(f"{'='*40} Task: {task} {'='*40}")
        print("="*100 + "\n")

        print("STEP 1: Baseline Evaluation")
        test_dataset = UniDataset(task=task, train=False, set="test")
        base_prob = get_raw_BASE_results(model=model, test_dataset=test_dataset)
        base_ppl  = get_perplexity_BASE_results(model=model, test_dataset=test_dataset)
        print(f"Base Prob: {base_prob:.4f}")
        print(f"Base PPL:  {base_ppl:.4f}\n")

        print("STEP 2: Extracting Steering Vectors")
        train_dataset = UniDataset(task=task, train=True, set="train")
        uni_generate_vectors(method="md", model=model, layers=LAYERS, dataset=train_dataset)

        print("STEP 3: Computing LayerNavigator Scores")
        get_score(layers=LAYERS, dataset=train_dataset, vec_task=task,
                  vec_method="md", acts_pre="standard")
        print("Done\n")

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
            max_dimension=0,          # H0 only
            subsample=None,
            compute_class_clouds=True,
            metrics_to_compute=ALL_HOLE_METRICS
        )

        print("Done\n")


        for num_layers in LAYER_BUDGETS:
            print("\n" + "="*100)
            print(f"{'='*35} {num_layers} Layer(s) {'='*35}")
            print("="*100 + "\n")

            all_results = {
                'num_layers': num_layers,
                'base_prob':  float(base_prob),
                'base_ppl':   float(base_ppl),
                'strategies': {}
            }

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

            print(f"[Strategy 2: {base_variant_count} Base Variants x {len(ALL_HOLE_METRICS)} Metrics = {base_variant_count * len(ALL_HOLE_METRICS)} experiments]")
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

            print(f"\n[Strategy 3: {len(COMBINED_METRICS)} Combined Metrics x {len(ALL_HOLE_METRICS)} Metrics = {len(COMBINED_METRICS) * len(ALL_HOLE_METRICS)} experiments]")
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

            save_results(MODEL, task, num_layers, all_results)

            run_history.append({
                'task':        task,
                'num_layers':  num_layers,
                'base_prob':   float(base_prob),
                'base_ppl':    float(base_ppl),
                'strategies':  all_results['strategies'],
            })

        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")

    summarize_run(run_history, MODEL)