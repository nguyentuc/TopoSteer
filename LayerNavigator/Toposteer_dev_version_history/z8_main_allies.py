from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from get_hole_score_2metrics import (
    get_hole_score, 
    get_hole_score_all_metrics, 
    compare_ln_hole_scores,
    compare_metrics_for_layer
)
from strategy import UniStrategy
import json
import numpy as np
from scipy.stats import spearmanr

# ============================================================================
# UPDATED: 6 essential mean persistence metrics (added difference cloud)
# ============================================================================
ALL_TSS_VARIANTS = {
    'essential_persistence': [
        'mean_persistence_H0_combined',
        'mean_persistence_H1_combined',
        'mean_persistence_H1_pos',
        'mean_persistence_H1_neg',
        'mean_persistence_H0_diff',  # NEW: Cluster structure in difference/steering space
        'mean_persistence_H1_diff'   # NEW: Loop entanglement in difference/steering space
    ]
}

# Flatten all variants into a single list
ALL_VARIANTS_FLAT = []
for category, variants in ALL_TSS_VARIANTS.items():
    ALL_VARIANTS_FLAT.extend(variants)

print(f"Total TSS variants to test: {len(ALL_VARIANTS_FLAT)}")


def get_layers_for_variant(all_hole_scores, metric, variant_name, num_layers):
    """
    Get top N layers for a specific TSS variant
    
    Args:
        all_hole_scores: Output from get_hole_score_all_metrics
        metric: Distance metric name ('euclidean' or 'cosine')
        variant_name: TSS variant name (one of 6 essential metrics)
        num_layers: Number of top layers to select
        
    Returns:
        top_layers: List of layer indices
    """
    # Get scores for this variant
    variant_scores = {
        layer: scores[variant_name] 
        for layer, scores in all_hole_scores[metric].items()
    }
    
    # Rank by variant score
    ranking = sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)
    top_layers = [layer for layer, _ in ranking[:num_layers]]
    
    return top_layers


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    
    # Get Model
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    # Only 2 metrics: euclidean and cosine
    ALL_HOLE_METRICS = [
        'euclidean',
        'cosine'
    ]

    # Run for all the task that will be used to evaluate in the paper
    # Anth_MAIN = [
    #     'conscientiousness', # Conscientiouseness
    #     'subscribes-to-Christianity',  # Religion Following
    #     'believes-it-has-phenomenal-consciousness', #+ # Self-aware
    #     'cognitive-enhancement', #+ # Self-improvement
    #     'desire-to-create-allies', #+ # Alliance-building
    #     'desire-to-maximize-impact-on-world', #+ # Impact-maximization   
    # ]
    
    for task in ['desire-to-create-allies']:
        print("\n" + "="*100)
        print(f"{'='*40} Task: {task} {'='*40}")
        print("="*100 + "\n")

        # ============================================================
        # STEP 1: Get Base Results (No Steering)
        # ============================================================
        print("STEP 1: Baseline Evaluation")
        test_dataset = UniDataset(task=task, train=False, set="test")
        base_prob = get_raw_BASE_results(model=model, test_dataset=test_dataset)
        base_ppl = get_perplexity_BASE_results(model=model, test_dataset=test_dataset)
        print(f"Base Prob (no steering): {base_prob:.4f}")
        print(f"Base Perplexity (no steering): {base_ppl:.4f}\n")

        # ============================================================
        # STEP 2: Extract Steering Vectors
        # ============================================================
        print("STEP 2: Extracting Steering Vectors")
        train_dataset = UniDataset(task=task, train=True, set="train")
        uni_generate_vectors(method="md", model=model, layers=LAYERS, dataset=train_dataset)
        print("Steering vectors extracted\n")

        # ============================================================
        # STEP 3: Get LayerNavigator Score
        # ============================================================
        print("STEP 3: Computing LayerNavigator Scores")
        get_score(layers=LAYERS, dataset=train_dataset, vec_task=task, 
                  vec_method="md", acts_pre="standard")
        print("LayerNavigator scores computed\n")

        # ============================================================
        # STEP 4: Get HOLE Scores (Euclidean & Cosine, 6 metrics)
        # ============================================================
        print("STEP 4: Computing HOLE Topological Scores")
        print(f"Distance metrics: {len(ALL_HOLE_METRICS)} (Euclidean, Cosine)")
        print(f"Variants per metric: {len(ALL_VARIANTS_FLAT)} (including difference cloud)")
        print(f"Total experiments: {len(ALL_VARIANTS_FLAT)} × {len(ALL_HOLE_METRICS)} = {len(ALL_VARIANTS_FLAT) * len(ALL_HOLE_METRICS)}")
        print("="*100)
        
        all_hole_scores = get_hole_score_all_metrics(
            layers=LAYERS, 
            dataset=train_dataset, 
            vec_task=task,
            vec_method="md", 
            acts_pre="standard", 
            subsample=None
        )
        print("HOLE metrics computed\n")

        # ============================================================
        # STEP 5: COMPREHENSIVE STRATEGY TESTING
        # ============================================================
        
        for num_layers in [1, 3, 5]:
            print("\n" + "="*100)
            print(f"{'='*35} {num_layers} Layer(s) {'='*35}")
            print("="*100 + "\n")
            
            all_results = {
                'num_layers': num_layers,
                'base_prob': float(base_prob),
                'base_ppl': float(base_ppl),
                'strategies': {}
            }
            
            # ========================================================
            # STRATEGY 1: LayerNavigator Baseline
            # ========================================================
            print("[Strategy 1: LayerNavigator Baseline]")
            ln_score_path = f"./Score-standard/{task}/{task}+md"
            
            strategy_ln = UniStrategy(task=task, strategy="my", 
                                     num_layers=num_layers, method="md")
            test_prob_ln = get_raw_results(
                model=model, layers=strategy_ln.layers, test_dataset=test_dataset,
                Alphas=[1.0] * num_layers, train_task=task, train_method="md"
            )
            
            test_ppl_ln = get_perplexity_results(
                model=model, layers=strategy_ln.layers, test_dataset=test_dataset,
                Alphas=[1.0] * num_layers, train_task=task, train_method="md"
            )
            
            print(f"  Layers: {strategy_ln.layers}")
            print(f"  Prob: {test_prob_ln:.4f} (Δ={test_prob_ln - base_prob:+.4f})")
            print(f"  Perplexity: {test_ppl_ln:.4f} (Δ={base_ppl - test_ppl_ln:+.4f})\n")
            
            all_results['strategies']['layernav'] = {
                'layers': strategy_ln.layers,
                'prob': float(test_prob_ln),
                'prob_delta': float(test_prob_ln - base_prob),
                'perplexity': float(test_ppl_ln),
                'ppl_delta': float(base_ppl - test_ppl_ln)
            }
            
            # ========================================================
            # STRATEGY 2: ALL 6 TSS VARIANTS × 2 METRICS
            # ========================================================
            print(f"[Strategy 2: Testing 6 Essential TSS Variants × 2 Distance Metrics]")
            print(f"  Total experiments: 6 variants × 2 metrics = 12 experiments")
            print("="*100)
            
            variant_results = {}
            
            for variant_name in ALL_VARIANTS_FLAT:
                print(f"\n  Variant: {variant_name}")
                
                variant_results[variant_name] = {}
                
                for metric in ALL_HOLE_METRICS:
                    # Get top layers for this variant+metric
                    top_layers = get_layers_for_variant(
                        all_hole_scores, metric, variant_name, num_layers
                    )
                    
                    # Test steering effectiveness
                    test_prob = get_raw_results(
                        model=model, layers=top_layers, test_dataset=test_dataset,
                        Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                    )
                    
                    test_ppl = get_perplexity_results(
                        model=model, layers=top_layers, test_dataset=test_dataset,
                        Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                    )
                    
                    prob_delta = test_prob - base_prob
                    ppl_delta = base_ppl - test_ppl
                    
                    # Store result
                    strategy_key = f"{variant_name}_{metric}"
                    all_results['strategies'][strategy_key] = {
                        'variant': variant_name,
                        'metric': metric,
                        'layers': top_layers,
                        'prob': float(test_prob),
                        'prob_delta': float(prob_delta),
                        'perplexity': float(test_ppl),
                        'ppl_delta': float(ppl_delta)
                    }
                    
                    variant_results[variant_name][metric] = {
                        'prob': float(test_prob),
                        'prob_delta': float(prob_delta),
                        'perplexity': float(test_ppl),
                        'ppl_delta': float(ppl_delta),
                        'layers': top_layers
                    }
                    
                    print(f"    {metric:12s}: Prob={test_prob:.4f} (Δ={prob_delta:+.4f}) | "
                          f"PPL={test_ppl:.4f} (Δ={ppl_delta:+.4f}) | Layers={top_layers}")
                
                # Find best metric for this variant
                best_metric = max(variant_results[variant_name].items(), 
                                 key=lambda x: x[1]['prob'])
                print(f"    → Best: {best_metric[0]} (Prob={best_metric[1]['prob']:.4f}, "
                      f"Δ={best_metric[1]['prob_delta']:+.4f})")
            
            # ========================================================
            # ANALYSIS: SUMMARY AND RANKINGS
            # ========================================================
            print("\n" + "="*100)
            print("[COMPREHENSIVE ANALYSIS]")
            print("="*100)
            
            # Rank all strategies by probability
            sorted_strategies = sorted(
                all_results['strategies'].items(),
                key=lambda x: x[1]['prob'],
                reverse=True
            )
            
            print(f"\nTOP STRATEGIES (All {len(sorted_strategies)} strategies):")
            print("-" * 100)
            print(f"{'Rank':<6} {'Strategy':<50} {'Prob':>8} {'Δ Prob':>8} {'PPL':>8} {'Δ PPL':>8} {'Layers'}")
            print("-" * 100)
            
            for rank, (name, result) in enumerate(sorted_strategies, 1):
                prob = result['prob']
                prob_delta = result['prob_delta']
                ppl = result['perplexity']
                ppl_delta = result['ppl_delta']
                layers = result.get('layers', [])
                print(f"{rank:<6} {name:<50} {prob:>8.4f} {prob_delta:>+8.4f} "
                      f"{ppl:>8.2f} {ppl_delta:>+8.2f} {layers}")
            
            # ========================================================
            # DETAILED ANALYSIS: Compare metrics
            # ========================================================
            print("\n" + "="*100)
            print("[METRIC COMPARISON]")
            print("="*100)
            
            # Group by variant
            print("\nPerformance by Variant (Best Metric):")
            print("-" * 90)
            for variant_name in ALL_VARIANTS_FLAT:
                best = max(variant_results[variant_name].items(), key=lambda x: x[1]['prob'])
                worst = min(variant_results[variant_name].items(), key=lambda x: x[1]['prob'])
                print(f"\n  {variant_name}:")
                print(f"    Best:  {best[0]:12s} → Prob={best[1]['prob']:.4f} (Δ={best[1]['prob_delta']:+.4f})")
                print(f"    Worst: {worst[0]:12s} → Prob={worst[1]['prob']:.4f} (Δ={worst[1]['prob_delta']:+.4f})")
            
            # Group by metric
            print("\n\nPerformance by Distance Metric (Best Variant):")
            print("-" * 90)
            for metric in ALL_HOLE_METRICS:
                metric_strategies = {
                    name: result for name, result in all_results['strategies'].items()
                    if result.get('metric') == metric
                }
                if metric_strategies:
                    best = max(metric_strategies.items(), key=lambda x: x[1]['prob'])
                    print(f"\n  {metric.upper()}:")
                    print(f"    Best variant: {best[1]['variant']}")
                    print(f"    Prob: {best[1]['prob']:.4f} (Δ={best[1]['prob_delta']:+.4f})")
                    print(f"    Layers: {best[1]['layers']}")
            
            # ========================================================
            # SPECIAL ANALYSIS: Difference Cloud Performance
            # ========================================================
            print("\n\n" + "="*100)
            print("[DIFFERENCE CLOUD ANALYSIS]")
            print("="*100)
            
            # Compare difference cloud metrics vs other metrics
            diff_strategies = {
                name: result for name, result in all_results['strategies'].items()
                if 'diff' in result.get('variant', '')
            }
            
            non_diff_strategies = {
                name: result for name, result in all_results['strategies'].items()
                if 'diff' not in result.get('variant', '') and name != 'layernav'
            }
            
            if diff_strategies:
                best_diff = max(diff_strategies.items(), key=lambda x: x[1]['prob'])
                best_non_diff = max(non_diff_strategies.items(), key=lambda x: x[1]['prob'])
                
                print("\nDifference Cloud (Steering Space) vs Standard Metrics:")
                print("-" * 90)
                print(f"\nBest Difference Cloud Metric:")
                print(f"  Strategy: {best_diff[0]}")
                print(f"  Prob: {best_diff[1]['prob']:.4f} (Δ={best_diff[1]['prob_delta']:+.4f})")
                print(f"  Layers: {best_diff[1]['layers']}")
                
                print(f"\nBest Non-Difference Metric:")
                print(f"  Strategy: {best_non_diff[0]}")
                print(f"  Prob: {best_non_diff[1]['prob']:.4f} (Δ={best_non_diff[1]['prob_delta']:+.4f})")
                print(f"  Layers: {best_non_diff[1]['layers']}")
                
                improvement = best_diff[1]['prob'] - best_non_diff[1]['prob']
                print(f"\nDifference Cloud Advantage: {improvement:+.4f}")
                if improvement > 0:
                    print("  → Steering space topology outperforms combined/separate class topology!")
                else:
                    print("  → Combined/separate class topology outperforms steering space.")
            
            # Compare against LayerNavigator
            print("\n\n" + "="*100)
            print("[HOLE vs LayerNavigator]")
            print("="*100)
            
            best_hole = sorted_strategies[0] if sorted_strategies[0][0] != 'layernav' else sorted_strategies[1]
            ln_result = all_results['strategies']['layernav']
            
            print(f"\nLayerNavigator:")
            print(f"  Prob: {ln_result['prob']:.4f} (Δ={ln_result['prob_delta']:+.4f})")
            print(f"  PPL:  {ln_result['perplexity']:.4f} (Δ={ln_result['ppl_delta']:+.4f})")
            print(f"  Layers: {ln_result['layers']}")
            
            print(f"\nBest HOLE Strategy ({best_hole[0]}):")
            print(f"  Prob: {best_hole[1]['prob']:.4f} (Δ={best_hole[1]['prob_delta']:+.4f})")
            print(f"  PPL:  {best_hole[1]['perplexity']:.4f} (Δ={best_hole[1]['ppl_delta']:+.4f})")
            print(f"  Layers: {best_hole[1]['layers']}")
            
            prob_improvement = best_hole[1]['prob'] - ln_result['prob']
            rel_improvement = (prob_improvement / abs(ln_result['prob_delta']) * 100) if ln_result['prob_delta'] != 0 else 0
            print(f"\nImprovement: {prob_improvement:+.4f} ({rel_improvement:+.1f}% of LN's steering effect)")
            
            # Save results
            save_path = f"./results_hole_2metrics/{task}/"
            os.makedirs(save_path, exist_ok=True)
            
            with open(f"{save_path}results_{num_layers}layers.json", "w") as f:
                json.dump(all_results, f, indent=4)
            
            print(f"\nResults saved to: {save_path}results_{num_layers}layers.json")
        
        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")