from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from get_multiscale_layernavigator_score import (
    get_multiscale_layernavigator_score,
    rank_layers_multiscale,
    compare_with_original_layernavigator
)
from strategy import UniStrategy
import json
import numpy as np
from scipy.stats import spearmanr

# ============================================================================
# MULTI-SCALE LAYERNAVIGATOR RANKING METHODS
# ============================================================================

ALL_MULTISCALE_METHODS = {
    # ===== ORIGINAL METHODS =====
    'original_methods': [
        'layernavigator',  # S(ε*) - original LayerNavigator at normalized scale
        'max',             # max_ε S(ε) - best possible score
        'robust',          # S(ε*) × p_0/(1+p_0) - robustness-weighted
        'persistence'      # p_0 only - pure robustness
    ],
    
    # ===== NEW: SCALE-INTEGRATED METHODS =====
    'scale_integrated_methods': [
        'total',           # Sum of S(ε) across all scales - versatility
        'avg',             # Average S(ε) - normalized versatility
        'integrated',      # ∫S(ε)dε - proper trapezoidal integral
        'log_weighted',    # Log-space weighted sum - scale-invariant
        'robust_total'     # total_S × p_0/(1+p_0) - BEST (versatile + stable)
    ]
}

# Flatten all methods into a single list
ALL_METHODS_FLAT = []
for category, methods in ALL_MULTISCALE_METHODS.items():
    ALL_METHODS_FLAT.extend(methods)

print(f"Total Multi-Scale Ranking Methods to test: {len(ALL_METHODS_FLAT)}")


def get_layers_for_ranking_method(multiscale_scores, metric, ranking_method, num_layers):
    """
    Get top N layers for a specific multi-scale ranking method
    
    Args:
        multiscale_scores: Output from get_multiscale_layernavigator_score
        metric: Distance metric name (e.g., 'cosine')
        ranking_method: Ranking method name (e.g., 'robust_total')
        num_layers: Number of top layers to select
        
    Returns:
        top_layers: List of layer indices
    """
    scores = multiscale_scores[metric]
    
    # Compute score for each layer based on ranking method
    layer_scores = {}
    
    for layer, data in scores.items():
        if ranking_method == 'layernavigator':
            score = data['layernavigator_S']
        elif ranking_method == 'max':
            score = data['max_S']
        elif ranking_method == 'robust':
            p_0 = data['persistence_length']
            S_ln = data['layernavigator_S']
            score = S_ln * (p_0 / (1 + p_0))
        elif ranking_method == 'persistence':
            score = data['persistence_length']
        elif ranking_method == 'total':
            score = data['total_S']
        elif ranking_method == 'avg':
            score = data['avg_S']
        elif ranking_method == 'integrated':
            score = data['integrated_S']
        elif ranking_method == 'log_weighted':
            score = data['log_weighted_S']
        elif ranking_method == 'robust_total':
            p_0 = data['persistence_length']
            total_S = data['total_S']
            score = total_S * (p_0 / (1 + p_0))
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        layer_scores[layer] = score
    
    # Rank by score
    ranking = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    top_layers = [layer for layer, _ in ranking[:num_layers]]
    
    return top_layers


def compute_method_ensemble(multiscale_scores, metric, methods_list, layers):
    """
    Compute ensemble score across multiple ranking methods for a single metric
    
    Args:
        multiscale_scores: Output from get_multiscale_layernavigator_score
        metric: Distance metric name
        methods_list: List of ranking method names to ensemble
        layers: List of all layer indices
        
    Returns:
        ensemble_scores: Dict[layer -> float]
    """
    all_normalized_scores = []
    
    for method in methods_list:
        # Get scores for this method
        method_vals = []
        for layer in layers:
            layer_scores = get_layers_for_ranking_method(
                {metric: multiscale_scores[metric]}, 
                metric, method, len(layers)
            )
            # Use rank as score (inverse)
            rank = layer_scores.index(layer) if layer in layer_scores else len(layers)
            method_vals.append(len(layers) - rank)
        
        method_vals = np.array(method_vals)
        
        # Normalize to [0, 1]
        method_norm = (method_vals - method_vals.min()) / (
            method_vals.max() - method_vals.min() + 1e-8
        )
        
        all_normalized_scores.append(method_norm)
    
    # Average across methods
    ensemble_scores = {}
    for idx, layer in enumerate(layers):
        ensemble_scores[layer] = np.mean([
            scores[idx] for scores in all_normalized_scores
        ])
    
    return ensemble_scores


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

    # Define distance metrics (same as HOLE for comparison)
    ALL_DISTANCE_METRICS = [
        'euclidean',
        'cosine',
        'mahalanobis'
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
        print(f"{'='*35} Task: {task} {'='*35}")
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
        # STEP 3: Get Original LayerNavigator Score (for comparison)
        # ============================================================
        print("STEP 3: Computing Original LayerNavigator Scores")
        get_score(layers=LAYERS, dataset=train_dataset, vec_task=task, 
                  vec_method="md", acts_pre="standard")
        print("LayerNavigator scores computed\n")

        # ============================================================
        # STEP 4: Get Multi-Scale LayerNavigator Scores (ALL METRICS)
        # ============================================================
        print("STEP 4: Computing Multi-Scale LayerNavigator Scores (ALL DISTANCE METRICS)")
        print(f"Distance Metrics: {ALL_DISTANCE_METRICS}")
        print(f"Ranking Methods per metric: {len(ALL_METHODS_FLAT)}")
        print(f"Total experiments: {len(ALL_METHODS_FLAT)} × {len(ALL_DISTANCE_METRICS)} = {len(ALL_METHODS_FLAT) * len(ALL_DISTANCE_METRICS)}")
        print("="*100)
        
        # Compute multi-scale scores for each distance metric
        all_multiscale_scores = {}
        
        for metric in ALL_DISTANCE_METRICS:
            print(f"\nComputing multi-scale scores with {metric} distance...")
            scores = get_multiscale_layernavigator_score(
                layers=LAYERS,
                dataset=train_dataset,
                vec_task=task,
                vec_method="md",
                acts_pre="standard",
                metric=metric,
                num_scales=50,  # Sample 50 scales
                subsample=None,  # Use all data (or set to 500 for speed)
                save_results=True
            )
            all_multiscale_scores[metric] = scores
            print(f"✓ {metric} completed")
        
        print("\nAll multi-scale scores computed\n")

        # ============================================================
        # STEP 5: COMPREHENSIVE STRATEGY TESTING
        # ============================================================
        
        for num_layers in [1, 3, 5]:
            print("\n" + "="*100)
            print(f"{'='*40} {num_layers} Layer(s) {'='*40}")
            print("="*100 + "\n")
            
            all_results = {
                'num_layers': num_layers,
                'base_prob': float(base_prob),
                'base_ppl': float(base_ppl),
                'strategies': {}
            }
            
            # ========================================================
            # STRATEGY 1: Original LayerNavigator Baseline
            # ========================================================
            print("[Strategy 1: Original LayerNavigator Baseline]")
            
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
            
            all_results['strategies']['original_layernav'] = {
                'method': 'original_layernav',
                'metric': 'N/A',
                'category': 'baseline',
                'layers': strategy_ln.layers,
                'prob': float(test_prob_ln),
                'prob_delta': float(test_prob_ln - base_prob),
                'perplexity': float(test_ppl_ln),
                'ppl_delta': float(base_ppl - test_ppl_ln)
            }
            
            # ========================================================
            # STRATEGY 2: ALL MULTI-SCALE METHODS × ALL METRICS
            # ========================================================
            print(f"[Strategy 2: Testing ALL {len(ALL_METHODS_FLAT)} Multi-Scale Methods × {len(ALL_DISTANCE_METRICS)} Distance Metrics]")
            print(f"  Total experiments: {len(ALL_METHODS_FLAT)} × {len(ALL_DISTANCE_METRICS)} = {len(ALL_METHODS_FLAT) * len(ALL_DISTANCE_METRICS)}")
            print("="*100)
            
            method_results = {}
            
            for category_name, method_list in ALL_MULTISCALE_METHODS.items():
                print(f"\nCATEGORY: {category_name.upper()} ({len(method_list)} methods)")
                print("-" * 100)
                
                for method_name in method_list:
                    print(f"\n  Ranking Method: {method_name}")
                    
                    method_results[method_name] = {}
                    
                    for metric in ALL_DISTANCE_METRICS:
                        # Get top layers for this method+metric
                        top_layers = get_layers_for_ranking_method(
                            all_multiscale_scores, metric, method_name, num_layers
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
                        strategy_key = f"{method_name}_{metric}"
                        all_results['strategies'][strategy_key] = {
                            'method': method_name,
                            'metric': metric,
                            'category': category_name,
                            'layers': top_layers,
                            'prob': float(test_prob),
                            'prob_delta': float(prob_delta),
                            'perplexity': float(test_ppl),
                            'ppl_delta': float(ppl_delta)
                        }
                        
                        method_results[method_name][metric] = {
                            'prob': float(test_prob),
                            'prob_delta': float(prob_delta),
                            'perplexity': float(test_ppl),
                            'ppl_delta': float(ppl_delta),
                            'layers': top_layers
                        }
                        
                        print(f"    {metric:15s}: Prob={test_prob:.4f} (Δ={prob_delta:+.4f}) | PPL={test_ppl:.4f} (Δ={ppl_delta:+.4f}) | Layers={top_layers}")
                    
                    # Find best metric for this method
                    best_metric = max(method_results[method_name].items(), 
                                     key=lambda x: x[1]['prob'])
                    print(f"    → Best metric: {best_metric[0]} (Prob={best_metric[1]['prob']:.4f}, Δ={best_metric[1]['prob_delta']:+.4f})")
            
            # ========================================================
            # STRATEGY 3: ENSEMBLE METHODS (Optional)
            # ========================================================
            print("\n" + "-"*100)
            print("[Strategy 3: Ensemble Methods]")
            print("-"*100)
            
            # Ensemble: All original methods
            print("\n  Ensemble 1: All Original Methods (layernav, max, robust, persistence)")
            for metric in ALL_DISTANCE_METRICS:
                ensemble_scores = compute_method_ensemble(
                    all_multiscale_scores, metric,
                    ALL_MULTISCALE_METHODS['original_methods'],
                    LAYERS
                )
                
                # Get top layers from ensemble
                top_layers_ensemble = sorted(
                    ensemble_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:num_layers]
                top_layers = [layer for layer, _ in top_layers_ensemble]
                
                # Test ensemble
                test_prob_ens = get_raw_results(
                    model=model, layers=top_layers, test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                )
                
                test_ppl_ens = get_perplexity_results(
                    model=model, layers=top_layers, test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                )
                
                prob_delta_ens = test_prob_ens - base_prob
                ppl_delta_ens = base_ppl - test_ppl_ens
                
                print(f"    {metric:15s}: Prob={test_prob_ens:.4f} (Δ={prob_delta_ens:+.4f}) | PPL={test_ppl_ens:.4f} (Δ={ppl_delta_ens:+.4f}) | Layers={top_layers}")
                
                all_results['strategies'][f'ensemble_original_{metric}'] = {
                    'method': 'ensemble_original',
                    'metric': metric,
                    'category': 'ensemble',
                    'layers': top_layers,
                    'prob': float(test_prob_ens),
                    'prob_delta': float(prob_delta_ens),
                    'perplexity': float(test_ppl_ens),
                    'ppl_delta': float(ppl_delta_ens)
                }
            
            # Ensemble: All scale-integrated methods
            print("\n  Ensemble 2: All Scale-Integrated Methods (total, avg, integrated, log_weighted, robust_total)")
            for metric in ALL_DISTANCE_METRICS:
                ensemble_scores = compute_method_ensemble(
                    all_multiscale_scores, metric,
                    ALL_MULTISCALE_METHODS['scale_integrated_methods'],
                    LAYERS
                )
                
                # Get top layers from ensemble
                top_layers_ensemble = sorted(
                    ensemble_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:num_layers]
                top_layers = [layer for layer, _ in top_layers_ensemble]
                
                # Test ensemble
                test_prob_ens = get_raw_results(
                    model=model, layers=top_layers, test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                )
                
                test_ppl_ens = get_perplexity_results(
                    model=model, layers=top_layers, test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers, train_task=task, train_method="md"
                )
                
                prob_delta_ens = test_prob_ens - base_prob
                ppl_delta_ens = base_ppl - test_ppl_ens
                
                print(f"    {metric:15s}: Prob={test_prob_ens:.4f} (Δ={prob_delta_ens:+.4f}) | PPL={test_ppl_ens:.4f} (Δ={ppl_delta_ens:+.4f}) | Layers={top_layers}")
                
                all_results['strategies'][f'ensemble_integrated_{metric}'] = {
                    'method': 'ensemble_integrated',
                    'metric': metric,
                    'category': 'ensemble',
                    'layers': top_layers,
                    'prob': float(test_prob_ens),
                    'prob_delta': float(prob_delta_ens),
                    'perplexity': float(test_ppl_ens),
                    'ppl_delta': float(ppl_delta_ens)
                }
            
            # ========================================================
            # ANALYSIS: SUMMARY AND RANKINGS
            # ========================================================
            print("\n" + "="*100)
            print("[COMPREHENSIVE ANALYSIS]")
            print("="*100)
            
            # Top 20 strategies overall
            print("\nTOP 20 STRATEGIES (by Probability):")
            print("-"*100)
            sorted_by_prob = sorted(
                all_results['strategies'].items(),
                key=lambda x: x[1]['prob'],
                reverse=True
            )
            
            for rank, (name, result) in enumerate(sorted_by_prob[:20], 1):
                method = result.get('method', 'N/A')
                metric = result.get('metric', 'N/A')
                prob = result['prob']
                prob_delta = result['prob_delta']
                ppl = result['perplexity']
                ppl_delta = result['ppl_delta']
                layers = result.get('layers', [])
                
                print(f"  {rank:2d}. {name:40s} | Method={method:20s} Metric={metric:15s}")
                print(f"      Prob={prob:.4f} (Δ={prob_delta:+.4f}) | PPL={ppl:.4f} (Δ={ppl_delta:+.4f}) | Layers={layers}")
            
            # Top strategies by category
            print("\n" + "-"*100)
            print("BEST STRATEGY PER CATEGORY:")
            print("-"*100)
            
            categories = {}
            for name, result in all_results['strategies'].items():
                cat = result.get('category', 'other')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((name, result))
            
            for cat, strategies in sorted(categories.items()):
                best = max(strategies, key=lambda x: x[1]['prob'])
                name, result = best
                print(f"\n{cat.upper()}:")
                print(f"  Best: {name}")
                print(f"  Prob={result['prob']:.4f} (Δ={result['prob_delta']:+.4f})")
                print(f"  PPL={result['perplexity']:.4f} (Δ={result['ppl_delta']:+.4f})")
                print(f"  Layers={result['layers']}")
            
            # Analysis: Compare original vs. new methods
            print("\n" + "-"*100)
            print("ORIGINAL vs. SCALE-INTEGRATED METHODS:")
            print("-"*100)
            
            original_best = max(
                [r for n, r in all_results['strategies'].items() 
                 if r.get('category') == 'original_methods'],
                key=lambda x: x['prob'],
                default=None
            )
            
            integrated_best = max(
                [r for n, r in all_results['strategies'].items() 
                 if r.get('category') == 'scale_integrated_methods'],
                key=lambda x: x['prob'],
                default=None
            )
            
            if original_best and integrated_best:
                print(f"\nBest Original Method:")
                print(f"  Method: {original_best['method']}")
                print(f"  Prob={original_best['prob']:.4f} (Δ={original_best['prob_delta']:+.4f})")
                print(f"  PPL={original_best['perplexity']:.4f} (Δ={original_best['ppl_delta']:+.4f})")
                
                print(f"\nBest Scale-Integrated Method:")
                print(f"  Method: {integrated_best['method']}")
                print(f"  Prob={integrated_best['prob']:.4f} (Δ={integrated_best['prob_delta']:+.4f})")
                print(f"  PPL={integrated_best['perplexity']:.4f} (Δ={integrated_best['ppl_delta']:+.4f})")
                
                improvement = integrated_best['prob'] - original_best['prob']
                print(f"\nImprovement: {improvement:+.4f} ({improvement/original_best['prob']*100:+.2f}%)")
            
            # Save results to JSON
            save_path = f"./results_multiscale_{task}_{num_layers}layers.json"
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"\nResults saved to: {save_path}")
        
        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")