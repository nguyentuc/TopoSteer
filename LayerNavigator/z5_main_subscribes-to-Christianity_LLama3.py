from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from get_hole_score_7metrics import (
    get_hole_score, 
    get_hole_score_all_metrics, 
    compare_ln_hole_scores,
    compare_metrics_for_layer
)
from strategy import UniStrategy
import json
import numpy as np
from scipy.stats import spearmanr

ALL_TSS_VARIANTS = {
    # ===== G1: BETTI NUMBERS =====
    'betti_numbers': [
        'beta0',
        'beta1',
        'beta2',
        'sum_beta_counts'
    ],
            
    # ===== G2: BASIC PERSISTENCE STATISTICS =====
    'persistence_stats': [
        # Mean persistence
        'mean_persistence_H0',
        'mean_persistence_H1',
        'mean_persistence_H2',
        'count_H0',
        
        # Max persistence
        'max_persistence_H0',
        'max_persistence_H1',
        'max_persistence_H2',
        'count_H1',
        
        # Total persistence
        'total_persistence_H0',
        'total_persistence_H1',
        'total_persistence_H2',
        'count_H2',
        'total_persistence',
        
        # Single dimension persistence
        'h0_persistence_only',
        'h1_persistence_only',
        'h2_persistence_only',
        
        # Combined persistence
        'mean_all_persistence'
    ],
    
    # ===== G2 EXTENDED: PARADOXICAL VARIANTS =====
    'paradoxical_persistence': [
        'robust_clusters_fragile_loops',
        'robust_clusters_fragile_voids',
        'persistence_inverseH1_inverseH2'
    ],
    
    # ===== G3: PERSISTENCE ENTROPY =====
    'entropy': [
        # Raw entropy
        'entropy_H0',
        'entropy_H1',
        'entropy_H2',
        
        # Inverse entropy (higher = better)
        'inverse_h0_entropy',
        'inverse_h1_entropy',
        'inverse_h2_entropy'
    ],
    
    # ===== G4: BETTI CURVE AUC (Multi-scale robustness) =====
    'multi_scale_robustness': [
        'betti_curve_auc_H0',
        'betti_curve_auc_H1',
        'betti_curve_auc_H2'
    ],
    
    # ===== G5: STRONG FEATURES =====
    'strong_features': [
        # Loops
        'strong_loops_count',
        'inverse_strong_loops_count'
        
        # Clusters
        'strong_cluster_count',
        'inverse_strong_cluster_count',
        
        # Voids
        'strong_voids_count',
        'inverse_strong_voids_count'
    ],
    
    # ===== G6: PERSISTENCE DOMINANCE & RATIOS =====
    'dominance_ratios': [
        'portion_of_persistence_dominance_h0',
        'portion_of_persistence_dominance_h1',
        'portion_of_persistence_dominance_h2',
        'avg_max_persistence_score',
        # Inverse (control/opposite hypothesis)
        'inverse_mean_H0',
        'inverse_mean_H1',
        'inverse_mean_H2'
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
        metric: Distance metric name (e.g., 'cosine')
        variant_name: TSS variant name (e.g., 'tss_robust_clusters_fragile_loops')
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


def compute_variant_ensemble(all_hole_scores, metric, variants_list, layers):
    """
    Compute ensemble score across multiple TSS variants for a single metric
    
    Args:
        all_hole_scores: Output from get_hole_score_all_metrics
        metric: Distance metric name
        variants_list: List of variant names to ensemble
        layers: List of all layer indices
        
    Returns:
        ensemble_scores: Dict[layer -> float]
    """
    all_normalized_scores = []
    
    for variant in variants_list:
        # Get scores for this variant
        variant_vals = np.array([
            all_hole_scores[metric][l][variant] for l in layers
        ])
        
        # Normalize to [0, 1]
        variant_norm = (variant_vals - variant_vals.min()) / (
            variant_vals.max() - variant_vals.min() + 1e-8
        )
        
        all_normalized_scores.append(variant_norm)
    
    # Average across variants
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

    # Define all 7 metrics from HOLE paper
    ALL_HOLE_METRICS = [
        'euclidean',
        'cosine',
        'mahalanobis',
        'geodesic',
        'dens_norm_euclidean',
        'dens_norm_cosine',
        'dens_norm_mahalanobis'
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
    
    for task in ['subscribes-to-Christianity']:
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
        print(f"Base Prob (no steering): {base_prob:.4f}\n")
        print(f"Base Perplexity (no steering): {base_ppl:.4f}")

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
        # STEP 4: Get HOLE Scores (ALL 7 METRICS)
        # ============================================================
        print("STEP 4: Computing HOLE Topological Scores (ALL 7 DISTANCE METRICS)")
        print(f"Total variants per metric: {len(ALL_VARIANTS_FLAT)}")
        print("="*100)
        
        all_hole_scores = get_hole_score_all_metrics(
            layers=LAYERS, dataset=train_dataset, vec_task=task,
            vec_method="md", acts_pre="standard", max_dimension=2, subsample=500
        )
        print("All 7 HOLE metrics computed\n")

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
            print(f"  Prob: {test_prob_ln:.4f} (delta={test_prob_ln - base_prob:+.4f})\n")
            print(f"  Perplexity: {test_ppl_ln:.4f} (delta={base_ppl - test_ppl_ln:+.4f})")
            
            all_results['strategies']['layernav'] = {
                'layers': strategy_ln.layers,
                'prob': float(test_prob_ln),
                'delta': float(test_prob_ln - base_prob),
                'prob_delta': float(test_prob_ln - base_prob),
                'perplexity': float(test_ppl_ln),
                'ppl_delta': float(base_ppl - test_ppl_ln)
                
            }
            
            # ========================================================
            # STRATEGY 2: ALL TSS VARIANTS × ALL METRICS
            # ========================================================
            print(f"[Strategy 2: Testing ALL {len(ALL_VARIANTS_FLAT)} TSS Variants X 7 Distance Metrics]")
            print(f"  Total experiments: {len(ALL_VARIANTS_FLAT)} X 7 = {len(ALL_VARIANTS_FLAT) * 7}")
            print("="*100)
            
            variant_results = {}
            
            for category_name, variant_list in ALL_TSS_VARIANTS.items():
                print(f"\nCATEGORY: {category_name.upper()} ({len(variant_list)} variants)")
                print("-" * 100)
                
                for variant_name in variant_list:
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
                        
                        delta = test_prob - base_prob
                        ppl_delta = base_ppl - test_ppl
                        
                        # Store result
                        strategy_key = f"{variant_name}_{metric}"
                        all_results['strategies'][strategy_key] = {
                            'variant': variant_name,
                            'metric': metric,
                            'category': category_name,
                            'layers': top_layers,
                            'prob': float(test_prob),
                            'delta': float(delta),
                            'prob_delta': float(prob_delta),
                            'perplexity': float(test_ppl),
                            'ppl_delta': float(ppl_delta)
                        }
                        
                        variant_results[variant_name][metric] = {
                            'prob': float(test_prob),
                            'delta': float(delta),
                            'prob_delta': float(prob_delta),
                            'perplexity': float(test_ppl),
                            'ppl_delta': float(ppl_delta),
                            'layers': top_layers
                        }
                        
                        print(f"    {metric:25s}: Prob={test_prob:.4f} (delta={delta:+.4f}) | PPL={test_ppl:.4f} (PPL_delta={ppl_delta:+.4f}) | Layers={top_layers}")
                    
                    # Find best metric for this variant
                    best_metric = max(variant_results[variant_name].items(), 
                                     key=lambda x: x[1]['prob'])
                    print(f"Best metric: {best_metric[0]} (Prob={best_metric[1]['prob']:.4f})")
            
            # ========================================================
            # ANALYSIS: SUMMARY AND RANKINGS
            # ========================================================
            print("\n" + "="*100)
            print("[COMPREHENSIVE ANALYSIS]")
            print("="*100)
            
            # Top 30 strategies overall
            print("TOP 30 STRATEGIES (All Types):")
            sorted_strategies = sorted(
                all_results['strategies'].items(),
                key=lambda x: x[1]['prob'],
                reverse=True
            )
            
            for rank, (name, result) in enumerate(sorted_strategies[:30], 1):
                prob = result['prob']
                delta = result['delta']
                ppl = result['perplexity']
                ppl_delta = result['ppl_delta']
                layers = result.get('layers', [])
                print(f"  {rank:2d}. {name:60s} | Prob={prob:.4f} (delta={delta:+.4f}) | PPL={ppl:.4f} (PPL_delta={ppl_delta:+.4f}) | Layers={layers}")
        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")