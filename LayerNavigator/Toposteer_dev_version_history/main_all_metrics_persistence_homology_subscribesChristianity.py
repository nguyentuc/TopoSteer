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

if __name__ == "__main__":
    ### Step 1.
    ### Complement the model path to the first line of the *globalenv.py* file.

    ### Step 2.
    ### Run the following code to get the main results in our paper.

    # Get Model
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

    # Define all 7 metrics from HOLE paper
    ALL_HOLE_METRICS = [
        'euclidean',           # Metric 1: Raw geometric distance
        'cosine',              # Metric 2: Directional similarity (best for LLMs)
        'mahalanobis',         # Metric 3: Covariance-aware
        'geodesic',            # Metric 4: Manifold distance
        'dens_norm_euclidean', # Metric 5: Multi-scale geometric
        'dens_norm_cosine',    # Metric 6: Multi-scale directional
        'dens_norm_mahalanobis' # Metric 7: Multi-scale covariance
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
        test_dataset = UniDataset(
            task=task,
            train=False,
            set="test",
        )
        base_prob = get_raw_BASE_results(
            model=model,
            test_dataset=test_dataset,
        )
        print(f"Base Prob (no steering): {base_prob:.4f}\n")

        # ============================================================
        # STEP 2: Extract Steering Vectors and Activations
        # ============================================================
        print("STEP 2: Extracting Steering Vectors")
        train_dataset = UniDataset(
            task=task,
            train=True,
            set="train",
        )
        
        # Extract mean difference steering vector
        uni_generate_vectors(
            method="md",
            model=model,
            layers=LAYERS,
            dataset=train_dataset,
        )
        print("Steering vectors extracted\n")

        # ============================================================
        # STEP 3A: Get LayerNavigator Score (Original Method)
        # ============================================================
        print("STEP 3A: Computing LayerNavigator Scores")
        get_score(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
        )
        print("LayerNavigator scores computed\n")

        # ============================================================
        # STEP 3B: Get HOLE Topological Scores (ALL 7 METRICS!)
        # ============================================================
        print("STEP 3B: Computing HOLE Topological Scores (ALL 7 METRICS)")
        print("="*100)
        
        all_hole_scores = get_hole_score_all_metrics(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
            max_dimension=2,
            subsample=500,  # Adjust based on dataset size and compute budget
        )
        
        print("All 7 HOLE metrics computed\n")

        # ============================================================
        # STEP 3C: Compare LayerNavigator vs HOLE Scores (All Metrics)
        # ============================================================
        print("STEP 3C: Comparing LayerNavigator with All HOLE Metrics")
        print("="*100)
        
        ln_score_path = f"./Score-standard/{task}/{task}+md"
        
        all_comparisons = {}
        
        for metric in ALL_HOLE_METRICS:
            print(f"\n--- Metric: {metric} ---")
            hole_score_path = f"./Score_HOLE-standard-{metric}/{task}/{task}+md"
            
            comparison = compare_ln_hole_scores(
                layers=LAYERS,
                ln_score_path=ln_score_path,
                hole_score_path=hole_score_path,
            )
            
            all_comparisons[metric] = comparison
            
            print(f"  Spearman Correlation: {comparison['spearman_correlation']:.3f} (p={comparison['spearman_pvalue']:.4f})")
            print(f"  Top-10 Jaccard Similarity: {comparison['jaccard_top10']:.3f}")
            print(f"  Agreement: {len(comparison['agreement'])}/10 layers")
        
        # Save all comparisons
        comparison_path = f"./Comparison/{task}/"
        os.makedirs(comparison_path, exist_ok=True)
        with open(f"{comparison_path}ln_vs_all_hole_metrics.json", "w") as f:
            json.dump(all_comparisons, f, indent=4)
        
        print("All comparisons saved\n")

        # ============================================================
        # STEP 3D: Analyze Metric Consensus
        # ============================================================
        print("STEP 3D: Analyzing Metric Consensus")
        print("="*100)
        
        # Find layers that are highly ranked across multiple metrics
        metric_rankings = {}
        for metric in ALL_HOLE_METRICS:
            ranking = sorted(
                all_hole_scores[metric].items(),
                key=lambda x: x[1]['tss'],
                reverse=True
            )
            metric_rankings[metric] = [layer for layer, _ in ranking]
        
        # Compute consensus: layers in top-10 for multiple metrics
        top_k = 10
        consensus_count = {}
        for layer in LAYERS:
            consensus_count[layer] = sum(
                1 for metric_ranking in metric_rankings.values()
                if layer in metric_ranking[:top_k]
            )
        
        # Sort by consensus
        consensus_ranking = sorted(
            consensus_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\nLayers by Metric Consensus (top-10 appearances):")
        print("-" * 60)
        for layer, count in consensus_ranking[:15]:
            print(f"  Layer {layer:2d}: Appears in top-10 of {count}/7 metrics")
        
        # Save consensus analysis
        with open(f"{comparison_path}metric_consensus.json", "w") as f:
            json.dump({
                'consensus_count': consensus_count,
                'consensus_ranking': [layer for layer, _ in consensus_ranking],
                'top_consensus_layers': [layer for layer, count in consensus_ranking[:10]]
            }, f, indent=4)

        # ============================================================
        # STEP 4: Test Layer Selection Strategies (All Metrics + Ensemble)
        # ============================================================
        print("\n>>> STEP 4: Testing Layer Selection Strategies")
        print("="*100)
        
        for num_layers in [1, 3, 5]:
            print(f"\n{'='*40} {num_layers} Layer(s) {'='*40}")
            
            all_results = {
                'num_layers': num_layers,
                'base_prob': float(base_prob),
                'strategies': {}
            }
            
            # ============================================================
            # Strategy 1: LayerNavigator Only (Baseline)
            # ============================================================
            print(f"\n[Strategy 1: LayerNavigator Only]")
            strategy_ln = UniStrategy(
                task=task,
                strategy="my",
                num_layers=num_layers,
                method="md",
            )
            test_prob_ln = get_raw_results(
                model=model,
                layers=strategy_ln.layers,
                test_dataset=test_dataset,
                Alphas=[1.0] * num_layers,
                train_task=task,
                train_method="md",
            )
            print(f"  Layers: {strategy_ln.layers}")
            print(f"  Test Prob: {test_prob_ln:.4f} (Δ={test_prob_ln - base_prob:+.4f})")
            
            all_results['strategies']['layernav'] = {
                'layers': strategy_ln.layers,
                'prob': float(test_prob_ln),
                'delta': float(test_prob_ln - base_prob)
            }
            
            # ============================================================
            # Strategy 2: Each HOLE Metric Individually
            # ============================================================
            print(f"\n[Strategy 2: Individual HOLE Metrics]")
            
            for metric in ALL_HOLE_METRICS:
                # Rank layers by this metric's TSS score
                hole_ranking = sorted(
                    all_hole_scores[metric].items(),
                    key=lambda x: x[1]['tss'],
                    reverse=True
                )
                hole_top_layers = [layer for layer, _ in hole_ranking[:num_layers]]
                
                test_prob_hole = get_raw_results(
                    model=model,
                    layers=hole_top_layers,
                    test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers,
                    train_task=task,
                    train_method="md",
                )
                
                print(f"  {metric:25s}: Prob={test_prob_hole:.4f} (Δ={test_prob_hole - base_prob:+.4f}) | Layers={hole_top_layers}")
                
                all_results['strategies'][f'hole_{metric}'] = {
                    'layers': hole_top_layers,
                    'prob': float(test_prob_hole),
                    'delta': float(test_prob_hole - base_prob)
                }
            
            # ============================================================
            # Strategy 3: Hybrid (LN + Best HOLE Metric)
            # ============================================================
            print(f"\n[Strategy 3: Hybrid Strategies]")
            
            # Find best performing HOLE metric
            best_hole_metric = max(
                ALL_HOLE_METRICS,
                key=lambda m: all_results['strategies'][f'hole_{m}']['prob']
            )
            print(f"  Best HOLE metric: {best_hole_metric}")
            
            # Load LN scores
            ln_scores_dict = {}
            for layer in LAYERS:
                with open(f"{ln_score_path}/L{layer}.json", "r") as f:
                    ln_scores_dict[layer] = json.load(f)['s_score']
            
            # Test different alpha values
            for alpha in [0.3, 0.5, 0.7]:
                # Normalize both scores to [0, 1]
                ln_vals = np.array([ln_scores_dict[l] for l in LAYERS])
                hole_vals = np.array([all_hole_scores[best_hole_metric][l]['tss'] for l in LAYERS])
                
                ln_norm = (ln_vals - ln_vals.min()) / (ln_vals.max() - ln_vals.min() + 1e-8)
                hole_norm = (hole_vals - hole_vals.min()) / (hole_vals.max() - hole_vals.min() + 1e-8)
                
                # Hybrid score: weighted combination
                hybrid_scores = {}
                for idx, layer in enumerate(LAYERS):
                    hybrid_scores[layer] = alpha * ln_norm[idx] + (1 - alpha) * hole_norm[idx]
                
                # Rank by hybrid score
                hybrid_ranking = sorted(
                    hybrid_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                hybrid_top_layers = [layer for layer, _ in hybrid_ranking[:num_layers]]
                
                test_prob_hybrid = get_raw_results(
                    model=model,
                    layers=hybrid_top_layers,
                    test_dataset=test_dataset,
                    Alphas=[1.0] * num_layers,
                    train_task=task,
                    train_method="md",
                )
                
                print(f"  Hybrid α={alpha} (LN+{best_hole_metric}): Prob={test_prob_hybrid:.4f} (Δ={test_prob_hybrid - base_prob:+.4f}) | Layers={hybrid_top_layers}")
                
                all_results['strategies'][f'hybrid_alpha{alpha}'] = {
                    'layers': hybrid_top_layers,
                    'prob': float(test_prob_hybrid),
                    'delta': float(test_prob_hybrid - base_prob),
                    'alpha': alpha,
                    'hole_metric': best_hole_metric
                }
            
            # ============================================================
            # Strategy 4: Ensemble Across All HOLE Metrics
            # ============================================================
            print(f"\n[Strategy 4: Ensemble Across All Metrics]")
            
            # Average normalized scores across all HOLE metrics
            all_normalized_scores = []
            for metric in ALL_HOLE_METRICS:
                hole_vals = np.array([all_hole_scores[metric][l]['tss'] for l in LAYERS])
                hole_norm = (hole_vals - hole_vals.min()) / (hole_vals.max() - hole_vals.min() + 1e-8)
                all_normalized_scores.append(hole_norm)
            
            # Ensemble: average across all metrics
            ensemble_scores = {}
            for idx, layer in enumerate(LAYERS):
                ensemble_scores[layer] = np.mean([scores[idx] for scores in all_normalized_scores])
            
            # Rank by ensemble score
            ensemble_ranking = sorted(
                ensemble_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            ensemble_top_layers = [layer for layer, _ in ensemble_ranking[:num_layers]]
            
            test_prob_ensemble = get_raw_results(
                model=model,
                layers=ensemble_top_layers,
                test_dataset=test_dataset,
                Alphas=[1.0] * num_layers,
                train_task=task,
                train_method="md",
            )
            
            print(f"  Ensemble (avg of 7): Prob={test_prob_ensemble:.4f} (Δ={test_prob_ensemble - base_prob:+.4f}) | Layers={ensemble_top_layers}")
            
            all_results['strategies']['ensemble_all_metrics'] = {
                'layers': ensemble_top_layers,
                'prob': float(test_prob_ensemble),
                'delta': float(test_prob_ensemble - base_prob)
            }
            
            # ============================================================
            # Strategy 5: Consensus-Based Selection
            # ============================================================
            print(f"\n[Strategy 5: Consensus-Based Selection]")
            
            # Use layers with highest consensus count
            consensus_top_layers = [layer for layer, _ in consensus_ranking[:num_layers]]
            
            test_prob_consensus = get_raw_results(
                model=model,
                layers=consensus_top_layers,
                test_dataset=test_dataset,
                Alphas=[1.0] * num_layers,
                train_task=task,
                train_method="md",
            )
            
            print(f"  Consensus: Prob={test_prob_consensus:.4f} (Δ={test_prob_consensus - base_prob:+.4f}) | Layers={consensus_top_layers}")
            
            all_results['strategies']['consensus'] = {
                'layers': consensus_top_layers,
                'prob': float(test_prob_consensus),
                'delta': float(test_prob_consensus - base_prob)
            }
            
            # ============================================================
            # Summary and Winner
            # ============================================================
            print(f"\n{'='*40} SUMMARY {'='*40}")
            
            # Find best strategy
            best_strategy = max(
                all_results['strategies'].items(),
                key=lambda x: x[1]['prob']
            )
            
            print(f"\nBEST STRATEGY: {best_strategy[0]}")
            print(f"   Probability: {best_strategy[1]['prob']:.4f}")
            print(f"   Improvement: {best_strategy[1]['delta']:+.4f}")
            print(f"   Layers: {best_strategy[1]['layers']}")
            
            # Top 5 strategies
            print(f"\nTop 5 Strategies:")
            sorted_strategies = sorted(
                all_results['strategies'].items(),
                key=lambda x: x[1]['prob'],
                reverse=True
            )
            for rank, (name, result) in enumerate(sorted_strategies[:5], 1):
                print(f"  {rank}. {name:30s}: {result['prob']:.4f} (Δ={result['delta']:+.4f})")
            
            all_results['best_strategy'] = best_strategy[0]
            all_results['best_prob'] = best_strategy[1]['prob']
            
            # Save detailed results
            results_path = f"./Results/{task}/"
            os.makedirs(results_path, exist_ok=True)
            with open(f"{results_path}all_strategies_{num_layers}layers.json", "w") as f:
                json.dump(all_results, f, indent=4)
        
        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")
    
    print("\n" + "="*100)
    print("ALL TASKS COMPLETED!")
    print("="*100)