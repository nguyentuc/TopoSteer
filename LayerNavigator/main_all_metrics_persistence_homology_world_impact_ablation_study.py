from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from get_hole_score import compare_ln_hole_scores
from strategy import UniStrategy
import json
import numpy as np
import os


def load_precomputed_hole_scores(task: str, metric: str, vec_method: str = "md") -> dict:
    """Load precomputed HOLE scores from JSON files"""
    score_path = f"./Score_HOLE-standard-{metric}/{task}/{task}+{vec_method}/all_layers.json"
    
    with open(score_path, 'r') as f:
        scores = json.load(f)
    
    scores = {int(k): v for k, v in scores.items()}
    
    print(f"Loaded precomputed HOLE scores for {metric}")
    print(f"   Found {len(scores)} layers")
    
    return scores


def load_all_precomputed_metrics(task: str, metrics: list, vec_method: str = "md") -> dict:
    """Load precomputed scores for all metrics"""
    all_scores = {}
    
    print("\n" + "="*80)
    print("LOADING PRECOMPUTED HOLE SCORES")
    print("="*80)
    
    for metric in metrics:
        try:
            scores = load_precomputed_hole_scores(task, metric, vec_method)
            all_scores[metric] = scores
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print(f"   Skipping {metric}")
            continue
    
    if not all_scores:
        raise FileNotFoundError(f"No precomputed HOLE scores found for task '{task}'!")
    
    print(f"\nSuccessfully loaded {len(all_scores)}/{len(metrics)} metrics")
    print("="*80 + "\n")
    
    return all_scores


def compute_component_scores(all_hole_scores: dict, layers: list) -> dict:
    """Compute alternative scoring strategies based on individual components"""
    component_scores = {
        'purity_only': {},
        'separability_only': {},
        'persistence_only': {},
        'beta1_inverse': {},
        'purity_sep': {},
        'purity_pers': {},
        'sep_pers': {},
        'tss_no_penalty': {},
        'tss_original': {},
        'tss_equal_weights': {},
        'tss_sep_heavy': {},
        'tss_purity_heavy': {},
    }
    
    for metric in all_hole_scores.keys():
        for strategy in component_scores.keys():
            component_scores[strategy][metric] = {}
        
        for layer in layers:
            if layer not in all_hole_scores[metric]:
                continue
            
            score = all_hole_scores[metric][layer]
            
            purity = score['purity']
            separability = score['separability']
            persistence = score['mean_persistence_H0']
            beta1 = score['beta1']
            entanglement_penalty = 1.0 / (1.0 + 0.1 * beta1)
            
            component_scores['purity_only'][metric][layer] = purity
            component_scores['separability_only'][metric][layer] = separability
            component_scores['persistence_only'][metric][layer] = persistence
            component_scores['beta1_inverse'][metric][layer] = entanglement_penalty
            component_scores['purity_sep'][metric][layer] = 0.5 * purity + 0.5 * separability
            component_scores['purity_pers'][metric][layer] = 0.5 * purity + 0.5 * persistence
            component_scores['sep_pers'][metric][layer] = 0.5 * separability + 0.5 * persistence
            
            base_score = 0.35 * purity + 0.35 * separability + 0.30 * persistence
            component_scores['tss_no_penalty'][metric][layer] = base_score
            component_scores['tss_original'][metric][layer] = score['tss']
            component_scores['tss_equal_weights'][metric][layer] = (0.333 * purity + 0.333 * separability + 0.333 * persistence) * entanglement_penalty
            component_scores['tss_sep_heavy'][metric][layer] = (0.2 * purity + 0.6 * separability + 0.2 * persistence) * entanglement_penalty
            component_scores['tss_purity_heavy'][metric][layer] = (0.6 * purity + 0.2 * separability + 0.2 * persistence) * entanglement_penalty
    
    return component_scores


if __name__ == "__main__":
    if "Llama" in MODEL:
        model = LlamaWrapper(MODEL)
    elif "Qwen" in MODEL:
        model = QwenWrapper(MODEL)
    else:
        raise NotImplementedError("Model Not Implemented")

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
    
    for task in ['desire-to-maximize-impact-on-world']:
        print("\n" + "="*100)
        print(f"{'='*40} Task: {task} {'='*40}")
        print("="*100 + "\n")

        print("STEP 1: Baseline Evaluation")
        test_dataset = UniDataset(task=task, train=False, set="test")
        base_prob = get_raw_BASE_results(model=model, test_dataset=test_dataset)
        print(f"Base Prob (no steering): {base_prob:.4f}\n")

        print("STEP 2: Checking for Steering Vectors")
        vec_root = f"./Vectors/{task}/md"
        
        if not os.path.exists(f"{vec_root}/acts.pt"):
            print("Steering vectors not found. Extracting steering vector...")
            train_dataset = UniDataset(task=task, train=True, set="train")
            uni_generate_vectors(method="md", model=model, layers=LAYERS, dataset=train_dataset)
            print("Steering vectors extracted\n")
        else:
            print("Steering vectors already exist\n")
            train_dataset = UniDataset(task=task, train=True, set="train")

        print("STEP 3A: Checking for LayerNavigator Scores")
        ln_score_path = f"./Score-standard/{task}/{task}+md"
        
        if not os.path.exists(f"{ln_score_path}/all_layers.json"):
            print("LayerNavigator scores not found. Computing now...")
            get_score(layers=LAYERS, dataset=train_dataset, vec_task=task, vec_method="md", acts_pre="standard")
            print("LayerNavigator scores computed\n")
        else:
            print("LayerNavigator scores already exist\n")

        print("STEP 3B: Loading Precomputed HOLE Scores")
        print("="*100)
        
        try:
            all_hole_scores = load_all_precomputed_metrics(task=task, metrics=ALL_HOLE_METRICS, vec_method="md")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("\nSkipping this task...")
            continue

        print("STEP 3C: Computing Component-Based Scores")
        print("="*100)
        
        component_scores = compute_component_scores(all_hole_scores, LAYERS)
        
        print("Component strategies available:")
        for strategy in component_scores.keys():
            print(f"  - {strategy}")
        print()

        print("STEP 3D: Comparing LayerNavigator with HOLE Metrics")
        print("="*100)
        
        all_comparisons = {}
        for metric in all_hole_scores.keys():
            hole_score_path = f"./Score_HOLE-standard-{metric}/{task}/{task}+md"
            comparison = compare_ln_hole_scores(layers=LAYERS, ln_score_path=ln_score_path, hole_score_path=hole_score_path)
            all_comparisons[metric] = comparison
            print(f"--- Metric: {metric} ---")
            print(f"  Spearman Correlation: {comparison['spearman_correlation']:.3f} (p={comparison['spearman_pvalue']:.4f})")
        
        comparison_path = f"./Comparison/{task}/"
        os.makedirs(comparison_path, exist_ok=True)
        with open(f"{comparison_path}ln_vs_all_hole_metrics.json", "w") as f:
            json.dump(all_comparisons, f, indent=4)
        print()

        print("\n>>> STEP 4: Testing Layer Selection Strategies (Component-Based)")
        print("="*100)
        
        for num_layers in [1, 3, 5]:
            print(f"\n{'='*40} {num_layers} Layer(s) {'='*40}")
            
            all_results = {
                'num_layers': num_layers,
                'base_prob': float(base_prob),
                'strategies': {}
            }
            
            print(f"\n[Strategy 1: LayerNavigator Only]")
            strategy_ln = UniStrategy(task=task, strategy="my", num_layers=num_layers, method="md")
            test_prob_ln = get_raw_results(model=model, layers=strategy_ln.layers, test_dataset=test_dataset, Alphas=[1.0] * num_layers, train_task=task, train_method="md")
            print(f"  Layers: {strategy_ln.layers}")
            print(f"  Test Prob: {test_prob_ln:.4f} (delta={test_prob_ln - base_prob:+.4f})")
            
            all_results['strategies']['layernav'] = {
                'layers': strategy_ln.layers,
                'prob': float(test_prob_ln),
                'delta': float(test_prob_ln - base_prob)
            }
            
            print(f"\n[Strategy 2: Component-Based Strategies]")
            print("Testing each component strategy with each metric...\n")
            
            for strategy_name in component_scores.keys():
                print(f"--- {strategy_name.upper()} ---")
                
                for metric in all_hole_scores.keys():
                    ranking = sorted(component_scores[strategy_name][metric].items(), key=lambda x: x[1], reverse=True)
                    top_layers = [layer for layer, _ in ranking[:num_layers]]
                    
                    test_prob = get_raw_results(model=model, layers=top_layers, test_dataset=test_dataset, Alphas=[1.0] * num_layers, train_task=task, train_method="md")
                    
                    strategy_key = f'{strategy_name}_{metric}'
                    
                    print(f"  {metric:25s}: Prob={test_prob:.4f} (delta={test_prob - base_prob:+.4f}) | Layers={top_layers}")
                    
                    all_results['strategies'][strategy_key] = {
                        'component': strategy_name,
                        'metric': metric,
                        'layers': top_layers,
                        'prob': float(test_prob),
                        'delta': float(test_prob - base_prob)
                    }
                
                print()
            
            print(f"\n[Strategy 3: Ensemble Within Each Component Strategy]")
            print("Averaging across all metrics for each component strategy...\n")
            
            for strategy_name in component_scores.keys():
                all_normalized_scores = []
                for metric in all_hole_scores.keys():
                    vals = np.array([component_scores[strategy_name][metric][l] for l in LAYERS])
                    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
                    all_normalized_scores.append(norm)
                
                ensemble_scores = {}
                for idx, layer in enumerate(LAYERS):
                    ensemble_scores[layer] = np.mean([scores[idx] for scores in all_normalized_scores])
                
                ranking = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
                top_layers = [layer for layer, _ in ranking[:num_layers]]
                
                test_prob = get_raw_results(model=model, layers=top_layers, test_dataset=test_dataset, Alphas=[1.0] * num_layers, train_task=task, train_method="md")
                
                strategy_key = f'ensemble_{strategy_name}'
                
                print(f"  {strategy_name:30s}: Prob={test_prob:.4f} (delta={test_prob - base_prob:+.4f}) | Layers={top_layers}")
                
                all_results['strategies'][strategy_key] = {
                    'component': strategy_name,
                    'metric': 'ensemble_all',
                    'layers': top_layers,
                    'prob': float(test_prob),
                    'delta': float(test_prob - base_prob),
                    'num_metrics': len(all_hole_scores)
                }
            
            print(f"\n{'='*40} SUMMARY {'='*40}")
            
            best_strategy = max(all_results['strategies'].items(), key=lambda x: x[1]['prob'])
            
            print(f"\nBEST STRATEGY: {best_strategy[0]}")
            print(f"   Probability: {best_strategy[1]['prob']:.4f}")
            print(f"   Improvement: {best_strategy[1]['delta']:+.4f}")
            print(f"   Layers: {best_strategy[1]['layers']}")
            
            print(f"\nTop 10 Strategies:")
            sorted_strategies = sorted(all_results['strategies'].items(), key=lambda x: x[1]['prob'], reverse=True)
            for rank, (name, result) in enumerate(sorted_strategies[:10], 1):
                print(f"  {rank:2d}. {name:40s}: {result['prob']:.4f} (Δ={result['delta']:+.4f})")
            
            print(f"\nBest Performer by Component:")
            component_best = {}
            for strategy_name in component_scores.keys():
                component_strategies = {k: v for k, v in all_results['strategies'].items() if k.startswith(strategy_name)}
                if component_strategies:
                    best = max(component_strategies.items(), key=lambda x: x[1]['prob'])
                    component_best[strategy_name] = best
                    print(f"  {strategy_name:30s}: {best[1]['prob']:.4f} ({best[0]})")
            
            all_results['best_strategy'] = best_strategy[0]
            all_results['best_prob'] = best_strategy[1]['prob']
            all_results['component_best'] = {k: v[0] for k, v in component_best.items()}
            
            results_path = f"./Results/{task}/"
            os.makedirs(results_path, exist_ok=True)
            with open(f"{results_path}component_strategies_{num_layers}layers.json", "w") as f:
                json.dump(all_results, f, indent=4)
        
        del test_dataset, train_dataset
        print("\n" + "="*100)
        print(f"Task {task} completed")
        print("="*100 + "\n")
    
    print("\n" + "="*100)
    print("ALL TASKS COMPLETED!")
    print("="*100)