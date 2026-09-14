from model_wrapper import LlamaWrapper, QwenWrapper
from dataset import UniDataset
from get_results import *
from get_vec import *
from globalenv import *
from get_score import *
from get_hole_score import get_hole_score, compare_ln_hole_scores
from strategy import UniStrategy
import json

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

    # Run for all the task that will be used to evaluate in the paper
    for task in Anth_MAIN:
        print(f"############## Task: {task} ################")

        # Get Base Results
        test_dataset = UniDataset(
            task=task,
            train=False,
            set="test",
        )
        base_prob = get_raw_BASE_results(
            model=model,
            test_dataset=test_dataset,
        )
        print(f"Base Prob: {base_prob}")

        # Get Vectors: Implements steering vector extraction using Mean Difference or PCA methods, and saves activations used during computation.
        train_dataset = UniDataset(
            task=task,
            train=True,
            set="train",
        )
        
        # Extract mean different steering vector: LAYERS in range(32)
        uni_generate_vectors(
            method="md",
            model=model,
            layers=LAYERS,
            dataset=train_dataset,
        )

        # ============================================================
        # STEP 3A: Get LayerNavigator Score (Original Method)
        # ============================================================
        # print("Computing LayerNavigator Scores")
        get_score(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
        )

        # ============================================================
        # STEP 3B: Get HOLE Topological Scores (NEW!)
        # ============================================================
        print("\n Computing HOLE Topological Scores...")
        hole_scores = get_hole_score(
            layers=LAYERS,
            dataset=train_dataset,
            vec_task=task,
            vec_method="md",
            acts_pre="standard",
            metric="mahalanobis",  # "cosine", "euclidean", "mahalanobis"
            max_dimension=2,  # Compute H0, H1, H2
            subsample=500,    # Subsample for efficiency (adjust based on dataset size)
        )

        # ============================================================
        # STEP 3C: Compare LayerNavigator vs HOLE Scores
        # ============================================================
        print("Comparing LayerNavigator and HOLE Rankings")
        ln_score_path = f"./Score-standard/{task}/{task}+md"
        hole_score_path = f"./Score_HOLE-standard-mahalanobis/{task}/{task}+md"
    
        comparison = compare_ln_hole_scores(
            layers=LAYERS,
            ln_score_path=ln_score_path,
            hole_score_path=hole_score_path,
        )
        
        print(f"  Spearman Correlation: {comparison['spearman_correlation']:.3f} (p={comparison['spearman_pvalue']:.4f})")
        print(f"  Top-10 Jaccard Similarity: {comparison['jaccard_top10']:.3f}")
        print(f"  LN Top-10:   {comparison['ln_top10']}")
        print(f"  HOLE Top-10: {comparison['hole_top10']}")
        print(f"  Agreement:   {comparison['agreement']}")
        print(f"  LN only:     {comparison['ln_only']}")
        print(f"  HOLE only:   {comparison['hole_only']}")
        
        # Save comparison
        comparison_path = f"./Comparison/{task}/"
        os.makedirs(comparison_path, exist_ok=True)
        with open(f"{comparison_path}ln_vs_hole.json", "w") as f:
            json.dump(comparison, f, indent=4)

        # ============================================================
        # STEP 4: Test Different Layer Selection Strategies
        # ============================================================
        print("Testing Layer Selection Strategies...")
        
        for num_layers in [1, 3, 5]:
            print(f"\n--- Testing with {num_layers} steering layer(s) ---")
            
            # Strategy 1: LayerNavigator Only (Original)
            print(f"\n[Strategy 1: LayerNavigator Only]")
            strategy_ln = UniStrategy(
                task=task,
                strategy="my",  # "my" uses steerability scores
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
            print(f"  LN Layers: {strategy_ln.layers}")
            print(f"  LN Test Prob: {test_prob_ln:.4f}")
            
            # Strategy 2: HOLE Only
            print(f"\n[Strategy 2: HOLE Topological Only]")
            # Rank layers by HOLE TSS score
            hole_ranking = sorted(
                hole_scores.items(),
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
            print(f"  HOLE Layers: {hole_top_layers}")
            print(f"  HOLE Test Prob: {test_prob_hole:.4f}")
            
            # Strategy 3: Hybrid (LN + HOLE weighted combination)
            print(f"\n[Strategy 3: Hybrid (LN + HOLE)]")
            # Load LN scores
            ln_scores = {}
            for layer in LAYERS:
                with open(f"{ln_score_path}/L{layer}.json", "r") as f:
                    ln_scores[layer] = json.load(f)['s_score']
            
            # Normalize both scores to [0, 1]
            ln_vals = np.array(list(ln_scores.values()))
            hole_vals = np.array([hole_scores[l]['tss'] for l in LAYERS])
            
            ln_norm = (ln_vals - ln_vals.min()) / (ln_vals.max() - ln_vals.min() + 1e-8)
            hole_norm = (hole_vals - hole_vals.min()) / (hole_vals.max() - hole_vals.min() + 1e-8)
            
            # Hybrid score: weighted combination
            alpha = 0.5  # Weight for LayerNavigator (0.5 LN + 0.5 HOLE)
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
            print(f"  Hybrid Layers (alpha={alpha}): {hybrid_top_layers}")
            print(f"  Hybrid Test Prob: {test_prob_hybrid:.4f}")
            
            # ============================================================
            # Compare Results
            # ============================================================
            print(f"[Comparison for {num_layers} layer(s)]")
            print(f"  Base (no steering):  {base_prob:.4f}")
            print(f"  LayerNavigator:      {test_prob_ln:.4f} (delta = {test_prob_ln - base_prob:+.4f})")
            print(f"  HOLE Topological:    {test_prob_hole:.4f} (delta = {test_prob_hole - base_prob:+.4f})")
            print(f"  Hybrid: {test_prob_hybrid:.4f} (delta = {test_prob_hybrid - base_prob:+.4f})")
            
            # Determine winner
            best_method = max(
                [("base", base_prob),("LN", test_prob_ln), ("HOLE", test_prob_hole), ("Hybrid", test_prob_hybrid)],
                key=lambda x: x[1]
            )
            print(f"Best: {best_method[0]} with prob={best_method[1]:.4f}")
            
            # Save detailed results
            results = {
                'num_layers': num_layers,
                'base_prob': float(base_prob),
                'layernav': {
                    'layers': strategy_ln.layers,
                    'prob': float(test_prob_ln),
                    'delta': float(test_prob_ln - base_prob),
                },
                'hole': {
                    'layers': hole_top_layers,
                    'prob': float(test_prob_hole),
                    'delta': float(test_prob_hole - base_prob),
                },
                'hybrid': {
                    'layers': hybrid_top_layers,
                    'prob': float(test_prob_hybrid),
                    'delta': float(test_prob_hybrid - base_prob),
                    'alpha': alpha,
                },
                'best_method': best_method[0],
            }
            
            results_path = f"./Results/{task}/"
            os.makedirs(results_path, exist_ok=True)
            with open(f"{results_path}strategies_{num_layers}layers.json", "w") as f:
                json.dump(results, f, indent=4)
        
        del test_dataset, train_dataset
        print("\n" + "="*80 + "\n")