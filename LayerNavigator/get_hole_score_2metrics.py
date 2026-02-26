# get_hole_score.py
"""
HOLE (Homological Observation of Latent Embeddings) Metric Computation
MINIMAL VERSION: 6 essential mean persistence metrics

Metrics computed:
  1. mean_persistence_H0_combined - Cluster lifetime (combined cloud)
  2. mean_persistence_H1_combined - Loop lifetime (combined cloud)
  3. mean_persistence_H1_pos - Loop lifetime (positive cloud)
  4. mean_persistence_H1_neg - Loop lifetime (negative cloud)
  5. mean_persistence_H0_diff - Cluster lifetime (difference cloud)
  6. mean_persistence_H1_diff - Loop lifetime (difference cloud)
"""

import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import os
import json
import gudhi
from sklearn.metrics import pairwise_distances


def compute_persistence_diagram(
    activations: np.ndarray,
    metric: str = 'cosine',
    max_dimension: int = 1
) -> Dict:
    """
    Compute persistence diagram using GUDHI
    MINIMAL: Only computes H0 and H1 mean persistence
    
    Args:
        activations: (N, hidden_dim)
        metric: Distance metric - either 'euclidean' or 'cosine'
        max_dimension: Maximum homology dimension (set to 1 for H0 and H1 only)
        
    Returns:
        persistence_data: Dict with H0 and H1 mean persistence only
    """
    # ============================================
    # STEP 1: COMPUTE DISTANCE MATRIX
    # ============================================
    
    if metric == 'euclidean':
        distances = pairwise_distances(activations, metric='euclidean')
    elif metric == 'cosine':
        distances = pairwise_distances(activations, metric='cosine')
    else:
        raise ValueError(f"Unsupported metric: {metric}. Only 'euclidean' and 'cosine' are supported.")
    
    # ============================================
    # STEP 2: CREATE RIPS COMPLEX AND COMPUTE PERSISTENCE
    # ============================================
    
    rips_complex = gudhi.RipsComplex(distance_matrix=distances, max_edge_length=np.inf)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    # Compute persistence
    persistence = simplex_tree.persistence()
    
    # Extract persistence pairs by dimension
    persistence_by_dim = {i: [] for i in range(max_dimension + 1)}
    for dim, (birth, death) in persistence:
        persistence_by_dim[dim].append((birth, death))
    
    # ============================================
    # STEP 3: COMPUTE MEAN PERSISTENCE FOR H0 AND H1
    # ============================================
    persistence_stats = {}
    
    for dim in range(max_dimension + 1):
        pairs = persistence_by_dim[dim]
        if len(pairs) > 0:
            # Filter out infinite persistence
            finite_pairs = [(b, d) for b, d in pairs if d != np.inf]
            
            if finite_pairs:
                persistences = [d - b for b, d in finite_pairs]
                persistence_stats[f'H{dim}_mean'] = np.mean(persistences)
            else:
                persistence_stats[f'H{dim}_mean'] = 0.0
        else:
            persistence_stats[f'H{dim}_mean'] = 0.0
    
    return {
        'persistence_stats': persistence_stats
    }


def get_hole_score(
    layers: List[int],
    dataset,
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    metric: str = "cosine",
    subsample: int = None
):
    """
    Compute 6 essential HOLE metrics for each layer:
      1. mean_persistence_H0_combined
      2. mean_persistence_H1_combined
      3. mean_persistence_H1_pos
      4. mean_persistence_H1_neg
      5. mean_persistence_H0_diff
      6. mean_persistence_H1_diff
    
    Args:
        layers: List of layer indices to analyze
        dataset: Dataset object (must have train=True)
        vec_task: Task name
        vec_method: Vector extraction method
        acts_pre: Preprocessing method ('standard' for z-score)
        metric: Distance metric - 'euclidean' or 'cosine'
        subsample: Subsample N points per class for efficiency
        
    Returns:
        hole_score_info: Dict with 6 metrics per layer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert dataset.train == True, "Only Use Train Dataset"
    
    vec_root = f"./Vectors/{vec_task}/{vec_method}"
    svec_path = vec_root[10:].replace("/", "+")
    
    # Score Save Path
    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    metric_str = f"-{metric}"
    
    save_root = f"./Score_HOLE{acts_pre_str}{metric_str}/{dataset.task}/{svec_path}/"
    os.makedirs(save_root, exist_ok=True)
    
    ans_num = 2  # Binary classification
    
    # Load activations
    print("Loading activations...")
    acts = torch.load(f"{vec_root}/acts.pt")
    
    hole_score_info = {}
    
    # Process each layer
    for l in tqdm(layers, desc=f"Computing HOLE Scores ({metric}, 6 metrics)"):
        # ========================================================
        # STEP 1: PREPARE ACTIVATIONS
        # ========================================================
        
        # Separate by label
        pos_acts_list = []  # Label 1
        neg_acts_list = []  # Label 0
        
        for i in range(ans_num):
            if i == 1:
                pos_acts_list.append(torch.stack(acts[i][l]))
            else:
                neg_acts_list.append(torch.stack(acts[i][l]))
        
        # Convert to numpy
        pos_acts = torch.cat(pos_acts_list, dim=0).float().cpu().numpy()
        neg_acts = torch.cat(neg_acts_list, dim=0).float().cpu().numpy()
        combined_acts = np.vstack([pos_acts, neg_acts])
        
        # Subsample if specified
        if subsample is not None:
            if len(pos_acts) > subsample:
                pos_indices = np.random.choice(len(pos_acts), subsample, replace=False)
                pos_acts = pos_acts[pos_indices]
            
            if len(neg_acts) > subsample:
                neg_indices = np.random.choice(len(neg_acts), subsample, replace=False)
                neg_acts = neg_acts[neg_indices]
            
            combined_acts = np.vstack([pos_acts, neg_acts])
        
        # ========================================================
        # STEP 2: CREATE DIFFERENCE CLOUD
        # Each positive activation minus mean of negative activations
        # Each negative activation minus mean of positive activations
        # This represents "deviation from opposite class center"
        # ========================================================
        
        pos_mean = pos_acts.mean(axis=0)
        neg_mean = neg_acts.mean(axis=0)
        
        # Difference cloud: deviations from opposite class
        diff_from_neg = pos_acts - neg_acts  # How each positive deviates from negative center
        # diff_from_pos = neg_acts - pos_mean  # How each negative deviates from positive center
        # diff_acts = np.vstack([diff_from_neg, diff_from_pos])
        diff_acts = diff_from_neg
        
        print(f"  Layer {l}: Created difference cloud with {len(diff_acts)} points")
        
        # ========================================================
        # STEP 3: NORMALIZE ACTIVATIONS
        # ========================================================
        
        if acts_pre == "standard":
            # Normalize combined (maintains between-class relationships)
            combined_mean = combined_acts.mean(axis=0)
            combined_std = combined_acts.std(axis=0) + 1e-8
            
            combined_acts_norm = (combined_acts - combined_mean) / combined_std
            pos_acts_norm = (pos_acts - combined_mean) / combined_std
            neg_acts_norm = (neg_acts - combined_mean) / combined_std
            
            # Normalize difference cloud separately (represents steering direction space)
            diff_mean = diff_acts.mean(axis=0)
            diff_std = diff_acts.std(axis=0) + 1e-8
            diff_acts_norm = (diff_acts - diff_mean) / diff_std
        else:
            combined_acts_norm = combined_acts
            pos_acts_norm = pos_acts
            neg_acts_norm = neg_acts
            diff_acts_norm = diff_acts
        
        # ========================================================
        # STEP 4: COMPUTE PERSISTENCE FOR FOUR CLOUDS
        # ========================================================
        
        print(f"  Layer {l}: Computing persistence ({metric})...")
        print(f"    - Combined: {len(combined_acts_norm)} points")
        print(f"    - Positive: {len(pos_acts_norm)} points")
        print(f"    - Negative: {len(neg_acts_norm)} points")
        print(f"    - Difference: {len(diff_acts_norm)} points")
        
        # Combined: need H0 and H1
        persistence_combined = compute_persistence_diagram(
            combined_acts_norm,
            metric=metric,
            max_dimension=1
        )
        
        # Positive: need H1 only (but we compute H0 too, it's fast)
        persistence_pos = compute_persistence_diagram(
            pos_acts_norm,
            metric=metric,
            max_dimension=1
        )
        
        # Negative: need H1 only (but we compute H0 too, it's fast)
        persistence_neg = compute_persistence_diagram(
            neg_acts_norm,
            metric=metric,
            max_dimension=1
        )
        
        # Difference: need H0 and H1 (steering direction topology)
        persistence_diff = compute_persistence_diagram(
            diff_acts_norm,
            metric=metric,
            max_dimension=1
        )
        
        # ========================================================
        # STEP 5: EXTRACT THE 6 REQUIRED METRICS
        # ========================================================
        
        layer_metrics = {
            # From combined cloud
            'mean_persistence_H0_combined': float(persistence_combined['persistence_stats'].get('H0_mean', 0.0)),
            'mean_persistence_H1_combined': float(persistence_combined['persistence_stats'].get('H1_mean', 0.0)),
            
            # From positive cloud
            'mean_persistence_H1_pos': float(persistence_pos['persistence_stats'].get('H1_mean', 0.0)),
            
            # From negative cloud
            'mean_persistence_H1_neg': float(persistence_neg['persistence_stats'].get('H1_mean', 0.0)),
            
            # From difference cloud (steering direction space)
            'mean_persistence_H0_diff': float(persistence_diff['persistence_stats'].get('H0_mean', 0.0)),
            'mean_persistence_H1_diff': float(persistence_diff['persistence_stats'].get('H1_mean', 0.0)),
            
            # Metadata
            'metric': metric,
            'n_samples_combined': int(len(combined_acts_norm)),
            'n_samples_pos': int(len(pos_acts_norm)),
            'n_samples_neg': int(len(neg_acts_norm)),
            'n_samples_diff': int(len(diff_acts_norm))
        }
        
        hole_score_info[l] = layer_metrics
        
        # Save per-layer results
        with open(f"{save_root}L{l}.json", "w") as f:
            json.dump(hole_score_info[l], f, indent=4)
    
    # Save complete results
    with open(f"{save_root}all_layers.json", "w") as f:
        json.dump(hole_score_info, f, indent=4)
    
    print(f"\nHOLE scores saved to: {save_root}")
    print(f"Metrics per layer: 6 essential mean persistence values")
    
    return hole_score_info


def get_hole_score_all_metrics(
    layers: List[int],
    dataset,
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    subsample: int = None
):
    """
    Compute 6 essential metrics using both Euclidean and Cosine
    
    Total: 6 metrics × 2 distance metrics = 12 values per layer
    
    Args:
        layers: List of layer indices
        dataset: Dataset object (train=True)
        vec_task: Task name
        vec_method: Vector extraction method
        acts_pre: Preprocessing method
        subsample: Subsample size per class
        
    Returns:
        all_metric_scores: {
            'euclidean': {layer: {6 metrics}},
            'cosine': {layer: {6 metrics}}
        }
    """
    all_metrics = ['euclidean', 'cosine']
    
    all_metric_scores = {}
    
    print("\n" + "="*80)
    print("COMPUTING 6 ESSENTIAL HOLE METRICS: EUCLIDEAN AND COSINE")
    print("="*80)
    
    for metric in all_metrics:
        print(f"\n Metric: {metric}")
        print("-" * 80)
        
        scores = get_hole_score(
            layers=layers,
            dataset=dataset,
            vec_task=vec_task,
            vec_method=vec_method,
            acts_pre=acts_pre,
            metric=metric,
            subsample=subsample
        )
        
        all_metric_scores[metric] = scores
    
    # Save comprehensive results
    vec_root = f"./Vectors/{vec_task}/{vec_method}"
    svec_path = vec_root[10:].replace("/", "+")
    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    
    save_path = f"./Score_HOLE{acts_pre_str}_ALL_METRICS/{dataset.task}/{svec_path}/"
    os.makedirs(save_path, exist_ok=True)
    
    with open(f"{save_path}all_metrics_all_layers.json", "w") as f:
        json.dump(all_metric_scores, f, indent=4)
    
    print("\n" + "="*80)
    print(f"All metrics saved to: {save_path}")
    print(f"Total: 6 metrics × 2 distance metrics = 12 values per layer")
    print("="*80)
    
    return all_metric_scores


def compare_ln_hole_scores(
    layers: List[int],
    ln_score_path: str,
    hole_score_path: str
):
    """Compare LayerNavigator and HOLE scores"""
    from scipy.stats import spearmanr, kendalltau
    
    ln_scores = []
    hole_scores = []
    
    for l in layers:
        with open(f"{ln_score_path}/L{l}.json", "r") as f:
            ln_data = json.load(f)
            ln_scores.append(ln_data['s_score'])
        
        with open(f"{hole_score_path}/L{l}.json", "r") as f:
            hole_data = json.load(f)
            # Use H1_combined as primary metric
            hole_scores.append(hole_data.get('mean_persistence_H1_combined', 0.0))
    
    spearman_corr, spearman_p = spearmanr(ln_scores, hole_scores)
    kendall_corr, kendall_p = kendalltau(ln_scores, hole_scores)
    
    ln_ranking = np.argsort(ln_scores)[::-1]
    hole_ranking = np.argsort(hole_scores)[::-1]
    
    top_k = min(10, len(layers))
    ln_top_k = set([layers[i] for i in ln_ranking[:top_k]])
    hole_top_k = set([layers[i] for i in hole_ranking[:top_k]])
    jaccard = len(ln_top_k & hole_top_k) / len(ln_top_k | hole_top_k)
    
    comparison = {
        'spearman_correlation': float(spearman_corr),
        'spearman_pvalue': float(spearman_p),
        'kendall_correlation': float(kendall_corr),
        'kendall_pvalue': float(kendall_p),
        'jaccard_top10': float(jaccard),
        'ln_top10': sorted(list(ln_top_k)),
        'hole_top10': sorted(list(hole_top_k)),
        'agreement': sorted(list(ln_top_k & hole_top_k)),
        'ln_only': sorted(list(ln_top_k - hole_top_k)),
        'hole_only': sorted(list(hole_top_k - ln_top_k))
    }
    
    return comparison


def compare_metrics_for_layer(
    layer_scores: Dict,
    layer_idx: int
):
    """Compare the 6 essential metrics between euclidean and cosine"""
    comparison = {
        'layer': layer_idx,
        'metrics': {}
    }
    
    for metric_name, scores in layer_scores.items():
        if layer_idx in scores:
            comparison['metrics'][metric_name] = {
                'mean_persistence_H0_combined': scores[layer_idx].get('mean_persistence_H0_combined', 0.0),
                'mean_persistence_H1_combined': scores[layer_idx].get('mean_persistence_H1_combined', 0.0),
                'mean_persistence_H1_pos': scores[layer_idx].get('mean_persistence_H1_pos', 0.0),
                'mean_persistence_H1_neg': scores[layer_idx].get('mean_persistence_H1_neg', 0.0),
                'mean_persistence_H0_diff': scores[layer_idx].get('mean_persistence_H0_diff', 0.0),
                'mean_persistence_H1_diff': scores[layer_idx].get('mean_persistence_H1_diff', 0.0),
            }
    
    print(f"\nLayer {layer_idx} - Essential Mean Persistence (Euclidean vs Cosine):")
    print("-" * 90)
    print(f"{'Metric':<12} {'H0_comb':>11} {'H1_comb':>11} {'H1_pos':>11} {'H1_neg':>11} {'H0_diff':>11} {'H1_diff':>11}")
    print("-" * 90)
    
    for metric, scores in comparison['metrics'].items():
        print(f"{metric:<12} "
              f"{scores['mean_persistence_H0_combined']:>11.6f} "
              f"{scores['mean_persistence_H1_combined']:>11.6f} "
              f"{scores['mean_persistence_H1_pos']:>11.6f} "
              f"{scores['mean_persistence_H1_neg']:>11.6f} "
              f"{scores['mean_persistence_H0_diff']:>11.6f} "
              f"{scores['mean_persistence_H1_diff']:>11.6f}")
    
    return comparison