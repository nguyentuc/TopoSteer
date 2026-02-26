"""
Multi-Scale LayerNavigator Score Computation
==============================================

Computes LayerNavigator's Discriminability (D) and Consistency (C) scores 
at every scale ε from the Vietoris-Rips filtration using persistent homology.

This generalizes LayerNavigator from single-scale to multi-scale:
- Original LayerNavigator: D and C at one fixed scale (Z-score normalized)
- This implementation: D(ε) and C(ε) at every scale ε

Key outputs:
- D(ε), C(ε), S(ε) curves across all scales
- Persistence length p_0 (robustness measure)
- Comparison with original LayerNavigator
- Layer rankings by robustness

Author: Based on LayerNavigator + HOLE frameworks
Date: 2025
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import os
import json
import gudhi
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr, kendalltau


# ============================================================================
# PART 1: CLUSTERING AT SCALE ε
# ============================================================================

def get_clusters_at_scale(distance_matrix: np.ndarray, epsilon: float) -> List[List[int]]:
    """
    Perform single-linkage hierarchical clustering at scale ε
    
    Points i and j are in the same cluster iff d(i,j) ≤ 2ε
    This corresponds to connected components in the Vietoris-Rips complex
    
    Args:
        distance_matrix: (N, N) pairwise distance matrix
        epsilon: Scale parameter ε
        
    Returns:
        clusters: List of clusters, where each cluster is a list of point indices
                  Example: [[0, 1, 5], [2, 3, 4, 6]] means 2 clusters
    """
    N = distance_matrix.shape[0]
    
    # Convert full distance matrix to condensed form for scipy
    condensed_distances = distance_matrix[np.triu_indices(N, k=1)]
    
    # Single-linkage hierarchical clustering
    Z = linkage(condensed_distances, method='single')
    
    # Cut dendrogram at height 2ε (diameter threshold for Vietoris-Rips)
    labels = fcluster(Z, t=2*epsilon, criterion='distance')
    
    # Group point indices by cluster label
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    return list(clusters.values())


# ============================================================================
# PART 2: COVARIANCE COMPUTATION AT SCALE ε
# ============================================================================

def compute_covariances_for_clustering(
    activations: np.ndarray,
    clusters: List[List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute between-cluster (S_b) and within-cluster (S_w) covariance matrices
    for a given clustering
    
    This generalizes LayerNavigator's covariance computation from fixed 
    binary clustering to arbitrary clustering at scale ε
    
    Args:
        activations: (N, d) activation matrix
        clusters: List of clusters from get_clusters_at_scale()
        
    Returns:
        S_b: (d, d) between-cluster covariance matrix
        S_w: (d, d) within-cluster covariance matrix
        
    Mathematical formulas:
        S_b = Σ_k n_k (μ_k - μ)(μ_k - μ)^T
        S_w = Σ_k Σ_{i ∈ C_k} (x_i - μ_k)(x_i - μ_k)^T
    """
    N, d = activations.shape
    
    # Global mean
    mu_global = activations.mean(axis=0)  # (d,)
    
    # Initialize covariance matrices
    S_b = np.zeros((d, d))
    S_w = np.zeros((d, d))
    
    for cluster_indices in clusters:
        # Get activations for this cluster
        cluster_acts = activations[cluster_indices]  # (n_cluster, d)
        n_cluster = len(cluster_indices)
        
        # Cluster mean
        mu_cluster = cluster_acts.mean(axis=0)  # (d,)
        
        # Between-cluster covariance: n_k * (μ_k - μ)(μ_k - μ)^T
        diff_cluster = mu_cluster - mu_global  # (d,)
        S_b += n_cluster * np.outer(diff_cluster, diff_cluster)
        
        # Within-cluster covariance: Σ_i (x_i - μ_k)(x_i - μ_k)^T
        for act in cluster_acts:
            diff_point = act - mu_cluster  # (d,)
            S_w += np.outer(diff_point, diff_point)
    
    return S_b, S_w


# ============================================================================
# PART 3: STEERING VECTOR COMPUTATION AT SCALE ε
# ============================================================================

def compute_steering_vector_at_scale(
    activations: np.ndarray,
    clusters: List[List[int]],
    labels: np.ndarray,
    method: str = 'mean_diff'
) -> np.ndarray:
    """
    Compute steering vector at scale ε based on clustering
    
    Args:
        activations: (N, d) activation matrix
        clusters: Clustering at this scale
        labels: (N,) original binary labels (0 or 1)
        method: 'mean_diff' or 'pca'
            - 'mean_diff': If exactly 2 clusters, use mean difference (like LayerNavigator)
            - 'pca': Use first principal component of S_b + S_w
            
    Returns:
        v: (d,) steering vector (unit normalized)
    """
    d = activations.shape[1]
    
    # Compute covariances at this scale
    S_b, S_w = compute_covariances_for_clustering(activations, clusters)
    
    if method == 'mean_diff' and len(clusters) == 2:
        # Binary clustering: use mean difference between two clusters
        # This matches LayerNavigator's approach when there are exactly 2 clusters
        cluster0_indices = clusters[0]
        cluster1_indices = clusters[1]
        
        mu0 = activations[cluster0_indices].mean(axis=0)
        mu1 = activations[cluster1_indices].mean(axis=0)
        
        v = mu0 - mu1  # Direction separating the two clusters
        
    else:
        # Multi-cluster or PCA method: use first eigenvector of S_total
        S_total = S_b + S_w
        
        # Add small regularization for numerical stability
        S_total += 1e-8 * np.eye(d)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(S_total)
        
        # Get first principal component (largest eigenvalue)
        v = eigenvectors[:, -1]  # Last column corresponds to largest eigenvalue
    
    # Normalize to unit vector
    v = v / (np.linalg.norm(v) + 1e-8)
    
    return v


# ============================================================================
# PART 4: DISCRIMINABILITY D(ε) AT SCALE ε
# ============================================================================

def compute_D_score_at_scale(
    S_b: np.ndarray,
    S_w: np.ndarray,
    v: np.ndarray
) -> float:
    """
    Compute discriminability D(ε) at scale ε
    
    This is EXACTLY the same formula as LayerNavigator's D-score,
    but computed for the clustering at scale ε
    
    Args:
        S_b: (d, d) between-cluster covariance at scale ε
        S_w: (d, d) within-cluster covariance at scale ε
        v: (d,) steering vector at scale ε
        
    Returns:
        D_score: Discriminability score ∈ [0, 1]
        
    Formula:
        D(ε) = v^T S_b(ε) v / (v^T (S_b(ε) + S_w(ε)) v)
        
    Interpretation:
        - D(ε) ≈ 1: Clusters well-separated along v
        - D(ε) ≈ 0: Clusters not separable
    """
    v = v.reshape(-1, 1)  # (d, 1) column vector
    
    # Numerator: between-cluster variance along v
    numerator = (v.T @ S_b @ v)[0, 0]
    
    # Denominator: total variance along v
    denominator = (v.T @ (S_b + S_w) @ v)[0, 0]
    
    if denominator < 1e-10:
        return 0.0
    
    D_score = numerator / denominator
    
    return float(D_score)


# ============================================================================
# PART 5: CONSISTENCY C(ε) AT SCALE ε
# ============================================================================

def compute_C_score_at_scale(
    activations: np.ndarray,
    clusters: List[List[int]],
    v: np.ndarray
) -> float:
    """
    Compute consistency C(ε) at scale ε
    
    This generalizes LayerNavigator's consistency score to arbitrary clustering
    
    Args:
        activations: (N, d) activation matrix
        clusters: Clustering at scale ε
        v: (d,) global steering vector (can also use cluster-specific directions)
        
    Returns:
        C_score: Consistency score (average alignment)
        
    Formula:
        C(ε) = (1/N) Σ_k Σ_{i ∈ C_k} cos_similarity(x_i, μ_k)
        
    where μ_k is the mean direction of cluster k
    
    Interpretation:
        - C(ε) ≈ 1: Points strongly aligned with their cluster means
        - C(ε) ≈ 0: Points not aligned
    """
    N = activations.shape[0]
    total_consistency = 0.0
    
    for cluster_indices in clusters:
        cluster_acts = activations[cluster_indices]
        
        # Cluster mean direction
        v_cluster = cluster_acts.mean(axis=0)
        v_cluster_norm = np.linalg.norm(v_cluster)
        
        if v_cluster_norm < 1e-8:
            continue
        
        # For each point in cluster, compute alignment with cluster mean
        for act in cluster_acts:
            act_norm = np.linalg.norm(act)
            
            if act_norm < 1e-8:
                continue
            
            # Cosine similarity: (x · μ) / (||x|| ||μ||)
            similarity = np.dot(act, v_cluster) / (act_norm * v_cluster_norm)
            total_consistency += similarity
    
    C_score = total_consistency / N if N > 0 else 0.0
    
    return float(C_score)


# ============================================================================
# PART 6: MAIN FUNCTION - MULTI-SCALE SCORE COMPUTATION
# ============================================================================

def get_multiscale_layernavigator_score(
    layers: List[int],
    dataset,  # UniDataset object
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    metric: str = "euclidean",
    num_scales: int = 50,
    subsample: Optional[int] = None,
    save_results: bool = True
) -> Dict:
    """
    Compute multi-scale LayerNavigator scores (D(ε), C(ε)) for each layer
    
    This is the MAIN FUNCTION that orchestrates the entire pipeline:
    1. Load activations
    2. Normalize (Z-score)
    3. Compute distance matrix
    4. Build Vietoris-Rips complex with GUDHI
    5. Extract critical scales from persistent homology
    6. For each scale ε:
       - Cluster points at scale ε
       - Compute S_b(ε), S_w(ε)
       - Compute steering vector v(ε)
       - Compute D(ε), C(ε), S(ε)
    7. Extract persistence interval [ε_b, ε_merge]
    8. Identify LayerNavigator's scale ε*
    9. Save results
    
    Args:
        layers: List of layer indices to analyze
        dataset: Dataset object (must have train=True)
        vec_task: Task name (e.g., 'sycophancy')
        vec_method: Vector extraction method (e.g., 'pca_diff')
        acts_pre: Preprocessing method ('standard' for z-score normalization)
        metric: Distance metric - one of:
            - 'euclidean': L2 distance
            - 'cosine': Cosine distance (recommended for LLMs)
            - 'mahalanobis': Covariance-aware distance
        num_scales: Number of scales to sample (for efficiency)
        subsample: Optional subsample size (None = use all data)
        save_results: Whether to save results to disk
        
    Returns:
        multiscale_score_info: {
            layer: {
                'scales': [ε_0, ε_1, ..., ε_M],
                'D_scores': [D(ε_0), D(ε_1), ..., D(ε_M)],
                'C_scores': [C(ε_0), C(ε_1), ..., C(ε_M)],
                'S_scores': [S(ε_0), S(ε_1), ..., S(ε_M)],
                'num_clusters': [|C(ε_0)|, |C(ε_1)|, ..., |C(ε_M)|],
                'persistence_interval': (ε_b, ε_merge),
                'persistence_length': p_0,
                'layernavigator_scale': ε*,
                'layernavigator_D': D(ε*),
                'layernavigator_C': C(ε*),
                'layernavigator_S': S(ε*),
                'max_S': max_ε S(ε),
                'optimal_scale': ε_opt where S(ε) is maximized,
                'metric': distance metric used,
                'n_samples': number of samples,
                'n_features': feature dimension
            }
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert dataset.train == True, "Only Use Train Dataset"
    
    vec_root = f"./Vectors/{vec_task}/{vec_method}"
    svec_path = vec_root[10:].replace("/", "+")
    
    # Setup save path
    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    metric_str = f"-{metric}"
    
    save_root = f"./Score_MultiScale{acts_pre_str}{metric_str}/{dataset.task}/{svec_path}/"
    if save_results:
        os.makedirs(save_root, exist_ok=True)
    
    ans_num = 2  # Binary classification
    
    # Load activations
    print("Loading activations...")
    acts = torch.load(f"{vec_root}/acts.pt")
    
    multiscale_score_info = {}
    
    # Process each layer
    for l in tqdm(layers, desc="Computing Multi-Scale LayerNavigator Scores"):
        # ====================================================================
        # STEP 1: PREPARE ACTIVATIONS
        # ====================================================================
        all_acts = []
        all_labels = []
        
        for i in range(ans_num):
            all_acts.append(torch.stack(acts[i][l]))
            all_labels.append(torch.ones(all_acts[i].shape[0]) * i)
        
        all_acts = torch.cat(all_acts, dim=0).float().cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        
        # Subsample if specified
        if subsample is not None and len(all_acts) > subsample:
            indices = np.random.choice(len(all_acts), subsample, replace=False)
            all_acts = all_acts[indices]
            all_labels = all_labels[indices]
        
        # Z-score normalization (same as LayerNavigator)
        if acts_pre == "standard":
            all_acts = (all_acts - all_acts.mean(axis=0)) / (all_acts.std(axis=0) + 1e-8)
        
        N, d = all_acts.shape
        
        print(f"\n  Layer {l}: N={N} samples, d={d} features")
        
        # ====================================================================
        # STEP 2: COMPUTE DISTANCE MATRIX
        # ====================================================================
        print(f"    Computing {metric} distance matrix...")
        
        if metric == 'euclidean':
            distances = pairwise_distances(all_acts, metric='euclidean')
        elif metric == 'cosine':
            distances = pairwise_distances(all_acts, metric='cosine')
        elif metric == 'mahalanobis':
            cov = np.cov(all_acts.T)
            inv_cov = np.linalg.pinv(cov)
            distances = pairwise_distances(all_acts, metric='mahalanobis', VI=inv_cov)
        else:
            # Default to euclidean
            distances = pairwise_distances(all_acts, metric='euclidean')
        
        # ====================================================================
        # STEP 3: BUILD VIETORIS-RIPS COMPLEX
        # ====================================================================
        print(f"    Building Vietoris-Rips complex...")
        
        rips_complex = gudhi.RipsComplex(distance_matrix=distances, max_edge_length=np.inf)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
        
        # Get all filtration values (critical scales where topology changes)
        filtration_list = list(simplex_tree.get_filtration())
        all_filtration_values = sorted(set([fval for _, fval in filtration_list]))
        
        print(f"    Total critical scales: {len(all_filtration_values)}")
        
        # ====================================================================
        # STEP 4: SAMPLE SCALES FOR EFFICIENCY
        # ====================================================================
        if len(all_filtration_values) > num_scales:
            min_scale = 0.0
            max_scale = all_filtration_values[-1]
            
            sampled_scales = np.linspace(min_scale, max_scale, num_scales)
            
            # Remove exact zeros if any
            sampled_scales = sampled_scales[sampled_scales >= 1e-10]
            
            print(f"    Uniform sampling: {len(sampled_scales)} scales from {sampled_scales[0]:.6f} to {sampled_scales[-1]:.6f}")
        else:
            sampled_scales = np.array(all_filtration_values)
        
        # ====================================================================
        # STEP 5: COMPUTE PERSISTENT HOMOLOGY
        # ====================================================================
        print(f"    Computing persistence diagram...")
        
        persistence = simplex_tree.persistence()
        
        # Extract β₀ features (connected components)
        beta_0_features = [(b, d) for dim, (b, d) in persistence if dim == 0 and d != np.inf]
        
        # Find the longest-lived β₀ feature (class separation)
        if beta_0_features:
            eps_b, eps_merge = max(beta_0_features, key=lambda x: x[1] - x[0])
            p_0 = eps_merge - eps_b
        else:
            eps_b, eps_merge, p_0 = 0.0, 0.0, 0.0
        
        print(f"    Persistence interval: [{eps_b:.6f}, {eps_merge:.6f}], p_0 = {p_0:.6f}")
        
        # ====================================================================
        # STEP 6: COMPUTE D(ε), C(ε), S(ε) AT EACH SCALE
        # ====================================================================
        print(f"    Computing D(ε), C(ε), S(ε) at {len(sampled_scales)} scales...")
        
        D_scores = []
        C_scores = []
        S_scores = []
        num_clusters_list = []
        
        for epsilon in tqdm(sampled_scales, desc=f"      Layer {l}", leave=False):
            # Get clustering at scale ε
            clusters = get_clusters_at_scale(distances, epsilon)
            num_clusters = len(clusters)
            
            # Compute covariances at this scale
            S_b, S_w = compute_covariances_for_clustering(all_acts, clusters)
            
            # Compute steering vector at this scale
            v = compute_steering_vector_at_scale(all_acts, clusters, all_labels, method='mean_diff')
            
            # Compute D(ε) - discriminability at scale ε
            D_eps = compute_D_score_at_scale(S_b, S_w, v)
            
            # Compute C(ε) - consistency at scale ε
            C_eps = compute_C_score_at_scale(all_acts, clusters, v)
            
            # Compute S(ε) - steerability at scale ε
            S_eps = D_eps + C_eps
            
            # Store results
            D_scores.append(D_eps)
            C_scores.append(C_eps)
            S_scores.append(S_eps)
            num_clusters_list.append(num_clusters)
        
        # ====================================================================
        # STEP 6.5: COMPUTE INTEGRATED/TOTAL S SCORE ACROSS ALL SCALES
        # ====================================================================
        # Sum of S(ε) across all scales (discrete approximation of integral)
        total_S = sum(S_scores)
        
        # Average S(ε) across all scales
        avg_S = np.mean(S_scores) if S_scores else 0.0
        
        # Weighted integral using trapezoidal rule (accounts for scale spacing)
        # ∫ S(ε) dε ≈ Σ (S(εᵢ) + S(εᵢ₊₁))/2 × (εᵢ₊₁ - εᵢ)
        if len(sampled_scales) > 1:
            scale_diffs = np.diff(sampled_scales)  # [ε₁-ε₀, ε₂-ε₁, ...]
            scale_avgs = [(S_scores[i] + S_scores[i+1])/2 for i in range(len(S_scores)-1)]
            integrated_S = sum(scale_avg * scale_diff for scale_avg, scale_diff in zip(scale_avgs, scale_diffs))
        else:
            integrated_S = S_scores[0] if S_scores else 0.0
        
        # Log-weighted sum (emphasizes behavior at different scales)
        # Useful because scales are sampled in log-space
        if len(sampled_scales) > 1:
            log_scale_diffs = np.diff(np.log10(sampled_scales + 1e-10))
            log_weighted_S = sum(s * ld for s, ld in zip(S_scores[:-1], log_scale_diffs))
        else:
            log_weighted_S = S_scores[0] if S_scores else 0.0
        
        # ====================================================================
        # STEP 7: IDENTIFY LAYERNAVIGATOR'S SCALE ε*
        # ====================================================================
        # LayerNavigator operates at the scale where 2 clusters first form
        layernavigator_scale_idx = None
        for idx, nc in enumerate(num_clusters_list):
            if nc == 2:
                layernavigator_scale_idx = idx
                break
        
        if layernavigator_scale_idx is not None:
            eps_star = sampled_scales[layernavigator_scale_idx]
            D_ln = D_scores[layernavigator_scale_idx]
            C_ln = C_scores[layernavigator_scale_idx]
        else:
            # Fallback: use scale closest to eps_b
            layernavigator_scale_idx = np.argmin(np.abs(sampled_scales - eps_b))
            eps_star = sampled_scales[layernavigator_scale_idx]
            D_ln = D_scores[layernavigator_scale_idx]
            C_ln = C_scores[layernavigator_scale_idx]
        
        # ====================================================================
        # STEP 8: FIND OPTIMAL SCALE
        # ====================================================================
        max_S_idx = np.argmax(S_scores)
        max_S = S_scores[max_S_idx]
        optimal_scale = sampled_scales[max_S_idx]
        
        # ====================================================================
        # STEP 9: STORE RESULTS
        # ====================================================================
        multiscale_score_info[l] = {
            # Multi-scale curves
            'scales': sampled_scales.tolist(),
            'D_scores': D_scores,
            'C_scores': C_scores,
            'S_scores': S_scores,
            'num_clusters': num_clusters_list,
            
            # Persistence information
            'persistence_interval': (float(eps_b), float(eps_merge)),
            'persistence_length': float(p_0),
            
            # LayerNavigator comparison
            'layernavigator_scale': float(eps_star),
            'layernavigator_scale_idx': int(layernavigator_scale_idx),
            'layernavigator_D': float(D_ln),
            'layernavigator_C': float(C_ln),
            'layernavigator_S': float(D_ln + C_ln),
            
            # Optimal scale
            'max_S': float(max_S),
            'optimal_scale': float(optimal_scale),
            'optimal_scale_idx': int(max_S_idx),
            
            # NEW: Aggregate scores across all scales
            'total_S': float(total_S),                    # Sum of all S(ε)
            'avg_S': float(avg_S),                        # Average S(ε)
            'integrated_S': float(integrated_S),          # Trapezoidal integral ∫S(ε)dε
            'log_weighted_S': float(log_weighted_S),      # Log-space weighted sum
            
            # Metadata
            'metric': metric,
            'n_samples': int(N),
            'n_features': int(d)
        }
        
        # Save per-layer results
        if save_results:
            with open(f"{save_root}L{l}.json", "w") as f:
                json.dump(multiscale_score_info[l], f, indent=4)
        
        # Print summary
        print(f"    LayerNavigator: ε* = {eps_star:.6f}, S(ε*) = {D_ln + C_ln:.6f}")
        print(f"    Optimal:        ε  = {optimal_scale:.6f}, S(ε)  = {max_S:.6f}")
        print(f"    Persistence:    p_0 = {p_0:.6f}")
    
    # Save complete results
    if save_results:
        with open(f"{save_root}all_layers.json", "w") as f:
            json.dump(multiscale_score_info, f, indent=4)
        
        print(f"\nMulti-scale scores saved to: {save_root}")
    
    return multiscale_score_info


# ============================================================================
# PART 7: COMPARISON WITH ORIGINAL LAYERNAVIGATOR
# ============================================================================

def compare_with_original_layernavigator(
    layers: List[int],
    ln_score_path: str,
    multiscale_score_path: str
) -> Dict:
    """
    Compare original LayerNavigator scores with multi-scale analysis
    
    Args:
        layers: List of layer indices
        ln_score_path: Path to original LayerNavigator scores (e.g., "./Score-standard/task/method/")
        multiscale_score_path: Path to multi-scale scores
        
    Returns:
        comparison: {
            'layers': [...],
            'ln_scores': [...],
            'multiscale_ln_scores': [...],  # S(ε*)
            'max_multiscale_scores': [...],  # max_ε S(ε)
            'persistence_lengths': [...],
            'in_robust_interval': [...],
            'correlations': {...}
        }
    """
    comparison = {
        'layers': [],
        'ln_scores': [],
        'multiscale_ln_scores': [],
        'max_multiscale_scores': [],
        'persistence_lengths': [],
        'in_robust_interval': [],
        'ln_vs_multiscale_diff': []
    }
    
    for l in layers:
        # Load LayerNavigator score
        with open(f"{ln_score_path}/L{l}.json", "r") as f:
            ln_data = json.load(f)
            ln_s_score = ln_data['s_score']
        
        # Load multi-scale score
        with open(f"{multiscale_score_path}/L{l}.json", "r") as f:
            ms_data = json.load(f)
            ms_ln_score = ms_data['layernavigator_S']
            max_s = ms_data['max_S']
            p_0 = ms_data['persistence_length']
            eps_star = ms_data['layernavigator_scale']
            eps_b, eps_merge = ms_data['persistence_interval']
        
        # Check if LayerNavigator's scale is in robust interval
        in_robust = (eps_b <= eps_star <= eps_merge) if p_0 > 0 else False
        
        comparison['layers'].append(l)
        comparison['ln_scores'].append(ln_s_score)
        comparison['multiscale_ln_scores'].append(ms_ln_score)
        comparison['max_multiscale_scores'].append(max_s)
        comparison['persistence_lengths'].append(p_0)
        comparison['in_robust_interval'].append(in_robust)
        comparison['ln_vs_multiscale_diff'].append(abs(ln_s_score - ms_ln_score))
    
    # Compute correlations
    if len(comparison['ln_scores']) > 2:
        spearman_corr, spearman_p = spearmanr(comparison['ln_scores'], comparison['multiscale_ln_scores'])
        kendall_corr, kendall_p = kendalltau(comparison['ln_scores'], comparison['multiscale_ln_scores'])
        
        comparison['correlations'] = {
            'spearman': float(spearman_corr),
            'spearman_pvalue': float(spearman_p),
            'kendall': float(kendall_corr),
            'kendall_pvalue': float(kendall_p)
        }
    
    # Print summary
    print("\n" + "="*80)
    print("LAYERNAVIGATOR VS MULTI-SCALE COMPARISON")
    print("="*80)
    print(f"{'Layer':<8} {'LN S':<10} {'MS S(ε*)':<10} {'Max S(ε)':<10} {'p_0':<10} {'Robust?':<10}")
    print("-"*80)
    
    for i, l in enumerate(layers):
        print(f"{l:<8} {comparison['ln_scores'][i]:<10.4f} "
              f"{comparison['multiscale_ln_scores'][i]:<10.4f} "
              f"{comparison['max_multiscale_scores'][i]:<10.4f} "
              f"{comparison['persistence_lengths'][i]:<10.4f} "
              f"{'✓' if comparison['in_robust_interval'][i] else '✗':<10}")
    
    return comparison


# ============================================================================
# PART 8: LAYER RANKING BY DIFFERENT CRITERIA
# ============================================================================

def rank_layers_multiscale(
    multiscale_score_path: str,
    layers: List[int],
    ranking_method: str = 'robust'
) -> List[Tuple[int, float]]:
    """
    Rank layers using multi-scale analysis
    
    Args:
        multiscale_score_path: Path to multi-scale scores
        layers: List of layer indices
        ranking_method: One of:
            - 'layernavigator': Use S(ε*) (original LayerNavigator at normalized scale)
            - 'max': Use max_ε S(ε) (best possible score)
            - 'robust': Use S(ε*) × p_0/(1+p_0) (robustness-weighted) ← RECOMMENDED
            - 'persistence': Use p_0 only (robustness)
            - 'total': Use sum of S(ε) across all scales ← NEW
            - 'avg': Use average S(ε) across all scales ← NEW
            - 'integrated': Use trapezoidal integral ∫S(ε)dε ← NEW
            - 'log_weighted': Use log-space weighted sum ← NEW
            - 'robust_total': Use total_S × p_0/(1+p_0) ← NEW (combines both)
            
    Returns:
        ranking: Sorted list of (layer, score) tuples in descending order
    """
    scores = []
    
    for l in layers:
        with open(f"{multiscale_score_path}/L{l}.json", "r") as f:
            data = json.load(f)
        
        if ranking_method == 'layernavigator':
            score = data['layernavigator_S']
        elif ranking_method == 'max':
            score = data['max_S']
        elif ranking_method == 'robust':
            p_0 = data['persistence_length']
            S_ln = data['layernavigator_S']
            score = S_ln * (p_0 / (1 + p_0))  # Robustness-weighted
        elif ranking_method == 'persistence':
            score = data['persistence_length']
        
        # NEW RANKING METHODS
        elif ranking_method == 'total':
            score = data['total_S']  # Sum of all S(ε)
        elif ranking_method == 'avg':
            score = data['avg_S']  # Average S(ε)
        elif ranking_method == 'integrated':
            score = data['integrated_S']  # Trapezoidal integral
        elif ranking_method == 'log_weighted':
            score = data['log_weighted_S']  # Log-space weighted
        elif ranking_method == 'robust_total':
            # Combines total steerability with robustness
            p_0 = data['persistence_length']
            total_S = data['total_S']
            score = total_S * (p_0 / (1 + p_0))
        
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        
        scores.append((l, score))
    
    # Sort by score (descending)
    ranking = sorted(scores, key=lambda x: x[1], reverse=True)
    
    print(f"\nLayer Ranking ({ranking_method}):")
    print("-" * 40)
    for rank, (layer, score) in enumerate(ranking, 1):
        print(f"{rank:2d}. Layer {layer:2d}: {score:.6f}")
    
    return ranking


# ============================================================================
# PART 9: VISUALIZATION (requires matplotlib)
# ============================================================================

def visualize_multiscale_profile(
    layer_idx: int,
    multiscale_score_path: str,
    save_plot: bool = True
):
    """
    Visualize D(ε), C(ε), S(ε) curves for a specific layer
    
    Args:
        layer_idx: Which layer to visualize
        multiscale_score_path: Path to multi-scale scores
        save_plot: Whether to save the plot
        
    Creates:
        - Top panel: D(ε), C(ε), S(ε) curves with persistence interval
        - Bottom panel: Number of clusters vs scale
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None
    
    # Load data
    with open(f"{multiscale_score_path}/L{layer_idx}.json", "r") as f:
        data = json.load(f)
    
    scales = np.array(data['scales'])
    D_scores = np.array(data['D_scores'])
    C_scores = np.array(data['C_scores'])
    S_scores = np.array(data['S_scores'])
    num_clusters = np.array(data['num_clusters'])
    
    eps_b, eps_merge = data['persistence_interval']
    p_0 = data['persistence_length']
    eps_star = data['layernavigator_scale']
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: D(ε), C(ε), S(ε)
    ax1 = axes[0]
    ax1.plot(scales, D_scores, 'b-', label='D(ε) - Discriminability', linewidth=2)
    ax1.plot(scales, C_scores, 'g-', label='C(ε) - Consistency', linewidth=2)
    ax1.plot(scales, S_scores, 'r-', label='S(ε) = D(ε) + C(ε)', linewidth=2)
    
    # Mark persistence interval
    ax1.axvline(eps_b, color='purple', linestyle='--', alpha=0.5, label=f'Birth ε_b = {eps_b:.3f}')
    ax1.axvline(eps_merge, color='orange', linestyle='--', alpha=0.5, label=f'Death ε_d = {eps_merge:.3f}')
    ax1.axvspan(eps_b, eps_merge, alpha=0.2, color='yellow', label=f'Persistent (p_0 = {p_0:.3f})')
    
    # Mark LayerNavigator scale
    ax1.axvline(eps_star, color='black', linestyle=':', linewidth=2, label=f'LayerNavigator ε* = {eps_star:.3f}')
    
    ax1.set_xlabel('Scale ε', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'Layer {layer_idx}: Multi-Scale Steerability Profile', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Number of clusters
    ax2 = axes[1]
    ax2.plot(scales, num_clusters, 'k-', linewidth=2)
    ax2.axvline(eps_b, color='purple', linestyle='--', alpha=0.5)
    ax2.axvline(eps_merge, color='orange', linestyle='--', alpha=0.5)
    ax2.axvspan(eps_b, eps_merge, alpha=0.2, color='yellow')
    ax2.axvline(eps_star, color='black', linestyle=':', linewidth=2)
    ax2.axhline(2, color='red', linestyle='--', alpha=0.5, label='Binary Clustering')
    
    ax2.set_xlabel('Scale ε', fontsize=12)
    ax2.set_ylabel('Number of Clusters', fontsize=12)
    ax2.set_title(f'Layer {layer_idx}: Cluster Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    plt.tight_layout()
    if save_plot:
        plot_path = f"{multiscale_score_path}/L{layer_idx}_profile.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    plt.show()
    return fig