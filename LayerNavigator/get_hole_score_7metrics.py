"""
HOLE (Homological Observation of Latent Embeddings) Metric Computation
Comprehensive Version with Class-Specific Cloud Analysis

Companion to LayerNavigator's get_score.py

Features:
- 7 distance metrics (euclidean, cosine, mahalanobis, geodesic, density-normalized variants)
- 50+ topological metrics from combined cloud (G1-G6)
- 21 class-specific metrics (positive, negative, difference clouds) (G7)
- Multi-metric comparison and agreement analysis
- Wasserstein distance for layer stability analysis
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import json
import gudhi
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


# ============================================
# DISTANCE COMPUTATION FUNCTIONS
# ============================================

def compute_geodesic_distance(activations: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute geodesic distance using k-NN graph and shortest paths
    
    Args:
        activations: (N, hidden_dim)
        k: number of nearest neighbors for graph construction
        
    Returns:
        geodesic_distances: (N, N) distance matrix
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(activations)
    distances, indices = nbrs.kneighbors(activations)
    
    # Create sparse adjacency matrix
    n = len(activations)
    row_ind = np.repeat(np.arange(n), k)
    col_ind = indices.flatten()
    data = distances.flatten()
    
    adjacency = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    
    # Make symmetric (take minimum distance if asymmetric)
    adjacency = adjacency.minimum(adjacency.T)
    
    # Compute shortest paths (geodesic distances)
    geodesic_dist = shortest_path(adjacency, directed=False, method='auto')
    
    return geodesic_dist


def compute_local_scales(activations: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute local scale for each point (average distance to k nearest neighbors)
    Used for density normalization
    
    Args:
        activations: (N, hidden_dim)
        k: number of nearest neighbors
        
    Returns:
        local_scales: (N,) array of local scales μᵢ
    """
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(activations)
    distances, _ = nbrs.kneighbors(activations)
    
    # Average distance to k nearest neighbors
    local_scales = distances.mean(axis=1)
    
    return local_scales


def apply_density_normalization(distance_matrix: np.ndarray, local_scales: np.ndarray) -> np.ndarray:
    """
    Apply density normalization to distance matrix
    
    Formula: d_normalized[i,j] = d[i,j] / sqrt(μᵢ * μⱼ)
    
    Args:
        distance_matrix: (N, N) pairwise distances
        local_scales: (N,) local scale for each point
        
    Returns:
        normalized_distances: (N, N) density-normalized distances
    """
    # Compute normalization matrix: sqrt(μᵢ * μⱼ) for all pairs
    norm_matrix = np.sqrt(np.outer(local_scales, local_scales))
    
    # Apply normalization
    normalized_distances = distance_matrix / (norm_matrix + 1e-8)
    
    return normalized_distances


# ============================================
# PERSISTENCE COMPUTATION FUNCTION
# ============================================

def compute_persistence_diagram(
    activations: np.ndarray,
    metric: str = 'cosine',
    max_dimension: int = 2,
    max_edge_length: float = np.inf,
    k_neighbors: int = 10,  # For geodesic and density normalization
    num_filtration_steps: int = 50  # For Betti curves
) -> Dict:
    """
    Compute persistence diagram using GUDHI with enhanced topological metrics
    
    Args:
        activations: (N, hidden_dim)
        metric: Distance metric - one of:
            - 'euclidean': Euclidean distance
            - 'cosine': Cosine distance
            - 'mahalanobis': Mahalanobis distance
            - 'geodesic': Geodesic distance via k-NN graph
            - 'dens_norm_euclidean': Density-normalized Euclidean
            - 'dens_norm_cosine': Density-normalized Cosine
            - 'dens_norm_mahalanobis': Density-normalized Mahalanobis
        max_dimension: maximum homology dimension to compute
        max_edge_length: maximum distance for Rips complex
        k_neighbors: k for geodesic (k-NN graph) and density normalization
        num_filtration_steps: number of steps for Betti curve computation
        
    Returns:
        persistence_data: {
            'diagram': list of (dimension, (birth, death)),
            'betti_numbers': {0: beta_0, 1: beta_1, 2: beta_2},
            'persistence_stats': {...},
            'persistence_entropy': {0: H0_entropy, 1: H1_entropy, ...},
            'betti_curves': {0: [(eta, beta_0), ...], 1: [(eta, beta_1), ...], ...},
            'betti_curve_auc': {0: AUC_beta_0, 1: AUC_beta_1, ...},
            'strong_loops_count': int,
        }
    """
    # ============================================
    # STEP 1: COMPUTE BASE DISTANCE MATRIX
    # ============================================
    
    if metric == 'euclidean':
        distances = pairwise_distances(activations, metric='euclidean')
        
    elif metric == 'cosine':
        distances = pairwise_distances(activations, metric='cosine')
        
    elif metric == 'mahalanobis':
        # Compute covariance
        cov = np.cov(activations.T)
        inv_cov = np.linalg.pinv(cov)
        distances = pairwise_distances(activations, metric='mahalanobis', VI=inv_cov)
        
    elif metric == 'geodesic':
        # Geodesic distance via k-NN graph
        distances = compute_geodesic_distance(activations, k=k_neighbors)
        
    elif metric.startswith('dens_norm_'):
        # Density-normalized variants
        base_metric = metric.replace('dens_norm_', '')
        
        # Compute base distance matrix
        if base_metric == 'euclidean':
            distances = pairwise_distances(activations, metric='euclidean')
        elif base_metric == 'cosine':
            distances = pairwise_distances(activations, metric='cosine')
        elif base_metric == 'mahalanobis':
            cov = np.cov(activations.T)
            inv_cov = np.linalg.pinv(cov)
            distances = pairwise_distances(activations, metric='mahalanobis', VI=inv_cov)
        else:
            raise ValueError(f"Unknown base metric for density normalization: {base_metric}")
        
        # Apply density normalization
        local_scales = compute_local_scales(activations, k=k_neighbors)
        distances = apply_density_normalization(distances, local_scales)
    
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # ============================================
    # STEP 2: CREATE RIPS COMPLEX AND COMPUTE PERSISTENCE HOMOLOGY
    # ============================================
    
    # Create Rips complex
    rips_complex = gudhi.RipsComplex(distance_matrix=distances, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    
    # Compute persistence
    persistence = simplex_tree.persistence()
    
    # Extract persistence pairs by dimension
    persistence_by_dim = {i: [] for i in range(max_dimension + 1)}
    for dim, (birth, death) in persistence:
        persistence_by_dim[dim].append((birth, death))
    
    # Compute Betti numbers (at final filtration value)
    betti_numbers = simplex_tree.betti_numbers()
    # Pad with zeros if needed
    while len(betti_numbers) <= max_dimension:
        betti_numbers.append(0)
    
    # ============================================
    # STEP 3: COMPUTE BASIC PERSISTENCE STATISTICS
    # ============================================
    persistence_stats = {}
    for dim in range(max_dimension + 1):
        pairs = persistence_by_dim[dim]
        if len(pairs) > 0:
            # Filter out infinite persistence (never dies)
            finite_pairs = [(b, d) for b, d in pairs if d != np.inf]
            
            if finite_pairs:
                persistences = [d - b for b, d in finite_pairs]
                persistence_stats[f'H{dim}_mean'] = np.mean(persistences)
                persistence_stats[f'H{dim}_max'] = np.max(persistences)
                persistence_stats[f'H{dim}_total'] = np.sum(persistences)
                persistence_stats[f'H{dim}_count'] = len(finite_pairs)
            else:
                persistence_stats[f'H{dim}_mean'] = 0.0
                persistence_stats[f'H{dim}_max'] = 0.0
                persistence_stats[f'H{dim}_total'] = 0.0
                persistence_stats[f'H{dim}_count'] = 0
        else:
            persistence_stats[f'H{dim}_mean'] = 0.0
            persistence_stats[f'H{dim}_max'] = 0.0
            persistence_stats[f'H{dim}_total'] = 0.0
            persistence_stats[f'H{dim}_count'] = 0
    
    # ============================================
    # STEP 4: PERSISTENCE ENTROPY
    # ============================================
    
    persistence_entropy = {}
    for dim in range(max_dimension + 1):
        pairs = persistence_by_dim[dim]
        finite_pairs = [(b, d) for b, d in pairs if d != np.inf]
        
        if len(finite_pairs) > 1:
            persistences = np.array([d - b for b, d in finite_pairs])
            total_persistence = np.sum(persistences)
            
            if total_persistence > 1e-10:
                # Compute normalized probabilities
                probabilities = persistences / total_persistence
                # Compute entropy: H = -sum(p_i * log(p_i))
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                persistence_entropy[dim] = float(entropy)
            else:
                persistence_entropy[dim] = 0.0
        else:
            persistence_entropy[dim] = 0.0
    
    # ============================================
    # STEP 5: COMPUTE BETTI CURVES
    # ============================================
    filtration_list = list(simplex_tree.get_filtration())
    if len(filtration_list) > 0:
        max_filtration = filtration_list[-1][1]
    else:
        max_filtration = 1.0

    min_filtration = 0.0

    # Create filtration thresholds
    filtration_values = np.linspace(min_filtration, max_filtration, num_filtration_steps)

    betti_curves = {dim: [] for dim in range(max_dimension + 1)}

    for epsilon in filtration_values:
        # Count features alive at this epsilon
        for dim in range(max_dimension + 1):
            pairs = persistence_by_dim[dim]
            alive_count = sum(1 for b, d in pairs if b <= epsilon and (d > epsilon or d == np.inf))
            betti_curves[dim].append((float(epsilon), alive_count))
    
    # Compute Area Under Curve (AUC) for each Betti curve
    betti_curve_auc = {}
    for dim in range(max_dimension + 1):
        if len(betti_curves[dim]) > 1:
            epsilons = np.array([eps for eps, _ in betti_curves[dim]])
            betti_values = np.array([beta for _, beta in betti_curves[dim]])
            # Trapezoidal integration
            auc = np.trapz(betti_values, epsilons)
            betti_curve_auc[dim] = float(auc)
        else:
            betti_curve_auc[dim] = 0.0
    
    # ============================================
    # STEP 6: COMPUTE H_1 ENHANCED STATISTICS
    # ============================================
    
    # Count "strong loops" - loops with high persistence
    h1_pairs = persistence_by_dim[1]
    finite_h1_pairs = [(b, d) for b, d in h1_pairs if d != np.inf]
    
    if finite_h1_pairs:
        h1_persistences = np.array([d - b for b, d in finite_h1_pairs])
        
        if len(h1_persistences) > 0:
            strong_loop_threshold = np.median(h1_persistences) if len(h1_persistences) > 1 else h1_persistences[0]
            strong_loops_count = int(np.sum(h1_persistences > strong_loop_threshold))
        else:
            strong_loops_count = 0
    else:
        strong_loops_count = 0
    
    # ============================================
    # STEP 7: RETURN COMPREHENSIVE RESULTS
    # ============================================
    
    return {
        # Original outputs
        'diagram': persistence,
        'persistence_by_dim': persistence_by_dim,
        'betti_numbers': {i: betti_numbers[i] if i < len(betti_numbers) else 0 
                          for i in range(max_dimension + 1)},
        'persistence_stats': persistence_stats,
        
        # Persistence Entropy
        'persistence_entropy': persistence_entropy,
        
        # Betti Curves and AUC
        'betti_curves': betti_curves,
        'betti_curve_auc': betti_curve_auc,
        
        # H1 Enhanced Statistics
        'strong_loops_count': strong_loops_count
    }


# ============================================
# WASSERSTEIN DISTANCE FUNCTIONS
# ============================================

def compute_wasserstein_distance(diagram1: Dict, diagram2: Dict, 
                                  dimension: int = 1, order: int = 2) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams
    
    Args:
        diagram1: Output from compute_persistence_diagram for layer i
        diagram2: Output from compute_persistence_diagram for layer j
        dimension: Which homology dimension to compare (default: H₁)
        order: Wasserstein order (1 or 2)
        
    Returns:
        wasserstein_distance: float
    """
    import gudhi.wasserstein
    
    # Extract persistence pairs for the specified dimension
    pairs1 = diagram1['persistence_by_dim'][dimension]
    pairs2 = diagram2['persistence_by_dim'][dimension]
    
    # Filter out infinite persistence
    finite_pairs1 = np.array([[b, d] for b, d in pairs1 if d != np.inf])
    finite_pairs2 = np.array([[b, d] for b, d in pairs2 if d != np.inf])
    
    # Handle empty diagrams
    if len(finite_pairs1) == 0 or len(finite_pairs2) == 0:
        return 0.0
    
    # Compute Wasserstein distance
    distance = gudhi.wasserstein.wasserstein_distance(
        finite_pairs1, 
        finite_pairs2, 
        order=order
    )
    
    return float(distance)

# ============================================
# MAIN HOLE SCORE COMPUTATION FUNCTION
# ============================================

def get_hole_score(
    layers: List[int],
    dataset,
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    metric: str = "cosine",
    max_dimension: int = 2,
    subsample: int = None,
    compute_class_clouds: bool = True
):
    """
    Compute HOLE metrics for each layer
    
    Computes persistence on 4 point clouds:
      1. Combined cloud (all activations together) - Original 50+ metrics
      2. Positive cloud (label=1 activations only) - Class-specific metrics
      3. Negative cloud (label=0 activations only) - Class-specific metrics
      4. Difference cloud (pos - neg_mean) - Steering direction topology
    
    Args:
        layers: List of layer indices to analyze
        dataset: Dataset object (must have train=True)
        vec_task: Task name
        vec_method: Vector extraction method
        acts_pre: Preprocessing method ('standard' for z-score)
        metric: Distance metric (euclidean, cosine, mahalanobis, geodesic, dens_norm_*)
        max_dimension: Maximum homology dimension (0, 1, 2)
        subsample: Subsample N points PER CLASS for efficiency (None = use all)
        compute_class_clouds: If True, compute pos/neg/diff clouds (adds 21 metrics)
        
    Returns:
        hole_score_info: Dict with comprehensive metrics (~71 total if compute_class_clouds=True)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    assert dataset.train == True, "Only Use Train Dataset"
    
    vec_root = f"./Vectors/{vec_task}/{vec_method}"
    svec_path = vec_root[10:].replace("/", "+")
    
    # Score Save Path
    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    metric_str = f"-{metric}"
    cloud_suffix = "_WITH_CLASS_CLOUDS" if compute_class_clouds else ""
    
    save_root = f"./Score_HOLE{acts_pre_str}{metric_str}{cloud_suffix}/{dataset.task}/{svec_path}/"
    os.makedirs(save_root, exist_ok=True)
    
    ans_num = 2  # Binary classification
    
    # Load activations
    print("Loading activations...")
    acts = torch.load(f"{vec_root}/acts.pt")
    
    hole_score_info = {}
    
    # Process each layer
    for l in tqdm(layers, desc=f"Computing HOLE Scores ({metric})"):
        # ========================================================
        # STEP 1: PREPARE ACTIVATIONS - SEPARATE BY CLASS
        # ========================================================
        
        # Separate by label for class-specific analysis
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
        
        # Subsample if specified (per class to maintain balance)
        if subsample is not None:
            if len(pos_acts) > subsample:
                pos_indices = np.random.choice(len(pos_acts), subsample, replace=False)
                pos_acts = pos_acts[pos_indices]
            
            if len(neg_acts) > subsample:
                neg_indices = np.random.choice(len(neg_acts), subsample, replace=False)
                neg_acts = neg_acts[neg_indices]
        
        # Combined cloud (for original metrics)
        combined_acts = np.vstack([pos_acts, neg_acts])
        
        # ========================================================
        # STEP 2: CREATE DIFFERENCE CLOUD (STEERING DIRECTION SPACE)
        # ========================================================
        
        if compute_class_clouds:
            # Difference cloud: How each positive deviates from negative center
            neg_mean = neg_acts.mean(axis=0)
            diff_acts = pos_acts - neg_mean
            
            print(f"  Layer {l}: Point clouds created:")
            print(f"    - Combined: {len(combined_acts)} points")
            print(f"    - Positive: {len(pos_acts)} points")
            print(f"    - Negative: {len(neg_acts)} points")
            print(f"    - Difference: {len(diff_acts)} points")
        
        # ========================================================
        # STEP 3: NORMALIZE ACTIVATIONS
        # ========================================================
        
        if acts_pre == "standard":
            # Normalize combined (maintains between-class relationships)
            combined_mean = combined_acts.mean(axis=0)
            combined_std = combined_acts.std(axis=0) + 1e-8
            
            combined_acts_norm = (combined_acts - combined_mean) / combined_std
            
            if compute_class_clouds:
                # Normalize pos/neg using COMBINED statistics (maintains relative positions)
                pos_acts_norm = (pos_acts - combined_mean) / combined_std
                neg_acts_norm = (neg_acts - combined_mean) / combined_std
                
                # Normalize difference cloud SEPARATELY (represents steering direction space)
                diff_mean = diff_acts.mean(axis=0)
                diff_std = diff_acts.std(axis=0) + 1e-8
                diff_acts_norm = (diff_acts - diff_mean) / diff_std
        else:
            combined_acts_norm = combined_acts
            if compute_class_clouds:
                pos_acts_norm = pos_acts
                neg_acts_norm = neg_acts
                diff_acts_norm = diff_acts
        
        # ========================================================
        # STEP 4: COMPUTE PERSISTENCE - COMBINED CLOUD (ORIGINAL)
        # ========================================================
        
        print(f"  Layer {l}: Computing persistent homology with {metric} distance...")
        
        persistence_combined = compute_persistence_diagram(
            combined_acts_norm,
            metric=metric,
            max_dimension=max_dimension,
            max_edge_length=np.inf
        )
        
        # Extract all computed metrics from persistent data
        betti_numbers = persistence_combined['betti_numbers']
        persistence_stats = persistence_combined['persistence_stats']
        persistence_entropy = persistence_combined['persistence_entropy']
        betti_curve_auc = persistence_combined['betti_curve_auc']
        strong_loops_count = persistence_combined['strong_loops_count']
        persistence_by_dim = persistence_combined['persistence_by_dim']
        
        # G1: Betti numbers
        beta0 = betti_numbers.get(0, 0)
        beta1 = betti_numbers.get(1, 0)
        beta2 = betti_numbers.get(2, 0)
        sum_beta_counts = float(beta0 + beta1 + beta2)
        
        # G2: Extract basic persistence statistics for all dimensions
        mean_pers_h0 = persistence_stats.get('H0_mean', 0.0)
        mean_pers_h1 = persistence_stats.get('H1_mean', 0.0)
        mean_pers_h2 = persistence_stats.get('H2_mean', 0.0)
        
        max_pers_h0 = persistence_stats.get('H0_max', 0.0)
        max_pers_h1 = persistence_stats.get('H1_max', 0.0)
        max_pers_h2 = persistence_stats.get('H2_max', 0.0)
        
        total_pers_h0 = persistence_stats.get('H0_total', 0.0)
        total_pers_h1 = persistence_stats.get('H1_total', 0.0)
        total_pers_h2 = persistence_stats.get('H2_total', 0.0)
        
        count_h0 = persistence_stats.get('H0_count', 0)
        count_h1 = persistence_stats.get('H1_count', 0)
        count_h2 = persistence_stats.get('H2_count', 0)
        
        # persistence only (robust clustering)
        h0_persistence_only = mean_pers_h0
        h1_persistence_only = mean_pers_h1
        h2_persistence_only = mean_pers_h2
        
        # All persistence
        mean_all_persistence = (
            mean_pers_h0 +
            mean_pers_h1 +
            mean_pers_h2
        ) / 3.0

        # Robust clusters, fragile loops
        h1_fragility_bonus = 1.0 / (1.0 + mean_pers_h1) if mean_pers_h1 > 0 else 1.0
        robust_clusters_fragile_loops = mean_pers_h0 * h1_fragility_bonus
        
        # Robust clusters, fragile voids
        h2_fragility_bonus = 1.0 / (1.0 + mean_pers_h2) if mean_pers_h2 > 0 else 1.0
        robust_clusters_fragile_voids = mean_pers_h0 * h2_fragility_bonus
        
        # H0 up, H1 down, H2 down
        persistence_inverseH1_inverseH2 = (
            mean_pers_h0 +
            (1.0 / (1.0 + mean_pers_h1)) +
            (1.0 / (1.0 + mean_pers_h2))
        ) / 3.0
        
        total_pers = sum([persistence_stats.get(f'H{d}_total', 0.0) 
                         for d in range(max_dimension + 1)])
        
        # G3: Extract raw entropy (no normalization)
        entropy_h0 = persistence_entropy.get(0, 0.0)
        entropy_h1 = persistence_entropy.get(1, 0.0)
        entropy_h2 = persistence_entropy.get(2, 0.0)
        inverse_h0_entropy = 1.0 / (1.0 + entropy_h0) if entropy_h0 > 0 else 1.0
        inverse_h1_entropy = 1.0 / (1.0 + entropy_h1) if entropy_h1 > 0 else 1.0
        inverse_h2_entropy = 1.0 / (1.0 + entropy_h2) if entropy_h2 > 0 else 1.0
        
        # G4: Betti Curve
        auc_h0 = betti_curve_auc.get(0, 0.0)
        auc_h1 = betti_curve_auc.get(1, 0.0)
        auc_h2 = betti_curve_auc.get(2, 0.0)
        
        # G5: Strong loop count
        strong_loops_count = float(strong_loops_count)
        inverse_strong_loops_count = 1.0 / (1.0 + strong_loops_count)
        
        # Strong cluster
        finite_h0_pairs = [(b, d) for b, d in persistence_by_dim[0] if d != np.inf]
        if finite_h0_pairs:
            h0_persistences = np.array([d - b for b, d in finite_h0_pairs])
            if len(h0_persistences) > 0:
                strong_threshold = np.median(h0_persistences) if len(h0_persistences) > 1 else h0_persistences[0]
                strong_cluster_count = int(np.sum(h0_persistences > strong_threshold))
            else:
                strong_cluster_count = 0
        else:
            strong_cluster_count = 0
        inverse_strong_cluster_count = 1.0 / (1.0 + strong_cluster_count)
        
        # Strong voids
        finite_h2_pairs = [(b, d) for b, d in persistence_by_dim[2] if d != np.inf]
        if finite_h2_pairs:
            h2_persistences = np.array([d - b for b, d in finite_h2_pairs])
            if len(h2_persistences) > 0:
                strong_threshold = np.median(h2_persistences) if len(h2_persistences) > 1 else h2_persistences[0]
                strong_voids_count = int(np.sum(h2_persistences > strong_threshold))
            else:
                strong_voids_count = 0
        else:
            strong_voids_count = 0
        inverse_strong_voids_count = 1.0 / (1.0 + strong_voids_count)
        
        # G6: Persistence dominance ratio
        portion_of_persistence_dominance_h0 = max_pers_h0 / (mean_pers_h0 + 0.1)
        portion_of_persistence_dominance_h1 = max_pers_h1 / (mean_pers_h1 + 0.1)
        portion_of_persistence_dominance_h2 = max_pers_h2 / (mean_pers_h2 + 0.1)
        avg_max_persistence_score = (
            max_pers_h0 +
            max_pers_h1 +
            max_pers_h2
        ) / 3.0
        
        # inverse
        inverse_mean_H0 = 1.0 / (mean_pers_h0 + 0.1)
        inverse_mean_H1 = 1.0 / (mean_pers_h1 + 0.1)
        inverse_mean_H2 = 1.0 / (mean_pers_h2 + 0.1)
        
        # ========================================================
        # STEP 5: COMPUTE PERSISTENCE - CLASS-SPECIFIC CLOUDS (NEW)
        # ========================================================
        
        if compute_class_clouds:
            print(f"    Computing class-specific persistence...")
            
            # === POSITIVE CLOUD ===
            persistence_pos = compute_persistence_diagram(
                pos_acts_norm,
                metric=metric,
                max_dimension=max_dimension,
                max_edge_length=np.inf
            )
            
            # === NEGATIVE CLOUD ===
            persistence_neg = compute_persistence_diagram(
                neg_acts_norm,
                metric=metric,
                max_dimension=max_dimension,
                max_edge_length=np.inf
            )
            
            # === DIFFERENCE CLOUD ===
            persistence_diff = compute_persistence_diagram(
                diff_acts_norm,
                metric=metric,
                max_dimension=max_dimension,
                max_edge_length=np.inf
            )
            
            # Extract class-specific metrics for all dimensions
            mean_persistence_H0_pos = float(persistence_pos['persistence_stats'].get('H0_mean', 0.0))
            mean_persistence_H1_pos = float(persistence_pos['persistence_stats'].get('H1_mean', 0.0))
            mean_persistence_H2_pos = float(persistence_pos['persistence_stats'].get('H2_mean', 0.0))
            max_persistence_H0_pos = float(persistence_pos['persistence_stats'].get('H0_max', 0.0))
            max_persistence_H1_pos = float(persistence_pos['persistence_stats'].get('H1_max', 0.0))
            max_persistence_H2_pos = float(persistence_pos['persistence_stats'].get('H2_max', 0.0))
            
            mean_persistence_H0_neg = float(persistence_neg['persistence_stats'].get('H0_mean', 0.0))
            mean_persistence_H1_neg = float(persistence_neg['persistence_stats'].get('H1_mean', 0.0))
            mean_persistence_H2_neg = float(persistence_neg['persistence_stats'].get('H2_mean', 0.0))
            max_persistence_H0_neg = float(persistence_neg['persistence_stats'].get('H0_max', 0.0))
            max_persistence_H1_neg = float(persistence_neg['persistence_stats'].get('H1_max', 0.0))
            max_persistence_H2_neg = float(persistence_neg['persistence_stats'].get('H2_max', 0.0))
            
            mean_persistence_H0_diff = float(persistence_diff['persistence_stats'].get('H0_mean', 0.0))
            mean_persistence_H1_diff = float(persistence_diff['persistence_stats'].get('H1_mean', 0.0))
            mean_persistence_H2_diff = float(persistence_diff['persistence_stats'].get('H2_mean', 0.0))
            max_persistence_H0_diff = float(persistence_diff['persistence_stats'].get('H0_max', 0.0))
            max_persistence_H1_diff = float(persistence_diff['persistence_stats'].get('H1_max', 0.0))
            max_persistence_H2_diff = float(persistence_diff['persistence_stats'].get('H2_max', 0.0))
            
            # Compute derived class-specific metrics
            h1_separation = abs(mean_persistence_H1_pos - mean_persistence_H1_neg)
            steering_quality = mean_persistence_H0_diff / (1.0 + mean_persistence_H1_diff)
            class_consistency = 1.0 / (1.0 + h1_separation)
        
        # ========================================================
        # STEP 6: ASSEMBLE COMPLETE METRICS DICTIONARY
        # ========================================================
        
        hole_score_info[l] = {
            # ===== G1 BETTI NUMBERS =====
            'beta0': int(beta0),
            'beta1': int(beta1),
            'beta2': int(beta2),
            'sum_beta_counts': float(sum_beta_counts),
            
            # ===== G2 BASIC PERSISTENCE STATISTICS =====
            # H0 (Connected Components)
            'mean_persistence_H0': float(mean_pers_h0),
            'max_persistence_H0': float(max_pers_h0),
            'total_persistence_H0': float(total_pers_h0),
            'count_H0': int(count_h0),
            # H1 (Loops)
            'mean_persistence_H1': float(mean_pers_h1),
            'max_persistence_H1': float(max_pers_h1),
            'total_persistence_H1': float(total_pers_h1),
            'count_H1': int(count_h1),
            # H2 (Voids)
            'mean_persistence_H2': float(mean_pers_h2),
            'max_persistence_H2': float(max_pers_h2),
            'total_persistence_H2': float(total_pers_h2),
            'count_H2': int(count_h2),
            # total
            'total_persistence': float(total_pers),
            # other variants
            'h0_persistence_only': float(h0_persistence_only),
            'h1_persistence_only': float(h1_persistence_only),
            'h2_persistence_only': float(h2_persistence_only),
            'mean_all_persistence': float(mean_all_persistence),
            'robust_clusters_fragile_loops': float(robust_clusters_fragile_loops),
            'robust_clusters_fragile_voids': float(robust_clusters_fragile_voids),
            'persistence_inverseH1_inverseH2': float(persistence_inverseH1_inverseH2),
            
            # =====G3 PERSISTENCE ENTROPY =====
            'entropy_H0': float(entropy_h0),
            'entropy_H1': float(entropy_h1),
            'entropy_H2': float(entropy_h2),
            'inverse_h0_entropy': float(inverse_h0_entropy),
            'inverse_h1_entropy': float(inverse_h1_entropy),
            'inverse_h2_entropy': float(inverse_h2_entropy),

            # ===== G4: BETTI CURVE AUC =====
            'betti_curve_auc_H0': float(auc_h0),
            'betti_curve_auc_H1': float(auc_h1),
            'betti_curve_auc_H2': float(auc_h2),

            # ===== G5: Strong Features =====
            'strong_loops_count': float(strong_loops_count),
            'inverse_strong_loops_count': float(inverse_strong_loops_count),
            'strong_cluster_count': float(strong_cluster_count),
            'inverse_strong_cluster_count': float(inverse_strong_cluster_count),
            'strong_voids_count': float(strong_voids_count),
            'inverse_strong_voids_count': float(inverse_strong_voids_count),
            
            # ===== G6: Dominance Ratios =====
            'portion_of_persistence_dominance_h0': float(portion_of_persistence_dominance_h0),
            'portion_of_persistence_dominance_h1': float(portion_of_persistence_dominance_h1),
            'portion_of_persistence_dominance_h2': float(portion_of_persistence_dominance_h2),
            'avg_max_persistence_score': float(avg_max_persistence_score),
            'inverse_mean_H0': float(inverse_mean_H0),
            'inverse_mean_H1': float(inverse_mean_H1),
            'inverse_mean_H2': float(inverse_mean_H2),

            # ===== G7: CLASS-SPECIFIC METRICS =====
            # Positive Cloud (Label 1)
            'mean_persistence_H0_pos': float(mean_persistence_H0_pos) if compute_class_clouds else None,
            'mean_persistence_H1_pos': float(mean_persistence_H1_pos) if compute_class_clouds else None,
            'mean_persistence_H2_pos': float(mean_persistence_H2_pos) if compute_class_clouds else None,
            'max_persistence_H0_pos': float(max_persistence_H0_pos) if compute_class_clouds else None,
            'max_persistence_H1_pos': float(max_persistence_H1_pos) if compute_class_clouds else None,
            'max_persistence_H2_pos': float(max_persistence_H2_pos) if compute_class_clouds else None,
            
            # Negative Cloud (Label 0)
            'mean_persistence_H0_neg': float(mean_persistence_H0_neg) if compute_class_clouds else None,
            'mean_persistence_H1_neg': float(mean_persistence_H1_neg) if compute_class_clouds else None,
            'mean_persistence_H2_neg': float(mean_persistence_H2_neg) if compute_class_clouds else None,
            'max_persistence_H0_neg': float(max_persistence_H0_neg) if compute_class_clouds else None,
            'max_persistence_H1_neg': float(max_persistence_H1_neg) if compute_class_clouds else None,
            'max_persistence_H2_neg': float(max_persistence_H2_neg) if compute_class_clouds else None,
            
            # Difference Cloud (Steering Direction: pos - neg_mean)
            'mean_persistence_H0_diff': float(mean_persistence_H0_diff) if compute_class_clouds else None,
            'mean_persistence_H1_diff': float(mean_persistence_H1_diff) if compute_class_clouds else None,
            'mean_persistence_H2_diff': float(mean_persistence_H2_diff) if compute_class_clouds else None,
            'max_persistence_H0_diff': float(max_persistence_H0_diff) if compute_class_clouds else None,
            'max_persistence_H1_diff': float(max_persistence_H1_diff) if compute_class_clouds else None,
            'max_persistence_H2_diff': float(max_persistence_H2_diff) if compute_class_clouds else None,
            
            # Derived Class-Specific Metrics
            'h1_separation': float(h1_separation) if compute_class_clouds else None,
            'steering_quality': float(steering_quality) if compute_class_clouds else None,
            'class_consistency': float(class_consistency) if compute_class_clouds else None,
            
            # ===== METADATA =====
            'metric': metric,
            'n_samples_combined': int(len(combined_acts_norm)),
            'n_samples_pos': int(len(pos_acts)) if compute_class_clouds else None,
            'n_samples_neg': int(len(neg_acts)) if compute_class_clouds else None,
            'n_samples_diff': int(len(diff_acts_norm)) if compute_class_clouds else None,
            'compute_class_clouds': compute_class_clouds
        }
        
        # Save per-layer results
        with open(f"{save_root}L{l}.json", "w") as f:
            json.dump(hole_score_info[l], f, indent=4)
    
    # Save complete results
    with open(f"{save_root}all_layers.json", "w") as f:
        json.dump(hole_score_info, f, indent=4)
    
    total_metrics = len(hole_score_info[layers[0]])
    print(f"\nHOLE scores saved to: {save_root}")
    print(f"Total metrics stored per layer: {total_metrics} fields")
    if compute_class_clouds:
        print(f"  - Combined cloud metrics: ~50")
        print(f"  - Class-specific metrics: 21")
        print(f"  - Metadata: 5")
    
    return hole_score_info


# ============================================
# ALL METRICS COMPUTATION
# ============================================

def get_hole_score_all_metrics(
    layers: List[int],
    dataset,
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    max_dimension: int = 2,
    subsample: int = None,
    compute_class_clouds: bool = True,
    metrics_to_compute: List[str] = None
):
    """
    Compute HOLE scores using ALL 7 distance metrics for comprehensive analysis
    
    Includes class-specific metrics (pos/neg/diff clouds) for each distance metric
    
    Args:
        layers: List of layer indices to analyze
        dataset: Dataset object (must have train=True)
        vec_task: Task name
        vec_method: Vector extraction method
        acts_pre: Preprocessing method
        max_dimension: Maximum homology dimension
        subsample: Subsample size PER CLASS (not total)
        compute_class_clouds: If True, compute pos/neg/diff clouds (+21 metrics per layer)
        metrics_to_compute: List of metrics to compute. If None, compute all 7.
                           Options: ['euclidean', 'cosine', 'mahalanobis', 'geodesic',
                                    'dens_norm_euclidean', 'dens_norm_cosine', 
                                    'dens_norm_mahalanobis']
        
    Returns:
        all_metric_scores: {
            metric_name: {
                layer: {score_dict with ~71 metrics if compute_class_clouds=True}
            }
        }
    """
    # Default to all metrics if not specified
    if metrics_to_compute is None:
        all_metrics = [
            'euclidean',
            # 'cosine',
            # 'mahalanobis',
            # 'geodesic',
            # 'dens_norm_euclidean',
            # 'dens_norm_cosine',
            # 'dens_norm_mahalanobis'
        ]
    else:
        all_metrics = metrics_to_compute
    
    all_metric_scores = {}
    
    print("\n" + "="*80)
    print(f"COMPUTING HOLE SCORES WITH {len(all_metrics)} DISTANCE METRICS")
    if compute_class_clouds:
        print("MODE: Comprehensive (Combined + Pos + Neg + Diff clouds)")
        print("      ~71 metrics per layer per distance metric")
    else:
        print("MODE: Combined cloud only")
        print("      ~50 metrics per layer per distance metric")
    print("="*80)
    
    for i, metric in enumerate(all_metrics, 1):
        print(f"\n[{i}/{len(all_metrics)}] Computing metric: {metric}")
        print("-" * 80)
        
        scores = get_hole_score(
            layers=layers,
            dataset=dataset,
            vec_task=vec_task,
            vec_method=vec_method,
            acts_pre=acts_pre,
            metric=metric,
            max_dimension=max_dimension,
            subsample=subsample,
            compute_class_clouds=compute_class_clouds
        )
        
        all_metric_scores[metric] = scores
        
        # Print summary for this metric
        sample_layer = layers[0]
        n_metrics = len(scores[sample_layer])
        print(f"Computed {n_metrics} metrics per layer")
    
    # Save comprehensive results
    vec_root = f"./Vectors/{vec_task}/{vec_method}"
    svec_path = vec_root[10:].replace("/", "+")
    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    
    cloud_suffix = "_WITH_CLASS_CLOUDS" if compute_class_clouds else ""
    save_path = f"./Score_HOLE{acts_pre_str}_ALL_METRICS{cloud_suffix}/{dataset.task}/{svec_path}/"
    os.makedirs(save_path, exist_ok=True)
    
    # Save JSON
    with open(f"{save_path}all_metrics_all_layers.json", "w") as f:
        json.dump(all_metric_scores, f, indent=4)
    
    # Save summary statistics
    summary = {
        'n_metrics': len(all_metrics),
        'metrics_computed': all_metrics,
        'n_layers': len(layers),
        'layers': layers,
        'compute_class_clouds': compute_class_clouds,
        'metrics_per_layer': len(all_metric_scores[all_metrics[0]][layers[0]]),
        'subsample_per_class': subsample,
        'max_dimension': max_dimension
    }
    
    with open(f"{save_path}computation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*80)
    print(f"All metrics saved to: {save_path}")
    print(f"  - Total distance metrics: {len(all_metrics)}")
    print(f"  - Metrics per layer: {summary['metrics_per_layer']}")
    print(f"  - Total data points: {len(all_metrics)} metrics × {len(layers)} layers × {summary['metrics_per_layer']} values")
    print("="*80)
    
    return all_metric_scores


# ============================================
# COMPARISON UTILITIES
# ============================================

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
            # Use robust_clusters_fragile_loops as primary metric
            hole_scores.append(hole_data.get('robust_clusters_fragile_loops', 0.0))
    
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
    layer_idx: int,
    show_class_metrics: bool = True
):
    """
    Compare how different distance metrics rank a specific layer
    
    Includes comparison of class-specific metrics across distance metrics
    
    Args:
        layer_scores: Output from get_hole_score_all_metrics()
        layer_idx: Which layer to compare
        show_class_metrics: If True, display class-specific metrics
        
    Returns:
        comparison: Dict with metric comparisons
    """
    comparison = {
        'layer': layer_idx,
        'metrics': {}
    }
    
    # Collect all metrics for this layer
    for metric_name, scores in layer_scores.items():
        if layer_idx in scores:
            layer_data = scores[layer_idx]
            
            # Core metrics (always present)
            comparison['metrics'][metric_name] = {
                'beta0': layer_data.get('beta0', 0),
                'beta1': layer_data.get('beta1', 0),
                'beta2': layer_data.get('beta2', 0),
                'mean_persistence_H0': layer_data.get('mean_persistence_H0', 0.0),
                'mean_persistence_H1': layer_data.get('mean_persistence_H1', 0.0),
                'mean_persistence_H2': layer_data.get('mean_persistence_H2', 0.0),
                'robust_clusters_fragile_loops': layer_data.get('robust_clusters_fragile_loops', 0.0),
            }
            
            # Class-specific metrics (if computed)
            if show_class_metrics and layer_data.get('compute_class_clouds', False):
                comparison['metrics'][metric_name].update({
                    'mean_persistence_H1_pos': layer_data.get('mean_persistence_H1_pos', 0.0),
                    'mean_persistence_H1_neg': layer_data.get('mean_persistence_H1_neg', 0.0),
                    'mean_persistence_H0_diff': layer_data.get('mean_persistence_H0_diff', 0.0),
                    'mean_persistence_H1_diff': layer_data.get('mean_persistence_H1_diff', 0.0),
                    'h1_separation': layer_data.get('h1_separation', 0.0),
                    'steering_quality': layer_data.get('steering_quality', 0.0),
                    'class_consistency': layer_data.get('class_consistency', 0.0),
                })
    
    # Print comprehensive comparison table
    print(f"\n{'='*100}")
    print(f"Layer {layer_idx} - Metric Comparison Across Distance Functions")
    print(f"{'='*100}")
    
    # Table 1: Core Combined Cloud Metrics
    print(f"\n--- COMBINED CLOUD METRICS ---")
    print(f"{'Metric':<25} {'β₀':>6} {'β₁':>6} {'β₂':>6} {'H0_mean':>10} {'H1_mean':>10} {'H2_mean':>10} {'RobustFrag':>10}")
    print("-" * 100)
    
    for metric_name, data in comparison['metrics'].items():
        print(f"{metric_name:<25} "
              f"{data['beta0']:>6} "
              f"{data['beta1']:>6} "
              f"{data['beta2']:>6} "
              f"{data['mean_persistence_H0']:>10.5f} "
              f"{data['mean_persistence_H1']:>10.5f} "
              f"{data['mean_persistence_H2']:>10.5f} "
              f"{data['robust_clusters_fragile_loops']:>10.5f}")
    
    # Table 2: Class-Specific Metrics (if available)
    if show_class_metrics:
        has_class_metrics = any(
            'mean_persistence_H1_pos' in data 
            for data in comparison['metrics'].values()
        )
        
        if has_class_metrics:
            print(f"\n--- CLASS-SPECIFIC METRICS ---")
            print(f"{'Metric':<25} {'H1_pos':>10} {'H1_neg':>10} {'H0_diff':>10} {'H1_diff':>10} {'H1_sep':>10} {'Steer_Q':>10} {'Consist':>10}")
            print("-" * 100)
            
            for metric_name, data in comparison['metrics'].items():
                if 'mean_persistence_H1_pos' in data:
                    print(f"{metric_name:<25} "
                          f"{data['mean_persistence_H1_pos']:>10.5f} "
                          f"{data['mean_persistence_H1_neg']:>10.5f} "
                          f"{data['mean_persistence_H0_diff']:>10.5f} "
                          f"{data['mean_persistence_H1_diff']:>10.5f} "
                          f"{data['h1_separation']:>10.5f} "
                          f"{data['steering_quality']:>10.5f} "
                          f"{data['class_consistency']:>10.5f}")
    
    # Ranking analysis
    print(f"\n--- RANKINGS ---")
    
    # Rank by robust_clusters_fragile_loops
    ranking_robust = sorted(
        comparison['metrics'].items(),
        key=lambda x: x[1]['robust_clusters_fragile_loops'],
        reverse=True
    )
    
    print("\nBy Robust Clusters Fragile Loops (High H0, Low H1):")
    for rank, (metric, data) in enumerate(ranking_robust, 1):
        print(f"  {rank}. {metric:25s}: {data['robust_clusters_fragile_loops']:.5f}")
    
    # Rank by steering_quality (if available)
    if show_class_metrics and has_class_metrics:
        ranking_steering = sorted(
            [(m, d) for m, d in comparison['metrics'].items() if 'steering_quality' in d],
            key=lambda x: x[1]['steering_quality'],
            reverse=True
        )
        
        if ranking_steering:
            print("\nBy Steering Quality (High H0_diff, Low H1_diff):")
            for rank, (metric, data) in enumerate(ranking_steering, 1):
                print(f"  {rank}. {metric:25s}: {data['steering_quality']:.5f}")
    
    comparison['ranking_robust_clusters_fragile_loops'] = [m for m, _ in ranking_robust]
    
    return comparison