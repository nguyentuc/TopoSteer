# get_hole_score.py
"""
HOLE (Homological Observation of Latent Embeddings) Metric Computation
Companion to LayerNavigator's get_score.py
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
            'strong_loops_count': int, # H1 features with persistence > threshold
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
                persistences = [d - b for b, d in finite_pairs] # compute the living period of each pair
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
    # Metric 4: PERSISTENCE ENTROPY (on each dimension d)
    # Persistence Entropy = -sum(p_i × log(p_i)) where p_i = (persistence of feature i) / (total persistence)
    # Uniform structure -> reliable steering
    # Low entropy -> Uniform, reliable topological structure -> Better steering
    # High entropy -> Mixed feature quality -> Unreliable steering
    # ============================================
    
    persistence_entropy = {}
    for dim in range(max_dimension + 1):
        pairs = persistence_by_dim[dim]
        finite_pairs = [(b, d) for b, d in pairs if d != np.inf]
        
        if len(finite_pairs) > 1:
            persistences = np.array([d - b for b, d in finite_pairs])
            total_persistence = np.sum(persistences)
            
            if total_persistence > 1e-10:  # Avoid division by zero
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
    # beta_i only at final filtration scale, missing how entanglement evolves across scales.
    # betti show number of loops at filtration threshold epsilon
    # entanglement_penalty = 1.0 / (1.0 + 0.1 * beta1_AUC)
    # ============================================
    filtration_list = list(simplex_tree.get_filtration())
    if len(filtration_list) > 0:
        max_filtration = filtration_list[-1][1]  # Last simplex's filtration value
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
            # Count features where birth <= epsilon < death
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
        
        # Define "strong loop" threshold (e.g., 50th percentile or absolute threshold)
        if len(h1_persistences) > 0:
            # Use median as threshold, or you can use a fixed value like 0.5
            strong_loop_threshold = np.median(h1_persistences) if len(h1_persistences) > 1 else h1_persistences[0]
            strong_loops_count = int(np.sum(h1_persistences > strong_loop_threshold))
        else:
            strong_loops_count = 0
    else:
        strong_loops_count = 0
    
    # Compute weighted entanglement score: beta_1 weighted by mean H₁ persistence
    mean_h1_persistence = persistence_stats.get('H1_mean', 0.0)
    beta1 = betti_numbers[1] if len(betti_numbers) > 1 else 0
    
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
        
        # NEW METRIC 1: Persistence Entropy
        'persistence_entropy': persistence_entropy,
        
        # NEW METRIC 3: Betti Curves and AUC
        'betti_curves': betti_curves,
        'betti_curve_auc': betti_curve_auc,
        
        # NEW METRIC 4: H1 Enhanced Statistics
        'strong_loops_count': strong_loops_count
    }


# ============================================
# HELPER FUNCTION FOR METRIC 2: WASSERSTEIN DISTANCE
# ============================================

def compute_wasserstein_distance(diagram1: Dict, diagram2: Dict, 
                                  dimension: int = 1, order: int = 2) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams
    
    This is NEW METRIC 2 - used to compare topology between layers
    
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

# Current layers are analyzed independently, missing how topology changes across layers.
# Compute topology similarity across adjacent layers
# W(layer_14, layer_15) = 0.05  # Small change (stable region)
# W(layer_16, layer_17) = 0.38  # Large value (transition region)

def compute_layer_stability(layer_diagrams: List[Dict], 
                            dimension: int = 1,
                            order: int = 2) -> Dict:
    """
    Compute topological stability across layers using Wasserstein distances
    
    Args:
        layer_diagrams: List of persistence diagrams for consecutive layers
        dimension: Homology dimension to analyze
        order: Wasserstein order
        
    Returns:
        stability_stats: {
            'wasserstein_distances': list of distances between adjacent layers,
            'avg_stability': average stability (lower Wasserstein = more stable),
            'max_transition': maximum topological change,
            'stable_regions': list of (start_layer, end_layer) for stable regions
        }
    """
    wasserstein_distances = []
    
    # Compute Wasserstein distance between consecutive layers
    for i in range(len(layer_diagrams) - 1):
        dist = compute_wasserstein_distance(
            layer_diagrams[i], 
            layer_diagrams[i + 1],
            dimension=dimension,
            order=order
        )
        wasserstein_distances.append(dist)
    
    if not wasserstein_distances:
        return {
            'wasserstein_distances': [],
            'avg_stability': 0.0,
            'max_transition': 0.0,
            'stable_regions': []
        }
    
    # Compute statistics
    avg_stability = np.mean(wasserstein_distances)
    max_transition = np.max(wasserstein_distances)
    
    # Identify stable regions (where Wasserstein distance is below threshold)
    stability_threshold = np.median(wasserstein_distances)
    stable_regions = []
    
    current_region_start = 0
    for i, dist in enumerate(wasserstein_distances):
        if dist > stability_threshold:
            # End of stable region
            if i > current_region_start:
                stable_regions.append((current_region_start, i))
            current_region_start = i + 1
    
    # Add final region if stable
    if len(wasserstein_distances) > current_region_start:
        stable_regions.append((current_region_start, len(wasserstein_distances)))
    
    return {
        'wasserstein_distances': wasserstein_distances,
        'avg_stability': float(avg_stability),
        'max_transition': float(max_transition),
        'stable_regions': stable_regions
    }

# main function that compute all persistent homology based distance
def get_hole_score(
    layers: List[int],
    dataset,  # UniDataset from LayerNavigator
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    metric: str = "cosine",
    max_dimension: int = 2,
    subsample: int = None  # Subsample for computational efficiency
):
    """
    Compute HOLE metrics for each layer
    
    Following LayerNavigator's get_score structure but with topological metrics
    
    Args:
        layers: List of layer indices to analyze
        dataset: Dataset object (must have train=True)
        vec_task: Task name (e.g., 'sycophancy')
        vec_method: Vector extraction method
        acts_pre: Preprocessing method ('standard' for z-score)
        metric: Distance metric - one of:
            - 'euclidean': Euclidean distance (ℓ₂-norm)
            - 'cosine': Cosine distance (directional similarity)
            - 'mahalanobis': Mahalanobis distance (covariance-aware)
            - 'geodesic': Geodesic distance via k-NN graph
            - 'dens_norm_euclidean': Density-normalized Euclidean
            - 'dens_norm_cosine': Density-normalized Cosine
            - 'dens_norm_mahalanobis': Density-normalized Mahalanobis
        max_dimension: Maximum homology dimension (0, 1, 2)
        subsample: Subsample N points for efficiency (None = use all)
        
    Returns:
        hole_score_info: Dict with comprehensive TSS variants
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
    
    # Load activations (same as LayerNavigator)
    print("Loading activations...")
    acts = torch.load(f"{vec_root}/acts.pt")
    
    hole_score_info = {}
    
    # Process each layer
    for l in tqdm(layers, desc="Computing HOLE Scores"):
        # Prepare activations and labels
        all_acts = []
        all_labels = []
        
        for i in range(ans_num):
            all_acts.append(torch.stack(acts[i][l]))
            all_labels.append(torch.ones(all_acts[i].shape[0]) * i) # activation vector is saved by label
        
        # Tuc: for Llama3
        # all_acts = torch.cat(all_acts, dim=0).cpu().numpy()  # Move to CPU for GUDHI
        # TUC: for Qwen
        all_acts = torch.cat(all_acts, dim=0).float().cpu().numpy()  # Convert to float32, move to CPU for GUDHI
        
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        
        # Subsample if specified (for large datasets)
        if subsample is not None and len(all_acts) > subsample:
            indices = np.random.choice(len(all_acts), subsample, replace=False)
            all_acts = all_acts[indices]
            all_labels = all_labels[indices]
        
        # Normalize activations (same as LayerNavigator)
        if acts_pre == "standard":
            all_acts = (all_acts - all_acts.mean(axis=0)) / (all_acts.std(axis=0) + 1e-8)
        
        # Persistent Homology
        print(f"  Layer {l}: Computing persistent homology with {metric} distance...")
    
        persistence_data = compute_persistence_diagram(
            all_acts,
            metric=metric,
            max_dimension=max_dimension,
            max_edge_length=np.inf
        )
        
        # Extract all computed metrics from persistent data
        betti_numbers = persistence_data['betti_numbers']
        persistence_stats = persistence_data['persistence_stats']
        persistence_entropy = persistence_data['persistence_entropy']
        betti_curve_auc = persistence_data['betti_curve_auc']
        strong_loops_count = persistence_data['strong_loops_count']
        persistence_by_dim = persistence_data['persistence_by_dim']
        
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
        # Hypothesis: Want HIGH H0 persistence, LOW H1 persistence
        h1_fragility_bonus = 1.0 / (1.0 + mean_pers_h1) if mean_pers_h1 > 0 else 1.0
        robust_clusters_fragile_loops = mean_pers_h0 * h1_fragility_bonus
        # Robust clusters, fragile voids
        h2_fragility_bonus = 1.0 / (1.0 + mean_pers_h2) if mean_pers_h2 > 0 else 1.0
        robust_clusters_fragile_voids = mean_pers_h0 * h2_fragility_bonus
        # H0 up, H1 down, H2 down
        persistence_inverseH1_inverseH2 = (
            mean_pers_h0 +
            (1.0 / (1.0 + mean_pers_h1 + 1e-8)) +
            (1.0 / (1.0 + mean_pers_h2 + 1e-8))
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
        
        # G5: Strong loop count (number of loop h1 with high persistent)
        strong_loops_count = float(strong_loops_count)
        # Inverse number of strong loop count
        inverse_strong_loops_count = 1.0 / (1.0 + strong_loops_count)
        
        # Strong cluster
        finite_h0_pairs = [(b, d) for b, d in persistence_by_dim[0] if d != np.inf]
        if finite_h0_pairs:
            h0_persistences = np.array([d - b for b, d in finite_h0_pairs])
            if len(h0_persistences) > 0:
                # Use median as threshold, or you can use a fixed value like 0.5
                strong_threshold = np.median(h0_persistences) if len(h0_persistences) > 1 else h0_persistences[0]
                strong_cluster_count = int(np.sum(h0_persistences > strong_threshold))
            else:
                strong_cluster_count = 0
        else:
            strong_cluster_count = 0
        # Inverse number of strong loop count
        inverse_strong_cluster_count = 1.0 / (1.0 + strong_cluster_count)
        
        # Strong voids
        finite_h2_pairs = [(b, d) for b, d in persistence_by_dim[2] if d != np.inf]
        if finite_h2_pairs:
            h2_persistences = np.array([d - b for b, d in finite_h2_pairs])
            if len(h2_persistences) > 0:
                # Use median as threshold, or you can use a fixed value like 0.5
                strong_threshold = np.median(h2_persistences) if len(h2_persistences) > 1 else h2_persistences[0]
                strong_voids_count = int(np.sum(h2_persistences > strong_threshold))
            else:
                strong_voids_count = 0
        else:
            strong_voids_count = 0
        # Inverse number of strong loop count
        inverse_strong_voids_count = 1.0 / (1.0 + strong_voids_count)
        
        
        # G6: Persistence dominance ratio (LONG-LIVED DOMINANCE)
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
        
        ## All the score will be used to rank the layers
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

            # ===== G5: Strong H1 =====
            'strong_loops_count': float(strong_loops_count),
            'inverse_strong_loops_count':float(inverse_strong_loops_count),
            'strong_cluster_count': float(strong_cluster_count),
            'inverse_strong_cluster_count': float(inverse_strong_cluster_count),
            'strong_voids_count': float(strong_voids_count),
            'inverse_strong_voids_count': float(inverse_strong_voids_count),
            
            # G6:
            'portion_of_persistence_dominance_h0': float(portion_of_persistence_dominance_h0),
            'portion_of_persistence_dominance_h1': float(portion_of_persistence_dominance_h1),
            'portion_of_persistence_dominance_h2': float(portion_of_persistence_dominance_h2),
            'avg_max_persistence_score': float(avg_max_persistence_score),
            'inverse_mean_H0': float(inverse_mean_H0),
            'inverse_mean_H1': float(inverse_mean_H1),
            'inverse_mean_H2': float(inverse_mean_H2),

            # ===== METADATA =====
            'metric': metric,
            'n_samples': int(len(all_acts))
        }
        
        # Save per-layer results
        with open(f"{save_root}L{l}.json", "w") as f:
            json.dump(hole_score_info[l], f, indent=4)
    
    # Save complete results
    with open(f"{save_root}all_layers.json", "w") as f:
        json.dump(hole_score_info, f, indent=4)
    
    print(f"\nHOLE scores saved to: {save_root}")
    print(f"Total metrics stored per layer: {len(hole_score_info[layers[0]])} fields")
    
    return hole_score_info


def compare_ln_hole_scores(
    layers: List[int],
    ln_score_path: str,
    hole_score_path: str
):
    """
    Compare LayerNavigator and HOLE scores
    
    Args:
        layers: List of layer indices
        ln_score_path: Path to LayerNavigator scores
        hole_score_path: Path to HOLE scores
        
    Returns:
        comparison: Dict with correlation analysis
    """
    from scipy.stats import spearmanr, kendalltau
    
    ln_scores = []
    hole_scores = []
    
    for l in layers:
        # Load LayerNavigator score
        with open(f"{ln_score_path}/L{l}.json", "r") as f:
            ln_data = json.load(f)
            ln_scores.append(ln_data['s_score'])
        
        # Load HOLE score
        with open(f"{hole_score_path}/L{l}.json", "r") as f:
            hole_data = json.load(f)
            hole_scores.append(hole_data['tss'])
    
    # Compute correlations
    spearman_corr, spearman_p = spearmanr(ln_scores, hole_scores)
    kendall_corr, kendall_p = kendalltau(ln_scores, hole_scores)
    
    # Rank comparison
    ln_ranking = np.argsort(ln_scores)[::-1]  # Descending
    hole_ranking = np.argsort(hole_scores)[::-1]
    
    # Top-K agreement
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


def get_hole_score_all_metrics(
    layers: List[int],
    dataset,
    vec_task: str,
    vec_method: str,
    acts_pre: str = "standard",
    max_dimension: int = 2,
    subsample: int = None
):
    """
    Compute HOLE scores using ALL 7 distance metrics for comprehensive analysis
    
    This follows the HOLE paper's approach of computing multiple metrics
    to reveal different geometric and semantic aspects.
    
    Args:
        layers: List of layer indices to analyze
        dataset: Dataset object (must have train=True)
        vec_task: Task name
        vec_method: Vector extraction method
        acts_pre: Preprocessing method
        max_dimension: Maximum homology dimension
        subsample: Subsample size
        
    Returns:
        all_metric_scores: {
            metric_name: {
                layer: {score_dict}
            }
        }
    """
    all_metrics = [
        'euclidean',
        'cosine',
        'mahalanobis',
        'geodesic',
        'dens_norm_euclidean',
        'dens_norm_cosine',
        'dens_norm_mahalanobis'
    ]
    
    all_metric_scores = {}
    
    print("\n" + "="*80)
    print("COMPUTING HOLE SCORES WITH ALL 7 DISTANCE METRICS")
    print("="*80)
    
    for metric in all_metrics:
        print(f" Metric: {metric}")
        print("-" * 80)
        
        scores = get_hole_score(
            layers=layers,
            dataset=dataset,
            vec_task=vec_task,
            vec_method=vec_method,
            acts_pre=acts_pre,
            metric=metric,
            max_dimension=max_dimension,
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
    print("="*80)
    
    return all_metric_scores


def compare_metrics_for_layer(
    layer_scores: Dict,
    layer_idx: int
    ):
    """
    Compare how different metrics rank a specific layer
    
    Args:
        layer_scores: Output from get_hole_score_all_metrics()
        layer_idx: Which layer to compare
        
    Returns:
        comparison: Dict with metric comparisons
    """
    comparison = {
        'layer': layer_idx,
        'metrics': {}
    }
    
    for metric_name, scores in layer_scores.items():
        if layer_idx in scores:
            comparison['metrics'][metric_name] = {
                'tss': scores[layer_idx]['tss'],
                'beta1': scores[layer_idx]['beta1'],
                'mean_persistence_H0': scores[layer_idx]['mean_persistence_H0']
            }
    
    # Rank by TSS score
    tss_ranking = sorted(
        comparison['metrics'].items(),
        key=lambda x: x[1]['tss'],
        reverse=True
    )
    
    comparison['ranking_by_tss'] = [metric for metric, _ in tss_ranking]
    
    print(f"\nLayer {layer_idx} - TSS Scores by Metric:")
    print("-" * 60)
    for rank, (metric, scores) in enumerate(tss_ranking, 1):
        print(f"{rank}. {metric:25s}: TSS={scores['tss']:.3f}, beta_1={scores['beta1']}")
    
    return comparison