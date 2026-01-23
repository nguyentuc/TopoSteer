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


def compute_clustering_purity(activations: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute clustering purity using connected components from H0
    
    HOLE paper: "Clustering purity dropped from 0.87 to 0.62 under speckle noise"
    
    Args:
        activations: (N, hidden_dim)
        labels: (N,) - class labels
        
    Returns:
        purity: float in [0, 1]
    """
    from sklearn.cluster import AgglomerativeClustering
    
    n_clusters = len(np.unique(labels))
    
    # Use same clustering as topological analysis would suggest
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    pred_labels = clustering.fit_predict(activations)
    
    # Compute purity
    purity = 0
    for cluster in range(n_clusters):
        cluster_mask = pred_labels == cluster
        if cluster_mask.sum() == 0:
            continue
        # Find most common true label in this cluster
        true_labels_in_cluster = labels[cluster_mask]
        most_common = np.bincount(true_labels_in_cluster.astype(int)).argmax()
        purity += (true_labels_in_cluster == most_common).sum()
    
    purity /= len(labels)
    return purity


def compute_class_separability(activations: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute class separability using mean inter-class vs intra-class distance
    
    Args:
        activations: (N, hidden_dim)
        labels: (N,) - class labels
        
    Returns:
        separability: float (higher = better separation)
    """
    unique_labels = np.unique(labels)
    
    # Compute class centroids
    centroids = {}
    for label in unique_labels:
        centroids[label] = activations[labels == label].mean(axis=0)
    
    # Inter-class distance (between centroids)
    if len(unique_labels) == 2:
        inter_class_dist = np.linalg.norm(centroids[unique_labels[0]] - centroids[unique_labels[1]])
    else:
        # Multiple classes: average pairwise centroid distances
        centroid_array = np.array([centroids[l] for l in unique_labels])
        inter_class_dist = pdist(centroid_array).mean()
    
    # Intra-class distance (within each class)
    intra_class_dists = []
    for label in unique_labels:
        class_acts = activations[labels == label]
        if len(class_acts) > 1:
            centroid = centroids[label]
            distances = np.linalg.norm(class_acts - centroid, axis=1)
            intra_class_dists.append(distances.mean())
    
    intra_class_dist = np.mean(intra_class_dists) if intra_class_dists else 1.0
    
    # Separability ratio
    separability = inter_class_dist / (intra_class_dist + 1e-8)
    
    return separability


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
    # STEP 4: COMPUTE PERSISTENCE ENTROPY 
    # Persistence Entropy = -sum(p_i × log(p_i)) where p_i = (persistence of feature i) / (total persistence)
    # Uniform structure -> reliable steering
    # Low entropy -> Uniform, reliable topological structure -> Better steering
    # High entropy -> Mixed feature quality -> Unreliable steering
    # Combine:  tss *= (1.0 - normalized_entropy)
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
                # Compute entropy: H = -Σ(p_i * log(p_i))
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
    weighted_entanglement = beta1 * mean_h1_persistence
    
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
        'strong_loops_count': strong_loops_count,
        'weighted_entanglement': weighted_entanglement,
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
            all_labels.append(torch.ones(all_acts[i].shape[0]) * i)
        
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
        
        # === TOPOLOGICAL METRICS ===
        
        # 1. Clustering Purity
        purity = compute_clustering_purity(all_acts, all_labels)
        
        # 2. Class Separability
        separability = compute_class_separability(all_acts, all_labels)
        
        # 3. Persistent Homology
        print(f"  Layer {l}: Computing persistent homology with {metric} distance...")
    
        persistence_data = compute_persistence_diagram(
            all_acts,
            metric=metric,
            max_dimension=max_dimension,
            max_edge_length=np.inf
        )
        
        # Extract all computed metrics
        betti_numbers = persistence_data['betti_numbers']
        persistence_stats = persistence_data['persistence_stats']
        persistence_entropy = persistence_data['persistence_entropy']
        betti_curve_auc = persistence_data['betti_curve_auc']
        strong_loops_count = persistence_data['strong_loops_count']
        weighted_entanglement = persistence_data['weighted_entanglement']
        persistence_by_dim = persistence_data['persistence_by_dim']
        
        # Extract Betti numbers
        beta0 = betti_numbers.get(0, 0)
        beta1 = betti_numbers.get(1, 0)
        beta2 = betti_numbers.get(2, 0)
        
        # Extract basic persistence statistics for all dimensions
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
        
        total_pers = sum([persistence_stats.get(f'H{d}_total', 0.0) 
                         for d in range(max_dimension + 1)])
        
        # Extract raw entropy (no normalization)
        entropy_h0 = persistence_entropy.get(0, 0.0)
        entropy_h1 = persistence_entropy.get(1, 0.0)
        entropy_h2 = persistence_entropy.get(2, 0.0)
        
        auc_h0 = betti_curve_auc.get(0, 0.0)
        auc_h1 = betti_curve_auc.get(1, 0.0)
        auc_h2 = betti_curve_auc.get(2, 0.0)
        
        # Get finite persistence pairs for fragility analysis
        finite_h0_pairs = [(b, d) for b, d in persistence_by_dim[0] if d != np.inf]
        finite_h1_pairs = [(b, d) for b, d in persistence_by_dim[1] if d != np.inf]
        finite_h2_pairs = [(b, d) for b, d in persistence_by_dim[2] if d != np.inf]

        # =====================================================================
        # COMPUTE PENALTIES AND FACTORS
        # =====================================================================

        # Basic penalties
        entanglement_penalty = 1.0 / (1.0 + 0.1 * beta1)  # beta1 count penalty
        beta1_penalty = entanglement_penalty  # Alias for clarity
        beta2_penalty = 1.0 / (1.0 + 0.05 * beta2)  # beta2 count penalty

        # Advanced penalties
        auc_penalty = 1.0 / (1.0 + 0.1 * auc_h1)  # Multi-scale beta1
        strong_loops_penalty = 1.0 / (1.0 + 0.2 * strong_loops_count)  # Real loops only
        weighted_penalty = 1.0 / (1.0 + 0.05 * weighted_entanglement)  # beta1 × persistence

        # Raw entropy scores (no normalization)
        # Lower entropy = more uniform = better
        h0_entropy_score = entropy_h0
        h1_entropy_score = entropy_h1
        h2_entropy_score = entropy_h2

        # Beta0 score (target = 2 for binary)
        beta0_score = 1.0 - min(abs(beta0 - 2.0) / 10.0, 1.0)

        # Topological simplicity
        topological_simplicity = (
            beta0_score +                    # Right number of components
            (1.0 / (1.0 + beta1)) +         # Few loops
            (1.0 / (1.0 + beta2))           # Few voids
        ) / 3.0

        # =====================================================================
        # CATEGORY 1: BASELINE VARIANTS
        # =====================================================================

        # Base components (no weighting)
        base_components = purity + separability + mean_pers_h0

        # VARIANT 1: No penalty
        tss_no_pen = base_components

        # VARIANT 2: Original with beta1 penalty (unweighted)
        tss = base_components * entanglement_penalty

        # VARIANT 3: Weighted (0.35, 0.35, 0.30)
        tss_weighted = (
            0.35 * purity +
            0.35 * separability +
            0.30 * mean_pers_h0
        ) * entanglement_penalty

        # VARIANT 4: Uniform weighted (1/3, 1/3, 1/3)
        tss_uniform_weighted = (base_components / 3.0) * entanglement_penalty

        # =====================================================================
        # CATEGORY 2: ALTERNATIVE PENALTIES
        # =====================================================================

        # VARIANT 5: Multi-scale entanglement (beta1 AUC)
        tss_beta1_auc = base_components * auc_penalty
        tss_beta1_auc_uniform = (base_components / 3.0) * auc_penalty

        # VARIANT 6: Strong loops only
        tss_strong_loops = base_components * strong_loops_penalty
        tss_strong_loops_uniform = (base_components / 3.0) * strong_loops_penalty

        # VARIANT 7: Weighted entanglement
        tss_weighted_entanglement = base_components * weighted_penalty
        tss_weighted_entanglement_uniform = (base_components / 3.0) * weighted_penalty

        # VARIANT 8: Entropy-enhanced (using raw H1 entropy)
        # Lower entropy = better, so use inverse
        entropy_factor = 1.0 / (1.0 + entropy_h1) if entropy_h1 > 0 else 1.0
        tss_with_entropy = (base_components / 3.0) * entanglement_penalty * entropy_factor

        # VARIANT 9: Comprehensive (all penalties combined)
        comprehensive_penalty = (
            entanglement_penalty * 0.25 +      # beta1 count
            auc_penalty * 0.25 +                # Multi-scale entanglement
            strong_loops_penalty * 0.25 +       # Real vs noise loops
            entropy_factor * 0.25               # Feature uniformity
        )
        tss_comprehensive_uniform = (base_components / 3.0) * comprehensive_penalty
        tss_comprehensive = base_components * comprehensive_penalty

        # =====================================================================
        # CATEGORY 3: EXPLICIT BETTI NUMBER FORMULATIONS
        # =====================================================================

        # VARIANT 10: beta0-aware (target beta0=2 for binary classification)
        tss_beta0_aware = (
            0.25 * purity +
            0.25 * separability +
            0.25 * mean_pers_h0 +
            0.25 * beta0_score
        ) * entanglement_penalty

        # VARIANT 11: Multi-Betti penalty (beta1 + beta2)
        multi_betti_penalty = (beta1_penalty + beta2_penalty) / 2.0
        tss_multi_betti = (base_components / 3.0) * multi_betti_penalty

        # VARIANT 12: beta2 penalty included
        tss_with_beta2 = base_components * beta1_penalty * beta2_penalty

        # =====================================================================
        # CATEGORY 4: PERSISTENCE-CENTRIC VARIANTS
        # =====================================================================

        # VARIANT 13: Max persistence instead of mean
        tss_max_persistence_h0 = (
            (purity + separability + max_pers_h0) / 3.0
        ) * entanglement_penalty

        # VARIANT 14: Total persistence (cumulative stability)
        normalized_total_pers_h0 = min(total_pers_h0 / 10.0, 1.0)
        tss_total_persistence = (
            (purity + separability + normalized_total_pers_h0) / 3.0
        ) * entanglement_penalty

        # VARIANT 15: Multi-scale persistence (H₀ AUC)
        normalized_auc_h0 = min(auc_h0 / 50.0, 1.0)
        tss_h0_auc = (
            (purity + separability + normalized_auc_h0) / 3.0
        ) * entanglement_penalty

        # =====================================================================
        # CATEGORY 5: PERSISTENCE HYPOTHESIS VARIANTS (OLD)
        # =====================================================================

        # VARIANT 16: H0 persistence only (robust clustering)
        h0_persistence_only = mean_pers_h0

        # VARIANT 17: H1 persistence only
        h1_persistence_only = mean_pers_h1

        # VARIANT 18: H2 persistence only
        h2_persistence_only = mean_pers_h2

        # VARIANT 19: All persistence (weighted) - OLD VERSION
        all_persistence_weighted_old = (
            mean_pers_h0 +
            mean_pers_h1 +
            mean_pers_h2
        ) / 3.0

        # VARIANT 20: Paradoxical - robust clusters, fragile loops
        # Hypothesis: Want HIGH H0 persistence, LOW H1 persistence
        h1_fragility_bonus = 1.0 / (1.0 + mean_pers_h1) if mean_pers_h1 > 0 else 1.0
        tss_robust_clusters_fragile_loops = mean_pers_h0 * h1_fragility_bonus

        # VARIANT 21: Paradoxical - robust clusters, fragile voids
        h2_fragility_bonus = 1.0 / (1.0 + mean_pers_h2) if mean_pers_h2 > 0 else 1.0
        tss_robust_clusters_fragile_voids = mean_pers_h0 * h2_fragility_bonus

        # VARIANT 22: Good persistence vs bad persistence
        # H0 persistence = good, H1+H2 persistence = bad
        bad_persistence = mean_pers_h1 + mean_pers_h2
        tss_good_vs_bad_persistence = max(0.0, mean_pers_h0 - 0.5 * bad_persistence)

        # =====================================================================
        # CATEGORY 6: THEORY-DRIVEN ALTERNATIVES
        # =====================================================================

        # VARIANT 23: Topological simplicity
        tss_topological_simplicity = (
            (separability + mean_pers_h0 + topological_simplicity) / 3.0
        )

        # VARIANT 24: Geometric-only (no topology)
        tss_geometric_only = 0.5 * purity + 0.5 * separability

        # VARIANT 25: Topology-only (no geometric clustering)
        tss_topology_only = mean_pers_h0 * entanglement_penalty

        # VARIANT 26: Pure persistence (selective)
        # H0 up, H1 down, H2 down
        tss_pure_persistence = (
            0.5 * mean_pers_h0 +
            0.3 * (1.0 / (1.0 + mean_pers_h1 + 1e-8)) +
            0.2 * (1.0 / (1.0 + mean_pers_h2 + 1e-8))
        )

        # =====================================================================
        # CATEGORY 7: HYPOTHESIS-ALIGNED VARIANTS (UPDATED)
        # =====================================================================
        
        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRIC BASELINES (Control - not topological)
        # ─────────────────────────────────────────────────────────────────────
        
        # VARIANT 27: Purity only (GEOMETRIC BASELINE)
        purity_only = purity
        
        # VARIANT 28: Separability only (GEOMETRIC BASELINE)
        separability_only = separability
        
        # ─────────────────────────────────────────────────────────────────────
        # PERSISTENCE-BASED VARIANTS (Directly test hypothesis)
        # ─────────────────────────────────────────────────────────────────────
        
        # VARIANT 29: H₀ persistence (LONG-LIVED CLUSTERS)
        persistence_h0_only = mean_pers_h0
        
        # VARIANT 30: H₁ persistence (LONG-LIVED LOOPS) - UPDATED
        persistence_h1_only = mean_pers_h1
        
        # VARIANT 31: H₂ persistence (LONG-LIVED VOIDS) - UPDATED
        persistence_h2_only = mean_pers_h2
        
        # VARIANT 32: All persistence weighted (COMPREHENSIVE LONGEVITY) - UPDATED
        all_persistence_weighted = (
            0.5 * mean_pers_h0 +  # Clusters (most important)
            0.3 * mean_pers_h1 +  # Loops (secondary)
            0.2 * mean_pers_h2    # Voids (tertiary)
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # MULTI-SCALE ROBUSTNESS (Scale-invariance from hypothesis)
        # ─────────────────────────────────────────────────────────────────────
        
        # VARIANT 33: H₀ multi-scale robustness (SCALE-INVARIANT CLUSTERS) - UPDATED
        h0_scale_invariance = auc_h0
        
        # VARIANT 34: H₁ multi-scale robustness (SCALE-INVARIANT LOOPS) - UPDATED
        h1_scale_invariance = auc_h1
        
        # VARIANT 35: H₂ multi-scale robustness (SCALE-INVARIANT VOIDS)
        h2_scale_invariance = auc_h2
        
        # VARIANT 36: Multi-scale robustness (ALL DIMENSIONS)
        multi_scale_robustness = (
            0.5 * auc_h0 +  # Cluster scale-invariance
            0.3 * auc_h1 +  # Loop scale-invariance
            0.2 * auc_h2    # Void scale-invariance
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # ROBUSTNESS vs FRAGILITY (Direct hypothesis test)
        # ─────────────────────────────────────────────────────────────────────
        
        # VARIANT 37: Strong features only (ROBUST TOPOLOGICAL FEATURES)
        strong_features_score = float(strong_loops_count)
        
        # VARIANT 38: Fragility detection (PENALIZE SHORT-LIVED FEATURES)
        if len(finite_h0_pairs) > 0:
            h0_persistences = np.array([d - b for b, d in finite_h0_pairs])
            max_h0_pers = np.max(h0_persistences) if len(h0_persistences) > 0 else 1.0
            fragility_threshold = 0.1 * max_h0_pers
            fragile_features_count = int(np.sum(h0_persistences < fragility_threshold))
            fragility_penalty_score = 1.0 / (1.0 + fragile_features_count)
        else:
            fragile_features_count = 0
            fragility_penalty_score = 1.0
        
        robustness_only = fragility_penalty_score
        
        # VARIANT 39: Persistence dominance ratio (LONG-LIVED DOMINANCE)
        persistence_dominance_h0 = max_pers_h0 / (mean_pers_h0 + 1e-8)
        
        # VARIANT 40: Persistence spread (UNIFORMLY ROBUST)
        total_persistence_normalized = min(total_pers_h0 / 10.0, 1.0)
        
        # ─────────────────────────────────────────────────────────────────────
        # ENTROPY-BASED (Feature quality uniformity)
        # ─────────────────────────────────────────────────────────────────────
        
        # VARIANT 41: H₀ persistence entropy (RAW - lower = more uniform)
        h0_entropy_only = entropy_h0
        
        # VARIANT 42: H₁ persistence entropy (RAW - lower = more uniform)
        h1_entropy_only = entropy_h1
        
        # VARIANT 43: H₂ persistence entropy (RAW - lower = more uniform)
        h2_entropy_only = entropy_h2
        
        # ─────────────────────────────────────────────────────────────────────
        # COMBINED PERSISTENCE METRICS (Sophisticated hypothesis tests)
        # ─────────────────────────────────────────────────────────────────────
        
        # VARIANT 44: Persistence × Purity (ROBUST GEOMETRY + TOPOLOGY)
        persistence_geometric_combined = mean_pers_h0 * purity
        
        # VARIANT 45: Persistence × Separability (ROBUST SEPARATION)
        persistence_separation_combined = mean_pers_h0 * separability
        
        # VARIANT 46: Max persistence focus (MOST ROBUST FEATURE)
        max_persistence_score = (
            0.5 * max_pers_h0 +
            0.3 * max_pers_h1 +
            0.2 * max_pers_h2
        )
        
        # VARIANT 47: Weighted entanglement (PERSISTENT STRUCTURE)
        weighted_entanglement_score = weighted_entanglement
        
        # ─────────────────────────────────────────────────────────────────────
        # HYPOTHESIS VARIANTS: CONTRADICTORY PREDICTIONS
        # ─────────────────────────────────────────────────────────────────────
        
        # VARIANT 48: Anti-persistence (CONTROL - opposite hypothesis)
        anti_persistence = 1.0 / (mean_pers_h0 + 0.1)
        
        # VARIANT 49: Fragility reward (CONTROL - opposite hypothesis)
        fragility_reward = float(fragile_features_count)
        
        # VARIANT 50: Beta counts only (CONTROL - count vs persistence)
        beta_counts_combined = float(beta0 + beta1 + beta2)

        # =====================================================================
        # LEGACY VARIANTS (For backward compatibility)
        # =====================================================================
        
        # These use OLD formulas (penalties instead of rewards)
        beta1_only = 1.0 / (1.0 + 0.1 * beta1)
        beta1_auc_only = 1.0 / (1.0 + 0.1 * auc_h1)
        beta2_only = 1.0 / (1.0 + 0.1 * beta2)
        beta0_only = beta0_score
        strong_loops_only = 1.0 / (1.0 + 0.2 * strong_loops_count)
        entropy_only = entropy_factor  # Uses inverse formula

        # =====================================================================
        # STORE ALL RESULTS
        # =====================================================================

        hole_score_info[l] = {
            # ===== CLUSTERING QUALITY =====
            'purity': float(purity),
            'separability': float(separability),

            # ===== BETTI NUMBERS =====
            'beta0': int(beta0),
            'beta1': int(beta1),
            'beta2': int(beta2),

            # ===== BASIC PERSISTENCE STATISTICS =====
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

            # Total across all dimensions
            'total_persistence': float(total_pers),

            # ===== PERSISTENCE ENTROPY (RAW VALUES) =====
            'persistence_entropy_H0': float(entropy_h0),
            'persistence_entropy_H1': float(entropy_h1),
            'persistence_entropy_H2': float(entropy_h2),

            # ===== BETTI CURVE AUC =====
            'betti_curve_auc_H0': float(auc_h0),
            'betti_curve_auc_H1': float(auc_h1),
            'betti_curve_auc_H2': float(auc_h2),

            # ===== H1 ENHANCED STATISTICS =====
            'strong_loops_count': int(strong_loops_count),
            'weighted_entanglement': float(weighted_entanglement),
            'fragile_features_count': int(fragile_features_count),

            # ===== CATEGORY 1: BASELINE VARIANTS =====
            'tss_no_pen': float(tss_no_pen),
            'tss': float(tss),
            'tss_weighted': float(tss_weighted),
            'tss_uniform_weighted': float(tss_uniform_weighted),

            # ===== CATEGORY 2: ALTERNATIVE PENALTIES =====
            'tss_beta1_auc': float(tss_beta1_auc),
            'tss_beta1_auc_uniform': float(tss_beta1_auc_uniform),
            'tss_strong_loops': float(tss_strong_loops),
            'tss_strong_loops_uniform': float(tss_strong_loops_uniform),
            'tss_weighted_entanglement': float(tss_weighted_entanglement),
            'tss_weighted_entanglement_uniform': float(tss_weighted_entanglement_uniform),
            'tss_with_entropy': float(tss_with_entropy),
            'tss_comprehensive': float(tss_comprehensive),
            'tss_comprehensive_uniform': float(tss_comprehensive_uniform),

            # ===== CATEGORY 3: EXPLICIT BETTI FORMULATIONS =====
            'tss_beta0_aware': float(tss_beta0_aware),
            'tss_multi_betti': float(tss_multi_betti),
            'tss_with_beta2': float(tss_with_beta2),

            # ===== CATEGORY 4: PERSISTENCE-CENTRIC =====
            'tss_max_persistence_h0': float(tss_max_persistence_h0),
            'tss_total_persistence': float(tss_total_persistence),
            'tss_h0_auc': float(tss_h0_auc),

            # ===== CATEGORY 5: PERSISTENCE HYPOTHESIS (OLD) =====
            'h0_persistence_only': float(h0_persistence_only),
            'h1_persistence_only': float(h1_persistence_only),
            'h2_persistence_only': float(h2_persistence_only),
            'all_persistence_weighted_old': float(all_persistence_weighted_old),
            'tss_robust_clusters_fragile_loops': float(tss_robust_clusters_fragile_loops),
            'tss_robust_clusters_fragile_voids': float(tss_robust_clusters_fragile_voids),
            'tss_good_vs_bad_persistence': float(tss_good_vs_bad_persistence),

            # ===== CATEGORY 6: THEORY-DRIVEN =====
            'tss_topological_simplicity': float(tss_topological_simplicity),
            'tss_geometric_only': float(tss_geometric_only),
            'tss_topology_only': float(tss_topology_only),
            'tss_pure_persistence': float(tss_pure_persistence),

            # ===== CATEGORY 7: HYPOTHESIS-ALIGNED VARIANTS (NEW) =====
            # Geometric baselines
            'purity_only': float(purity_only),
            'separability_only': float(separability_only),
            
            # Persistence by dimension
            'persistence_h0_only': float(persistence_h0_only),
            'persistence_h1_only': float(persistence_h1_only),
            'persistence_h2_only': float(persistence_h2_only),
            'all_persistence_weighted': float(all_persistence_weighted),
            
            # Multi-scale robustness
            'h0_scale_invariance': float(h0_scale_invariance),
            'h1_scale_invariance': float(h1_scale_invariance),
            'h2_scale_invariance': float(h2_scale_invariance),
            'multi_scale_robustness': float(multi_scale_robustness),
            
            # Robustness vs fragility
            'strong_features_score': float(strong_features_score),
            'robustness_only': float(robustness_only),
            'persistence_dominance_h0': float(persistence_dominance_h0),
            'total_persistence_normalized': float(total_persistence_normalized),
            
            # Entropy (raw values)
            'h0_entropy_only': float(h0_entropy_only),
            'h1_entropy_only': float(h1_entropy_only),
            'h2_entropy_only': float(h2_entropy_only),
            
            # Combined
            'persistence_geometric_combined': float(persistence_geometric_combined),
            'persistence_separation_combined': float(persistence_separation_combined),
            'max_persistence_score': float(max_persistence_score),
            'weighted_entanglement_score': float(weighted_entanglement_score),
            
            # Controls (opposite hypothesis)
            'anti_persistence': float(anti_persistence),
            'fragility_reward': float(fragility_reward),
            'beta_counts_combined': float(beta_counts_combined),
            
            # ===== LEGACY VARIANTS (OLD FORMULAS) =====
            'beta1_only': float(beta1_only),
            'beta1_auc_only': float(beta1_auc_only),
            'beta2_only': float(beta2_only),
            'beta0_only': float(beta0_only),
            'strong_loops_only': float(strong_loops_only),
            'entropy_only': float(entropy_only),

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
        # 'dens_norm_euclidean',
        # 'dens_norm_cosine',
        # 'dens_norm_mahalanobis'
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
                'purity': scores[layer_idx]['purity'],
                'separability': scores[layer_idx]['separability'],
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