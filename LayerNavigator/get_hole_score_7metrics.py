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
    k_neighbors: int = 10  # For geodesic and density normalization
) -> Dict:
    """
    Compute persistence diagram using GUDHI
    
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
        
    Returns:
        persistence_data: {
            'diagram': list of (dimension, (birth, death)),
            'betti_numbers': {0: β0, 1: β1, 2: β2},
            'persistence_stats': {...}
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
        raise ValueError(
            f"Unknown metric: {metric}. "
            f"Valid options: 'euclidean', 'cosine', 'mahalanobis', 'geodesic', "
            f"'dens_norm_euclidean', 'dens_norm_cosine', 'dens_norm_mahalanobis'"
        )
    
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
    
    # Compute persistence statistics
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
    
    return {
        'diagram': persistence,
        'persistence_by_dim': persistence_by_dim,
        'betti_numbers': {i: betti_numbers[i] if i < len(betti_numbers) else 0 
                          for i in range(max_dimension + 1)},
        'persistence_stats': persistence_stats
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
        hole_score_info: {
            layer: {
                'purity': float,
                'separability': float,
                'beta0': int,
                'beta1': int,
                'beta2': int,
                'mean_persistence_H0': float,
                'mean_persistence_H1': float,
                'total_persistence': float,
                'tss': float  # Topological Steering Score
            }
        }
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
        
        all_acts = torch.cat(all_acts, dim=0).cpu().numpy()  # Move to CPU for GUDHI
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
        
        try:
            persistence_data = compute_persistence_diagram(
                all_acts,
                metric=metric,
                max_dimension=max_dimension,
                max_edge_length=np.inf
            )
            
            betti_numbers = persistence_data['betti_numbers']
            persistence_stats = persistence_data['persistence_stats']
            
            # Extract key metrics
            beta0 = betti_numbers.get(0, 0)
            beta1 = betti_numbers.get(1, 0)
            beta2 = betti_numbers.get(2, 0)
            
            mean_pers_h0 = persistence_stats.get('H0_mean', 0.0)
            mean_pers_h1 = persistence_stats.get('H1_mean', 0.0)
            total_pers = sum([persistence_stats.get(f'H{d}_total', 0.0) 
                             for d in range(max_dimension + 1)])
            
        except Exception as e:
            print(f"  Warning: Persistence computation failed for layer {l}: {e}")
            beta0, beta1, beta2 = 0, 0, 0
            mean_pers_h0, mean_pers_h1, total_pers = 0.0, 0.0, 0.0
        
        # === TOPOLOGICAL STEERING SCORE (TSS) ===
        # Weighted combination of metrics
        # Higher purity = better
        # Higher separability = better
        # Higher H0 persistence = more stable features
        # Lower β1 = less entanglement (penalize loops)
        
        entanglement_penalty = 1.0 / (1.0 + 0.1 * beta1)  # Reduce score if loops exist
        
        tss = (
            0.35 * purity +
            0.35 * separability +
            0.30 * mean_pers_h0
        ) * entanglement_penalty
        
        # Store results
        hole_score_info[l] = {
            # Clustering quality
            'purity': float(purity),
            'separability': float(separability),
            
            # Topological features
            'beta0': int(beta0),
            'beta1': int(beta1),
            'beta2': int(beta2),
            
            # Persistence measures
            'mean_persistence_H0': float(mean_pers_h0),
            'mean_persistence_H1': float(mean_pers_h1),
            'total_persistence': float(total_pers),
            
            # Combined score
            'tss': float(tss),
            
            # Metadata
            'metric': metric,
            'n_samples': int(len(all_acts))
        }
        
        # Save per-layer results
        with open(f"{save_root}L{l}.json", "w") as f:
            json.dump(hole_score_info[l], f, indent=4)
        
        print(f"  Layer {l}: TSS={tss:.3f}, Purity={purity:.3f}, Sep={separability:.3f}, β1={beta1}")
    
    # Save complete results
    with open(f"{save_root}all_layers.json", "w") as f:
        json.dump(hole_score_info, f, indent=4)
    
    print(f"HOLE scores saved to: {save_root}")
    
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
        print(f"{rank}. {metric:25s}: TSS={scores['tss']:.3f}, β₁={scores['beta1']}")
    
    return comparison