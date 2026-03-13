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
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(activations)
    distances, indices = nbrs.kneighbors(activations)
    n = len(activations)
    row_ind = np.repeat(np.arange(n), k)
    col_ind = indices.flatten()
    data = distances.flatten()
    adjacency = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    adjacency = adjacency.minimum(adjacency.T)
    geodesic_dist = shortest_path(adjacency, directed=False, method='auto')
    return geodesic_dist


def compute_local_scales(activations: np.ndarray, k: int = 10) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(activations)
    distances, _ = nbrs.kneighbors(activations)
    local_scales = distances.mean(axis=1)
    return local_scales


def apply_density_normalization(distance_matrix: np.ndarray, local_scales: np.ndarray) -> np.ndarray:
    norm_matrix = np.sqrt(np.outer(local_scales, local_scales))
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
    k_neighbors: int = 10,
    num_filtration_steps: int = 50
) -> Dict:
    if metric == 'euclidean':
        distances = pairwise_distances(activations, metric='euclidean')
    elif metric == 'cosine':
        distances = pairwise_distances(activations, metric='cosine')
    elif metric == 'mahalanobis':
        cov = np.cov(activations.T)
        inv_cov = np.linalg.pinv(cov)
        distances = pairwise_distances(activations, metric='mahalanobis', VI=inv_cov)
    elif metric == 'geodesic':
        distances = compute_geodesic_distance(activations, k=k_neighbors)
    elif metric.startswith('dens_norm_'):
        base_metric = metric.replace('dens_norm_', '')
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
        local_scales = compute_local_scales(activations, k=k_neighbors)
        distances = apply_density_normalization(distances, local_scales)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    rips_complex = gudhi.RipsComplex(distance_matrix=distances, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    persistence = simplex_tree.persistence()

    persistence_by_dim = {i: [] for i in range(max_dimension + 1)}
    for dim, (birth, death) in persistence:
        persistence_by_dim[dim].append((birth, death))

    betti_numbers = simplex_tree.betti_numbers()
    while len(betti_numbers) <= max_dimension:
        betti_numbers.append(0)

    persistence_stats = {}
    for dim in range(max_dimension + 1):
        pairs = persistence_by_dim[dim]
        if len(pairs) > 0:
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
        'persistence_stats': persistence_stats,
    }


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert dataset.train == True, "Only Use Train Dataset"

    vec_root = f"./Vectors/{vec_task}/{vec_method}"
    svec_path = vec_root[10:].replace("/", "+")

    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    metric_str = f"-{metric}"
    cloud_suffix = "_WITH_CLASS_CLOUDS" if compute_class_clouds else ""

    save_root = f"./Score_HOLE{acts_pre_str}{metric_str}{cloud_suffix}/{dataset.task}/{svec_path}/"
    os.makedirs(save_root, exist_ok=True)

    ans_num = 2

    print("Loading activations...")
    acts = torch.load(f"{vec_root}/acts.pt")

    hole_score_info = {}

    for l in tqdm(layers, desc=f"Computing HOLE Scores ({metric})"):
        # ========================================================
        # STEP 1: PREPARE ACTIVATIONS
        # ========================================================
        pos_acts_list = []
        neg_acts_list = []

        for i in range(ans_num):
            if i == 1:
                pos_acts_list.append(torch.stack(acts[i][l]))
            else:
                neg_acts_list.append(torch.stack(acts[i][l]))

        pos_acts = torch.cat(pos_acts_list, dim=0).float().cpu().numpy()
        neg_acts = torch.cat(neg_acts_list, dim=0).float().cpu().numpy()

        if subsample is not None:
            # Must use SAME indices for pos and neg to preserve per-pair correspondence
            # (pos_i - neg_i requires pos[k] and neg[k] to be the k-th contrastive pair)
            n_pairs = min(len(pos_acts), len(neg_acts))
            if n_pairs > subsample:
                shared_indices = np.random.choice(n_pairs, subsample, replace=False)
                pos_acts = pos_acts[shared_indices]
                neg_acts = neg_acts[shared_indices]
            else:
                pos_acts = pos_acts[:n_pairs]
                neg_acts = neg_acts[:n_pairs]

        combined_acts = np.vstack([pos_acts, neg_acts])

        # ========================================================
        # STEP 2: VERIFY PER-PAIR ALIGNMENT
        # ========================================================
        if compute_class_clouds:
            # Diff cloud is computed AFTER normalization in STEP 3,
            # so pos and neg must have matching length before we proceed.
            assert len(pos_acts) == len(neg_acts), (
                f"Layer {l}: pos/neg size mismatch ({len(pos_acts)} vs {len(neg_acts)}). "
                f"Per-pair diff requires aligned contrastive pairs."
            )

        # ========================================================
        # STEP 3: NORMALIZE ACTIVATIONS
        # ========================================================
        if acts_pre == "standard":
            # Z-score using combined statistics — preserves between-class geometry
            combined_mean = combined_acts.mean(axis=0)
            combined_std  = combined_acts.std(axis=0) + 1e-8
            combined_acts_norm = (combined_acts - combined_mean) / combined_std

            if compute_class_clouds:
                # pos and neg normalized in the SAME space (combined stats)
                pos_acts_norm = (pos_acts - combined_mean) / combined_std
                neg_acts_norm = (neg_acts - combined_mean) / combined_std

                # Diff computed from already-normalized activations:
                # diff_acts_norm[i] = pos_acts_norm[i] - neg_acts_norm[i]
                # No further re-scaling — diff is in the same space as combined cloud.
                diff_acts_norm = pos_acts_norm - neg_acts_norm  # shape (N, d)

                # L2-normalize each diff vector onto unit sphere (angular-only topology)
                # Near-zero vectors at early layers safely handled by +1e-8
                diff_l2_norms  = np.linalg.norm(diff_acts_norm, axis=1, keepdims=True)
                diff_acts_l2   = diff_acts_norm / (diff_l2_norms + 1e-8)

                print(f"  Layer {l}: Point clouds built:")
                print(f"    - Combined (vstack):  {len(combined_acts_norm)} points")
                print(f"    - Diff (raw norm):     {len(diff_acts_norm)} points  (pos_norm_i - neg_norm_i)")
                print(f"    - Diff (L2 unit):      {len(diff_acts_l2)} points  (each row on unit sphere)")
        else:
            combined_acts_norm = combined_acts
            if compute_class_clouds:
                pos_acts_norm  = pos_acts
                neg_acts_norm  = neg_acts
                diff_acts_norm = pos_acts_norm - neg_acts_norm  # same space, no raw re-scaling
                diff_l2_norms  = np.linalg.norm(diff_acts_norm, axis=1, keepdims=True)
                diff_acts_l2   = diff_acts_norm / (diff_l2_norms + 1e-8)

        # ========================================================
        # STEP 4: COMPUTE PERSISTENCE - COMBINED CLOUD
        # ========================================================
        print(f"  Layer {l}: Computing persistent homology with {metric} distance...")

        persistence_combined = compute_persistence_diagram(
            combined_acts_norm,
            metric=metric,
            max_dimension=max_dimension,
            max_edge_length=np.inf
        )

        persistence_stats = persistence_combined['persistence_stats']
        persistence_by_dim = persistence_combined['persistence_by_dim']

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

        h0_persistence_only = mean_pers_h0
        h1_persistence_only = mean_pers_h1
        h2_persistence_only = mean_pers_h2

        mean_all_persistence = (mean_pers_h0 + mean_pers_h1 + mean_pers_h2) / 3.0
        total_pers = sum([persistence_stats.get(f'H{d}_total', 0.0)
                         for d in range(max_dimension + 1)])

        inverse_mean_H0 = 1.0 / (mean_pers_h0 + 0.1)
        inverse_mean_H1 = 1.0 / (mean_pers_h1 + 0.1)
        inverse_mean_H2 = 1.0 / (mean_pers_h2 + 0.1)

        # ========================================================
        # STEP 5: COMPUTE PERSISTENCE - RAW DIFF CLOUD (z-score only)
        # ========================================================
        persistence_diff = compute_persistence_diagram(
            diff_acts_norm,
            metric=metric,
            max_dimension=max_dimension,
            max_edge_length=np.inf
        )

        mean_persistence_H0_diff  = float(persistence_diff['persistence_stats'].get('H0_mean',  0.0))
        mean_persistence_H1_diff  = float(persistence_diff['persistence_stats'].get('H1_mean',  0.0))
        mean_persistence_H2_diff  = float(persistence_diff['persistence_stats'].get('H2_mean',  0.0))
        max_persistence_H0_diff   = float(persistence_diff['persistence_stats'].get('H0_max',   0.0))
        max_persistence_H1_diff   = float(persistence_diff['persistence_stats'].get('H1_max',   0.0))
        max_persistence_H2_diff   = float(persistence_diff['persistence_stats'].get('H2_max',   0.0))
        total_persistence_H0_diff = float(persistence_diff['persistence_stats'].get('H0_total', 0.0))
        total_persistence_H1_diff = float(persistence_diff['persistence_stats'].get('H1_total', 0.0))
        total_persistence_H2_diff = float(persistence_diff['persistence_stats'].get('H2_total', 0.0))

        # ========================================================
        # STEP 5b: COMPUTE PERSISTENCE - L2-NORMALIZED DIFF CLOUD
        # [NEW] Pure angular topology: all vectors on unit sphere.
        # Directly comparable to C-score's cosine space.
        # ========================================================
        persistence_diff_l2 = compute_persistence_diagram(
            diff_acts_l2,
            metric=metric,
            max_dimension=max_dimension,
            max_edge_length=np.inf
        )

        mean_persistence_H0_diff_l2  = float(persistence_diff_l2['persistence_stats'].get('H0_mean',  0.0))
        mean_persistence_H1_diff_l2  = float(persistence_diff_l2['persistence_stats'].get('H1_mean',  0.0))
        mean_persistence_H2_diff_l2  = float(persistence_diff_l2['persistence_stats'].get('H2_mean',  0.0))
        max_persistence_H0_diff_l2   = float(persistence_diff_l2['persistence_stats'].get('H0_max',   0.0))
        max_persistence_H1_diff_l2   = float(persistence_diff_l2['persistence_stats'].get('H1_max',   0.0))
        max_persistence_H2_diff_l2   = float(persistence_diff_l2['persistence_stats'].get('H2_max',   0.0))
        total_persistence_H0_diff_l2 = float(persistence_diff_l2['persistence_stats'].get('H0_total', 0.0))
        total_persistence_H1_diff_l2 = float(persistence_diff_l2['persistence_stats'].get('H1_total', 0.0))
        total_persistence_H2_diff_l2 = float(persistence_diff_l2['persistence_stats'].get('H2_total', 0.0))

        # ========================================================
        # STEP 6: ASSEMBLE COMPLETE METRICS DICTIONARY
        # ========================================================
        hole_score_info[l] = {
            # ===== COMBINED CLOUD BASIC PERSISTENCE =====
            'mean_persistence_H0':  float(mean_pers_h0),
            'max_persistence_H0':   float(max_pers_h0),
            'total_persistence_H0': float(total_pers_h0),
            'count_H0': int(count_h0),
            'mean_persistence_H1':  float(mean_pers_h1),
            'max_persistence_H1':   float(max_pers_h1),
            'total_persistence_H1': float(total_pers_h1),
            'count_H1': int(count_h1),
            'mean_persistence_H2':  float(mean_pers_h2),
            'max_persistence_H2':   float(max_pers_h2),
            'total_persistence_H2': float(total_pers_h2),
            'count_H2': int(count_h2),
            'total_persistence':     float(total_pers),
            'h0_persistence_only':   float(h0_persistence_only),
            'h1_persistence_only':   float(h1_persistence_only),
            'h2_persistence_only':   float(h2_persistence_only),
            'mean_all_persistence':  float(mean_all_persistence),

            # ===== RAW DIFF CLOUD (z-score normalized) =====
            'mean_persistence_H0_diff':  float(mean_persistence_H0_diff)  if compute_class_clouds else None,
            'mean_persistence_H1_diff':  float(mean_persistence_H1_diff)  if compute_class_clouds else None,
            'mean_persistence_H2_diff':  float(mean_persistence_H2_diff)  if compute_class_clouds else None,
            'max_persistence_H0_diff':   float(max_persistence_H0_diff)   if compute_class_clouds else None,
            'max_persistence_H1_diff':   float(max_persistence_H1_diff)   if compute_class_clouds else None,
            'max_persistence_H2_diff':   float(max_persistence_H2_diff)   if compute_class_clouds else None,
            'total_persistence_H0_diff': float(total_persistence_H0_diff) if compute_class_clouds else None,
            'total_persistence_H1_diff': float(total_persistence_H1_diff) if compute_class_clouds else None,
            'total_persistence_H2_diff': float(total_persistence_H2_diff) if compute_class_clouds else None,

            # ===== L2-NORMALIZED DIFF CLOUD (unit sphere / angular-only) =====
            # [NEW] Topology of steering directions on unit sphere.
            # Low H0 = all directions cluster tightly (consistent steering).
            # Low H1 = no angular loops (no destructive interference).
            'mean_persistence_H0_diff_l2':  float(mean_persistence_H0_diff_l2)  if compute_class_clouds else None,
            'mean_persistence_H1_diff_l2':  float(mean_persistence_H1_diff_l2)  if compute_class_clouds else None,
            'mean_persistence_H2_diff_l2':  float(mean_persistence_H2_diff_l2)  if compute_class_clouds else None,
            'max_persistence_H0_diff_l2':   float(max_persistence_H0_diff_l2)   if compute_class_clouds else None,
            'max_persistence_H1_diff_l2':   float(max_persistence_H1_diff_l2)   if compute_class_clouds else None,
            'max_persistence_H2_diff_l2':   float(max_persistence_H2_diff_l2)   if compute_class_clouds else None,
            'total_persistence_H0_diff_l2': float(total_persistence_H0_diff_l2) if compute_class_clouds else None,
            'total_persistence_H1_diff_l2': float(total_persistence_H1_diff_l2) if compute_class_clouds else None,
            'total_persistence_H2_diff_l2': float(total_persistence_H2_diff_l2) if compute_class_clouds else None,

            # ===== METADATA =====
            'metric': metric,
            'n_samples_combined': int(len(combined_acts_norm)),
            'n_samples_pos':  int(len(pos_acts))       if compute_class_clouds else None,
            'n_samples_neg':  int(len(neg_acts))       if compute_class_clouds else None,
            'n_samples_diff': int(len(diff_acts_norm)) if compute_class_clouds else None,
            'compute_class_clouds': compute_class_clouds
        }

        with open(f"{save_root}L{l}.json", "w") as f:
            json.dump(hole_score_info[l], f, indent=4)

    with open(f"{save_root}all_layers.json", "w") as f:
        json.dump(hole_score_info, f, indent=4)

    total_metrics = len(hole_score_info[layers[0]])
    print(f"\nHOLE scores saved to: {save_root}")
    print(f"Total metrics stored per layer: {total_metrics} fields")

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
        print("MODE: Comprehensive (Combined + Diff + L2-Diff clouds)")
    else:
        print("MODE: Combined cloud only")
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

        sample_layer = layers[0]
        n_metrics = len(scores[sample_layer])
        print(f"Computed {n_metrics} metrics per layer")

    vec_root = f"./Vectors/{vec_task}/{vec_method}"
    svec_path = vec_root[10:].replace("/", "+")
    acts_pre_str = f"-{acts_pre}" if acts_pre is not None else ""
    cloud_suffix = "_WITH_CLASS_CLOUDS" if compute_class_clouds else ""
    save_path = f"./Score_HOLE{acts_pre_str}_ALL_METRICS{cloud_suffix}/{dataset.task}/{svec_path}/"
    os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}all_metrics_all_layers.json", "w") as f:
        json.dump(all_metric_scores, f, indent=4)

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
    print("="*80)
    return all_metric_scores