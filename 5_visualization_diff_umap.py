#!/usr/bin/env python3
"""
HOLE Layer Analysis - Difference Vector UMAP Visualization
Plots per-sample difference vectors (acts[0][l][i] - acts[1][l][i]) via UMAP.

Unlike t-SNE, UMAP preserves both local AND global structure, making inter-cluster
distances meaningful — more faithful to the topological structure (H0/H1) your
research is built on.

Usage:
    python 5_visualization_diff_umap.py --task believes-it-has-phenomenal-consciousness --model Llama3-8B --metric euclidean --score-key mean_persistence_H0
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from umap import UMAP
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

root_folder = '/data/project/le-lab/Adaptive_Layer_Steering/'


class HOLEDiffVectorAnalyzer:

    def __init__(self, task: str, model_name: str = "Llama3-8B", metric: str = "euclidean",
                 vec_method: str = "md", n_neighbors: int = 15, min_dist: float = 0.1):
        self.task = task
        self.model_name = model_name
        self.metric = metric
        self.vec_method = vec_method
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

        if "Llama3-8B" in model_name or "LLama3_8B" in model_name:
            self.base_dir = Path(root_folder) / "LayerNavigator" / "1_LLama3_8B"
            _model_folder = "LLama3-8B"
        elif "Qwen2.5-7B" in model_name:
            self.base_dir = Path(root_folder) / "LayerNavigator" / "2_Qwen2.5_7B"
            _model_folder = model_name
        elif "Qwen2.5-32B" in model_name:
            self.base_dir = Path(root_folder) / "LayerNavigator" / "3_Qwen2.5_32B"
            _model_folder = model_name
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.hole_score_dir = self.base_dir / f"Score_HOLE-standard-{metric}"
        self.vector_dir     = self.base_dir / "Vector_Aggregated" / f"Vectors-{_model_folder}"
        self.viz_dir        = Path(root_folder) / "Visualizations_Diff_UMAP" / task
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"HOLE Diff-Vector Analyzer (UMAP) | Task: {task} | Model: {model_name} | Metric: {metric}")
        print(f"UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
        print(f"Acts path: {self.vector_dir / task / vec_method / 'acts.pt'}")
        print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_hole_scores(self, score_key: str = 'mean_persistence_H0') -> Dict[int, float]:
        scores = {}
        score_path = self.hole_score_dir / self.task / f"{self.task}+{self.vec_method}"

        if not score_path.exists():
            raise FileNotFoundError(
                f"HOLE scores not found at: {score_path}\n"
                f"Please run get_hole_score_all_metrics() first."
            )

        available_keys = None
        for layer_file in sorted(score_path.glob("L*.json")):
            layer_idx = int(layer_file.stem[1:])
            with open(layer_file, 'r') as f:
                layer_data = json.load(f)

            if available_keys is None:
                available_keys = list(layer_data.keys())

            if score_key in layer_data:
                value = layer_data[score_key]
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    scores[layer_idx] = float(value)

        if not scores:
            raise ValueError(
                f"No valid scores found for '{score_key}'.\n"
                f"Available keys:\n" + "\n".join(f"  - {k}" for k in sorted(available_keys))
            )

        print(f"✓ Loaded {len(scores)} layers with score_key='{score_key}'")
        return scores

    def load_difference_vectors(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        acts_path = self.vector_dir / self.task / self.vec_method / "acts.pt"
        if not acts_path.exists():
            raise FileNotFoundError(
                f"Activations not found at: {acts_path}\n"
                f"Please run uni_generate_vectors() first."
            )

        acts = torch.load(acts_path, map_location='cpu')
        acts_0 = torch.stack(acts[0][layer]).numpy()  # (N, hidden_dim) - negative
        acts_1 = torch.stack(acts[1][layer]).numpy()  # (N, hidden_dim) - positive

        diff_vectors = acts_0 - acts_1               # (N, hidden_dim)
        norms = np.linalg.norm(diff_vectors, axis=1) # (N,)

        return diff_vectors, norms

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_diff_layer(
        self,
        layer: int,
        diff_vectors: np.ndarray,
        norms: np.ndarray,
        score: float,
        ax=None
    ):
        """
        UMAP projection of the N difference vectors, colored by per-sample L2 norm.

        Unlike t-SNE, inter-cluster distances are meaningful here:
          - Clusters that appear far apart are genuinely distant in activation space
          - Elongated or connected shapes reflect true manifold topology (H0/H1 structure)
          - n_neighbors controls local vs global balance (higher = more global)
          - min_dist controls how tightly points are packed in 2D

        Cyan star = centroid = the saved steering vector L{layer}.pt.
        """
        reducer = UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=42,
            verbose=False
        )
        embeddings = reducer.fit_transform(diff_vectors)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(10, 8))

        norm_colors = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
        sc = ax.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=norm_colors, cmap='plasma',
            alpha=0.7, s=60, edgecolors='black', linewidth=0.4
        )

        centroid = embeddings.mean(axis=0)
        ax.scatter(*centroid, c='cyan', s=200, marker='*',
                   edgecolors='black', linewidth=1.2, zorder=5,
                   label='Centroid (≈ steering vec)')

        ax.set_xlabel('UMAP-1', fontsize=11)
        ax.set_ylabel('UMAP-2', fontsize=11)
        ax.set_title(f'Layer {layer} | Diff Cloud (UMAP) | Score: {score:.4f}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if standalone:
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label('Normalized ||diff||', fontsize=10)
            plt.tight_layout()
            save_path = self.viz_dir / f"diff_umap_layer{layer}_score{score:.4f}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path

    # ------------------------------------------------------------------
    # Comparison grid
    # ------------------------------------------------------------------

    def create_comparison_grid(self, top_k: int = 32,
                               score_key: str = 'mean_persistence_H0',
                               descending: bool = True):
        """4-column grid of UMAP diff-cloud plots"""
        scores = self.load_hole_scores(score_key)
        top_layers = sorted(scores.items(), key=lambda x: x[1], reverse=descending)[:top_k]

        grid_cols = 4
        grid_rows = int(np.ceil(top_k / grid_cols))
        fig, axes = plt.subplots(grid_rows, grid_cols,
                                 figsize=(5 * grid_cols, 4 * grid_rows))
        axes = axes.flatten() if top_k > 1 else [axes]

        sort_label = "Highest" if descending else "Lowest"
        print(f"\nBuilding comparison grid ({top_k} layers, {sort_label} {score_key})...")
        for idx, (layer, score) in enumerate(tqdm(top_layers)):
            diff_vectors, norms = self.load_difference_vectors(layer)
            self.visualize_diff_layer(layer, diff_vectors, norms, score, ax=axes[idx])

        for idx in range(top_k, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle(
            f'Difference Cloud (UMAP) — {top_k} Layers Ranked by {score_key}\n'
            f'Task: {self.task} | Metric: {self.metric} | n_neighbors={self.n_neighbors} | min_dist={self.min_dist}',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        save_path = self.viz_dir / f"diff_umap_grid_{top_k}layers_{score_key}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved grid: {save_path}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_complete_analysis(self, top_k: int = 32,
                              score_key: str = 'mean_persistence_H0',
                              descending: bool = True):
        scores = self.load_hole_scores(score_key)
        top_layers = sorted(scores.items(), key=lambda x: x[1], reverse=descending)[:top_k]

        sort_label = "highest" if descending else "lowest"
        print(f"Top {top_k} layers ({sort_label} {score_key}):")
        for rank, (layer, score) in enumerate(top_layers, 1):
            print(f"  {rank:2d}. Layer {layer:2d}: {score:.4f}")

        print("\nGenerating UMAP visualizations...")
        for layer, score in tqdm(top_layers):
            diff_vectors, norms = self.load_difference_vectors(layer)
            self.visualize_diff_layer(layer, diff_vectors, norms, score)

        self.create_comparison_grid(top_k=top_k, score_key=score_key, descending=descending)

        print(f"\n✅ Done. Visualizations saved to: {self.viz_dir}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HOLE Difference-Vector UMAP Visualization",
        epilog="""
Plots per-sample difference vectors (acts[0][l][i] - acts[1][l][i]) via UMAP.
Color = per-sample ||diff|| norm. Cyan star = steering vector centroid.

UMAP vs t-SNE: inter-cluster distances are meaningful in UMAP — more faithful
to the topological structure (H0/H1) that HOLE is built on.

n_neighbors: controls local vs global structure balance
  - Low  (5-15)  → preserves fine local clusters
  - High (30-50) → preserves global topology, better for H1 loops

min_dist: controls point packing in 2D
  - Low  (0.0-0.1) → tighter clusters, good for seeing cluster boundaries
  - High (0.3-0.8) → more spread out, better for seeing overall manifold shape

Examples:
  python 5_visualization_diff_umap.py --task believes-it-has-phenomenal-consciousness --model Llama3-8B
  python 5_visualization_diff_umap.py --task sycophancy --score-key beta1 --ascending --n-neighbors 30 --min-dist 0.0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--task',        type=str, required=True)
    parser.add_argument('--model',       type=str, default='Llama3-8B',
                        choices=['Llama3-8B', 'Qwen2.5-7B', 'Qwen2.5-32B'])
    parser.add_argument('--metric',      type=str, default='euclidean',
                        choices=['euclidean', 'cosine', 'mahalanobis', 'geodesic',
                                 'dens_norm_euclidean', 'dens_norm_cosine', 'dens_norm_mahalanobis'])
    parser.add_argument('--score-key',   type=str, default='mean_persistence_H0')
    parser.add_argument('--top-k',       type=int, default=32)
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='UMAP n_neighbors: low=local structure, high=global topology (default: 15)')
    parser.add_argument('--min-dist',    type=float, default=0.1,
                        help='UMAP min_dist: low=tight clusters, high=spread manifold (default: 0.1)')
    parser.add_argument('--ascending',   action='store_true',
                        help='Use for metrics where lower=better (e.g. beta1)')

    args = parser.parse_args()
    analyzer = HOLEDiffVectorAnalyzer(
        task=args.task, model_name=args.model, metric=args.metric,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist
    )
    analyzer.run_complete_analysis(
        top_k=args.top_k,
        score_key=args.score_key,
        descending=not args.ascending
    )


if __name__ == "__main__":
    main()