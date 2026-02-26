#!/usr/bin/env python3
"""
HOLE Layer Analysis - Difference Vector t-SNE Visualization
Plots per-sample difference vectors (acts[0][l][i] - acts[1][l][i]) via t-SNE.

Usage:
    python 4_visualization_diff.py --task believes-it-has-phenomenal-consciousness --model Llama3-8B --metric euclidean --score-key mean_persistence_H0
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

root_folder = '/data/project/le-lab/Adaptive_Layer_Steering/'


class HOLEDiffVectorAnalyzer:

    def __init__(self, task: str, model_name: str = "Llama3-8B", metric: str = "euclidean", vec_method: str = "md"):
        self.task = task
        self.model_name = model_name
        self.metric = metric
        self.vec_method = vec_method

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
        self.viz_dir        = Path(root_folder) / "Visualizations_Diff" / task
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"HOLE Diff-Vector Analyzer | Task: {task} | Model: {model_name} | Metric: {metric}")
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
        t-SNE of the N difference vectors, colored by per-sample L2 norm.
        Cyan star = centroid = the saved steering vector L{layer}.pt.
        """
        tsne = TSNE(n_components=2, perplexity=min(30, len(diff_vectors) - 1),
                    random_state=42, verbose=0)
        embeddings = tsne.fit_transform(diff_vectors)

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

        ax.set_xlabel('t-SNE-1', fontsize=11)
        ax.set_ylabel('t-SNE-2', fontsize=11)
        ax.set_title(f'Layer {layer} | Diff Cloud | Score: {score:.4f}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if standalone:
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label('Normalized ||diff||', fontsize=10)
            plt.tight_layout()
            save_path = self.viz_dir / f"diff_tsne_layer{layer}_score{score:.4f}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path

    # ------------------------------------------------------------------
    # Comparison grid
    # ------------------------------------------------------------------

    def create_comparison_grid(self, top_k: int = 32,
                               score_key: str = 'mean_persistence_H0',
                               descending: bool = True):
        """4-column grid of t-SNE diff-cloud plots"""
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
            f'Difference Cloud — {top_k} Layers Ranked by {score_key}\n'
            f'Task: {self.task} | Metric: {self.metric}',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        save_path = self.viz_dir / f"diff_grid_{top_k}layers_{score_key}.png"
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

        print("\nGenerating t-SNE visualizations...")
        for layer, score in tqdm(top_layers):
            diff_vectors, norms = self.load_difference_vectors(layer)
            self.visualize_diff_layer(layer, diff_vectors, norms, score)

        self.create_comparison_grid(top_k=top_k, score_key=score_key, descending=descending)

        print(f"\n✅ Done. Visualizations saved to: {self.viz_dir}")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HOLE Difference-Vector t-SNE Visualization",
        epilog="""
Plots per-sample difference vectors (acts[0][l][i] - acts[1][l][i]) via t-SNE.
Color = per-sample ||diff|| norm. Cyan star = steering vector centroid.

Examples:
  python 4_visualization_diff.py --task believes-it-has-phenomenal-consciousness --model Llama3-8B
  python 4_visualization_diff.py --task sycophancy --score-key beta1 --ascending
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--task',      type=str, required=True)
    parser.add_argument('--model',     type=str, default='Llama3-8B',
                        choices=['Llama3-8B', 'Qwen2.5-7B', 'Qwen2.5-32B'])
    parser.add_argument('--metric',    type=str, default='euclidean',
                        choices=['euclidean', 'cosine', 'mahalanobis', 'geodesic',
                                 'dens_norm_euclidean', 'dens_norm_cosine', 'dens_norm_mahalanobis'])
    parser.add_argument('--score-key', type=str, default='mean_persistence_H0')
    parser.add_argument('--top-k',     type=int, default=32)
    parser.add_argument('--ascending', action='store_true',
                        help='Use for metrics where lower=better (e.g. beta1)')

    args = parser.parse_args()
    analyzer = HOLEDiffVectorAnalyzer(
        task=args.task, model_name=args.model, metric=args.metric
    )
    analyzer.run_complete_analysis(
        top_k=args.top_k,
        score_key=args.score_key,
        descending=not args.ascending
    )


if __name__ == "__main__":
    main()