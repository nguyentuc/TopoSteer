#!/usr/bin/env python3
"""
Plot heatmap dendrogram of activation vectors for all layers.

Usage:
    python plot_activations.py --task cognitive-enhancement
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import argparse
from tqdm import tqdm

VECTORS_ROOT = Path('/data/project/le-lab/Adaptive_Layer_Steering/LayerNavigator/Vectors')
OUTPUT_ROOT  = Path('/data/project/le-lab/Adaptive_Layer_Steering/Visualizations')


def load_all_layers(task: str, vec_method: str = 'md'):
    acts_path = VECTORS_ROOT / task / vec_method / 'acts.pt'
    if not acts_path.exists():
        raise FileNotFoundError(f"acts.pt not found at: {acts_path}")
    acts = torch.load(acts_path, map_location='cpu')
    n_layers = len(acts[0])
    print(f"Found {n_layers} layers, loading from: {acts_path}")
    return acts, n_layers


def plot_heatmap_dendrogram(layer: int, activations: np.ndarray, labels: np.ndarray, out_dir: Path, task: str):
    """Plot and save heatmap dendrogram for a single layer"""
    distances = pdist(activations, metric='euclidean')
    distance_matrix = squareform(distances)
    linkage_matrix = linkage(distances, method='average')

    dendro_order = leaves_list(linkage_matrix)
    reordered_matrix = distance_matrix[dendro_order, :][:, dendro_order]
    reordered_labels = labels[dendro_order]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1],
                          hspace=0.02, wspace=0.02)
    ax_dendro_top  = fig.add_subplot(gs[0, 1])
    ax_dendro_left = fig.add_subplot(gs[1, 0])
    ax_heatmap     = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[0, 0]).axis('off')

    # Top dendrogram
    dendrogram(linkage_matrix, ax=ax_dendro_top, no_labels=True, color_threshold=0)
    ax_dendro_top.axis('off')

    # Left dendrogram
    dendrogram(linkage_matrix, ax=ax_dendro_left, no_labels=True, orientation='left', color_threshold=0)
    ax_dendro_left.axis('off')

    # Heatmap
    im = ax_heatmap.imshow(reordered_matrix, cmap='viridis', aspect='auto', interpolation='nearest')

    # Red lines at class boundaries
    for change_idx in (np.where(np.diff(reordered_labels) != 0)[0] + 1):
        ax_heatmap.axhline(y=change_idx - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax_heatmap.axvline(x=change_idx - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.7)

    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Pairwise Distance', rotation=270, labelpad=20)

    n = len(reordered_labels)
    ticks = np.arange(0, n, max(1, n // 10))
    ax_heatmap.set_xticks(ticks); ax_heatmap.set_xticklabels(ticks, fontsize=8)
    ax_heatmap.set_yticks(ticks); ax_heatmap.set_yticklabels(ticks, fontsize=8)
    ax_heatmap.set_xlabel('Sample Index (Reordered)', fontsize=10)
    ax_heatmap.set_ylabel('Sample Index (Reordered)', fontsize=10)

    fig.suptitle(f'Heatmap Dendrogram — Layer {layer} | Task: {task}',
                 fontsize=14, fontweight='bold', y=0.98)

    save_path = out_dir / f'layer_{layer:03d}.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_all_layers(task: str, vec_method: str = 'md'):
    acts, n_layers = load_all_layers(task, vec_method)

    out_dir = OUTPUT_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting heatmap dendrograms for {n_layers} layers...")
    for layer in tqdm(range(n_layers)):
        class_0 = torch.stack(acts[0][layer]).float().numpy()
        class_1 = torch.stack(acts[1][layer]).float().numpy()
        activations = np.vstack([class_0, class_1])
        labels = np.concatenate([np.zeros(len(class_0)), np.ones(len(class_1))])
        plot_heatmap_dendrogram(layer, activations, labels, out_dir, task)

    print(f"Saved {n_layers} plots to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot heatmap dendrogram of activations for all layers')
    parser.add_argument('--task',       type=str, required=True, help='Task name')
    parser.add_argument('--vec-method', type=str, default='md',  help='Vector method subfolder (default: md)')
    args = parser.parse_args()
    plot_all_layers(task=args.task, vec_method=args.vec_method)


if __name__ == '__main__':
    main()