#!/usr/bin/env python3
"""
Plot t-SNE of activation vectors for all layers.

Usage:
    python plot_activations.py --task cognitive-enhancement
    python plot_activations.py --task cognitive-enhancement --vec-method md
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
    # acts[class_idx][layer_idx] -> list of tensors
    n_layers = len(acts[0])
    print(f"Found {n_layers} layers, loading from: {acts_path}")
    return acts, n_layers


def plot_all_layers(task: str, vec_method: str = 'md'):
    acts, n_layers = load_all_layers(task, vec_method)

    out_dir = OUTPUT_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting {n_layers} layers...")
    for layer in tqdm(range(n_layers)):
        class_0 = torch.stack(acts[0][layer]).float().numpy()
        class_1 = torch.stack(acts[1][layer]).float().numpy()
        activations = np.vstack([class_0, class_1])
        labels = np.concatenate([np.zeros(len(class_0)), np.ones(len(class_1))])

        perplexity = min(30, len(activations) - 1)
        embeddings = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(activations)

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in [0, 1]:
            mask = labels == cls
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=f'C{int(cls)}', label=f'Class {int(cls)}',
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.set_title(f'Layer {layer} | Task: {task}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.axis('off')

        save_path = out_dir / f'layer_{layer:03d}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot t-SNE of activations for all layers')
    parser.add_argument('--task',       type=str, required=True, help='Task name')
    parser.add_argument('--vec-method', type=str, default='md',  help='Vector method subfolder (default: md)')
    args = parser.parse_args()

    plot_all_layers(task=args.task, vec_method=args.vec_method)


if __name__ == '__main__':
    main()