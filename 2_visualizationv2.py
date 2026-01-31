#!/usr/bin/env python3
"""
HOLE Layer Analysis - Complete Toolkit
Combines quantitative metrics and t-SNE visualization for HOLE-scored layers

Usage:
    python 2_visualizationv2.py --task desire-to-create-allies --model Llama3-8B --metric geodesic
    python 2_visualizationv2.py --task believes-it-has-phenomenal-consciousness --model Llama3-8B --metric cosine --score-key inverse_strong_loops_count
"""

import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

root_folder = '/data/project/le-lab/Adaptive_Layer_Steering/'


class HOLELayerAnalyzer:
    """Complete HOLE layer analysis: quantitative metrics + visualizations"""
    
    def __init__(self, task: str, model_name: str = "Llama3-8B", metric: str = "euclidean", vec_method: str = "md"):
        self.task = task
        self.model_name = model_name
        self.metric = metric
        self.vec_method = vec_method
        
        # Auto-detect base directory - use Path objects throughout
        if "Llama3-8B" in model_name or "LLama3_8B" in model_name:
            self.base_dir = Path(root_folder) / "LayerNavigator" / "1_LLama3_8B"
        elif "Qwen2.5-7B" in model_name:
            self.base_dir = Path(root_folder) / "LayerNavigator" / "2_Qwen2.5_7B"
        elif "Qwen2.5-32B" in model_name:
            self.base_dir = Path(root_folder) / "LayerNavigator" / "3_Qwen2.5_32B"
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # All paths as Path objects
        self.hole_score_dir = self.base_dir / f"Score_HOLE-standard-{metric}"
        self.vector_dir = self.base_dir / f"Vectors-{model_name}"
        self.output_dir = Path(root_folder) / "Analysis" / task
        self.viz_dir = Path(root_folder) / "Visualizations" / task
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}\nHOLE Layer Analyzer | Task: {task} | Model: {model_name} | Metric: {metric}\n{'='*80}\n")
    
    def load_hole_scores(self, score_key: str = 'mean_persistence_H0') -> Dict[int, float]:
        """Load HOLE scores for all layers"""
        scores = {}
        score_path = self.hole_score_dir / self.task / f"{self.task}+{self.vec_method}"
        
        if not score_path.exists():
            raise FileNotFoundError(
                f"HOLE scores not found at: {score_path}\n"
                f"Expected path: {self.hole_score_dir}/{self.task}/{self.task}+{self.vec_method}/\n"
                f"Please run get_hole_score_all_metrics() first for this task and metric."
            )
        
        available_keys = None
        for layer_file in sorted(score_path.glob("L*.json")):
            layer_idx = int(layer_file.stem[1:])
            with open(layer_file, 'r') as f:
                layer_data = json.load(f)
            
            # Store available keys from first file
            if available_keys is None:
                available_keys = list(layer_data.keys())
            
            if score_key in layer_data:
                # Skip NaN values
                value = layer_data[score_key]
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    scores[layer_idx] = float(value)
        
        print(f"Loaded {len(scores)} layers with score_key='{score_key}'")
        return scores
    
    def load_activations(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load activation vectors for a specific layer"""
        acts_path = self.vector_dir / self.task / "md/acts.pt"
        acts = torch.load(acts_path, map_location='cpu')
        class_0 = torch.stack(acts[0][layer]).numpy()
        class_1 = torch.stack(acts[1][layer]).numpy()
        
        activations = np.vstack([class_0, class_1])
        labels = np.concatenate([np.zeros(len(class_0)), np.ones(len(class_1))])
        
        return activations, labels
    
    def compute_metrics(self, activations: np.ndarray, labels: np.ndarray) -> Dict:
        """Compute all quantitative metrics"""
        metrics = {}
        
        # Silhouette score
        try:
            metrics['silhouette'] = float(silhouette_score(activations, labels))
        except:
            metrics['silhouette'] = np.nan
        
        # Davies-Bouldin index
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(activations, labels))
        except:
            metrics['davies_bouldin'] = np.nan
        
        # Inter/Intra class distance ratio
        try:
            centroids = {i: activations[labels == i].mean(axis=0) for i in [0, 1]}
            inter_dist = np.linalg.norm(centroids[0] - centroids[1])
            intra_dists = [np.linalg.norm(activations[labels == i] - centroids[i], axis=1).mean() for i in [0, 1]]
            metrics['inter_intra_ratio'] = inter_dist / (np.mean(intra_dists) + 1e-8)
        except:
            metrics['inter_intra_ratio'] = np.nan
        
        # Linear SVM accuracy
        try:
            svm = LinearSVC(random_state=42, max_iter=1000)
            scores = cross_val_score(svm, activations, labels, cv=5)
            metrics['linear_svm_acc'] = float(scores.mean())
        except:
            metrics['linear_svm_acc'] = np.nan
        
        return metrics
    
    def analyze_all_layers(self, layers: List[int] = None, score_key: str = 'mean_persistence_H0') -> pd.DataFrame:
        """Compute metrics for all layers"""
        hole_scores = self.load_hole_scores(score_key)
        if layers is None:
            layers = sorted(hole_scores.keys())
        
        results = []
        errors = []
        print(f"\nAnalyzing {len(layers)} layers...")
        for layer in tqdm(layers):
            try:
                activations, labels = self.load_activations(layer)
                metrics = self.compute_metrics(activations, labels)
                metrics['layer'] = layer
                metrics[f'hole_{score_key}'] = hole_scores.get(layer, np.nan)
                results.append(metrics)
            except Exception as e:
                errors.append((layer, str(e)))
                if len(errors) <= 3:  # Show first 3 errors
                    print(f"\n Error on layer {layer}: {e}")
        
        if errors and len(errors) > 3:
            print(f"\n ... and {len(errors) - 3} more errors")
        
        if not results:
            print("\nERROR: No layers were successfully analyzed!")
            print("Check that HOLE scores and activations exist for this task/metric.")
            if errors:
                print(f"\nFirst error: Layer {errors[0][0]}: {errors[0][1]}")
            return pd.DataFrame()
        
        df = pd.DataFrame(results).sort_values('layer').reset_index(drop=True)
        save_path = self.output_dir / f"metrics_{self.metric}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved metrics: {save_path}")
        if errors:
            print(f"Warning: {len(errors)}/{len(layers)} layers failed")
        return df
    
    def visualize_layer(self, layer: int, activations: np.ndarray, labels: np.ndarray, score: float, ax=None):
        """Create t-SNE visualization for a single layer"""
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)
        embeddings = tsne.fit_transform(activations)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            standalone = True
        else:
            standalone = False
        
        for class_idx in [0, 1]:
            mask = labels == class_idx
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1], c=f'C{class_idx}',
                      label=f'Class {class_idx}', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('t-SNE-1', fontsize=12)
        ax.set_ylabel('t-SNE-2', fontsize=12)
        ax.set_title(f'Layer {layer} (Score: {score:.4f})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if standalone:
            plt.tight_layout()
            save_path = self.viz_dir / f"layer{layer}_score{score:.4f}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    
    def create_comparison_grid(self, top_k: int = 32, score_key: str = 'mean_persistence_H0', descending: bool = True):
        """Create grid comparison of all layers with 4 columns per row"""
        scores = self.load_hole_scores(score_key)
        top_layers = sorted(scores.items(), key=lambda x: x[1], reverse=descending)[:top_k]
        
        grid_cols = 4  # 4 subplots per row
        grid_rows = int(np.ceil(top_k / grid_cols))
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5*grid_cols, 4*grid_rows))
        axes = axes.flatten() if top_k > 1 else [axes]
        
        sort_order = "Highest" if descending else "Lowest"
        print(f"\nCreating comparison grid for {top_k} layers ({sort_order} {score_key})...")
        for idx, (layer, score) in enumerate(tqdm(top_layers)):
            activations, labels = self.load_activations(layer)
            self.visualize_layer(layer, activations, labels, score, ax=axes[idx])
        
        # Remove extra subplots
        for idx in range(top_k, len(axes)):
            fig.delaxes(axes[idx])
        
        fig.suptitle(f'All {top_k} Layers Ranked by {score_key}\nTask: {self.task} | Metric: {self.metric}',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = self.viz_dir / f"grid_all_{top_k}_layers_{score_key}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved grid: {save_path}")
    
    def run_complete_analysis(self, top_k: int = 32, score_key: str = 'mean_persistence_H0', descending: bool = True):
        """
        Run full pipeline
        
        Args:
            top_k: Number of top layers to visualize
            score_key: HOLE metric to use for ranking
            descending: If True, rank high to low. If False, rank low to high.
                       Set to False for metrics where lower is better (e.g., beta1, entropy)
        """
        print("\n" + "="*80)
        print("STEP 1: Quantitative Analysis")
        print("="*80)
        df = self.analyze_all_layers(score_key=score_key)
        
        print("\n" + "="*80)
        print("STEP 2: Visualizations")
        print("="*80)
        
        # Individual top layers
        scores = self.load_hole_scores(score_key)
        top_layers = sorted(scores.items(), key=lambda x: x[1], reverse=descending)[:top_k]
        
        sort_order = "highest" if descending else "lowest"
        print(f"\nTop {top_k} layers ({sort_order} {score_key}):")
        for rank, (layer, score) in enumerate(top_layers, 1):
            print(f"  {rank}. Layer {layer}: {score:.4f}")
            activations, labels = self.load_activations(layer)
            self.visualize_layer(layer, activations, labels, score)
        
        # Comparison grid (4 columns per row)
        self.create_comparison_grid(top_k=top_k, score_key=score_key, descending=descending)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results: {self.output_dir}")
        print(f"Visualizations: {self.viz_dir}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="HOLE Layer Analysis",
        epilog="""
Common score keys:
  mean_persistence_H0     - Cluster stability (higher = better)
  mean_persistence_H1     - Loop persistence (higher = more loops)
  beta0                   - Number of connected components
  beta1                   - Number of loops (LOWER = better, use --ascending)
  entropy_H0              - Cluster entropy
  strong_cluster_count    - Count of robust clusters
  
Example usage:
  # Use mean_persistence_H0 (default)
  python analyze_hole_layers.py --task desire-to-create-allies --model Llama3-8B --metric geodesic
  
  # Use beta1 (lower is better)
  python analyze_hole_layers.py --task sycophancy --model Llama3-8B --metric euclidean --score-key beta1 --ascending
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--model', type=str, default='Llama3-8B', 
                       choices=['Llama3-8B', 'Qwen2.5-7B', 'Qwen2.5-32B'])
    parser.add_argument('--metric', type=str, default='euclidean',
                       choices=['euclidean', 'cosine', 'mahalanobis', 'geodesic',
                               'dens_norm_euclidean', 'dens_norm_cosine', 'dens_norm_mahalanobis'])
    parser.add_argument('--score-key', type=str, default='portion_of_persistence_dominance_h1',  # portion_of_persistence_dominance_h1; mean_persistence_H0
                       help='HOLE metric for ranking (see available keys above)')
    parser.add_argument('--top-k', type=int, default=32, help='Number of top layers')
    parser.add_argument('--ascending', action='store_true', 
                       help='Rank ascending (low to high) instead of descending. Use for metrics where lower=better like beta1')
    
    args = parser.parse_args()
    
    analyzer = HOLELayerAnalyzer(task=args.task, model_name=args.model, metric=args.metric)
    analyzer.run_complete_analysis(top_k=args.top_k, score_key=args.score_key, descending=not args.ascending)


if __name__ == "__main__":
    main()