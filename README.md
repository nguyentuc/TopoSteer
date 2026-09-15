## Environment Setup
```
conda create --prefix /media/volume/h100_instance2/conda_env/Adaptive_LayerSteering python=3.10 -y
conda activate /media/volume/h100_instance2/conda_env/Adaptive_LayerSteering
pip install -r requirements.txt
pip install huggingface_hub
pip install gudhi
pip install scipy
pip install scikit-learn
pip install typing_extensions
pip install urllib3
pip install matplotlib
pip install seaborn
```

## Quick Start
We provide complete code and data to reproduce the **main results** from our paper.
You can run everything with just **two steps**:

1. **Set model path**
   In the first line of `globalenv.py`, modify the path to your model checkpoint.

2. **Download the model:**
```
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir /data/project/le-lab/cache/Llama-3-8B-Instruct --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /data/project/le-lab/cache/Qwen2.5-32B-Instruct --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /data/project/le-lab/cache/Qwen2.5-3B-Instruct --local-dir-use-symlinks False
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir /data/project/le-lab/cache/Llama-3-3B-Instruct --local-dir-use-symlinks False
```
2. **Run the script for the LayerNavigator:**
   ```bash
   python main.py
   ```

2. **Run the script for the TopoSteer:**
   ```bash
   python main.py
   ```

---

## Ablation Study
-  Implement HOLE metric to choose layer to steer with LayerNavigator code base.

| Feature Category | Full HOLE Library | My Implementation | Status |
|-----------------|-------------------|-------------------|---------|
| **Distance Metrics** | 4 (Euclidean, Cosine, Mahalanobis, Geodesic) | 3 (Euclidean, Cosine, Mahalanobis) | Partial |
| **Persistent Homology** | Full (H0, H1, H2, persistence diagrams) |  Full |  Complete |
| **Betti Numbers** |  (beta0, beta1, beta2) |  (beta0, beta1, beta2) |  Complete |
| **Clustering Metrics** | Purity, separability |  Purity, separability | Complete |
| **Visualization** | Dendrograms, heatmaps, blob graphs, Sankey |  None |  Missing |
| **Cross-Layer Analysis** |  Sankey evolution tracking |  Per-layer only |  Missing |
| **Robustness Testing** | Noise, pruning, quantization |  None |  Missing |
| **Purpose** | Comprehensive NN interpretation | Layer selection for steering | Different goal |

## Code dictionary note:
- 3_visualization.py: plot tsne visualization across all the tasks, each class plot all the layers
- 3_visualization_PCA.py: plot PCA visualization across all the tasks, each task plot on all the layers
- 7_activation_visualization_LLama8B_LayerNavigator_scores.py: layernavigator score
- 7_activation_visualization_mean_H0_union_LLama8B.py: mean persistence H0 (positive and negative)
- 8_activation_visualization_mean_H0_diff_LLama8B_diff.py: mean persistence H0 (positive - negative)
- 9_activation_visualization_LLama8B_of_l2diff_minus_l2mean.py: mean persistence H0 (l2 norm of (positive - negative) minus l2 norm of mean (positive - negative)).
- 10_activation_visualization_LLama8B_of_diff_minus_mean.py: mean persistence H0 ((positive - negative) - mean of (positive - negative))
- 11_visualization_across_all_steered_layers.py: plot the alignment of model steering on all the tasks.
- 12_visualization_across_all_steered_layers_LN_Scores.py: plot the aligment of steering on all the tasks with LN Scores
- 13_visualization_across_all_steered_layers_LN_Scores_HOLE_H0union.py: plot the aligment of steering on all the tasks with LN Scores and with the mean H0 score
- 14_TDA_simulation.py: do simulation on different dataset configuration, each configuration (plot the tsne and compute the mean persistence H0).
- 15/16/17_persistence_diagram_of_union_activation_visualization_LLama8B.py: compute the persistence diagram on the union of positive and negative pointclouds across all the layers (all 6 tasks) and then plot the visualization.
- 18_entropy_persistence_diagram.py: code that compute entropy of persistence.
- 19_barcode_persistence_union_visualization_LLama8B.py: load persitence pairs from 15/16/17 and plot the barcode for all datasets across layers.
- 21_text_cluster_analysis.py: load the text do tf-idf and then use K-Mean cluster