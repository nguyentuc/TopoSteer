## Environment Setup
```
conda create --prefix /data/project/le-lab/conda_env/Adaptive_LayerSteering python=3.10 -y
conda activate /data/project/le-lab/conda_env/Adaptive_LayerSteering
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
2. **Run the main script**
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

- Current runing HOLE to select layer based on the TSS score (combination score) with:
+ Cosine Metric(Done)
+ Euclidean Metric (DONE - H100_2): Performance not so good compated with using Cosine metric
+ Mahalanobis (Running on the H100_2)
+ Geodesic Distance (Next, need to be implement)

- Current HOLE score: tss = (0.35 * purity +0.35 * separability + 0.30 * mean_pers_h0) * entanglement_penalty

- Ablation Study on different metric.

## Running:
[LAIR] z5_main_allies_LLama3.sh + z5_main_conscientiousness_LLama3.sh are running on LAIR
[NEXT] Plot visualization for the steering score on each metrics (rank by larger to smaller) corresponding the activation:
+ provide visualization of tsne/umap to plot the 2D embeddings of the activation vectors.
+ 6e,6f visualization of heatmap dendrograms of activations vectors between layers.