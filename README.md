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

## Running:
- [H100_2] z8_main_v3.sh with LLama8B
- [LAIR] z8_main_v3.sh with Qwen32B 
- [H100-2] Qwen2.5-7B-Instruct