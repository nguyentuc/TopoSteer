#!/bin/bash

#SBATCH --account=le-lab
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=90GB
#SBATCH --time=336:00:00
#SBATCH --partition=general
#SBATCH --output=z11_v2_believes_plot_hole_scores-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tucnguye@iu.edu

nvidia-smi
eval "$(conda shell.bash hook)"
conda env list
conda activate /data/project/le-lab/conda_env/Adaptive_LayerSteering

gpu=0
cmd="CUDA_VISIBLE_DEVICES=$gpu python z11_v2_believes_plot_hole_scores.py"
echo "$cmd"
eval $cmd