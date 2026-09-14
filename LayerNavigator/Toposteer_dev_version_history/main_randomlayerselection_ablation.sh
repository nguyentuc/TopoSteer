#!/bin/bash

#SBATCH --account=le-lab
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=70GB
#SBATCH --time=336:00:00
#SBATCH --partition=general
#SBATCH --output=main_randomlayerselection_ablation_Qwen32B-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tucnguye@iu.edu

nvidia-smi
eval "$(conda shell.bash hook)"
conda activate /data/project/le-lab/conda_env/Adaptive_LayerSteering

cd /data/project/le-lab/Adaptive_Layer_Steering/LayerNavigator/

gpu=0
cmd="CUDA_VISIBLE_DEVICES=$gpu python main_randomlayerselection_ablation.py"

echo "$cmd"
eval $cmd
