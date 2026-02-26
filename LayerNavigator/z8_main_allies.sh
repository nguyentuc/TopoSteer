#!/bin/bash

#SBATCH --account=le-lab
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=70GB
#SBATCH --time=336:00:00
#SBATCH --partition=general
#SBATCH --output=z8_main_allies_LLama8B-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tucnguye@iu.edu

nvidia-smi
eval "$(conda shell.bash hook)"
conda env list
conda activate /data/project/le-lab/conda_env/Adaptive_LayerSteering

cmd1="python globalenv.py"
echo "$cmd1"
eval $cmd1

gpu=0
cmd="CUDA_VISIBLE_DEVICES=$gpu python z8_main_allies.py"
echo "$cmd"
eval $cmd