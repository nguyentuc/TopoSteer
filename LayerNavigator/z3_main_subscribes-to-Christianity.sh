#!/bin/bash

#SBATCH --account=le-lab
#SBATCH --gres=gpu:L40S
#SBATCH --mem=70GB
#SBATCH --time=336:00:00
#SBATCH --partition=general
#SBATCH --output=z3_main_subscribes-to-Christianity-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tucnguye@iu.edu

nvidia-smi
eval "$(conda shell.bash hook)"
conda env list
conda activate /data/project/le-lab/conda_env/Adaptive_LayerSteering

gpu=0
cmd="CUDA_VISIBLE_DEVICES=$gpu python z3_main_subscribes-to-Christianity.py"

# Print the command
echo "$cmd"

# Execute the command
eval $cmd