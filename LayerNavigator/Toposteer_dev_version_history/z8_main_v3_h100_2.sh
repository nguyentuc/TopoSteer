nvidia-smi
eval "$(conda shell.bash hook)"
conda env list
conda activate /media/volume/h100_instance2/conda_env/Adaptive_LayerSteering

cmd1="python globalenv.py"
echo "$cmd1"
eval $cmd1

gpu=0
cmd="CUDA_VISIBLE_DEVICES=$gpu python z8_main_v3.py"
echo "$cmd"
eval $cmd