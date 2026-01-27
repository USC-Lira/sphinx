#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --job-name=awe-hydra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --output=slurm_jobs/%x_%j.out
#SBATCH --error=slurm_jobs/%x_%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
# choose from A100, A40, V100, P100, K40
# eval "$(ssh-agent -s)" ssh-add ~/.ssh/id_rsa
# sinfo -t idle -o "%N %G"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/ubhuwani/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mujoco210
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

module purge
source /scratch1/ubhuwani/miniconda3/etc/profile.d/conda.sh
conda activate sphinx_env

# ../data_hydra/sphinx_compatible_awe/square/ph/err_0.005/data
# ../data_hydra/sphinx_compatible/square/ph/data

# python scripts/train_dense.py --config_path cfgs/dense/dp_square.yaml
# python scripts/train_dp3.py --config_path cfgs/dense/dp3_square.yaml
# python scripts/train_waypoint.py --config_path cfgs/waypoint/square.yaml
python -u scripts/train_hydra.py --config_path cfgs/hydra/square_hydra.yaml

# python -u interactive_scripts/record_sim.py --data_folder "data/auto/square" --task "square"

# python -u dataset_utils/reformat_hydra_data.py --in_file ../data_hydra/original_awe/square/ph/err_0.005/auto_labeled_mode_real2_v2_fast_human_square_30k_imgs.npz --out_dir ../data_hydra/sphinx_compatible_awe/square/ph/err_0.005/data
# python -u dataset_utils/reformat_hydra_data.py --in_file ../data_hydra/original/mode_real2_v2_fast_human_square_30k_imgs.npz --out_dir ../data_hydra/sphinx_compatible/square/ph/data
# python -u dataset_utils/reformat_hydra_data.py --in_file ../data_hydra/original_dense/square/ph/dense_labeled_mode_real2_v2_fast_human_square_30k_imgs.npz --out_dir ../data_hydra/sphinx_compatible_dense/square/ph/data

# python -u tools/visualize_hydra_dataset.py --in_dir ../data_hydra/sphinx_compatible_dense/square/ph/data --out_dir ../data_hydra/sphinx_compatible_dense/square/ph/visuals