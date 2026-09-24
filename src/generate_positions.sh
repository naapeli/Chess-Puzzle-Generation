#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --output=generations.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h200"
#SBATCH --cpus-per-task=16


module load mamba
module load triton/2024.1-gcc gcc/12.3.0  # needed for torch.compile
source activate environment
srun python src/generate_positions.py --run_type supervised --run_name final_model --checkpoint_name model_1000000.pt --n_fens 200000 --batch_size 10000 --temperature 1.0 --steps 256 --output_file arxiv_paper/test_move.csv --context_dataset test   # --generate_move_last
# srun python src/generate_positions.py --run_type rl --run_name final_large_runs --checkpoint_name ownThemeDistribution/run14/model_0025000.pt --n_fens 10000 --batch_size 32768 --temperature 1.0 --steps 128 --output_file final_model/rl/ownThemeDistribution/run14/positions_test_25000.csv --context_dataset test  # random  #--generate_move_last
# srun python src/generate_positions.py --run_type rl --run_name final_large_runs --checkpoint_name final_thesis_experiments/full_diversity11/model_0003500.pt --n_fens 50000 --batch_size 32768 --temperature 1.0 --steps 128 --output_file final_model/rl/full_diversity11_3500/positions_test_new_theme_match_check.csv --context_dataset test  # random  #--generate_move_last