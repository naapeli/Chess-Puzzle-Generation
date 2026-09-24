#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH --output=recompute_evaluations.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

module load mamba
source activate environment
# srun python src/recompute_evaluations.py --input_file Generate_positions/arxiv_true/run22/ --in_place
# srun python src/recompute_evaluations.py --input_file Generate_positions/arxiv_true/run22/model_0020000.csv --in_place
srun python src/recompute_evaluations.py --input_file Generate_positions/arxiv_true/old_rl_full_diversity11_3500.csv --in_place