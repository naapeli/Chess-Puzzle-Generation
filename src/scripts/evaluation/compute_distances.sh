#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --output=compute_distances.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32


module load mamba
source activate environment

srun python src/scripts/evaluation/compute_distances.py \
  --lichess_csv src/Generate_positions/arxiv_true/lichess.csv \
  --generated_csv src/Generate_positions/arxiv_true/run22/model_0020000.csv \
  --self_sample_size 40000 \
  --lichess_sample_size 100000 \
  --chunk_size 10000
