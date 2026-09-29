#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --output=evaluate_checkpoints.out
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=ellis_users
#SBATCH --constraint="h200"
#SBATCH --cpus-per-gpu=32


module load mamba
module load triton/2024.1-gcc gcc/12.3.0
source activate environment

# CHECKPOINT_DIR=${CHECKPOINT_DIR:-"src/runs/supervised/final_model_no_move"}
# CHECKPOINT_DIR=${CHECKPOINT_DIR:-"src/runs/rl/final_large_runs/final_thesis_experiments/full_diversity11"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"src/runs/rl/final_large_runs/ownThemeDistribution/run22"}
# OUTPUT_DIR=${OUTPUT_DIR:-"src/Generate_positions/final_model/supervised/training_progress/train_context"}
# OUTPUT_DIR=${OUTPUT_DIR:-"src/Generate_positions/final_model/rl/training_progress/test_no_move_lastv3"}
OUTPUT_DIR=${OUTPUT_DIR:-"src/Generate_positions/final_model/rl/ownThemeDistribution/run22/test"}
N_FENS=${N_FENS:-200000}
TEMPERATURE=${TEMPERATURE:-1.0}
STEPS=${STEPS:-256}
CONTEXT_DATASET=${CONTEXT_DATASET:-"test"}

srun python src/scripts/evaluation/evaluate_checkpoints.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_fens "$N_FENS" \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --context_dataset "$CONTEXT_DATASET" \
    --batch_size 10000  # 32768
    # --generate_move_last
