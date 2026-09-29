# Chess Puzzle Generation

This repository contains the official code for our paper `Conditional Generation of Creative Chess Puzzles with Diffusion Models`.

**Authors:** [Aatu Selkee](https://openreview.net/profile?id=~Aatu_Selkee1), [Severi Rissanen](https://openreview.net/profile?id=~Severi_Rissanen1), [Xidong Feng](https://openreview.net/profile?id=~Xidong_Feng1), [Tom Zahavy](https://openreview.net/profile?id=~Tom_Zahavy2), and [Eric Malmi](https://openreview.net/profile?id=~Eric_Malmi1).

## Overview

Procedurally generating high-quality, aesthetic, and controllable chess puzzles is a complex task. Traditional methods rely on filtering positions from human play. In this project, we train a generative model for chess puzzle generation. Furthermore, we employ Reinforcement Learning via Denoising Diffusion Policy Optimization (DDPO) for further model training.

### Key Contributions
- **Masked Diffusion for Chess:** A novel application of discrete diffusion models for generating chess positions.
- **RL-based Alignment (DDPO):** Utilizing Stockfish evaluations and theme heuristics as reward signals to steer the diffusion process towards better puzzles.
- **Controllable Generation:** Generating puzzles conditional on themes, ratings, partial board states and best moves.

---

## Master's Thesis

This paper extends Aatu Selkee's master's thesis. The thesis was written with guidance from Adjunct Professor Eric Malmi and Professor Nuutti Hyvönen as well as all other co-authors. The thesis and presentation are presented in [`Thesis/`](Thesis/) and [`Thesis Presentation/`](Thesis%20Presentation/) for interested readers.

---

## Environment Setup

The environment requirements are specified in `environment.yml`.

To set up the environment, run:

```bash
conda env create -f environment.yml
conda activate environment
```

**Note:** You must have the [Stockfish engine](https://stockfishchess.org/download/) installed and accessible, or compiled inside the project directory structure (`./Stockfish/src/stockfish` by default).

---

## Project Structure

- `src/MaskedDiffusion/model.py`: Masked Diffusion architecture, training loss, and sampling methods.
- `src/MaskingSchedule/MaskingSchedule.py`: Linear, cosine, polynomial, and geometric masking schedules.
- `src/Config.py`: Configuration dataclass for tokenization, model architecture, and supervised training.
- `src/tokenization/`: FEN and move tokenization, theme and rating preprocessing, and dataset preparation.
- `src/scripts/training/`: Supervised and DDPO training scripts, replay buffer initialization, and legacy ESPO training scripts.
- `src/rl/espo.py`: Legacy ESPO losses, ELBO estimation, and grouped sampling utilities.
- `src/metrics/theme_conditioning.py`: Shared theme and move conditioning generation, theme matching rewards, and extra-theme counting.
- `src/metrics/`: Stockfish-based puzzle evaluation, theme detection, reward components, diversity measures, and replay buffer utilities.
- `src/scripts/evaluation/`: Checkpoint and Lichess evaluation, recomputation of metrics for existing puzzle CSVs, and distances within and between puzzle datasets.
- `src/scripts/generation/`: Position generation, partial-board conditioning, and branching diffusion trajectories.
- Slurm job scripts (`.sh`) are stored alongside their Python entry points in `src/scripts/training/`, `src/scripts/evaluation/`, and `src/scripts/generation/`.
- `src/analysis.ipynb`, `src/plots.ipynb`, and `src/move_visualization.ipynb`: Notebooks for analysis, plotting, and move visualization.

---

## Usage

Run the following commands from the repository root on an HPC cluster with Slurm. Before submitting jobs, adjust the settings in the corresponding `.sh` scripts to match your setup.

**Dataset preparation:** Before training, prepare the Lichess puzzle data using the preprocessing and tokenization utilities in `src/tokenization/`, and save the training and test splits as PyTorch datasets. Each sample should contain FEN tokens, best-move tokens, encoded themes, and a scaled rating, in that order. These dataset files are not included in the repository.

### 1. Supervised Training

To train the base Masked Diffusion model on a dataset of chess puzzles:

```bash
sbatch src/scripts/training/supervised.sh
```

### 2. Reinforcement Learning (DDPO)

To train a pre-trained supervised model using DDPO:

```bash
sbatch src/scripts/training/train_rl_ddpo.sh
```

### 3. Evaluation

- **Checkpoint Evaluation:** Generate and evaluate positions from every `.pt` checkpoint in a directory, saving one CSV per checkpoint.

  ```bash
  sbatch src/scripts/evaluation/evaluate_checkpoints.sh
  ```

- **Lichess Evaluation:** Evaluate puzzle legality, solution uniqueness, and counter-intuitiveness for a Lichess dataset.

  ```bash
  sbatch src/scripts/evaluation/evaluate_lichess.sh
  ```

- **Diversity and Distance:** Compute board and solution-line distances within generated puzzles and from generated puzzles to a Lichess reference dataset.

  ```bash
  sbatch src/scripts/evaluation/compute_distances.sh
  ```

- **Recompute Evaluations:** Recalculate puzzle metrics for existing CSVs.

  ```bash
  sbatch src/scripts/evaluation/recompute_evaluations.sh
  ```

### 4. Generation and Conditioning Experiments

- **Position Generation:** Generate and evaluate positions from a single checkpoint.

  ```bash
  sbatch src/scripts/generation/generate_positions.sh
  ```

- **Partial-Board Conditioning:** Generate positions with a specified partial board, themes, rating, and optionally a best move.

  ```bash
  sbatch src/scripts/generation/generate_from_partial_board.sh
  ```

- **Diffusion Trajectories:** Generate branching diffusion trajectories and evaluate their final positions.

  ```bash
  sbatch src/scripts/generation/compute_diffusion_trajectories.sh
  ```
