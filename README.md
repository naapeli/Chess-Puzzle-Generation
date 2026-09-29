# Generating Open-Source Chess Puzzles

This repository contains the official code for our paper `Conditional Generation of Creative Chess Puzzles with Diffusion Models`.

## Overview

Procedurally generating high-quality, aesthetic, and controllable chess puzzles is a complex task. Traditional methods rely on filtering positions from human play. In this project, we train a generative model for chess puzzle generation. Furthermore, we employ Reinforcement Learning via Denoising Diffusion Policy Optimization (DDPO) for further model training.

### Key Contributions
- **Masked Diffusion for Chess:** A novel application of discrete diffusion models for generating chess positions.
- **RL-based Alignment (DDPO):** Utilizing Stockfish evaluations and theme heuristics as reward signals to steer the diffusion process towards better puzzles.
- **Controllable Generation:** Generating puzzles conditional on themes, ratings, partial board states and best moves.

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
- `src/supervised_training.py`: Supervised training of the base model.
- `src/train_rl_ddpo.py` and `src/train_rl_espo.py`: Reinforcement learning training entry points for DDPO and ESPO. ESPO is legacy code for old experiments.
- `src/rl/espo.py`: Shared RL utilities, including ELBO estimation, policy losses, grouped sampling, and theme rewards.
- `src/metrics/`: Stockfish-based puzzle evaluation, theme detection, reward components, diversity measures, and replay buffer utilities.
- `src/generate_positions.py` and `src/generate_from_partial_board.py`: Generate positions from checkpoints or complete partially specified boards.
- `src/evaluate_checkpoints.py`, `src/evaluate_lichess.py`, and `src/compute_distances.py`: Evaluate model checkpoints and Lichess puzzles, and measure distances within and between puzzle datasets.
- `src/compute_diffusion_trajectories.py`: Generates and evaluates branching diffusion trajectories.
- `src/*.sh`: Slurm job scripts for training, generation, and evaluation.
- `src/analysis.ipynb`, `src/plots.ipynb`, and `src/move_visualization.ipynb`: Notebooks for analysis, plotting, and move visualization.

---

## Usage

Run the following commands from the repository root on an HPC cluster with Slurm. Before submitting jobs, adjust the settings in the corresponding `.sh` scripts to match your setup.

**Dataset preparation:** Before training, prepare the Lichess puzzle data using the preprocessing and tokenization utilities in `src/tokenization/`, and save the training and test splits as PyTorch datasets. Each sample should contain FEN tokens, best-move tokens, encoded themes, and a scaled rating, in that order. These dataset files are not included in the repository.

### 1. Supervised Training

To train the base Masked Diffusion model on a dataset of chess puzzles:

```bash
sbatch src/supervised.sh
```

### 2. Reinforcement Learning (DDPO)

To train a pre-trained supervised model using DDPO:

```bash
sbatch src/train_rl_ddpo.sh
```

### 3. Evaluation

- **Checkpoint Evaluation:** Generate and evaluate positions from every `.pt` checkpoint in a directory, saving one CSV per checkpoint.

  ```bash
  sbatch src/evaluate_checkpoints.sh
  ```

- **Lichess Evaluation:** Evaluate puzzle legality, solution uniqueness, and counter-intuitiveness for a Lichess dataset.

  ```bash
  sbatch src/evaluate_lichess.sh
  ```

- **Diversity and Distance:** Compute board and solution-line distances within generated puzzles and from generated puzzles to a Lichess reference dataset.

  ```bash
  sbatch src/compute_distances.sh
  ```

### 4. Generation and Conditioning Experiments

- **Position Generation:** Generate and evaluate positions from a single checkpoint.

  ```bash
  sbatch src/generate_positions.sh
  ```

- **Partial-Board Conditioning:** Generate positions with a specified partial board, themes, rating, and optionally a best move.

  ```bash
  sbatch src/generate_from_partial_board.sh
  ```

- **Diffusion Trajectories:** Generate branching diffusion trajectories and evaluate their final positions.

  ```bash
  sbatch src/compute_diffusion_trajectories.sh
  ```
