import sys
from pathlib import Path

# Keep src imports available when this file is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import queue
import torch
import pandas as pd
from chess.engine import SimpleEngine
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import os
import re

from MaskedDiffusion.model import MaskedDiffusion
from metrics.theme_conditioning import generate_random_themes, theme_reward
from tokenization.tokenization import theme_preprocessor, scale_ratings, tokens_to_fen, tokens_to_move, unscale_ratings
from metrics.themes import legal, get_unique_puzzle_from_fen, counter_intuitive, uniqueness
from metrics.cook import cook
from MaskingSchedule.MaskingSchedule import string_to_schedule


torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint .pt files")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the evaluated CSVs")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=512)
parser.add_argument("--context_dataset", choices=["train", "test", "random"], default="test")
parser.add_argument("--generate_move_last", action="store_true")
parser.add_argument("--n_fens", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=32768)
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_path = Path("./src")

checkpoint_dir = Path(args.checkpoint_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Find all checkpoints
checkpoint_paths = list(checkpoint_dir.glob("*.pt"))
if not checkpoint_paths:
    raise ValueError(f"No checkpoint .pt files found in {checkpoint_dir}")

def get_checkpoint_step(path):
    match = re.search(r'\d+', path.name)
    return int(match.group()) if match else path.name

try:
    checkpoint_paths = sorted(checkpoint_paths, key=get_checkpoint_step)
except Exception:
    checkpoint_paths = sorted(checkpoint_paths)

print(f"Found {len(checkpoint_paths)} checkpoints to evaluate in order:")
for p in checkpoint_paths:
    print(f" - {p.name}")

# Setup Stockfish pool once
slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("SLURM_CPUS_PER_GPU")
if slurm_cpus is not None:
    n_jobs = int(slurm_cpus) - 2
else:
    n_jobs = max(1, os.cpu_count() - 2)

stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"
engine_pool = queue.Queue()
for _ in range(n_jobs):
    engine = SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 32})
    engine_pool.put(engine)

# Dataset caching
dataset = None
def get_dataset(use_context, context_dataset):
    global dataset
    if dataset is not None:
        return dataset
    if use_context and context_dataset in ["train", "test"]:
        dataset_name = "trainset.pt" if context_dataset == "train" else "testset.pt"
        dataset = torch.load(base_path / "dataset" / "with_best_move" / dataset_name, weights_only=False, map_location="cpu")
        return dataset
    return None

def process_puzzle(fen_tokens, move_tokens, base_theme, base_rating, device):
    entry = {
        "target_themes": base_theme,
        "target_rating": base_rating,
        "fen": None,
        "best_move": None,
        "is_legal": False,
        "is_puzzle": False,
        "counter_intuitive": None,
        "counter_intuitive_value": None,
        "actual_themes": None,
        "themes_match": None,
        "main_line": None
    }

    engine = engine_pool.get()
    try:
        try:
            fen = tokens_to_fen(fen_tokens)
            entry["fen"] = fen
            if move_tokens is not None:
                move = tokens_to_move(move_tokens)
                entry["best_move"] = move
        except:
            return entry

        if not legal(fen):
            return entry
        
        entry["is_legal"] = True
        
        try:
            engine.configure({"Clear Hash": None})
            entry["counter_intuitive"], entry["counter_intuitive_value"] = counter_intuitive(fen, engine, return_value=True)
            entry["is_puzzle"] = uniqueness(fen, engine)
            
            if entry["is_puzzle"]:
                puzzle = get_unique_puzzle_from_fen(fen, engine)
                if puzzle is not None:
                    entry["main_line"] = " ".join([move.uci() for move in puzzle.mainline])
                    existing_themes = cook(puzzle, engine)
                    entry["actual_themes"] = existing_themes
                    
                    if base_theme is not None:
                        entry["themes_match"] = theme_reward(base_theme, existing_themes)
        except Exception:
            try:
                engine.quit()
            except Exception:
                pass
            engine = SimpleEngine.popen_uci(stockfish_path)
            engine.configure({"Threads": 1, "Hash": 32})

        return entry
    finally:
        engine_pool.put(engine)

executor = ThreadPoolExecutor(max_workers=n_jobs)
try:
    for cp_path in checkpoint_paths:
        output_file_name = cp_path.with_suffix(".csv").name
        output_path = output_dir / output_file_name
        
        existing_count = 0
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_count = max(0, sum(1 for _ in f) - 1)
            except Exception:
                existing_count = 0

        if existing_count >= args.n_fens:
            print(f"\nSkipping {cp_path.name} as output already has {existing_count}/{args.n_fens} positions at {output_path}", flush=True)
            continue
            
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
        
        config = checkpoint["config"]
        model = MaskedDiffusion(config)
        state_dict = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state_dict)
        model.to(device=device)
        model.eval()
        
        remaining = args.n_fens - existing_count
        batch_size = args.batch_size
        iteration = 0
        total_iterations = (remaining + batch_size - 1) // batch_size
        
        if existing_count > 0:
            print(f"\n=== Resuming checkpoint {cp_path.name}: {existing_count}/{args.n_fens} already evaluated, generating remaining {remaining} positions ===", flush=True)
        else:
            print(f"\n=== Evaluating checkpoint: {cp_path.name} ===", flush=True)
        
        import threading
        prev_cpu_thread = None
        prev_results = []

        while remaining > 0 or prev_cpu_thread is not None:
            if remaining > 0:
                current_batch_size = min(batch_size, remaining)
                iteration += 1
                print(f"Iteration {iteration}/{total_iterations} (sampling GPU batch of size {current_batch_size})", flush=True)
                
                if config.use_context:
                    if args.context_dataset == "random":
                        themes, ratings = generate_random_themes(current_batch_size, lichess_distribution=False)
                        base_themes = themes
                        base_ratings = ratings.tolist()
                        themes_one_hot = torch.from_numpy(theme_preprocessor.transform(themes)).to(device=device, dtype=torch.float32)
                        scaled_ratings = scale_ratings(ratings).to(device=device, dtype=torch.float32)
                    else:
                        ds = get_dataset(config.use_context, args.context_dataset)
                        indices = torch.randint(0, len(ds), (current_batch_size,)).tolist()
                        sampled_items = [ds[idx] for idx in indices]
                        sampled_themes = torch.stack([torch.as_tensor(item[2]) for item in sampled_items])
                        sampled_ratings = torch.stack([torch.as_tensor(item[3]) for item in sampled_items])
                        
                        base_themes = theme_preprocessor.inverse_transform(sampled_themes.numpy())
                        base_ratings = unscale_ratings(sampled_ratings).tolist()
                        
                        themes_one_hot = sampled_themes.to(device=device, dtype=torch.float32)
                        scaled_ratings = sampled_ratings.to(device=device, dtype=torch.float32)
                else:
                    themes_one_hot = None
                    scaled_ratings = None
                    base_themes = None
                    base_ratings = None
                    
                module = model.module if hasattr(model, "module") else model
                start = perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    tokens = module.sample(
                        themes_one_hot,
                        scaled_ratings,
                        batch_size=current_batch_size,
                        steps=args.steps,
                        temperature=args.temperature,
                        generate_move_last=args.generate_move_last
                    )
                print(f"GPU Sampling time: {perf_counter() - start:.4f}s", flush=True)
                
                tokens_cpu = tokens.cpu()
                if config.predict_moves:
                    fen_tokens = tokens_cpu[:, :config.fen_length]
                    move_tokens = tokens_cpu[:, config.fen_length:]
                else:
                    fen_tokens = tokens_cpu
                    move_tokens = None
                    
                args_list = [
                    (
                        fen_tokens[i],
                        move_tokens[i] if move_tokens is not None else None,
                        base_themes[i] if themes_one_hot is not None else None,
                        base_ratings[i] if scaled_ratings is not None else None,
                        device
                    ) for i in range(current_batch_size)
                ]
                
                remaining -= current_batch_size
                has_current_batch = True
            else:
                has_current_batch = False

            # Wait for previous batch's CPU processing to finish and write to CSV
            if prev_cpu_thread is not None:
                prev_cpu_thread.join()
                df_batch = pd.DataFrame(prev_results)
                write_header = not output_path.exists() or os.path.getsize(output_path) == 0
                df_batch.to_csv(output_path, mode='a', index=False, header=write_header)
                prev_cpu_thread = None

            # Launch CPU processing for current batch in background thread while GPU moves to next batch
            if has_current_batch:
                current_results = []
                def run_cpu_evaluation(al, res_out):
                    start_cpu = perf_counter()
                    res = list(executor.map(lambda p: process_puzzle(*p), al))
                    res_out.extend(res)
                    print(f"CPU Stockfish evaluation time: {perf_counter() - start_cpu:.4f}s", flush=True)

                prev_results = current_results
                prev_cpu_thread = threading.Thread(target=run_cpu_evaluation, args=(args_list, current_results))
                prev_cpu_thread.start()

        print(f"Saved results to {output_path}", flush=True)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

finally:
    executor.shutdown(wait=True)
    print("Shutting down Stockfish pool...", flush=True)
    while not engine_pool.empty():
        try:
            engine = engine_pool.get_nowait()
            engine.quit()
        except queue.Empty:
            break
    print("Done.", flush=True)


