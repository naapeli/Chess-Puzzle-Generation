import argparse
import ast
import os
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import pandas as pd
from chess.engine import SimpleEngine, Limit

import metrics.themes as themes
from metrics.themes import legal, uniqueness, counter_intuitive, get_unique_puzzle_from_fen
from metrics.cook import cook
from rl.espo import theme_reward


engine_pool = queue.Queue()
stockfish_path = None

COLUMNS_ORDER = [
    "target_themes",
    "target_rating",
    "fen",
    "best_move",
    "is_legal",
    "is_puzzle",
    "counter_intuitive",
    "counter_intuitive_value",
    "actual_themes",
    "themes_match",
    "main_line",
]


def parse_themes(target_themes_val):
    if target_themes_val is None or pd.isna(target_themes_val) or target_themes_val == "":
        return None
    if isinstance(target_themes_val, (list, tuple, set)):
        return list(target_themes_val)
    if isinstance(target_themes_val, str):
        target_themes_val = target_themes_val.strip()
        if not target_themes_val:
            return None
        try:
            val = ast.literal_eval(target_themes_val)
            if isinstance(val, (list, tuple, set)):
                return list(val)
        except Exception:
            pass
        if "," in target_themes_val:
            return [t.strip().strip("'\"") for t in target_themes_val.strip("()[]").split(",") if t.strip()]
        return [t.strip().strip("'\"") for t in target_themes_val.strip("()[]").split() if t.strip()]
    return None


def process_row(row_dict, fen_col):
    global stockfish_path
    entry = dict(row_dict)
    fen = str(entry.get(fen_col, "")).strip() if pd.notna(entry.get(fen_col)) else ""

    entry["is_legal"] = False
    entry["is_puzzle"] = False
    entry["counter_intuitive"] = False
    entry["counter_intuitive_value"] = None
    entry["actual_themes"] = None
    entry["themes_match"] = None
    entry["main_line"] = None

    if not fen or not legal(fen):
        return entry

    entry["is_legal"] = True
    engine = engine_pool.get()

    try:
        try:
            engine.configure({"Clear Hash": None})
            ci_bool, ci_val = counter_intuitive(fen, engine, return_value=True)
            entry["counter_intuitive"] = ci_bool
            entry["counter_intuitive_value"] = ci_val

            is_puz = uniqueness(fen, engine)
            entry["is_puzzle"] = is_puz

            if is_puz:
                puzzle = get_unique_puzzle_from_fen(fen, engine)
                if puzzle is not None:
                    entry["main_line"] = " ".join([move.uci() for move in puzzle.mainline])
                    existing_themes = cook(puzzle, engine)
                    entry["actual_themes"] = existing_themes

                    target_themes = parse_themes(entry.get("target_themes"))
                    if target_themes is not None:
                        entry["themes_match"] = theme_reward(target_themes, existing_themes)
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


def main():
    global stockfish_path

    parser = argparse.ArgumentParser(description="Recompute evaluation metrics (uniqueness, counter-intuitive, themes) on existing positions CSV.")
    parser.add_argument("--input_file", "--input_path", "-i", type=str, required=True, help="Path to input CSV file or directory containing CSVs.")
    parser.add_argument("--output_file", "--output_path", "-o", type=str, default=None, help="Path to output CSV (or directory). If omitted and not in_place, adds '_recomputed' suffix.")
    parser.add_argument("--in_place", action="store_true", help="Overwrite the input file directly when finished.")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern to match files when input is a directory (default: '*.csv').")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively search for CSV files in subdirectories.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for writing to disk.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of concurrent Stockfish workers.")
    parser.add_argument("--n_puzzles", type=int, default=None, help="Optional maximum number of puzzles per file to recompute.")

    args = parser.parse_args()

    base_path = Path("./src")
    stockfish_path = base_path / ".." / "Stockfish" / "src" / "stockfish"

    def resolve_input(p_str):
        p = Path(p_str)
        if p.exists():
            return p
        if (base_path / p_str).exists():
            return base_path / p_str
        if (base_path / "Generate_positions" / p_str).exists():
            return base_path / "Generate_positions" / p_str
        return p

    input_path = resolve_input(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {args.input_file} not found.")

    def resolve_output_dir(p_str):
        p = Path(p_str)
        if p.is_absolute() or str(p).startswith("src/"):
            return p
        return base_path / "Generate_positions" / p

    if input_path.is_dir():
        glob_fn = input_path.rglob if args.recursive else input_path.glob
        raw_files = sorted(glob_fn(args.pattern))
        # Filter out hidden files, temporary files, distance matrices, and existing recomputed files if not in-place
        input_files = [
            f for f in raw_files
            if f.is_file()
            and f.name != "distances.csv"
            and not f.name.startswith(".")
            and not f.name.endswith("_tmp.csv")
            and (args.in_place or args.output_file is not None or not f.stem.endswith("_recomputed"))
        ]
        if not input_files:
            raise FileNotFoundError(f"No matching CSV files found in {input_path} matching pattern '{args.pattern}'")

        if args.output_file:
            out_dir = resolve_output_dir(args.output_file)
        else:
            out_dir = input_path

        files_to_process = []
        for f_in in input_files:
            if args.in_place:
                f_out = f_in
            elif args.output_file:
                rel = f_in.relative_to(input_path) if args.recursive else Path(f_in.name)
                f_out = out_dir / rel
            else:
                f_out = f_in.with_name(f"{f_in.stem}_recomputed.csv")
            files_to_process.append((f_in, f_out))
    else:
        if args.in_place:
            f_out = input_path
        elif args.output_file:
            p_out = Path(args.output_file)
            if p_out.is_dir() or args.output_file.endswith("/"):
                out_dir = resolve_output_dir(args.output_file)
                f_out = out_dir / input_path.name
            else:
                f_out = resolve_output_dir(args.output_file) if not p_out.is_absolute() and not str(p_out).startswith("src/") else p_out
        else:
            f_out = input_path.with_name(f"{input_path.stem}_recomputed.csv")
        files_to_process = [(input_path, f_out)]

    print(f"\nFound {len(files_to_process)} file(s) to process:", flush=True)
    for idx, (f_in, f_out) in enumerate(files_to_process, 1):
        print(f" [{idx}/{len(files_to_process)}] {f_in} -> {f_out}", flush=True)

    # Determine CPU workers
    if args.n_jobs is None:
        slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("SLURM_CPUS_PER_GPU")
        if slurm_cpus:
            args.n_jobs = max(1, int(slurm_cpus) - 2)
        else:
            args.n_jobs = max(1, os.cpu_count() - 2)

    print(f"\nInitializing engine pool with {args.n_jobs} workers...", flush=True)
    for _ in range(args.n_jobs):
        engine = SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 1, "Hash": 128})
        engine_pool.put(engine)

    try:
        for file_idx, (f_in, f_out) in enumerate(files_to_process, 1):
            print(f"\n=======================================================", flush=True)
            print(f"[{file_idx}/{len(files_to_process)}] Processing: {f_in} -> {f_out}", flush=True)
            print(f"=======================================================", flush=True)

            f_out.parent.mkdir(parents=True, exist_ok=True)
            temp_out = f_out.with_name(f".{f_out.stem}_tmp.csv") if f_in == f_out else f_out

            try:
                df = pd.read_csv(f_in)
            except Exception as e:
                print(f"Error reading CSV file {f_in}: {e}. Skipping.", flush=True)
                continue

            target_len = len(df)
            if args.n_puzzles is not None and args.n_puzzles > 0 and target_len > args.n_puzzles:
                df = df.head(args.n_puzzles)
                target_len = args.n_puzzles

            fen_col = None
            for col in ["fen", "Puzzle_FEN"]:
                if col in df.columns:
                    fen_col = col
                    break
            if fen_col is None:
                fen_cols = [c for c in df.columns if "fen" in c.lower()]
                if fen_cols:
                    fen_col = fen_cols[0]
                else:
                    print(f"Error: No FEN column found in {f_in}. Columns: {list(df.columns)}. Skipping.", flush=True)
                    continue

            # Check if target output file already exists and is complete
            if f_in != f_out and f_out.exists() and f_out.stat().st_size > 0:
                try:
                    with open(f_out, "r", encoding="utf-8") as f:
                        completed_count = max(0, sum(1 for _ in f) - 1)
                    if completed_count >= target_len:
                        print(f"Skipping {f_in.name}: already fully recomputed at {f_out} ({completed_count}/{target_len} rows).", flush=True)
                        continue
                except Exception:
                    pass

            # Check for existing progress to resume in temp_out
            start_idx = 0
            if temp_out.exists() and temp_out.stat().st_size > 0:
                try:
                    with open(temp_out, "r", encoding="utf-8") as f:
                        start_idx = max(0, sum(1 for _ in f) - 1)
                    print(f"Found existing progress: {start_idx}/{target_len} rows already recomputed. Resuming...", flush=True)
                except Exception:
                    start_idx = 0

            if start_idx >= target_len:
                print(f"All {target_len} rows already recomputed in {temp_out}.", flush=True)
                if temp_out != f_out and temp_out.exists():
                    temp_out.replace(f_out)
                continue

            remaining_df = df.iloc[start_idx:]
            total_remaining = len(remaining_df)
            batch_size = args.batch_size
            num_batches = (total_remaining + batch_size - 1) // batch_size
            print(f"Recomputing remaining {total_remaining} positions in {num_batches} batches (batch size {batch_size})...", flush=True)

            start_time = perf_counter()
            executor = ThreadPoolExecutor(max_workers=args.n_jobs)
            processed_count = start_idx

            try:
                for b in range(num_batches):
                    b_start = perf_counter()
                    b_df = remaining_df.iloc[b * batch_size : (b + 1) * batch_size]
                    batch_rows = b_df.to_dict(orient="records")
                    batch_args = [(row, fen_col) for row in batch_rows]

                    batch_results = list(executor.map(lambda p: process_row(*p), batch_args))
                    df_batch = pd.DataFrame(batch_results)

                    # Preserve canonical column ordering
                    ordered_cols = [c for c in COLUMNS_ORDER if c in df_batch.columns]
                    other_cols = [c for c in df_batch.columns if c not in ordered_cols]
                    df_batch = df_batch[ordered_cols + other_cols]

                    write_header = (not temp_out.exists() or temp_out.stat().st_size == 0)
                    df_batch.to_csv(temp_out, mode="a", index=False, header=write_header)

                    processed_count += len(batch_results)
                    b_duration = perf_counter() - b_start
                    print(f"Batch {b + 1}/{num_batches} done in {b_duration:.2f}s. Saved: {processed_count}/{len(df)} -> {temp_out}", flush=True)
            finally:
                executor.shutdown(wait=True)

            if temp_out != f_out and temp_out.exists():
                temp_out.replace(f_out)

            total_duration = perf_counter() - start_time
            print(f"Successfully finished {f_out} ({processed_count} positions) in {total_duration:.2f}s.", flush=True)

    finally:
        print("Shutting down engine pool...", flush=True)
        while not engine_pool.empty():
            try:
                engine = engine_pool.get_nowait()
                engine.quit()
            except Exception:
                break
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
