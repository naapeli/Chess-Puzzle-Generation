---
license: mit
tags:
- diffusers
- chess
- custom-pipeline
---

# Chess Puzzle Generator

A masked diffusion model for generating chess puzzles conditioned on themes, ratings, moves and partial boards. This model is not guaranteed to generate a puzzle and the generations should be filtered afterward.

## Models

We provide two models both with 268M parameters. The main model, called V2, is trained mainly for generating as many positions with a unique solution that match the themes the user asked for. In contrast, the model called V1 was mainly trained to maximize counter-intuitivity and uniqueness instead of thematic accuracy. The two models can be obtained by using the revision parameter.


## Pipeline Documentation

### `ChessPuzzlePipeline.__call__`

```python
pipeline(
    themes: list[PuzzleTheme] | list[list[PuzzleTheme]] | None = None,
    rating: float | list[float] = 1500.0,
    partial_board: str | list[str] | None = None,
    best_move: str | list[str] | None = None,
    batch_size: int = 1,
    steps: int = 256,
    temperature: float = 1.0,
    schedule: Schedule = Schedule.linear,
    move_generation_order: MoveGenerationOrder = MoveGenerationOrder.simultaneous,
) -> list[Position]
```

### Parameters

- **`themes`** (`list[PuzzleTheme] | list[list[PuzzleTheme]]`, optional, default: `None`):  
  The thematic tags to condition the puzzle generation on as a list of `PuzzleTheme` enum members (e.g. `[pipeline.Theme.mateIn2, pipeline.Theme.middlegame]`). Can also be a list of theme lists per position when generating a batch (e.g. `[[pipeline.Theme.fork], [pipeline.Theme.mateIn1]]`).

- **`rating`** (`float | list[float]`, optional, default: `1500.0`):  
  Target puzzle difficulty rating. Scaled based on Lichess puzzle ratings. Can be a single float or a list of floats corresponding to each position in `batch_size`.

- **`partial_board`** (`str | list[str]`, optional, default: `None`):  
  A partial FEN string (or list of strings) to condition on, where unknown squares/fields are represented with `?` and empty squares with `.`.
  *Example:* `"?????rk./?????ppp/????????/????????/????????/???B????/????????/???????? w ??-- - ? ?"`

- **`best_move`** (`str | list[str]`, optional, default: `None`):  
  A UCI-format move string (or list of strings) to force as the solution (e.g. `"d3h7"`, `"e7e8q"`, `"e2??"`). Can also contain `?` for unknown characters.

- **`batch_size`** (`int`, optional, default: `1`):  
  Number of puzzle positions to generate in parallel. Automatically inferred if conditioning variables are provided as lists.

- **`steps`** (`int`, optional, default: `256`):  
  Number of discrete diffusion unmasking steps. Higher steps generally yield higher quality and more valid positions, but lower values work as well. Tested values between 16 and 256.

- **`temperature`** (`float`, optional, default: `1.0`):  
  Sampling temperature applied to the unmasking logits. Lower values make sampling more greedy/deterministic.

- **`schedule`** (`Schedule`, optional, default: `Schedule.linear`):  
  Noise schedule used for unmasking tokens using the `pipeline.Schedule` enum:
  - `pipeline.Schedule.linear`
  - `pipeline.Schedule.cosine`
  - `pipeline.Schedule.geometric`
  - `pipeline.Schedule.polynomial`

- **`move_generation_order`** (`MoveGenerationOrder`, optional, default: `MoveGenerationOrder.simultaneous`):  
  The order in which the board and move tokens are generated:
  - `MoveGenerationOrder.simultaneous`: Board and move tokens are generated in a single phase.
  - `MoveGenerationOrder.first`: Move tokens are generated first, then the board tokens.
  - `MoveGenerationOrder.last`: Board tokens are generated first, then the move tokens.

### Return Value

Returns a `list[Position]` of length `batch_size`, where each `Position` is a dataclass:
```python
@dataclass
class Position:
    fen: str
    move: str | None
```

You can access the generated FEN and move directly as attributes:
```python
position = results[0]
print(position.fen)   # "2nrb3/n2k2qp/1ppp4/4p3/5P2/R7/1PPBP1PP/R4NK1 w - - 0 22"
print(position.move)  # "a3a7"
```

---

### Available Themes

All 66 supported themes can be accessed via `pipeline.Theme.<name>` and passed as a list of theme enum objects:

| Category | Available Themes |
| :--- | :--- |
| **State-of-game** | `opening`, `middlegame`, `endgame` |
| **Type-of-endgame** | `pawnEndgame`, `bishopEndgame`, `knightEndgame`, `rookEndgame`, `queenEndgame`, `queenRookEndgame` |
| **Type-of-checkmate** | `mate`, `backRankMate`, `bodenMate`, `smotheredMate`, `hookMate`, `doubleBishopMate`, `arabianMate`, `dovetailMate`, `anastasiaMate`, `triangleMate`, `balestraMate`, `killBoxMate`, `blindSwineMate`, `cornerMate`, `vukovicMate` |
| **Length-of-checkmate** | `mateIn1`, `mateIn2`, `mateIn3`, `mateIn4`, `mateIn5` |
| **Length-of-puzzle** | `oneMove`, `short`, `long`, `veryLong` |
| **Winning** | `crushing`, `advantage` |
| **Other** | `hangingPiece`, `fork`, `interference`, `kingsideAttack`, `zugzwang`, `exposedKing`, `skewer`, `pin`, `quietMove`, `discoveredAttack`, `sacrifice`, `deflection`, `advancedPawn`, `attraction`, `promotion`, `queensideAttack`, `defensiveMove`, `attackingF2F7`, `clearance`, `intermezzo`, `equality`, `trappedPiece`, `xRayAttack`, `capturingDefender`, `doubleCheck`, `enPassant`, `castling`, `underPromotion`, `master`, `masterVsMaster`, `superGM` |

---

## Usage

```python
import torch
from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = DiffusionPipeline.from_pretrained(
    "naapeli/chess-puzzle-generator",
    trust_remote_code=True,
)
pipeline.to(device)

# pipeline = DiffusionPipeline.from_pretrained(
#     "naapeli/chess-puzzle-generator",
#     revision="V1",
#     trust_remote_code=True,
# )
# pipeline.to(device)

themes = pipeline.Theme
schedules = pipeline.Schedule
move_generation_orders = pipeline.MoveGenerationOrder

# 1. Unconditional generation conditioned on themes and rating
results = pipeline(
    themes=[themes.mateIn2, themes.middlegame],
    rating=1800,
    batch_size=1,
    steps=64,
    schedule=schedules.linear,
    move_generation_order=move_generation_orders.first
)
print(results[0].fen, results[0].move)

# 2. Condition on a partial board and a best move
partial_fen = "?????rk?/?????ppp/????????/????????/????????/???B????/????????/???????? w ??-- - ? ?"
best_move = "d3h7"
results = pipeline(
    themes=[themes.mate],
    rating=1600,
    partial_board=partial_fen,
    best_move=best_move,
    batch_size=1,
    steps=256,
    schedule=schedules.cosine,
)
print(results[0].fen, results[0].move)

# 3. Per-position conditioning variables in batch generation
results = pipeline(
    themes=[themes.middlegame, themes.long, themes.sacrifice],  # same themes
    best_move=["???3", "???4", "???1"],  # unique best_moves (not guaranteed, but used for conditioning)
    rating=2000,  # same ratings
    batch_size=3,
    steps=16,
)
for i, pos in enumerate(results, start=1):
    print(f"Batch {i}: {pos.fen} | move: {pos.move}")
```
