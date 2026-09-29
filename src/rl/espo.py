import torch
import numpy as np
import pandas as pd

import random


n_quadrature = 3
t_points, quadrature_weights = np.polynomial.legendre.leggauss(n_quadrature)  # estimate the integral with a gaussian quadrature (https://arxiv.org/pdf/2510.08554)
t_points, quadrature_weights = torch.from_numpy(t_points), torch.from_numpy(quadrature_weights)
t_points, quadrature_weights = (1 - 0) / 2 * t_points + (1 + 0) / 2, (1 - 0) / 2 * quadrature_weights  # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval


def kl_estimate(model_elbo, reference_elbo, kind="k2"):
    # k2 from http://joschu.net/blog/kl-approx.html and https://arxiv.org/pdf/2512.03759 (biased, but low variance estimator)
    if kind == "k2":
        return 0.5 * (model_elbo - reference_elbo) ** 2
    elif kind == "k3":
        return torch.exp(reference_elbo - model_elbo) - 1 - (reference_elbo - model_elbo)
    return model_elbo - reference_elbo  # unbiased basic MC estimate

def entropy(elbo, sequence_length):
    entropy_vals = -elbo / sequence_length
    return entropy_vals

def compute_elbo(model, fens, themes=None, ratings=None, mask=None, return_mask=False):
    device = fens.device
    n_samples = len(fens)

    config = model.module.config if hasattr(model, "module") else model.config
    module = model.module if hasattr(model, "module") else model

    t = t_points.repeat(n_samples).to(device)
    alpha_t = config.masking_schedule(t)
    quadrature_weight = quadrature_weights.unsqueeze(0).to(device)

    if themes is None:
        fens = fens.repeat_interleave(n_quadrature, dim=0)
    else:
        fens = fens.repeat_interleave(n_quadrature, dim=0)
        themes = themes.repeat_interleave(n_quadrature, dim=0)
        ratings = ratings.repeat_interleave(n_quadrature, dim=0)

    random_mask = torch.rand(fens.size(), device=device) < alpha_t.unsqueeze(1) if mask is None else mask
    masked_fens = torch.where(random_mask, fens, config.mask_token)

    logits = model(masked_fens, themes, ratings)
    elbo = module.elbo_loss(t, logits, fens, masked_fens)
    elbo = (quadrature_weight * elbo.reshape(n_samples, n_quadrature)).sum(dim=1)
    elbo = -elbo  # model.elbo_loss returns an upper bound of the negative log likelihood, which we minimized during supervised training
    assert (elbo <= 0).all(), f"elbo should be a lower bound of a probability, {elbo}"
    if return_mask:
        return elbo, random_mask, t
    return elbo

def compute_elbo_basic(model, fens, themes=None, ratings=None, mask=None, t=None, return_mask=False, n_mc=n_quadrature):
    device = fens.device
    n_samples = len(fens)

    config = model.module.config if hasattr(model, "module") else model.config
    module = model.module if hasattr(model, "module") else model

    if themes is None:
        fens = fens.repeat_interleave(n_mc, dim=0)
    else:
        fens = fens.repeat_interleave(n_mc, dim=0)
        themes = themes.repeat_interleave(n_mc, dim=0)
        ratings = ratings.repeat_interleave(n_mc, dim=0)

    total_samples = n_samples * n_mc
    t = ((torch.rand(1) + torch.arange(total_samples) / total_samples) % 1).to(device)[torch.randperm(total_samples, device=device)] if t is None else t
    alpha_t = config.masking_schedule(t)

    random_mask = torch.rand(fens.size(), device=device) < alpha_t.unsqueeze(1) if mask is None else mask
    masked_fens = torch.where(random_mask, fens, config.mask_token)

    logits = model(masked_fens, themes, ratings)
    elbo = module.elbo_loss(t, logits, fens, masked_fens)
    elbo = elbo.reshape(n_samples, n_mc).mean(dim=1)
    elbo = -elbo  # model.elbo_loss returns an upper bound of the negative log likelihood, which we minimized during supervised training
    assert (elbo <= 0).all(), f"elbo should be a lower bound of a probability, {elbo}"
    if return_mask:
        return elbo, random_mask, t
    return elbo

def espo_loss(model, reference_elbos, old_elbos, fens, themes, ratings, rewards, group_size, mask=None, t=None, eps=0.2, beta=0.1):
    n_samples, sequence_length = fens.shape  # fens.shape == (batch_size * group_size, sequence_length)  batch_size groups of size group_size
    assert n_samples % group_size == 0
    batch_size = n_samples // group_size

    device = fens.device
    elbo = compute_elbo(model, fens, themes, ratings, mask=mask, return_mask=False)
    # elbo = compute_elbo_basic(model, fens, themes, ratings, mask=mask, t=t, return_mask=False)

    rewards = rewards.reshape(batch_size, group_size)
    elbo = elbo.reshape(batch_size, group_size) / sequence_length
    reference_elbos = reference_elbos.reshape(batch_size, group_size) / sequence_length
    old_elbos = old_elbos.reshape(batch_size, group_size) / sequence_length
    # this assumes that elbo and old_elbos are negative (lower bounds of the log probability)
    rho = torch.exp(elbo - old_elbos)  # importance sampling (generate from old model, update current model)

    # (batch_size,)
    advantages = ((rewards - rewards.mean(dim=1, keepdim=True))).to(device)  # Dr GRPO (maybe do not normalize by the standard deviation) https://arxiv.org/pdf/2503.20783
    #  / (rewards.std(dim=1, keepdim=True) + 1e-8)
    coef_1 = rho * advantages
    coef_2 = torch.clamp(rho, 1 - eps, 1 + eps) * advantages
    loss = torch.minimum(coef_1, coef_2).mean(dim=1)
    is_clipped = (coef_2 < coef_1).flatten()
    kl = kl_estimate(elbo, reference_elbos).mean(dim=1)
    loss = loss - beta * kl
    return -loss, kl, is_clipped  # maximize the loss above

def critic_free_ppo_loss(model, reference_elbos, old_elbos, fens, themes, ratings, rewards, group_size, mask=None, t=None, eps=0.2, beta=0.1):
    _, sequence_length = fens.shape
    # we still use the ELBO instead of the log probability
    assert group_size == 1, "group_size must be 1 for critic-free PPO."

    device = ratings.device
    elbo = compute_elbo(model, fens, themes, ratings, mask=mask, return_mask=False)
    # elbo = compute_elbo_basic(model, fens, themes, ratings, mask=mask, t=t, return_mask=False)

    # this assumes that elbo and old_elbos are negative (lower bounds of the log probability)
    rho = torch.exp((elbo - old_elbos) / sequence_length)  # importance sampling (generate from old model, update current model)

    advantages = (rewards - rewards.mean(dim=0, keepdim=True)).to(device)
    coef_1 = rho * advantages
    coef_2 = torch.clamp(rho, 1 - eps, 1 + eps) * advantages
    loss = torch.minimum(coef_1, coef_2)
    is_clipped = coef_2 < coef_1
    kl = kl_estimate(elbo / sequence_length, reference_elbos / sequence_length)
    loss = loss - beta * kl
    return -loss, kl, is_clipped  # maximize the loss above


def generate_grouped_positions(model, themes, ratings, group_size, batch_size, steps=256, temperature=1.0, generate_move_last=True):
    if themes is not None:
        themes = themes.repeat_interleave(group_size, dim=0)
    if ratings is not None:
        ratings = ratings.repeat_interleave(group_size, dim=0)

    module = model.module if hasattr(model, "module") else model

    fens = module.sample(themes, ratings, steps=steps, batch_size=batch_size * group_size, temperature=temperature, generate_move_last=generate_move_last)
    return fens, themes, ratings

state_of_game_tokens = ("opening", "middlegame", "endgame")
endgames = ("pawnEndgame", "bishopEndgame", "knightEndgame", "rookEndgame", "queenEndgame", "queenRookEndgame")

is_mate = "mate"
mate_lengths = ("mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5")
types_of_mate = ("backRankMate", "bodenMate", "smotheredMate", "hookMate", "doubleBishopMate", "arabianMate", "dovetailMate", "anastasiaMate")

lengths = ("oneMove", "short", "long", "veryLong")

winnings = ("crushing", "advantage")
other = ("hangingPiece", "fork", "interference", "kingsideAttack", "zugzwang", "exposedKing", "skewer", "pin", "quietMove", "discoveredAttack", "sacrifice", "deflection", "advancedPawn", "attraction", "promotion", "queensideAttack", "defensiveMove", "attackingF2F7", "clearance", "intermezzo", "equality", "trappedPiece", "xRayAttack", "capturingDefender", "doubleCheck", "enPassant", "castling", "underPromotion")

dataset = None
# def generate_random_themes(batch_size, lichess_distribution=False):
#     global dataset
#     if lichess_distribution:
#         if dataset is None:
#             dataset = pd.read_csv("./src/dataset/dataset.csv")
#         rows = dataset.sample(n=batch_size)
#         themes = rows["Themes"].str.split(" ").to_list()
#         ratings = torch.from_numpy(rows["Rating"].to_numpy())
#     else:
#         themes = []
#         for _ in range(batch_size):
#             position_themes = [random.choice(lengths)]
#             state_of_game = random.choice(state_of_game_tokens)
#             position_themes.append(state_of_game)

#             if state_of_game == "endgame":
#                 position_themes.append(random.choice(endgames))
            
#             if torch.rand(1) < 0.1:
#                 position_themes.append(is_mate)
#                 position_themes.append(random.choice(mate_lengths))
#                 position_themes.append(random.choice(types_of_mate))
#             else:
#                 position_themes.append(random.choice(winnings))
#                 position_themes.append(random.choice(other))

#             themes.append(position_themes)
        
#         ratings = 3000 * torch.rand((batch_size,)) + 300

#     return themes, ratings
def generate_random_themes(batch_size, lichess_distribution=False):
    global dataset
    if lichess_distribution:
        if dataset is None:
            dataset = pd.read_csv("./src/dataset/dataset.csv")
        rows = dataset.sample(n=batch_size)
        themes = rows["Themes"].str.split(" ").to_list()
        ratings = torch.from_numpy(rows["Rating"].to_numpy())
    else:
        themes = []
        for _ in range(batch_size):
            position_themes = []
            # length = random.choice(lengths)
            # length = random.choices(lengths, weights=[15, 25, 30, 30], k=1)[0]
            length = random.choices(lengths, weights=[5, 15, 40, 40], k=1)[0]
            if torch.rand(1) < 0.8: position_themes.append(length)
            # state_of_game = random.choice(state_of_game_tokens)
            state_of_game = random.choices(state_of_game_tokens, weights=[20, 40, 40], k=1)[0]
            if torch.rand(1) < 0.8: position_themes.append(state_of_game)

            if state_of_game == "endgame" and torch.rand(1) < 0.8:
                position_themes.append(random.choice(endgames))
            
            if torch.rand(1) < 0.2:
                position_themes.append(is_mate)
                if torch.rand(1) < 0.8:
                    i = lengths.index(length) + (1 if length == "veryLong" and torch.rand(1) < 0.5 else 0)
                    position_themes.append(mate_lengths[i])
                if torch.rand(1) < 0.8: position_themes.append(random.choice(types_of_mate))
                n = random.choices([0, 1, 2], weights=[70, 25, 5])[0]
            else:
                if torch.rand(1) < 0.5: position_themes.append(random.choice(winnings))
                n = random.choices([1, 2], weights=[80, 20])[0]

            position_themes.extend(random.sample(other, n))

            themes.append(position_themes)
        
        # ratings = 3000 * torch.ones((batch_size,)) + 300
        ratings = 3000 * torch.rand((batch_size,)) + 300

    return themes, ratings

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]

KNIGHT_OFFSETS = [
    (1, 2), (2, 1), (-1, 2), (-2, 1),
    (1, -2), (2, -1), (-1, -2), (-2, -1)
]
BISHOP_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PAWN_OFFSETS = [
    (0, 1), (0, 2), (1, 1), (-1, 1),
    (0, -1), (0, -2), (1, -1), (-1, -1)
]
KING_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

def sample_move_squares(move_type, allowed_files=None, allowed_ranks=None):
    if allowed_files is None:
        allowed_files = FILES
    if allowed_ranks is None:
        allowed_ranks = RANKS

    f1, r1 = random.choice(allowed_files), random.choice(allowed_ranks)
    f1_idx, r1_idx = FILES.index(f1), RANKS.index(r1)

    valid_targets = []

    if move_type == "knight_move":
        for df, dr in KNIGHT_OFFSETS:
            f2_idx, r2_idx = f1_idx + df, r1_idx + dr
            if 0 <= f2_idx < 8 and 0 <= r2_idx < 8:
                cand_f, cand_r = FILES[f2_idx], RANKS[r2_idx]
                if cand_f in allowed_files and cand_r in allowed_ranks:
                    valid_targets.append((cand_f, cand_r))

    elif move_type == "bishop_move":
        for df, dr in BISHOP_DIRECTIONS:
            k = 1
            while True:
                f2_idx, r2_idx = f1_idx + k * df, r1_idx + k * dr
                if 0 <= f2_idx < 8 and 0 <= r2_idx < 8:
                    cand_f, cand_r = FILES[f2_idx], RANKS[r2_idx]
                    if cand_f in allowed_files and cand_r in allowed_ranks:
                        valid_targets.append((cand_f, cand_r))
                    k += 1
                else:
                    break

    elif move_type == "rook_move":
        for df, dr in ROOK_DIRECTIONS:
            k = 1
            while True:
                f2_idx, r2_idx = f1_idx + k * df, r1_idx + k * dr
                if 0 <= f2_idx < 8 and 0 <= r2_idx < 8:
                    cand_f, cand_r = FILES[f2_idx], RANKS[r2_idx]
                    if cand_f in allowed_files and cand_r in allowed_ranks:
                        valid_targets.append((cand_f, cand_r))
                    k += 1
                else:
                    break

    elif move_type == "pawn_move":
        for df, dr in PAWN_OFFSETS:
            f2_idx, r2_idx = f1_idx + df, r1_idx + dr
            if 0 <= f2_idx < 8 and 0 <= r2_idx < 8:
                cand_f, cand_r = FILES[f2_idx], RANKS[r2_idx]
                if cand_f in allowed_files and cand_r in allowed_ranks:
                    valid_targets.append((cand_f, cand_r))

    elif move_type == "king_move":
        for df, dr in KING_OFFSETS:
            f2_idx, r2_idx = f1_idx + df, r1_idx + dr
            if 0 <= f2_idx < 8 and 0 <= r2_idx < 8:
                cand_f, cand_r = FILES[f2_idx], RANKS[r2_idx]
                if cand_f in allowed_files and cand_r in allowed_ranks:
                    valid_targets.append((cand_f, cand_r))

    if valid_targets:
        f2, r2 = random.choice(valid_targets)
    else:
        f2, r2 = random.choice(allowed_files), random.choice(allowed_ranks)

    return f1, r1, f2, r2

SPECIAL_THEMES = {
    "smotheredMate": {"pattern_types": None},
    "backRankMate": {"pattern_types": ["from_square", "to_file"], "possible_files": FILES, "possible_ranks": RANKS},
    "bodenMate": {"pattern_types": ["from_square", "to_file", "to_rank"], "possible_files": FILES, "possible_ranks": RANKS},
    "hookMate": {"pattern_types": ["from_square", "to_file", "to_rank"], "possible_files": FILES, "possible_ranks": RANKS},
    "doubleBishopMate": {"pattern_types": ["from_square", "to_file", "to_rank"], "possible_files": FILES, "possible_ranks": RANKS},
    "arabianMate": {"pattern_types": ["to_file", "to_rank"], "possible_files": FILES, "possible_ranks": RANKS},
    "anastasiaMate": {"pattern_types": ["from_square", "to_file", "to_rank"], "possible_files": FILES, "possible_ranks": RANKS},
    "promotion": {"pattern_types": ["to_file"], "possible_files": FILES, "possible_ranks": RANKS},
    "underPromotion": {"pattern_types": None},
    "castling": {"pattern_types": None},
    "enpassant": {"pattern_types": ["to_file"], "possible_files": FILES, "possible_ranks": RANKS},
}

DEFAULT = {
    "pattern_types": ["from_square", "to_square", "files", "ranks"],  # "to_file", "to_rank", "from_file", "from_rank", "from_square", "files", "ranks", "full_move"
    "possible_files": FILES,
    "possible_ranks": RANKS,
}

MOVE_TYPES = ["knight_move", "bishop_move", "rook_move", "pawn_move", "king_move"]

def generate_random_moves(batch_size, themes=None):
    moves = []
    # pattern_types = ["from_file", "from_rank", "to_file", "to_rank", "from_square", "to_square"]  # , "full_move"

    for i in range(batch_size):
        sample_themes = themes[i] if (themes is not None and i < len(themes)) else []
        matching_rules = [SPECIAL_THEMES[t] for t in sample_themes if t in SPECIAL_THEMES]

        if matching_rules:
            rule = random.choice(matching_rules) 
        else:
            rule = DEFAULT

        pat_types = rule.get("pattern_types", DEFAULT["pattern_types"])
        if not pat_types:
            moves.append("-----")
            continue

        ptype = random.choice(pat_types)
        allowed_files = rule.get("possible_files") or FILES
        allowed_ranks = rule.get("possible_ranks") or RANKS

        # f1, r1 = random.choice(allowed_files), random.choice(allowed_ranks)
        # f2, r2 = random.choice(allowed_files), random.choice(allowed_ranks)
        # move_type = random.choice(MOVE_TYPES)
        move_type = random.choices(MOVE_TYPES, weights=[25, 25, 25, 12.5, 12.5])[0]
        f1, r1, f2, r2 = sample_move_squares(move_type, allowed_files, allowed_ranks)

        f_from = f1 if ptype in ("from_file", "from_square", "files", "full_move") else "-"
        r_from = r1 if ptype in ("from_rank", "from_square", "ranks", "full_move") else "-"
        f_to = f2 if ptype in ("to_file", "to_square", "files", "full_move") else "-"
        r_to = r2 if ptype in ("to_rank", "to_square", "ranks", "full_move") else "-"

        moves.append(f"{f_from}{r_from}{f_to}{r_to}-")

    return moves

def theme_reward(base_themes, puzzle_themes):
    base_set = set(base_themes)
    puzzle_set = set(puzzle_themes)

    base_state = base_set.intersection(state_of_game_tokens)
    if not base_state.issubset(puzzle_set):
        return False

    base_endgame = base_set.intersection(endgames)
    if not base_endgame.issubset(puzzle_set):
        return False
    
    if "mate" in base_set:
        if "mate" not in puzzle_set:
            return False
            
        base_mate_length = base_set.intersection(mate_lengths)
        if not base_mate_length.issubset(puzzle_set):
            return False
            
        base_mate_type = base_set.intersection(types_of_mate)
        if not base_mate_type.issubset(puzzle_set):
            return False
    else:
        if "mate" in puzzle_set:  # if model generated a checkmate when we did not ask it to do so, return False
            return False
            
    base_other = base_set.intersection(other)
    if not base_other.issubset(puzzle_set):
        return False

    return True

def count_extra_themes(base_themes, puzzle_themes):
    if base_themes is None or puzzle_themes is None:
        return 0
    base_set = set(base_themes)
    puzzle_set = set(puzzle_themes)
    
    base_other = base_set.intersection(other)
    puzzle_other = puzzle_set.intersection(other)
    
    extra_themes = puzzle_other - base_other
    return len(extra_themes)

