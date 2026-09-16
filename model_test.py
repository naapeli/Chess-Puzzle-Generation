from diffusers import DiffusionPipeline
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path = "model_main"
# path = "model_paper"
path = "model_run22"
# path = "naapeli/chess-puzzle-generator"
pipeline = DiffusionPipeline.from_pretrained(
    path,
    trust_remote_code=True,
)
# For exactly the same model as in the paper, use revision="paper":
# pipeline = DiffusionPipeline.from_pretrained(
#     "naapeli/chess-puzzle-generator",
#     revision="paper",
#     trust_remote_code=True,
# )
pipeline.to(device)

themes = pipeline.Theme
schedules = pipeline.Schedule
move_generation_orders = pipeline.MoveGenerationOrder

results = pipeline(
    themes=[themes.pin],
    # best_move = ["???1", "???2", "???3", "???4", "???5", "???6", "???7", "???8"],
    # best_move = ["??a?", "??b?", "??c?", "??d?", "??e?", "??f?", "??g?", "??h?"],
    rating=2000,
    batch_size=4,
    steps=16,
    move_generation_order=move_generation_orders.simultaneous
)
for i, pos in enumerate(results, start=1):
    print(f"Batch {i}: {pos.fen} | move: {pos.move}")
