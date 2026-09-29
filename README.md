# Chess Puzzle Generation

This branch contains the code for publishing our chess puzzle generation models to Huggingface diffusers.

---

## Environment Setup

The environment is specified in `uv.lock`.

To set up the environment, run:

```bash
uv venv
uv sync
```

## Project Structure

- `src/hf_export/`: Contains the tools for publishing the models.
- `model_test.py`: Contains manual tests for each new model before publishing.
- `uv.lock`: Contains the environment used for publishing and testing.

---

## Usage

To convert the torch checkpoint to the Huggingface format, run:

```bash
uv run --directory src python -m hf_export.convert_checkpoint some/path/model.pt some/export/directory
```

To publish the models determined in src/hf_export/model_registry.json, run:

```bash
uv run src/hf_export/publish.py
```
